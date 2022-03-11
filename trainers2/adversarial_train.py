import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from models.cifar10_resnet import *
from trainers.trade import trades_loss
from CAM.base_CAM import get_cam, get_cam_diff, get_cam_diff_plus, get_random_cam_weight
import cv2
from CAM.CAM_utils import cam_binarization


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def get_adv_example(model,
                    x_natural,
                    y,
                    optimizer,
                    device,
                    step_size=0.003,
                    epsilon=0.031,
                    perturb_steps=10,
                    distance='l_inf'):
    # PGD 生成对抗样本
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = criterion(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion(model(adv), y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    model.train()
    return x_adv


def adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        adv_data = get_adv_example(model, data, target, optimizer, device, step_size, epsilon, perturb_steps)
        adv_out = model(adv_data)
        loss = criterion(adv_out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def cam_weight_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps,
                                       mode=1, is_bool=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        adv_data = get_adv_example(model, data, target, optimizer, device, step_size, epsilon, perturb_steps)

        if mode == 0:
            # 正常对抗训练
            weight_data = adv_data

        if mode == 1:
            # 相当于偏重于在前景生成扰动。
            cam_weight = get_cam(model=model, inputs=data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight

        if mode == 2:
            # 相当于偏重于在背景生成扰动。
            cam_weight = get_cam(model=model, inputs=data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * (1 - cam_weight)

        if mode == 3:
            cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight

        if mode == 4:
            cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * (1 - cam_weight)

        if mode == 5:
            # 随机
            b,c,h,w = data.size()
            cam_weight = get_random_cam_weight(b,h,w)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight


        adv_out = model(weight_data)
        loss = criterion(adv_out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 历史版本 新版本精简模式为5种
# def cam_weight_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps,
#                                        mode=1,is_bool=False):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#
#         adv_data = get_adv_example(model, data, target, optimizer, device, step_size, epsilon, perturb_steps)
#         if mode == 1:
#             # 根据对抗样本预测激活最大的区域，认为这部分是扰动主要生效的区域，但是忽略了除了激活，不激活也是扰动的作用之一。
#             cam_weight = get_cam(model=model, inputs=adv_data)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 2:
#             # 没啥意义
#             cam_weight = get_cam(model=model, inputs=adv_data)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 3:
#             # target是adv_input的预测结果
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(adv_data)
#                 _, predicted = torch.max(outputs.data, 1)
#             cam_weight = get_cam(model=model, inputs=data, target=predicted)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#             model.train()
#         elif mode == 4:
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(adv_data)
#                 _, predicted = torch.max(outputs.data, 1)
#             cam_weight = get_cam(model=model, inputs=data, target=predicted)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#             model.train()
#         elif mode == 5:
#             cam_weight = get_cam(model=model, inputs=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 6:
#             cam_weight = get_cam(model=model, inputs=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 7:
#             cam_weight = get_cam(model=model, inputs=data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 8:
#             cam_weight = get_cam(model=model, inputs=data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 9:
#             # mode 9 开始 使用类别激活差异作为cam weighted的依据
#             # 以 ground true label 的激活差异作为指导
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 10:
#             # 以 预测 label 的激活差异作为指导
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 11:
#             # 综合了ground true label 和预测 label 的激活差异
#             cam_weight = get_cam_diff_plus(model=model, natural_data=data, adv_data=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 12:
#             b = data.size()[0]
#             cam_weight = get_random_cam_weight(b, 32, 32)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 13:
#             # mode 9
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 14:
#             # mode 10
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 15:
#             # mode 11
#             cam_weight = get_cam_diff_plus(model=model, natural_data=data, adv_data=adv_data, target=target)
#             if is_bool:
#                 cam_weight = cam_binarization(cam_weight)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#
#         else:
#             print("Not choose mode!!!")
#             return
#
#         adv_out = model(weight_data)
#         loss = criterion(adv_out, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # print progress
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            train_loss += F.cross_entropy(outputs, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, total, 100. * correct / total))
    training_accuracy = correct / total
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            train_loss += F.cross_entropy(outputs, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    train_loss /= len(test_loader.dataset)
    print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, total, 100. * correct / total))
    training_accuracy = correct / total
    return train_loss, training_accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--gpu', type=int, default=0,
                        help='disables CUDA training')
    parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                        help='save frequency')

    args = parser.parse_args()
    ####################################################################################################################
    step_size = 0.003
    epsilon = 0.031
    perturb_steps = 10
    ####################################################################################################################
    # settings
    model_dir = "../pth"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.manual_seed(9)
    device = torch.device(args.gpu)

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    # init model, ResNet18() can be also used here for training

    model = ResNet18().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4)

    for epoch in range(1, args.epochs + 1):

        # adversarial training
        adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'adversarial_model-epoch{}.pt'.format(epoch)))
