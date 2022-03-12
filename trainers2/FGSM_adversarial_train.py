from __future__ import print_function
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


# https://github.com/locuslab/fast_adversarial
# 首先添加随机扰动,然后再进行FGSM攻击
def get_fgsm_adv_example(model,
                         x_natural,
                         y,
                         optimizer,
                         device,
                         epsilon=8 / 255,
                         alpha=10 / 255):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    delta = torch.zeros_like(x_natural).to(device)
    # 改成均匀分布的噪声
    delta.uniform_(-epsilon, epsilon)

    x_adv = x_natural.detach() + delta.detach()
    x_adv = torch.clamp(x_adv, 0, 1)

    x_adv.requires_grad_()
    with torch.enable_grad():
        loss = criterion(model(x_adv), y)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    model.train()
    return x_adv


def fgsm_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, epsilon, alpha):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        adv_data = get_fgsm_adv_example(model, data, target, optimizer, device, epsilon, alpha)
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


def cam_weight_fgsm_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, epsilon, alpha,
                                            mode=1, is_bool=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        adv_data = get_fgsm_adv_example(model, data, target, optimizer, device, epsilon, alpha)

        if mode == 1:
            cam_weight = get_cam(model=model, inputs=data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight

        if mode == 2:
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
            b,c,h,w = data.size()
            cam_weight = get_random_cam_weight(b, h, w)
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


# def cam_weight_fgsm_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, epsilon, alpha,
#                                             mode=1):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#
#         adv_data = get_fgsm_adv_example(model, data, target, optimizer, device, epsilon, alpha)
#         if mode == 1:
#             # 根据对抗样本预测激活最大的区域，认为这部分是扰动主要生效的区域，但是忽略了除了激活，不激活也是扰动的作用之一。
#             cam_weight = get_cam(model=model, inputs=adv_data)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 2:
#             # 没啥意义
#             cam_weight = get_cam(model=model, inputs=adv_data)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 3:
#             # target是adv_input的预测结果
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(adv_data)
#                 _, predicted = torch.max(outputs.data, 1)
#             cam_weight = get_cam(model=model, inputs=data, target=predicted)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#             model.train()
#         elif mode == 4:
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(adv_data)
#                 _, predicted = torch.max(outputs.data, 1)
#             cam_weight = get_cam(model=model, inputs=data, target=predicted)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#             model.train()
#         elif mode == 5:
#             cam_weight = get_cam(model=model, inputs=adv_data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 6:
#             cam_weight = get_cam(model=model, inputs=adv_data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 7:
#             cam_weight = get_cam(model=model, inputs=data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 8:
#             cam_weight = get_cam(model=model, inputs=data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 9:
#             # mode 9 开始 使用类别激活差异作为cam weighted的依据
#             # 以 ground true label 的激活差异作为指导
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 10:
#             # 以 预测 label 的激活差异作为指导
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * cam_weight
#         elif mode == 11:
#             # 综合了ground true label 和预测 label 的激活差异
#             cam_weight = get_cam_diff_plus(model=model, natural_data=data, adv_data=adv_data, target=target)
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
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 14:
#             # mode 10
#             cam_weight = get_cam_diff(model=model, natural_data=data, adv_data=adv_data)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
#         elif mode == 15:
#             # mode 11
#             cam_weight = get_cam_diff_plus(model=model, natural_data=data, adv_data=adv_data, target=target)
#             cam_weight = cam_weight.to(device)
#             weight_data = data + (adv_data - data) * (1 - cam_weight)
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
    pass
