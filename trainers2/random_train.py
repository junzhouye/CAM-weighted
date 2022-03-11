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


# https://github.com/locuslab/fast_adversarial
# 首先添加随机扰动,然后再进行FGSM攻击
def get_random_example(x_natural, device, epsilon=8 / 255):

    x_adv = x_natural.detach() + epsilon * torch.randn(x_natural.shape).to(device).detach()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


def random_per_train_epoch(model, device, train_loader, optimizer, epoch,epsilon):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        adv_data = get_random_example(data,device,epsilon)
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
    trainset = torchvision.datasets.CIFAR10(root='../data/dataset', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='../data/dataset', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    # init model, ResNet18() can be also used here for training

    model = ResNet18().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4)

    for epoch in range(1, args.epochs + 1):

        # adversarial training
        random_per_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'adversarial_model-epoch{}.pt'.format(epoch)))
