"""
首先生成0-1CAM，然后使用这个CAM指导对抗样本的生成。
"""

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
from CAM.CAM_utils import cam_binarization
import cv2


# 这里是首先生成mask,然后生成对抗样本
# 这里使用了三种模式，分别对应前景，背景和随机。至于adversarial-train中的mode3,4则被阉割，因为这两种模式需要使用对抗样本的信息
# emmm，就先这样
def get_bool_adv_example(model,
                         x_natural,
                         y,
                         optimizer,
                         device,
                         step_size=0.003,
                         epsilon=0.031,
                         perturb_steps=10,
                         mode=0):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_size = len(x_natural)
    if mode == 0:
        cam = get_cam(model=model, inputs=x_natural, target=y)
        cam = cam_binarization(cam)

    if mode == 1:
        cam = get_cam(model=model, inputs=x_natural, target=y)
        cam = cam_binarization(cam)
        cam = 1 - cam

    if mode == 2:
        b, c,h, w = x_natural.size()
        cam_weight = get_random_cam_weight(b,h, w)
        cam = cam_binarization(cam_weight)

    cam = cam.to(device)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * cam * torch.randn(x_natural.shape).to(device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        # 这里将扰动也乘上一个特定的mask
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach()) * cam
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    model.train()
    return x_adv


def bool_pgd_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps,
                                     mode):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        adv_data = get_bool_adv_example(model, data, target, optimizer, device, step_size, epsilon, perturb_steps, mode)
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
    pass
