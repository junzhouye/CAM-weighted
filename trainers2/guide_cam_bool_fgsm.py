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


def guide_cam_weight_fgsm_adversarial_train_epoch(model,guide_model, device, train_loader, optimizer, epoch, epsilon, alpha,
                                            mode=1, is_bool=False):
    model.train()
    criterion = nn.CrossEntropyLoss()

    guide_model.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        adv_data = get_fgsm_adv_example(model, data, target, optimizer, device, epsilon, alpha)

        if mode == 1:
            cam_weight = get_cam(model=guide_model, inputs=data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight

        if mode == 2:
            cam_weight = get_cam(model=guide_model, inputs=data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * (1 - cam_weight)

        if mode == 3:
            cam_weight = get_cam_diff(model=guide_model, natural_data=data, adv_data=adv_data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * cam_weight

        if mode == 4:
            cam_weight = get_cam_diff(model=guide_model, natural_data=data, adv_data=adv_data, target=target)
            if is_bool:
                cam_weight = cam_binarization(cam_weight)
            cam_weight = cam_weight.to(device)
            weight_data = data + (adv_data - data) * (1 - cam_weight)

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
