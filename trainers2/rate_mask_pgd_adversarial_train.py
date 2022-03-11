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
from CAM.random_cam import get_random_rate_binarization_mask


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


def rate_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps,
                                 rate):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        b, c, h, w = data.size()
        data, target = data.to(device), target.to(device)

        # rate mask
        adv_data = get_adv_example(model, data, target, optimizer, device, step_size, epsilon, perturb_steps)
        random_cam_mask = get_random_rate_binarization_mask(b, h, w, rate)
        random_cam_mask = random_cam_mask.to(device)
        adv_weighted_data = data + (adv_data - data) * random_cam_mask
        adv_weighted_data = adv_weighted_data.clone().detach()

        adv_out = model(adv_weighted_data)
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
