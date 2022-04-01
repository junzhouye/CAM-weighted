import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from data.cifar10 import cifar10
from models.cifar10_resnet import ResNet18
import torch.nn.functional as F
import torch.optim as optim
from TrainFGSM.base import BaseTrain
import cv2


class Ladder(BaseTrain):
    def getMask(self, data, target, fgsm_perturbed, mode):
        grad = self.getFGM(data, target)
        sort_grad = torch.abs(grad.clone()).view(-1)
        len_sort_grad = sort_grad.size()[0]
        sorted_grad = torch.sort(sort_grad)
        mask = torch.zeros_like(data)
        for i in range(1, 11):
            rate_v = sorted_grad.values[int(len_sort_grad * i / 10) - 1]
            mask_add = (torch.abs(grad.clone()) > (torch.zeros_like(grad) + rate_v))
            mask += mask_add * 0.1
        mask = mask.to(self.device)
        return mask

    def getFGM(self, data, target):
        criterion = nn.CrossEntropyLoss()
        data.requires_grad = True
        self.model.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, target)
        loss.backward()
        data_grad = data.grad.data
        self.model.zero_grad()
        return data_grad


class Ladder2(BaseTrain):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='train epoch')
    parser.add_argument('--gpu', type=int, default=0, help='disables CUDA training')
    parser.add_argument('--epsilon', type=float, default=8 / 255, help='epsilon in FAST IS BETTER THAN FREE')
    parser.add_argument('--alpha', type=float, default=10 / 255, help='alpha in FAST IS BETTER THAN FREE')
    parser.add_argument('--dataroot', type=str, default="../data/dataset", help='data root')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--saveroot', type=str, default="./pth/final")
    parser.add_argument('--savename', type=str, default="default.pth")
    parser.add_argument('--mode', type=float, default=0.1)
    parser.add_argument("--random_init", type=int, default=0)

    args = parser.parse_args()

    camTrain = Ladder(args)
    camTrain.train()
