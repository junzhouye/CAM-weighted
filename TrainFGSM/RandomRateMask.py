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


class RandomRateTrain(BaseTrain):
    @torch.no_grad()
    def getMask(self, data, target, fgsm_perturbed, mode):
        b, c, h, w = data.size()
        assert 0.0 <= mode <= 1.0
        if mode == 0:
            return 0
        if mode == 1:
            return 1
        # torch.rand生成[0,1)均匀分布的随机数。
        cam = torch.rand(b, c, h, w)
        a = torch.zeros_like(cam) + (1 - mode)
        mask = cam > a
        mask = mask.to(self.device)
        return mask


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

    camTrain = RandomRateTrain(args)
    camTrain.train()
