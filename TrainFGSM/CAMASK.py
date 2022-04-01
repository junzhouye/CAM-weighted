"""
CAM加权的方式进行训练
关于 CAM。
tip1:突然想到，传统的做法是直接归一化到[0,1]之间，但是实际上，如果考虑到用在加权的时候，负值也是有意义的。
而其中的中间值反而是没有那么有意义，但是如果是归一化之后，中间的值反而会变成接近0.5，也就变得重要了。
先不管这个。这个只会影响到mode1和mode2
关于 model.eval()
tip2:在生成热力图的过程中使用的model.eval()。但是我们生成对抗样本或是训练过程中都是使用的 model.train()
我们知道，model.eval()会关闭dropout和BN，这里是只有BN。BN会引入随机的噪声，从而防止过拟合。
那么在生成热力图时，我们是否应该使用train()的模式呢？其实个人感觉差异不会很大，但是我觉得，用train()吧。
"""
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


class CAMTrain(BaseTrain):

    def getMask(self, data, target, fgsm_perturbed, mode):
        if mode == 1:
            mask = self.get_cam(data, target)

        if mode == 2:
            mask = 1 - self.get_cam(data, target)

        if mode == 3:
            mask = self.get_diff_cam(data, fgsm_perturbed, target)

        if mode == 4:
            mask = 1 - self.get_diff_cam(data, fgsm_perturbed, target)

        if mode == 5:
            b, c, h, w = data.size()
            cam_weight = np.random.rand(b, c, h, w)
            cam_weight = torch.from_numpy(cam_weight)
            mask = cam_weight.float()

        mask = mask.to(self.device)
        return mask

    @torch.no_grad()
    def get_cam(self, data, target):
        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

        def returnCAM(feature_conv, weight_softmax, class_idx):
            # get input (H,W) , to resize CAM
            size_upsample = (data.size()[-2], data.size()[-1])
            batch_size, nc, h, w = feature_conv.shape
            output_cam = []

            for i in range(batch_size):
                cam = weight_softmax[class_idx[i].item()].dot(feature_conv[i].reshape((nc, h * w)))
                cam = cam.reshape(h, w)
                cam = cv2.resize(cam, size_upsample)
                if cam.max() == cam.min():
                    continue
                # Scale between 0-1
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                output_cam.append(cam)
            return output_cam

        f, outputs = self.model(data, get_feature=True)
        feature = f.cpu().data.numpy()
        _, pred = torch.max(outputs.data, 1)
        output_cam = returnCAM(feature_conv=feature, weight_softmax=weight_softmax, class_idx=target)

        cam = np.array(output_cam)
        cam = np.expand_dims(cam, 1)
        cam = torch.from_numpy(cam)
        # shape of cam is (b,c,h,w) (b,1,32,32)
        return cam

    @torch.no_grad()
    def get_diff_cam(self, data, fgsm_perturbed, target):
        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

        def returnCAM(feature_conv_nature, feature_conv_adv, weight_softmax, class_idx):
            # get input (H,W) , to resize CAM
            size_upsample = (data.size()[-2], data.size()[-1])
            batch_size, nc, h, w = feature_conv_nature.shape
            output_cam = []

            for i in range(batch_size):
                cam1 = weight_softmax[class_idx[i].item()].dot(feature_conv_nature[i].reshape((nc, h * w)))
                cam2 = weight_softmax[class_idx[i].item()].dot(feature_conv_adv[i].reshape((nc, h * w)))
                cam = abs(cam1 - cam2)
                cam = cam.reshape(h, w)
                cam = cv2.resize(cam, size_upsample)
                if cam.max() == cam.min():
                    continue
                # Scale between 0-1
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                output_cam.append(cam)
            return output_cam

        f, outputs = self.model(data, get_feature=True)
        feature = f.cpu().data.numpy()

        adv_data = data.detach().clone() + fgsm_perturbed.detach().clone()
        f, outputs = self.model(adv_data, get_feature=True)
        adv_feature = f.cpu().data.numpy()

        output_cam = returnCAM(feature_conv_nature=feature, feature_conv_adv=adv_feature, weight_softmax=weight_softmax,
                               class_idx=target)

        cam = np.array(output_cam)
        cam = np.expand_dims(cam, 1)
        cam = torch.from_numpy(cam)
        # shape of cam is (b,c,h,w) (b,1,32,32)
        return cam


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
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument("--random_init",type=int,default=0)

    args = parser.parse_args()

    camTrain = CAMTrain(args)
    camTrain.train()
