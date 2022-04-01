"""
这里的所有的MASKED-ADVERSARIAL-SAMPLE都是这样计算的:
这个是针对在random_init之后的图像上进行获得mask操作的图像的
masked_adversarial_sample = original_data + clamp(random_init + MASK * epsilon)
之前的计算是直接在整个扰动——包括随机扰动上进行计算，这个可能是不科学的。因为随机的扰动的加权会破坏计算梯度时候的情况。

tip1:在计算 FGSM sample 的时候，需要在model.train()的模式下。因为这样模型会使用dropout以及BN。虽然这里是只有BN。但是BN的使用其实引入了噪声
起到了防止过拟合的作用。train_loader是shuffle的，随机的。BN的使用会起到正则的作用。
ps:model.train() model.eval()其实都会使用BN层。
但是前者是在一个batch上计算均值方差，后者则是全局。从防止过拟合的角度来看，发生的没有BN精度下降严重也就非常合理了。
而添加了mask提升了性能，也就是说明了mask起到了防止过拟合的作用。
但是FAST IS BETTER THAN FREE一文，没有提到这个点。我看了一下有些其他的论文也有生成FGSM也是在model.eval()的情况下。
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


class BaseTrain:
    def __init__(self, args):
        self.device = torch.device(args.gpu)
        self.model = ResNet18().to(self.device)
        self.train_loader = cifar10(dataRoot=args.dataroot, batchsize=128, train=True)
        self.test_loader = cifar10(dataRoot=args.dataroot, batchsize=128, train=False)
        self.lr = args.lr
        self.epsilon = args.epsilon
        self.epochs = args.epochs
        self.savename = args.savename
        self.mode = args.mode
        self.root = args.saveroot
        self.random_init = (args.random_init == 1)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-4)
        for epoch in range(1, self.epochs):
            self.adjust_lr(optimizer=optimizer, epoch=epoch)
            self.train_epoch(optimizer, epoch)
            self.eval_test()

        torch.save(self.model.state_dict(), os.path.join(self.root, self.savename))

    def train_epoch(self, optimizer, epoch):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.random_init:
                # add random init
                delta = torch.zeros_like(data).to(self.device)
                delta.uniform_(-self.epsilon, self.epsilon)
                data_init = data.detach() + delta.detach()
                data_init = torch.clamp(data_init, 0, 1)
            else:
                data_init = data.data

            # get MASK_FGSM Adversarial Sample
            fgsm_perturbed = self.getFGSM(data_init, target)
            mask = self.getMask(data_init, target, fgsm_perturbed, self.mode)
            data_input = data_init + mask * fgsm_perturbed
            data_input = torch.clamp(data_input, data.detach().clone()-self.epsilon, data.detach().clone()+self.epsilon)
            data_input = torch.clamp(data_input, 0, 1).detach().to(self.device)
            # Train
            output = self.model(data_input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

    def getFGSM(self, data, target):
        criterion = nn.CrossEntropyLoss()
        data.requires_grad = True
        self.model.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, target)
        loss.backward()
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        fgsm_perturbed = self.epsilon * sign_data_grad
        self.model.zero_grad()
        return fgsm_perturbed

    def getMask(self, data, target,fgsm_perturbed, mode):
        pass

    @torch.no_grad()
    def eval_test(self):
        self.model.eval()
        train_loss = 0
        total = 0
        correct = 0

        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            train_loss += F.cross_entropy(outputs, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        train_loss /= len(self.test_loader.dataset)
        print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, total, 100. * correct / total))
        training_accuracy = correct / total
        return train_loss, training_accuracy

    def adjust_lr(self, optimizer, epoch):
        lr = self.lr
        if epoch >= 30:
            lr /= 10
        if epoch >= 40:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

