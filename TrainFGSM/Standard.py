import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from data.cifar10 import cifar10
from models.cifar10_resnet import ResNet18,ResNet50
from models.cifar_vgg import VGG
import torch.nn.functional as F
import torch.optim as optim


class Standard:
    def __init__(self, model,savename, args):
        self.device = torch.device(args.gpu)
        self.model = model.to(self.device)
        self.train_loader = cifar10(dataRoot=args.dataroot, batchsize=128, train=True)
        self.test_loader = cifar10(dataRoot=args.dataroot, batchsize=128, train=False)
        self.lr = args.lr
        self.savename = savename
        self.epochs = args.epochs
        self.root = args.saveroot
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
            # Train
            output = self.model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='train epoch')
    parser.add_argument('--gpu', type=int, default=0, help='disables CUDA training')
    parser.add_argument('--dataroot', type=str, default="../data/dataset", help='data root')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--saveroot', type=str, default="./pth/final")
    parser.add_argument('--savename', type=str, default="default.pth")

    args = parser.parse_args()

    model = ResNet18()
    # model = VGG("VGG16")
    Train = Standard(model,args)
    Train.train()