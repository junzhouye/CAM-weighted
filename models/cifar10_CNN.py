"""
cifar10 model for CAM
因为使用CAM的话，对特征图的尺寸有一定要求。如果特征图的尺寸太小，会丢失太多的位置信息，从而导致生成的热力图无法使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        # inputs (B,C,H,W)
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.calssifier = nn.Linear(512, 10)

    def forward(self, x,get_feature=False):
        feature = self.features(x)
        x = F.adaptive_max_pool2d(feature, (1, 1))
        x = x.view(-1, 512)
        x = self.calssifier(x)
        if get_feature:
            return feature, x
        else:
            return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    model = Net()
    out = model(x,get_feature=True)
    print(out)
