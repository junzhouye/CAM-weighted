import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import cv2


def get_random_rate_binarization_mask(b, h, w, rate=0):
    """return (b,1,h,w) random mask"""
    assert 0 <= rate <= 1
    if rate == 0:
        return torch.zeros((b, 1, h, w))
    if rate == 1:
        return torch.ones((b, 1, h, w))
    # 第一个做法。torch.rand生成[0,1)均匀分布的随机数。所以生成随机数，然后使用rate进行分割，基本上就是对应的rate了。
    # 感觉够用了
    cam = torch.rand(b, 1, h, w)
    a = torch.zeros_like(cam) + (1 - rate)

    cam = cam > a
    return cam


if __name__ == "__main__":
    a = random_rate_binarization_mask(8, 32, 32, 0.6)
    print(a)
    print(a.sum())
    print(a.sum() / (8 * 32 * 32))
