import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def cifar10(dataRoot, batchsize=128, train=True):
    if train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root=dataRoot, train=True,
                                                download=True, transform=train_transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                 shuffle=True, num_workers=0)
    if not train:
        test_transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root=dataRoot, train=False,
                                               download=True, transform=test_transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                                 shuffle=False, num_workers=0)

    return dataloader


def imshow(img):
    # input shape : [N,C,H,W]
    img = torchvision.utils.make_grid(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_image(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


if __name__ == "__main__":
    dataRoot = "./dataset"
    testlodaer = cifar10(dataRoot=dataRoot, batchsize=1,train=False)
    dataiter = iter(testlodaer)
    images, labels = dataiter.next()
    print(labels)
    print(images)
    imshow(images)
