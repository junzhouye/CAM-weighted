import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_image(image, save_path=None):
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(255 * image)
    plt.imshow(image)
    plt.show()
    if save_path:
        plt.savefig(save_path)


def show_cam(cam: torch.Tensor, save_path=None):
    cam = cam.cpu().numpy()
    cam = np.transpose(cam, (1, 2, 0))
    cam_gray = np.uint8(255 * cam)
    img_color = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
    plt.imshow(img_color)
    plt.show()
    if save_path:
        plt.savefig(save_path)


def show_cam_image(cam, image, save_path=None):
    cam = cam.cpu().numpy()
    cam = np.transpose(cam, (1, 2, 0))
    cam_gray = np.uint8(255 * cam)
    img_color = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)

    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(255 * image)

    cam_image = image * 0.5 + img_color * 0.3
    cam_image = np.uint8(cam_image)

    plt.imshow(cam_image)
    plt.show()
    if save_path:
        plt.savefig(save_path)




if __name__ == "__main__":
    from CAM.base_CAM import get_cam,get_cam_diff
    from trainers.standard_train import *
    from data.cifar10 import cifar10
    from models.cifar10_CNN import Net
    import argparse
    from attacks.FGSM import fgsm_attack
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--gpu', action='store_true', default=0,
                        help='disables CUDA training')

    args = parser.parse_args()

    device = torch.device(args.gpu)

    # setup data loader
    test_loader = cifar10(dataRoot="../data/dataset", batchsize=1, train=False)
    pretrain_model_path = "../pth/standard_model-epoch30.pth"
    model = Net().to(device)
    model.load_state_dict(torch.load(pretrain_model_path))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        outputs = model(data)
        model.zero_grad()
        loss = F.nll_loss(outputs, target)
        loss.backward()
        data_grad = data.grad.data


        perturbed_data = fgsm_attack(data, 0.1, data_grad, min_val=0, max_val=1)
        final_outputs = model(perturbed_data)
        _, final_predicted = torch.max(final_outputs.data, 1)
        cam = get_cam_diff(model,data,perturbed_data,target=final_predicted)
        show_cam(cam[0])

        cam2 = get_cam(model,data,target=final_predicted)
        show_cam(cam2[0])

        cam3 = get_cam(model,perturbed_data,target=final_predicted)
        show_cam(cam3[0])

        show_image(image=data[0])


        break
