import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def show_image(image, save_path=None):
    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(255 * image)
    plt.imshow(image)
    if save_path:
        plt.savefig(save_path)
    plt.show()



def show_cam(cam: torch.Tensor, save_path=None):
    cam = cam.cpu().numpy()
    cam = np.transpose(cam, (1, 2, 0))
    cam_gray = np.uint8(255 * cam)
    img_color = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
    plt.imshow(img_color)
    if save_path:
        plt.savefig(save_path)
    plt.show()



def show_cam_image(cam, image, save_path=None):
    cam = cam.cpu().numpy()
    cam = np.transpose(cam, (1, 2, 0))
    cam_gray = np.uint8(255 * cam)
    img_color = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)

    image = image.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.uint8(255 * image)

    cam_image = image * 0.5 + img_color * 0.3
    cam_image = np.uint8(cam_image)

    plt.imshow(cam_image)
    if save_path:
        plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    from CAM.base_CAM import get_cam, get_cam_diff
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
    test_loader = cifar10(dataRoot="../data/dataset", batchsize=8, train=False)
    pretrain_model_path = "../pth/standard_model-epoch30.pth"
    model = Net().to(device)
    model.load_state_dict(torch.load(pretrain_model_path))

    save_path = "../result_image/cam_show"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        outputs = model(data)
        model.zero_grad()
        loss = F.nll_loss(outputs, target)
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, 0.03, data_grad, min_val=0, max_val=1)
        final_outputs = model(perturbed_data)
        _, final_predicted = torch.max(final_outputs.data, 1)

        cam_nature_target = get_cam(model=model, inputs=data, target=target)

        # for i in range(8):
        #     if target[i] != final_predicted[i]:
        #         nature_image_name = "{}-".format(i) + "real_class_{}".format(
        #             target[i].item()) + "nature" + ".jpg"
        #         show_image(image=data[i],save_path=os.path.join(save_path,nature_image_name))
        #
        #         adv_image_name = "{}-".format(i) + "per_class_{}".format(
        #             final_predicted[i].item()) + "adv" + ".jpg"
        #         show_image(image=perturbed_data[i],save_path=os.path.join(save_path,adv_image_name))

        for i in range(8):
            if target[i] != final_predicted[i]:
                save_name = "{}-".format(i) + "cam_nature_target" + ".jpg"
                # show_cam_image(cam=cam_nature_target[i], image=data[i],save_path=os.path.join(save_path,save_name))
                show_cam(cam=cam_nature_target[i])
        # cam_adv_target = get_cam(model=model, inputs=perturbed_data, target=target)
        # for i in range(8):
        #     if target[i] != final_predicted[i]:
        #         save_name = "{}-".format(i) + "cam_fgsm_adv_target" + ".jpg"
        #         show_cam_image(cam=cam_adv_target[i], image=perturbed_data[i],save_path=os.path.join(save_path,save_name))
        #
        # cam_nature_per = get_cam(model=model, inputs=data, target=final_predicted)
        # for i in range(8):
        #     if target[i] != final_predicted[i]:
        #         save_name = "{}-".format(i) + "cam_nature_per" + ".jpg"
        #         show_cam_image(cam=cam_nature_per[i], image=data[i],save_path=os.path.join(save_path,save_name))
        #
        # cam_adv_per = get_cam(model=model, inputs=perturbed_data, target=final_predicted)
        # for i in range(8):
        #     if target[i] != final_predicted[i]:
        #         save_name = "{}-".format(i) + "cam_adv_per" + ".jpg"
        #         show_cam_image(cam=cam_adv_per[i], image=perturbed_data[i],save_path=os.path.join(save_path,save_name))

        break
