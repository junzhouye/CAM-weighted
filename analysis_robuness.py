"""
混淆矩阵来说，这种方法并没有什么改善class-wise的效果
"""
from Analysis.accuracy_test import *
from models.cifar10_CNN import Net
from models.cifar10_resnet import ResNet18
import torch
from data.cifar10 import cifar10
import argparse
import os
from attacks.PGD import *
from attacks.FGSM import *
from utils.log import Log, LogCSV
from CAM.base_CAM import get_cam
from CAM.CAM_utils import cam_binarization


def model_robustness_accuracy_test_FGSM(root, pth_name, testloader, epsilon_list: list, device, Log=None):
    # set epsilon_list = [0,0.01,0.02,0.03,0.04,0.05]
    pretrain_model_path = os.path.join(root, pth_name)
    model = ResNet18()
    model.load_state_dict(torch.load(pretrain_model_path, map_location="cuda:0"))
    model.to(device)
    model.eval()

    acc_list = []
    for epsilon in epsilon_list:
        acc = fgsm_test(model=model, device=device, test_loader=testloader, epsilon=epsilon)
        acc_list.append(acc)

    if Log:
        Log([pth_name])
        # Log(str(epsilon_list))
        Log(acc_list)


def model_robustness_accuracy_test_PGD(root, pth_name, testloader, epsilon_list: list, device, Log=None):
    # set epsilon_list: [0.0,0.01,0.02,0.03,0.04,0.05]
    pretrain_model_path = os.path.join(root, pth_name)
    model = ResNet18()
    model.load_state_dict(torch.load(pretrain_model_path, map_location="cuda:0"))
    model.to(device)
    model.eval()
    acc_list = []
    for epsilon in epsilon_list:
        iters = int(epsilon / 0.003 + 1)
        acc = pgd_test(model=model, test_loader=testloader, device=device, eps=epsilon, alpha=0.003, iters=iters)
        acc_list.append(acc)

    if Log:
        Log([pth_name])
        # Log(str(epsilon_list))
        Log(acc_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device(args.gpu)

    root = "./pth/newRes"

    dataRoot = "./data/dataset"
    # 使用测试集
    testloader = cifar10(dataRoot=dataRoot, batchsize=128, train=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    log_fgsm = LogCSV("./result_image", "V2_2_FGSM_robustness_accuracy.csv")
    log_pgd = LogCSV("./result_image", "v2_2_PGD_robustness_accuracy.csv")

    a = [
        # 'v2_cam_fgsm_adversarial_mode1_epoch50.pth',
        #  'v2_cam_fgsm_adversarial_mode2_epoch50.pth',
        #  'v2_cam_fgsm_adversarial_mode3_epoch50.pth',
        #  'v2_cam_fgsm_adversarial_mode4_epoch50.pth',
        #  'v2_cam_fgsm_adversarial_mode5_epoch50.pth',

        # 'v2_cam_pgd_adversarial_mode1_epoch50.pth',
        # 'v2_cam_pgd_adversarial_mode2_epoch50.pth',
        # 'v2_cam_pgd_adversarial_mode3_epoch50.pth',
        # 'v2_cam_pgd_adversarial_mode4_epoch50.pth',
        # 'v2_cam_pgd_adversarial_mode5_epoch50.pth',
        #
        # 'v2_cam_pgd_bool_adversarial_mode1_epoch50.pth',
        # 'v2_cam_pgd_bool_adversarial_mode2_epoch50.pth',
        # 'v2_cam_pgd_bool_adversarial_mode3_epoch50.pth',
        # 'v2_cam_pgd_bool_adversarial_mode4_epoch50.pth',
        # 'v2_cam_pgd_bool_adversarial_mode5_epoch50.pth',

        # 'v2_fgsm_adversarial_epoch50.pth',
        # 'v2_pgd_adversarial_epoch50.pth',
        # 'v2_random_model_epoch50.pth',
        # 'v2_standard_model_epoch50.pth',
        # 'v3_cam_bool_pgd_adversarial_mode_0_epoch50.pth',
        # 'v3_cam_bool_pgd_adversarial_mode_1_epoch50.pth'
        # "v2_cam_fgsm_bool_adversarial_mode1_epoch50.pth",
        # "v2_cam_fgsm_bool_adversarial_mode2_epoch50.pth",
        # "v2_cam_fgsm_bool_adversarial_mode3_epoch50.pth",
        # "v2_cam_fgsm_bool_adversarial_mode4_epoch50.pth",
        # "v2_cam_fgsm_bool_adversarial_mode5_epoch50.pth"
        # 'v3_rate_fgsm_rate0.2_epoch50.pth',
        # 'v3_rate_fgsm_rate0.4_epoch50.pth',
        # 'v3_rate_fgsm_rate0.6_epoch50.pth',
        # 'v3_rate_fgsm_rate0.8_epoch50.pth',
        # 'v3_rate_pgd_rate0.2_epoch50.pth',
        # 'v3_rate_pgd_rate0.4_epoch50.pth',
        # 'v3_rate_pgd_rate0.6_epoch50.pth',
        # 'v3_rate_pgd_rate0.8_epoch50.pth',
        'v3_rate_fgsm_dynamic_epoch50.pth',
        # 'v3_rate_fgsm_random_epoch50.pth',
        'v3_rate_pgd_dynamic_epoch50.pth',
        # 'v3_rate_pgd_random_epoch50.pth',
        # "v3_guide_cam_fgsm_bool_mode1_epoch50.pth",
        # "v3_guide_cam_fgsm_bool_mode2_epoch50.pth",
        # "v3_guide_cam_fgsm_bool_mode3_epoch50.pth",
        # "v3_guide_cam_fgsm_bool_mode4_epoch50.pth",
    ]

    epsilon_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    for pth_name in a:
        model_robustness_accuracy_test_FGSM(root, pth_name, testloader, epsilon_list, device, log_fgsm)
        model_robustness_accuracy_test_PGD(root, pth_name, testloader, epsilon_list, device, log_pgd)
