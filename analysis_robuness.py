"""
混淆矩阵来说，这种方法并没有什么改善class-wise的效果
"""
from Analysis.accuracy_test import *
from models.cifar10_CNN import Net
import torch
from data.cifar10 import cifar10
import argparse
import os
from attacks.PGD import *
from attacks.FGSM import *
from utils.log import Log,LogCSV


def model_robustness_accuracy_test_FGSM(root, pth_name, testloader, epsilon_list:list, device, Log):
    # set epsilon_list = [0,0.01,0.02,0.03,0.04,0.05]
    pretrain_model_path = os.path.join(root, pth_name)
    model = Net()
    model.load_state_dict(torch.load(pretrain_model_path,map_location="cuda:0"))
    model.to(device)
    model.eval()

    acc_list = []
    for epsilon in epsilon_list:
        acc = fgsm_test(model=model,device=device,test_loader=testloader,epsilon=epsilon)
        acc_list.append(acc)

    if Log:
        Log([pth_name])
        # Log(str(epsilon_list))
        Log(acc_list)


def model_robustness_accuracy_test_PGD(root, pth_name, testloader,epsilon_list:list, device, Log):
    # set epsilon_list: [0.0,0.01,0.02,0.03,0.04,0.05]
    pretrain_model_path = os.path.join(root, pth_name)
    model = Net()
    model.load_state_dict(torch.load(pretrain_model_path,map_location="cuda:0"))
    model.to(device)
    model.eval()
    acc_list = []
    for epsilon in epsilon_list:
        iters = int(epsilon/0.003 + 1)
        acc = pgd_test(model=model,test_loader=testloader,device=device,eps=epsilon,alpha=0.003,iters=iters)
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

    root = "./pth"

    dataRoot = "./data/dataset"
    # 使用测试集
    testloader = cifar10(dataRoot=dataRoot, batchsize=128, train=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    log_fgsm = LogCSV("./result_image", "FGSM_robustness_accuracy.csv")
    log_pgd = LogCSV("./result_image", "PGD_robustness_accuracy.csv")

    # a = ['adversarial_model_epoch10.pth',
    #      'adversarial_model_epoch20.pth',
    #      'adversarial_model_epoch30.pth',
    #      'adversarial_model_epoch40.pth',
    #      'adversarial_model_epoch50.pth',
    #      'cam_weighted_adversarial_model_mode1_epoch10.pth',
    #      'cam_weighted_adversarial_model_mode1_epoch20.pth',
    #      'cam_weighted_adversarial_model_mode1_epoch30.pth',
    #      'cam_weighted_adversarial_model_mode1_epoch40.pth',
    #      'cam_weighted_adversarial_model_mode1_epoch50.pth',
    #      'cam_weighted_trade_model_mode_1_beta_6_epoch10.pth',
    #      'cam_weighted_trade_model_mode_1_beta_6_epoch20.pth',
    #      'cam_weighted_trade_model_mode_1_beta_6_epoch30.pth',
    #      'cam_weighted_trade_model_mode_1_beta_6_epoch40.pth',
    #      'cam_weighted_trade_model_mode_1_beta_6_epoch50.pth',
    #      'standard_model-epoch10.pth',
    #      'standard_model-epoch20.pth',
    #      'standard_model-epoch30.pth',
    #      'standard_model-epoch40.pth',
    #      'standard_model-epoch50.pth',
    #      'trade_model_beta_6-epoch10.pth',
    #      'trade_model_beta_6-epoch20.pth',
    #      'trade_model_beta_6-epoch30.pth',
    #      'trade_model_beta_6-epoch40.pth',
    #      'trade_model_beta_6-epoch50.pth'
    #      ]
    a = []
    epsilon_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    for pth_name in a:
        # model_robustness_accuracy_test_FGSM(root, pth_name,testloader,epsilon_list, device, log_fgsm)
        model_robustness_accuracy_test_PGD(root, pth_name,testloader,epsilon_list, device, log_pgd)
