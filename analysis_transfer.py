# 评估对抗样本的迁移性

from Analysis.accuracy_test import *
from models.cifar10_CNN import Net
import torch
from data.cifar10 import cifar10
import argparse
import os
from attacks.PGD import *
from attacks.FGSM import *
from utils.log import Log, LogCSV
from models.cifar10_resnet import *

   
def evl_FGSM_transfer(generator, victim, test_loader, device, epsilon):
    generator.to(device)
    generator.eval()
    victim.to(device)
    victim.eval()

    total = 0
    generator_acc = 0
    victim_acc = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        # get perturbed data by generator
        perturbed_data = get_fgsm_adversarial_sample(model=generator, device=device, data=data, target=target,
                                                     epsilon=epsilon)
        perturbed_data.to(device)

        # generator accuracy
        generator_output = generator(perturbed_data)
        _, generator_pred = torch.max(generator_output.data, 1)
        generator_acc += (generator_pred == target).sum().item()

        # victim accuracy
        victim_output = victim(perturbed_data)
        _, victim_pred = torch.max(victim_output.data, 1)
        victim_acc += (victim_pred == target).sum().item()

    generator_final_acc = generator_acc / total
    victim_final_acc = victim_acc / total

    return generator_final_acc, victim_final_acc


def eval_pgd_transfer(generator, victim, test_loader, device, epsilon, alpha, iters):
    generator.to(device)
    generator.eval()
    victim.to(device)
    victim.eval()

    total = 0
    generator_acc = 0
    victim_acc = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        # get perturbed data by generator
        perturbed_data = pgd_attack(model=generator, images=data, labels=target, device=device, eps=epsilon,
                                    alpha=alpha, iters=iters)
        perturbed_data.to(device)

        # victim accuracy
        victim_output = victim(perturbed_data)
        _, victim_pred = torch.max(victim_output.data, 1)
        victim_acc += (victim_pred == target).sum().item()
    victim_final_acc = victim_acc / total

    return victim_final_acc


def pgd_test_cifar10(root, generator_pth, victim_pth, testloader, device, epsilon, alpha, iters, Log):
    generator_model_pth = os.path.join(root, generator_pth)
    generator_model = ResNet18()
    generator_model.load_state_dict(torch.load(generator_model_pth, map_location="cuda:0"))
    generator_model.to(device)
    generator_model.eval()

    victim_model_pth = os.path.join(root, victim_pth)
    victim_model = ResNet18()
    victim_model.load_state_dict(torch.load(victim_model_pth, map_location="cuda:0"))
    victim_model.to(device)
    victim_model.eval()

    generator2victim_accuracy = eval_pgd_transfer(generator_model, victim_model, testloader, device, epsilon, alpha,
                                                  iters)
    print(generator2victim_accuracy)
    if Log:
        Log([generator_model_pth, "to", victim_model_pth])
        Log([generator2victim_accuracy])


def loader_model(root, model_pth, device):
    path = os.path.join(root, model_pth)
    model = ResNet18()
    model.load_state_dict(torch.load(path, map_location="cuda:0"))
    model.to(device)
    model.eval()
    return model


def pgd_transfer_test(root,victim_pth_list: list, test_loader, device, epsilon, alpha, iters, Log):

    # model number
    victim_list_len = len(victim_pth_list)

    # model list
    victim_model_list = []
    for i in range(victim_list_len):
        victim_model_list.append(loader_model(root, victim_pth_list[i], device))

    # chose a generator model with iter
    for i in range(victim_list_len):
        # set accuracy list
        victim_model_list_acc = [0] * victim_list_len
        total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            total += target.size(0)
            # generator adversarial sample by model victim_model_list[i] , a generator
            perturbed_data = pgd_attack(model=victim_model_list[i], images=data, labels=target, device=device, eps=epsilon,
                                        alpha=alpha, iters=iters)
            perturbed_data.to(device)

            # evaluation victim model for iter
            for j in range(victim_list_len):
                victim_output = victim_model_list[j](perturbed_data)
                _, victim_pred = torch.max(victim_output.data, 1)
                victim_model_list_acc[j] += (victim_pred == target).sum().item()

        for k in range(victim_list_len):
            victim_model_list_acc[k] = victim_model_list_acc[k] / total

        Log([victim_pth_list[i]]+victim_model_list_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device(args.gpu)
    model_root = "./pth/newRes"

    testloader = cifar10(dataRoot="./data/dataset", batchsize=128, train=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    a = [
        'v2_standard_model_epoch50.pth',
        'v2_fgsm_adversarial_epoch50.pth',
        'v2_pgd_adversarial_epoch50.pth',

        'v2_cam_pgd_bool_adversarial_mode1_epoch50.pth',
        'v2_cam_pgd_bool_adversarial_mode2_epoch50.pth',
        'v2_cam_pgd_bool_adversarial_mode3_epoch50.pth',
        'v2_cam_pgd_bool_adversarial_mode4_epoch50.pth',
        'v2_cam_pgd_bool_adversarial_mode5_epoch50.pth',

        'v2_cam_pgd_adversarial_mode1_epoch50.pth',
        'v2_cam_pgd_adversarial_mode2_epoch50.pth',
        'v2_cam_pgd_adversarial_mode3_epoch50.pth',
        'v2_cam_pgd_adversarial_mode4_epoch50.pth',
        'v2_cam_pgd_adversarial_mode5_epoch50.pth',

        'v2_cam_fgsm_bool_adversarial_mode1_epoch50.pth',
        'v2_cam_fgsm_bool_adversarial_mode2_epoch50.pth',
        'v2_cam_fgsm_bool_adversarial_mode3_epoch50.pth',
        'v2_cam_fgsm_bool_adversarial_mode4_epoch50.pth',
        'v2_cam_fgsm_bool_adversarial_mode5_epoch50.pth',

        'v2_cam_fgsm_adversarial_mode1_epoch50.pth',
        'v2_cam_fgsm_adversarial_mode2_epoch50.pth',
        'v2_cam_fgsm_adversarial_mode3_epoch50.pth',
        'v2_cam_fgsm_adversarial_mode4_epoch50.pth',
        'v2_cam_fgsm_adversarial_mode5_epoch50.pth',

        'v3_rate_pgd_rate0.2_epoch50.pth',
        'v3_rate_pgd_rate0.4_epoch50.pth',
        'v3_rate_pgd_rate0.6_epoch50.pth',
        'v3_rate_pgd_rate0.8_epoch50.pth',

        'v3_rate_fgsm_rate0.2_epoch50.pth',
        'v3_rate_fgsm_rate0.4_epoch50.pth',
        'v3_rate_fgsm_rate0.6_epoch50.pth',
        'v3_rate_fgsm_rate0.8_epoch50.pth',

    ]

    Log = LogCSV("./result_image", "transfer_robustness.csv")
    Log([""] + a)

    model_number = len(a)

    acc = pgd_transfer_test(root=model_root, victim_pth_list=a, test_loader=testloader, device=device, epsilon=0.03, alpha=0.003, iters=10, Log=Log)
