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


