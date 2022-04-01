import argparse
from models.cifar10_resnet import ResNet18
import torch
import torch.nn.functional as F
from data.cifar10 import cifar10
from utils.log import LogCSV
import os
import numpy as np
import torch.nn as nn
from utils.log import LogCSV
from attacks.CW import Attack
from models.cifar10_resnet import ResNet18,ResNet50
from models.cifar_vgg import VGG
from models.cifar10_wideresnet import get_model


class EvalAll:
    def __init__(self, root, model_pth, args):
        self.device = torch.device(args.gpu)
        self.model = self.loadModel(root, model_pth)
        self.testloader = cifar10(args.dataroot, train=False)
        self.log = LogCSV(".", "All_robustness.csv")
        self.log([model_pth])

    def main(self):
        # evaluation clean
        clean_acc = self.eval_clean()

        # evaluation FGSM
        epsilon_list = [2 / 255, 4 / 255, 6 / 255, 8 / 255, 10 / 255, 12 / 255]
        self.log(["FGSM", "2", "4", "6", "8", "10", "12"])
        fgsm_acc = [clean_acc]
        for epsilon in epsilon_list:
            fgsm_a = self.eval_FGSM(epsilon)
            fgsm_acc.append(fgsm_a)
        self.log(fgsm_acc)

        # evaluation PGD
        epsilon_pgd_list = [2 / 255, 4 / 255, 6 / 255, 8 / 255, 10 / 255, 12 / 255]
        self.log(["PGD", "2", "4", "6", "8", "10", "12"])
        pgd_acc = [clean_acc]
        for epsilon in epsilon_pgd_list:
            pgd_a = self.eval_PGD(epsilon, alpha=1.0/ 255, iters=20, random_init=True)
            pgd_acc.append(pgd_a)
        self.log(pgd_acc)

        cw_acc = self.eval_CW()
        self.log([cw_acc])

    def eval_FGSM(self, epsilon):
        self.model.eval()
        correct = 0
        total = 0
        for data, target in self.testloader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            outputs = self.model(data)
            total += target.size(0)
            self.model.zero_grad()
            loss = F.nll_loss(outputs, target)
            loss.backward()
            data_grad = data.grad.data
            if epsilon == 0:
                perturbed_image = data
            else:
                sign_data_grad = data_grad.sign()
                perturbed_image = data + epsilon * sign_data_grad
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
            with torch.no_grad():
                final_outputs = self.model(perturbed_image)
                _, final_predicted = torch.max(final_outputs.data, 1)
                correct += (final_predicted == target).sum().item()
        final_acc = correct / total
        print("FGSM Epsilon: {} \t Test Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
        return final_acc

    def eval_PGD(self, epsilon, alpha, iters, random_init=False):
        self.model.eval()
        correct = 0
        total = 0
        for data, target in self.testloader:
            data, target = data.to(self.device), target.to(self.device)
            total += target.size(0)
            ori_images = data.data

            if random_init:
                adv = data.data + torch.Tensor(np.random.uniform(epsilon, epsilon, data.shape)).type_as(data).to(self.device)
            else:
                adv = data.data.to(self.device)

            for iter in range(iters):
                adv.requires_grad = True
                outputs = self.model(adv)
                self.model.zero_grad()
                loss = F.nll_loss(outputs, target)
                loss.backward()
                sign_data_grad = adv.grad.data.sign()
                perturbed_image = adv + alpha * sign_data_grad
                perturbed = torch.clamp(perturbed_image - ori_images, -epsilon, epsilon)
                adv = torch.clamp(data + perturbed,0,1).detach_()

            with torch.no_grad():
                final_outputs = self.model(adv)
                _, final_predicted = torch.max(final_outputs.data, 1)
                correct += (final_predicted == target).sum().item()
        final_acc = correct / total
        print("PGD Epsilon: {} alpha : {} iters: {} \t Test Accuracy = {} / {} = {}".format(epsilon, alpha, iters,
                                                                                            correct, total, final_acc))
        return final_acc

    @torch.no_grad()
    def eval_clean(self):
        self.model.eval()
        correct = 0
        total = 0
        for data, target in self.testloader:
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            total += target.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
        final_acc = correct / total
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))
        return final_acc

    def eval_CW(self):
        self.model.eval()
        correct = 0
        total = 0
        epsilon = 0.03
        num_steps = 100
        step_size = 0.003
        cw_attack = Attack(epsilon=epsilon, num_steps=num_steps, step_size=step_size)
        for data, target in self.testloader:
            data, target = data.to(self.device), target.to(self.device)
            total += target.size(0)
            perturbed_data = cw_attack.attack(model=self.model, X=data, y=target, loss_type="cw_loss")
            with torch.no_grad():
                adv_outputs = self.model(perturbed_data)
                _, final_predicted = torch.max(adv_outputs.data, 1)
                correct += (final_predicted == target).sum().item()

        final_acc = correct / total
        print("CW Epsilon: {} \tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
        return final_acc

    def eval_black_box(self):
        # Understanding Catastrophic Overfitting in Single-step Adversarial Training
        # we also consider PGD50 adversarial image generator from Wide-ResNet 40-10 trained on clean image (Black-Box)
        pass

    def loadModel(self, root, model_pth):
        pretrain_model_path = os.path.join(root, model_pth)
        model = ResNet18()
        model.load_state_dict(torch.load(pretrain_model_path))# ,map_location="cuda:0"
        model.to(self.device)
        model.eval()
        return model


class TransferEval:
    def __init__(self, generator_root, victim_root,victim_type, victim: list, gpu, dataroot):
        self.device = torch.device(gpu)
        self.test_loader = cifar10(dataroot, train=False)
        self.generator_root = generator_root
        self.victim_number = len(victim)

        victim_model_list = []
        for i in range(self.victim_number):
            victim_model_list.append(self.loadModel(victim_root, victim[i],victim_type[i]))
        self.victims = victim_model_list

        self.log = LogCSV(".", "transfer.csv")
        self.log([""] + victim)

    def main(self, generator_pth):
        self.log([generator_pth])
        generator = self.loadModel(self.generator_root, generator_pth)
        total = 0
        victim_acc = [0] * (self.victim_number + 1)
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            total += target.size(0)
            perturbed_data = self.generator_pgd(generator,data, target, epsilon=8 / 255, alpha=1 / 255, iter=20,
                                                random_init=True)
            with torch.no_grad():
                generator_output = generator(perturbed_data)
                _, generator_pred = torch.max(generator_output.data, 1)
                victim_acc[0] += (generator_pred == target).sum().item()

                for victim_n in range(self.victim_number):
                    victim_output = self.victims[victim_n](perturbed_data)
                    _, victim_pred = torch.max(victim_output.data, 1)
                    victim_acc[victim_n+1] += (victim_pred == target).sum().item()

        for k in range(self.victim_number + 1):
            victim_acc[k] = victim_acc[k] / total
        self.log(victim_acc)

    def generator_pgd(self, generator,data, target, epsilon, alpha, iter, random_init=False):
        loss = nn.CrossEntropyLoss()
        ori_images = data.data
        if random_init:
            data = ori_images + torch.Tensor(np.random.uniform(epsilon, epsilon, data.shape)).type_as(data).to(
                self.device)
        for i in range(iter):
            data.requires_grad = True
            output = generator(data)
            generator.zero_grad()
            cost = loss(output, target)
            cost.backward()
            adv_images = data + alpha * data.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
            data = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        generator.zero_grad()
        data.to(self.device)
        return data

    def loadModel(self, root, model_pth,model_type="res18"):
        pretrain_model_path = os.path.join(root, model_pth)
        if model_type == "res18":
            model = ResNet18()

        if model_type == "res50":
            model = ResNet50()

        if model_type == "vgg16":
            model = VGG("VGG16")

        if model_type == "WRN40":
            model = get_model(name="WRN40", num_classes=10)

        model.load_state_dict(torch.load(pretrain_model_path))# ,map_location="cuda:0"
        model.to(self.device)
        model.eval()
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='disables CUDA training')
    parser.add_argument('--dataroot', type=str, default="../data/dataset", help='data root')
    parser.add_argument('--savename', type=str, default="default.pth")
    args = parser.parse_args()
