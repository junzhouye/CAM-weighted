import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def pgd_attack(model, images, labels, device, eps, alpha, iters, min_val=0, max_val=1):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=min_val, max=max_val).detach_()

    return images


def pgd_test(model, test_loader, device, eps=0.3, alpha=2 / 255, iters=40):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        if eps != 0:
            perturbed_data = pgd_attack(model, data, target, device, eps, alpha, iters)
        else:
            perturbed_data = data
        adv_outputs = model(perturbed_data)
        _, final_predicted = torch.max(adv_outputs.data, 1)
        correct += (final_predicted == target).sum().item()

    final_acc = correct / total
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, total, final_acc))
    return final_acc


def pgd_test_confusion_matrix(model, test_loader, device, class_number=10, eps=0.3, alpha=2 / 255, iters=40):
    model.eval()
    confusion_m = np.zeros((class_number, class_number))
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if eps != 0:
            perturbed_data = pgd_attack(model, data, target, device, eps, alpha, iters)
        else:
            perturbed_data = data
        adv_outputs = model(perturbed_data)
        _, final_predicted = torch.max(adv_outputs.data, 1)
        for t, p in zip(target, final_predicted):
            confusion_m[t][p] += 1
    return confusion_m


def save_pgd_confusion_matrix(model, test_loader, device,classes,title=" ",save_path=None, class_number=10, eps=0.3, alpha=2 / 255, iters=40):
    conf_matrix = pgd_test_confusion_matrix(model, test_loader, device, class_number, eps, alpha, iters)
    if classes:
        assert len(classes) == class_number
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    iters = np.reshape([[[i, j] for j in range(class_number)] for i in range(class_number)], (conf_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, int(conf_matrix[i, j]), va='center', ha='center')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    # plt.show()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig("default.jpg")