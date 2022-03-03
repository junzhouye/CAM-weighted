import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def fgsm_attack(image, epsilon, data_grad, min_val=0, max_val=1):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, min_val, max_val)
    return perturbed_image


def fgsm_test(model, device, test_loader, epsilon=0.1, min_val=0, max_val=1):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        outputs = model(data)
        total += target.size(0)
        model.zero_grad()
        loss = F.nll_loss(outputs, target)
        loss.backward()
        data_grad = data.grad.data
        if epsilon == 0:
            perturbed_data = data
        else:
            perturbed_data = fgsm_attack(data, epsilon, data_grad, min_val, max_val)
        final_outputs = model(perturbed_data)
        _, final_predicted = torch.max(final_outputs.data, 1)
        correct += (final_predicted == target).sum().item()

    final_acc = correct / total
    print("Epsilon: {} \t Test Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc


def fgsm_test_confusion_matrix(model, device, test_loader, class_number=10, epsilon=0.1, min_val=0, max_val=1):
    model.eval()
    confusion_m = np.zeros((class_number, class_number))
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        outputs = model(data)
        model.zero_grad()
        loss = F.nll_loss(outputs, target)
        loss.backward()
        data_grad = data.grad.data
        if epsilon == 0:
            perturbed_data = data
        else:
            perturbed_data = fgsm_attack(data, epsilon, data_grad, min_val, max_val)
        final_outputs = model(perturbed_data)
        _, final_predicted = torch.max(final_outputs.data, 1)

        for i, j in zip(target, final_predicted):
            confusion_m[i][j] += 1
    return confusion_m


def save_fgsm_confusion_matrix(model, device, test_loader, epsilon=0.1, title=" ", class_number=10, min_val=0,
                               max_val=1, save_path=None, classes=None):

    conf_matrix = fgsm_test_confusion_matrix(model, device, test_loader, class_number=10, epsilon=epsilon,
                                             min_val=min_val, max_val=max_val)
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


def get_fgsm_adversarial_sample(model, device, test_loader, epsilon, min_val, max_val):
    """
    返回的是攻击成功的样本list
    并且，给定batch_size = 1
    """
    model.eavl()
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        # 跳过自然分错样本
        if init_pred.item() != target.item():
            continue
        model.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        data_grad = data.grad.data
        if epsilon == 0:
            perturbed_data = data
        else:
            perturbed_data = fgsm_attack(data, epsilon, data_grad, min_val, max_val)
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            pass
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                break
    return adv_examples
