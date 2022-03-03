"""
1. global accuracy 全局准确率。 GA = correct/total
2. class-wise accuracy 1 类准确率1。 CA1.shape = [class_number], CA[i] = class_i_pred_true / ground_true_class_i_total
3. class-wise accuracy 2 类准确率1。 CA2.shape = [class_number], CA[i] = class_i_pred_true/ pred_class_i_total
4. confusion Matrix
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def global_accuracy(model, device, test_loader):
    model.eval()
    # test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # test_loss /= len(test_loader.dataset)
    print('Test: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def class_wise_accuracy1(model, device, test_loader, class_number=10):
    # 召回率
    class_pred_true = [0] * class_number
    class_groud_true = [0] * class_number
    model.eval()
    # test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)

            for i, j in zip(target, predicted):
                class_groud_true[i.item()] += 1
                if i.item() == j.item():
                    class_pred_true[j.item()] += 1
    class_wise_acc = [0] * 10
    for i in range(class_number):
        class_wise_acc[i] = 100 * class_pred_true[i] / class_groud_true[i]
    return class_wise_acc


def class_wise_accuracy2(model, device, test_loader, class_number=10):
    # 精准率
    class_pred_true = [0] * class_number
    class_pred_as = [0] * class_number
    model.eval()
    # test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)

            for i, j in zip(target, predicted):
                class_pred_as[j.item()] += 1
                if i.item() == j.item():
                    class_pred_true[j.item()] += 1
    class_wise_acc = [0] * 10
    for i in range(class_number):
        class_wise_acc[i] = round(100 * class_pred_true[i] / class_pred_as[i], 2)
    return class_wise_acc


def get_confusion_matrix(model, device, test_loader, class_number=10):
    model.eval()
    confusion_m = np.zeros((class_number, class_number))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)

            for i, j in zip(target, predicted):
                confusion_m[i][j] += 1

    return confusion_m


def save_confusion_matrix(model, device, test_loader, title=" ", class_number=10, save_path=None, classes=None):
    conf_matrix = get_confusion_matrix(model, device, test_loader, class_number=10)
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


def pred_count(model, device, test_loader):
    # 目的是讲预测结果为i类的数量，做一个统计
    model.eval()
    class_count = np.zeros(10)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            for i in predicted:
                class_count[i.item()] += 1
    return class_count
