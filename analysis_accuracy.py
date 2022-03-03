from Analysis.accuracy_test import *
from models.cifar10_CNN import Net
import torch
from data.cifar10 import cifar10
import argparse
import os
from utils.log import Log,LogCSV


def model_accuracy_test(root,pth_name,testloader,device,Log):
    pretrain_model_path = os.path.join(root, pth_name)
    model = Net()
    model.load_state_dict(torch.load(pretrain_model_path, map_location="cuda:0"))
    model.to(device)
    model.eval()

    global_acc = global_accuracy(model=model, device=device, test_loader=testloader)
    print("the accuracy of  "+pth_name+" is {}".format(global_acc))
    if Log:
        Log(pth_name)
        Log(str(global_acc))
        Log("===================================================================================")


def model_accuracy_test_confusion_Matrix(root,pth_name,result_save_root,testloader,device):
    pretrain_model_path = os.path.join(root, pth_name)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = Net()
    model.load_state_dict(torch.load(pretrain_model_path,map_location="cuda:0"))
    model.to(device)
    model.eval()

    title = pth_name.split(".")[0]
    image_name = title + ".jpg"
    save_path = os.path.join(result_save_root,image_name)
    save_confusion_matrix(model=model, device=device, test_loader=testloader, title=title, class_number=10,
                          save_path=save_path, classes=classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device(args.gpu)

    root = "./pth"

    result_save_root = "./result_image/model_confusion_Matrix"
    if not os.path.exists(result_save_root):
        os.makedirs(result_save_root)

    dataRoot = "./data/dataset"
    testloader = cifar10(dataRoot=dataRoot, batchsize=128, train=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    log = Log("./result_image","global_accuracy.txt")

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
    for pth_name in a:
        model_accuracy_test(root, pth_name,testloader, device,log)
    # model_accuracy_test_confusion_Matrix(root, pth_name, result_save_root, testloader, device)