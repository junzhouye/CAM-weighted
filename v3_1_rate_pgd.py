"""
PGD ADVERSARIAL TRAINING WITH BOOL CAM WEIGHTED
"""
from trainers2.rate_mask_pgd_adversarial_train import *
from trainers2.adjust_lr import adjust_learning_rate
from data.cifar10 import cifar10
from models.cifar10_resnet import ResNet18
import argparse
import torch
import torch.optim as optim

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Adversarial Training')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--gpu', type=int, default=0,
                    help='disables CUDA training')
parser.add_argument('--save-freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')
parser.add_argument("--v", default=2, type=int)

args = parser.parse_args()
# settings
####################################################################################################################
step_size = 0.003
epsilon = 0.031
perturb_steps = 10
lr = 0.1
v = args.v
####################################################################################################################
model_dir = "./pth/newRes"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.manual_seed(9)
device = torch.device(args.gpu)

train_loader = cifar10(dataRoot="./data/dataset", batchsize=128, train=True)
test_loader = cifar10(dataRoot="./data/dataset", batchsize=128, train=False)

model = ResNet18().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)

for epoch in range(1, args.epochs + 1):

    # adversarial training
    adjust_learning_rate(lr, optimizer, epoch)
    if v == 2:
        rate_adversarial_train_epoch_v2(model, device, train_loader, optimizer, epoch, step_size, epsilon,
                                        perturb_steps)
        # evaluation on natural examples
        print('================================================================')
        # eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if (epoch % args.save_freq == 0) or (epoch == args.epochs):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'v3_rate_pgd_dynamic_epoch{}.pth'.format(epoch)))
    if v == 3:
        rate_adversarial_train_epoch_v3(model, device, train_loader, optimizer, epoch, step_size, epsilon,
                                        perturb_steps)
        # evaluation on natural examples
        print('================================================================')
        # eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if (epoch % args.save_freq == 0) or (epoch == args.epochs):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'v3_rate_pgd_random_epoch{}.pth'.format(epoch)))
