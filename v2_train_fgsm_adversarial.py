"""
mode 1
mode 12
mode 9
mode 13
"""
from trainers2.FGSM_adversarial_train import *
from trainers2.adjust_lr import adjust_learning_rate
from data.cifar10 import cifar10
from models.cifar10_resnet import ResNet18
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--gpu', type=int, default=0,
                    help='disables CUDA training')
parser.add_argument('--save-freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()
# settings
####################################################################################################################
epsilon = 0.031
alpha = 10 / 255
lr = 0.1
####################################################################################################################
model_dir = "./pth/Resnet18"
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
    fgsm_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, epsilon, alpha)

    # evaluation on natural examples
    print('================================================================')
    # eval_train(model, device, train_loader)
    eval_test(model, device, test_loader)
    print('================================================================')

    # save checkpoint
    if (epoch % args.save_freq == 0) or (epoch == args.epochs):
        torch.save(model.state_dict(),
                   os.path.join(model_dir, 'v2_fgsm_uniform_adversarial_epoch{}.pth'.format(epoch)))