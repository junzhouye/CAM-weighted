"""
mode1 OK
mode 9 OK
mode 10 OK
mode 11 OK
"""
from trainers.adversarial_train import *
from data.cifar10 import cifar10
from models.cifar10_CNN import Net
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--gpu', type=int, default=0,
                    help='disables CUDA training')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument("--mode", type=int, default=1)

args = parser.parse_args()
# settings
####################################################################################################################
step_size = 0.003
epsilon = 0.031
perturb_steps = 10
mode = args.mode
####################################################################################################################
model_dir = "./pth"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.manual_seed(9)
device = torch.device(args.gpu)

train_loader = cifar10(dataRoot="./data/dataset", batchsize=128, train=True)
test_loader = cifar10(dataRoot="./data/dataset", batchsize=128, train=False)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4)

for epoch in range(1, args.epochs + 1):

    # adversarial training
    cam_weight_adversarial_train_epoch(model, device, train_loader, optimizer, epoch, step_size, epsilon, perturb_steps,
                                       mode=mode)

    # evaluation on natural examples
    print('================================================================')
    # eval_train(model, device, train_loader)
    eval_test(model, device, test_loader)
    print('================================================================')

    # save checkpoint
    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(),
                   os.path.join(model_dir, 'cam_weighted_adversarial_model_mode{}_epoch{}.pth'.format(mode, epoch)))
