import torch
import torch.nn.functional as F

def cam_clamp(cam):

    return cam


def cam_binarization(cam):
    mean = cam.mean(axis=-1).mean(axis=-1)

    return cam


if __name__ == "__main__":
    from CAM.base_CAM import get_cam
    from trainers.standard_train import *
    from data.cifar10 import cifar10
    from models.cifar10_CNN import Net
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--gpu', action='store_true', default=0,
                        help='disables CUDA training')

    args = parser.parse_args()

    device = torch.device(args.gpu)

    # setup data loader
    test_loader = cifar10(dataRoot="../data/dataset", batchsize=4, train=False)
    pretrain_model_path = "../pth/standard_model-epoch30.pth"
    model = Net().to(device)
    model.load_state_dict(torch.load(pretrain_model_path))
    model.eval()

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        cam = get_cam(model=model, inputs=data, target=target)
        print(cam)
        print("#"*100)
        cam = cam_clamp(cam)
        print(cam)

        break
