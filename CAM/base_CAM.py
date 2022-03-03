import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import cv2


def tensor_normal(orign_data: torch.Tensor):
    d_min = orign_data.min()
    d_max = orign_data.max()
    assert d_max > d_min
    normal_data = (orign_data - d_min) / (d_max - d_min)
    return normal_data.detach()


def get_original_cam(model, inputs, target=None):
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # get input (H,W) , to resize CAM
        size_upsample = (inputs.size()[-2], inputs.size()[-1])
        batch_size, nc, h, w = feature_conv.shape
        output_cam = []

        for i in range(batch_size):
            cam = weight_softmax[class_idx[i].item()].dot(feature_conv[i].reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cv2.resize(cam, size_upsample)
            output_cam.append(cam)
        return output_cam

    model.eval()
    # 这里不对获得的CAM图做归一化处理
    # weight_softmax shape : (class_number, xxx)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    f, outputs = model(inputs, get_feature=True)
    feature = f.cpu().data.numpy()
    _, pred = torch.max(outputs.data, 1)
    if target != None:
        class_idx = target
    else:
        class_idx = pred
    output_cam = returnCAM(feature_conv=feature, weight_softmax=weight_softmax, class_idx=class_idx)

    cam = np.array(output_cam)
    cam = np.expand_dims(cam, 1)
    cam = torch.from_numpy(cam)
    model.train()
    # shape of cam is (b,c,h,w) (b,1,32,32)
    return cam


def get_cam(model, inputs, target=None):
    model.eval()
    # weight_softmax shape : (class_number, xxx)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # get input (H,W) , to resize CAM
        size_upsample = (inputs.size()[-2], inputs.size()[-1])
        batch_size, nc, h, w = feature_conv.shape
        output_cam = []

        for i in range(batch_size):
            cam = weight_softmax[class_idx[i].item()].dot(feature_conv[i].reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cv2.resize(cam, size_upsample)
            if cam.max() == cam.min():
                continue
            # Scale between 0-1
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            output_cam.append(cam)
        return output_cam

    f, outputs = model(inputs, get_feature=True)
    feature = f.cpu().data.numpy()
    _, pred = torch.max(outputs.data, 1)

    if target != None:
        class_idx = target
    else:
        class_idx = pred

    output_cam = returnCAM(feature_conv=feature, weight_softmax=weight_softmax, class_idx=class_idx)

    cam = np.array(output_cam)
    cam = np.expand_dims(cam, 1)
    cam = torch.from_numpy(cam)
    model.train()
    # shape of cam is (b,c,h,w) (b,1,32,32)
    return cam


def get_cam_diff(model, natural_data, adv_data, target=None):
    if target==None:
        model.eval()
        outputs = model(adv_data)
        _, pred = torch.max(outputs.data, 1)
        target = pred
        model.train()

    cam_natural = get_original_cam(model, natural_data, target)
    cam_adv = get_original_cam(model, adv_data, target)
    cam_diff = abs(cam_natural - cam_adv)
    cam_diff = tensor_normal(cam_diff)
    return cam_diff


def get_cam_diff_plus(model, natural_data, adv_data, target):
    cam_natural = get_original_cam(model, natural_data, target)
    cam_adv = get_original_cam(model, adv_data, target)
    cam_diff1 = abs(cam_natural - cam_adv)
    # get pred
    model.eval()
    outputs = model(natural_data)
    _, pred = torch.max(outputs.data, 1)
    model.train()

    cam_natural_pred = get_original_cam(model, natural_data, pred)
    cam_adv_pred = get_original_cam(model, adv_data, pred)
    cam_diff2 = abs(cam_natural_pred - cam_adv_pred)

    cam_diff = tensor_normal(cam_diff1 + cam_diff2)
    return cam_diff


if __name__ == "__main__":
    pass
