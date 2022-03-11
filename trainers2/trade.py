import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from CAM.base_CAM import get_cam, get_cam_diff_plus, get_cam_diff


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                device,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    per = 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = x_natural.detach() + per
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def cam_weight_trades_loss(model,
                           x_natural,
                           y,
                           device,
                           optimizer,
                           step_size=0.003,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=1.0,
                           distance='l_inf',
                           mode=1):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    per = 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = x_natural.detach() + per
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if mode == 1:
        cam_weight = get_cam(model=model, inputs=x_adv)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 2:
        cam_weight = get_cam(model=model, inputs=x_adv)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * (1 - cam_weight)
    elif mode == 3:
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs.data, 1)
        cam_weight = get_cam(model=model, inputs=x_natural, target=predicted)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 4:
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs.data, 1)
        cam_weight = get_cam(model=model, inputs=x_natural, target=predicted)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * (1 - cam_weight)
    elif mode == 5:
        cam_weight = get_cam(model=model, inputs=x_adv, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 6:
        cam_weight = get_cam(model=model, inputs=x_adv, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * (1 - cam_weight)
    elif mode == 7:
        cam_weight = get_cam(model=model, inputs=x_natural, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 8:
        cam_weight = get_cam(model=model, inputs=x_natural, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * (1 - cam_weight)
    elif mode == 9:
        # mode 9 开始 使用类别激活差异作为cam weighted的依据
        cam_weight = get_cam_diff(model=model, natural_data=x_natural, adv_data=x_adv, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 10:
        cam_weight = get_cam_diff(model=model, natural_data=x_natural, adv_data=x_adv)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    elif mode == 11:
        cam_weight = get_cam_diff_plus(model=model, natural_data=x_natural, adv_data=x_adv, target=y)
        cam_weight = cam_weight.to(device)
        weight_data = x_natural + (x_adv - x_natural) * cam_weight
    else:
        weight_data = x_adv

    model.train()

    weight_data = Variable(torch.clamp(weight_data, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(weight_data), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss