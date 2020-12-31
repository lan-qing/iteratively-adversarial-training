import torch
import torch.nn as nn


def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40, half=True, double=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)  # .to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
