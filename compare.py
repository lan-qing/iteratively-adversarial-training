import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from models.resnet import resnet20
from model import ResNet20Wrapper
from attacks import pgd_attack
from dataset import MyDataset
model = ResNet20Wrapper()


path1 = f'results/20201231-100epoch-double_seed44/iter{1}_testset/'

dataset0 = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([transforms.ToTensor(),
]))
dataset1 = MyDataset(root=path1)

for idx in range(10, 500):
    a = dataset0[idx][0] - dataset1[idx][0]
    print(torch.max(a.flatten()))
# batch_size = 100
# workers = 4
# model.load('results/20201228_seed42/model0.pt')
# path = 'results/20201228_seed42/iter3_testset/'
# dataset = MyDataset(root=path)
# val_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=batch_size, shuffle=False,
#     num_workers=workers, pin_memory=True)
#
# model.validate(val_loader)

# idx = 101
# dataset = MyDataset(path)
# dataset.load()
# d = dataset[idx][0].cuda().half()
# out = model.model(d)
# print(out, torch.argmax(out))
#
# cifartrain = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([transforms.ToTensor(),
# ]))
# data = cifartrain[idx][0].cuda().half()
# # data = torch.einsum("ijk->kji", data).cuda()
# out = model.model(data)
# print(out, torch.argmax(out))
