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



path1 = f'results/20210101-100epoch-double_seed42/iter{1}_testset/'
path2 = f'results/20210108_seed55/iter5_testset/'
dataset0 = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([transforms.ToTensor(),
]))
dataset1 = MyDataset(root=path1)
dataset2 = MyDataset(root=path2)
for idx in range(10, 500):
    a = dataset0[idx][0] - dataset2[idx][0]
    print(torch.max(a.flatten()))

def validate(modelpath, datapath):
    model = ResNet20Wrapper()
    model.load(modelpath)
    batch_size = 100
    workers = 4
    dataset = MyDataset(root=datapath)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    model.validate(val_loader)


#
# model = ResNet20Wrapper()
# model.load(modelpath)
# batch_size = 100
# workers = 4
# dataset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#         ]))
# val_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=batch_size, shuffle=False,
#     num_workers=workers, pin_memory=True)
# model.validate(val_loader)

modelpath = f'results/20210114adv_seed50/model{1}.pt'
path = f"data/mycifartestset/"
validate(modelpath, path)

for iter in 1, 2, 3, 4, 5:
    modelpath = f'results/20210114adv_seed50/model{1}.pt'
    path = f'results/20210101-100epoch-double_seed42/iter{1}_testset/'
    validate(modelpath, path)

# path = f'results/20210101-100epoch-double_seed44/iter{1}_testset/'
# validate(modelpath, path)
# path2 = f'results/20210101-100epoch-0.09step_seed50/iter1_testset/'
# validate(modelpath, path2)
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
