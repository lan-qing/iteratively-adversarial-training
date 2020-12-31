import os
import random
import functools

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from model import ResNet20Wrapper
from attacks import pgd_attack
from dataset import MyDataset

batch_size = 100
workers = 4
seed = 44
signature = "20201231-100epoch"
rootpath = f"results/{signature}_seed{seed}/"
model = ResNet20Wrapper()
model.load(rootpath + "model0.pt")

testset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
]))
trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=batch_size, shuffle=False,
                                          num_workers=workers, pin_memory=True)

train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=workers, pin_memory=True)

attack = functools.partial(pgd_attack, eps=0.1, iters=1)

for i, (input_data, target) in enumerate(test_loader):
    break

model.model.eval()
input_data = input_data.double()
model.model.double()


a = attack(model.model, input_data, target)
