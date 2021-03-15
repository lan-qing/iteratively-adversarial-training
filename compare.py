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

