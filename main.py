import os
import random

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from model import ResNet20Wrapper
from attacks import pgd_attack
from dataset import MyDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    signature = "results/test20201228/"
    model = ResNet20Wrapper()
    model.load('checkpoints/resnet20test.pt')

    batch_size = 100
    workers = 4
    cifar10 = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_loader = torch.utils.data.DataLoader(cifar10,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    adv_data = model.generate_adv_data(pgd_attack, test_loader)
    dataset_iter1 = MyDataset(signature + 'iter1_testset_pgd0.3/')
    dataset_iter1.save(adv_data, cifar10.targets)


if __name__ == '__main__':
    main()
