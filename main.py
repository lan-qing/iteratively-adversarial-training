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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def data2model(savepath, trainset, valset, batch_size=100, workers=4):
    model = ResNet20Wrapper()

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    for epoch in range(200):
        lr = 0.1
        if epoch == 100 or epoch == 150 or epoch == 200:
            lr /= 10

        model.fit(train_loader, lr=lr, epoch=epoch)
        model.validate(val_loader)

    model.save(savepath)
    return model


def model2data(model, basetrainset, basetestset, trainset_path, testset_path, batch_size=100, workers=4, adveps=0.1):
    test_loader = torch.utils.data.DataLoader(basetestset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(basetrainset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    attack = functools.partial(pgd_attack, eps=adveps)

    adv_data = model.generate_adv_data(attack, test_loader)
    test_dataset = MyDataset(testset_path)
    test_dataset.save(adv_data, basetestset.targets)

    adv_data = model.generate_adv_data(attack, train_loader)
    train_dataset = MyDataset(trainset_path)
    train_dataset.save(adv_data, basetrainset.targets)
    return train_dataset, test_dataset


def main():
    seed = 42
    signature = "20201228"
    rootpath = f"results/{signature}_seed{seed}/"
    set_seed(seed)

    ## D0 -> M0

    savepath = rootpath + 'model0.pt'
    trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ]), download=True)
    valset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    model = data2model(savepath, trainset, valset)

    testset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    for it in 1, 2:
        tmpseed = random.randint(0, 2147483647)
        set_seed(tmpseed)

        trainset_path = rootpath + f'iter{it}_trainset/'
        testset_path = rootpath + f'iter{it}_testset/'
        trainset, testset = model2data(model, trainset, testset, trainset_path, testset_path)

        savepath = rootpath + f'model{it}.pt'
        trainset = MyDataset(root=trainset_path, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]))
        valset = MyDataset(root=testset_path, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
        model = data2model(savepath, trainset, valset)

    tmpseed = random.randint(0, 2147483647)
    set_seed(tmpseed)

    trainset_path = rootpath + f'iter{it + 1}_trainset/'
    testset_path = rootpath + f'iter{it + 1}_testset/'
    trainset, testset = model2data(model, trainset, testset, trainset_path, testset_path)


if __name__ == '__main__':
    main()
