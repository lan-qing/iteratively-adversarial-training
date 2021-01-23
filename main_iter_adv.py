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


def data2model(savepath, trainset, valset, batch_size=100, workers=4, adv=None):
    model = ResNet20Wrapper(half=False, double=True)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    initial_epochs = 100
    initial_batch = 50000 / batch_size
    total_batch = initial_batch * initial_epochs
    total_epoch = int(total_batch * batch_size / len(trainset))
    print("total_epoch: ", total_epoch)
    for epoch in range(total_epoch):
        lr = 0.1
        if epoch == int(total_epoch * 0.5) or epoch == int(total_epoch * 0.75):
            lr /= 10

        model.fit(train_loader, lr=lr, epoch=epoch, adv=adv)
        model.validate(val_loader)

    print("Saving model...")
    model.save(savepath)
    return model


def model2data(model, basetrainset, basetestset, trainset_path, testset_path, batch_size=100, workers=4, adveps=0.1):
    test_loader = torch.utils.data.DataLoader(basetestset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(basetrainset,
                                               batch_size=batch_size, shuffle=False,
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
    seed = 55
    print("Use random seed ", seed)
    step = 0.1
    signature = "20210116advtest"
    rootpath = f"results/{signature}_seed{seed}/"
    if not os.path.isdir(rootpath):
        os.mkdir(rootpath)
    set_seed(seed)

    trainset_path = "data/mycifartrainset/"
    trainset = MyDataset(trainset_path, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
    ]))
    testset_path = "data/mycifartestset/"
    testset = MyDataset(testset_path)
    current_trainset = MyDataset(rootpath + f"current_trainset/", transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
    ]))
    current_testset = MyDataset(rootpath + f"current_testset/")
    for it in [1]:
        tmpseed = random.randint(0, 2147483647)
        set_seed(tmpseed)

        current_trainset.concat_from(current_trainset, trainset)
        current_testset.concat_from(current_testset, testset)
        savepath = rootpath + f'model{it}.pt'
        model = data2model(savepath, current_trainset, current_testset, adv=pgd_attack)

        pre_trainset = trainset
        pre_testset = testset
        trainset_path = rootpath + f'iter{it}_trainset/'
        testset_path = rootpath + f'iter{it}_testset/'
        trainset, testset = model2data(model, pre_trainset, pre_testset, trainset_path, testset_path, adveps=step)


if __name__ == '__main__':
    main()
