import os

from PIL import Image
import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset
from utils import *


class MyDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, ):
        super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.path = root
        self.data = None
        self.targets = None
        self.loaded = False
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        elif os.path.isfile(self.path + 'data.npy') and os.path.isfile(self.path + 'targets.npy'):
            self.load()

    def __getitem__(self, index):
        assert self.loaded

        img, target = self.data[index], self.targets[index]

        img = torch.tensor(img)

        flag = False
        if img.dtype == torch.float16:
            flag = True
            img = img.float()

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if flag:
            img.half()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        assert self.loaded
        return len(self.data)

    def save(self, data, targets):
        assert not self.loaded
        self.data = data
        self.targets = targets
        np.save(self.path + 'data', data)
        np.save(self.path + 'targets', targets)
        self.loaded = True

    def load(self):
        assert not self.loaded
        self.data = np.load(self.path + 'data.npy')
        self.targets = np.load(self.path + 'targets.npy')
        self.loaded = True

    def tofile(self, index, path):
        assert self.loaded
        data2img(self.data[index], path)

    def concat_from(self, dataset1, dataset2):
        if not dataset1.loaded and not dataset2.loaded:
            return
        elif not dataset1.loaded:
            self.data = dataset2.data.copy()
            self.targets = dataset2.targets.copy()
            self.loaded = True
        elif not dataset2.loaded:
            self.data = dataset1.data.copy()
            self.targets = dataset1.targets.copy()
            self.loaded = True
        else:
            assert dataset1.data.shape[1:] == dataset2.data.shape[1:]
            assert dataset1.targets.shape[1:] == dataset2.targets.shape[1:]
            self.data = np.concatenate([dataset1.data, dataset2.data], axis=0)
            self.targets = np.concatenate([dataset1.targets, dataset2.targets], axis=0)
            self.loaded = True

    def fromvision(self, dataset):
        assert not self.loaded
        targets = np.array(dataset.targets)
        data = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
        self.save(data, targets)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    testset = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    trainset = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    mycifartrain = MyDataset("data/mycifartrainset/")
    mycifartrain.fromvision(trainset)
    mycifartest = MyDataset("data/mycifartestset/")
    mycifartest.fromvision(testset)
