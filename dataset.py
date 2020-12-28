import os

from PIL import Image
import numpy as np
from torchvision.datasets.vision import VisionDataset


class MyDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, ):
        super(MyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.path = root
        self.data = None
        self.targets = None

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def __getitem__(self, index):
        if self.data is None:
            print("Please firstly load data.")

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def save(self, data, targets):
        self.data = data
        self.targets = targets
        np.save(self.path + 'data', data)
        np.save(self.path + 'targets', targets)

    def load(self):
        self.data = np.load(self.path + 'data.npy')
        self.targets = np.load(self.path + 'targets.npy')


if __name__ == '__main__':
    test_dataset = MyDataset('data/')
    data = np.random.randn(20, 5, 5)
    labels = np.arange(0, 20)
    test_dataset.save(data, labels)
    test_dataset.load()
