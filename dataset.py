import os

from PIL import Image
import torch
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
        ## It seems a byg of pytorch.
        """
        >>> a = torch.tensor([1,2])
        >>> a.flip(-1)
        tensor([2, 1])
        >>> a = torch.tensor([1,2]).cuda()
        >>> a.flip(-1)
        tensor([2, 1], device='cuda:0')
        >>> a = torch.tensor([1,2]).half().cuda()
        >>> a.flip(-1)
        tensor([2., 1.], device='cuda:0', dtype=torch.float16)
        >>> a = torch.tensor([1,2]).half()
        >>> a.flip(-1)
        Traceback (most recent call last):
            File "<input>", line 1, in <module>
        RuntimeError: "flip_cpu" not implemented for 'Half'
        """

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
