from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import MyDataset

idx = [1000, 2000, 4000, 6000, 7000]
#
# cifartest = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
#     transforms.ToTensor(),
# ]))
# data = cifartest.data[idx]
# img = Image.fromarray(data)
# img.save(f'tmp/{idx}-iter0.png')

for id in idx:
    i = 1
    path = f'results/20210101-100epoch-double_seed44/iter{i}_testset/'
    dataset = MyDataset(path)
    dataset.tofile(id, f'tmp/{id}-direct-iter{i}.png')
