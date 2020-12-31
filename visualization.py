from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import MyDataset
idx = 2000

cifartest = datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
]))
data = cifartest.data[idx]
img = Image.fromarray(data)
img.save(f'tmp/{idx}-iter0.png')


for i in 1,2,3:
    path = f'results/20201231test_seed42/iter{i}_testset/'
    dataset = MyDataset(path)
    dataset.tofile(idx, f'tmp/{idx}-iter{i}.png')

