from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import MyDataset

path = 'results/test20201228/iter1_testset_pgd0.3/'

dataset = MyDataset(path)
dataset.load()
dataset.tofile(0, 'tmp.png')

cifartrain = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))
data = cifartrain.data[0]
img = Image.fromarray(data)
img.save('tmp2.png')