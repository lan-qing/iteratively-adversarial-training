import torch
import torch.nn as nn

from utils import *
from models.resnet import resnet20


class NetWrapper():
    def __init__(self):
        cprint('c', '\nNet:')
        self.model = None

    def fit(self, train_loader):
        raise NotImplementedError

    def predict(self, test_loader):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def save(self, filename='checkpoint.th'):
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])


class ResNet20Wrapper(NetWrapper):
    def __init__(self, half=True, cuda=True):
        super(ResNet20Wrapper).__init__()
        self.model = resnet20()
        self.half = half
        if self.half:
            self.model.half()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.1, weight_decay=0.0):
        optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, criterion, optimizer, epoch)

        return loss, prec

    def predict(self, test_loader):
        pass

    def validate(self, val_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion)
        return loss, prec


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    batch_size = 100
    workers = 4
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    net = ResNet20Wrapper()

    for epoch in range(200):
        lr = 0.0
        if epoch < 100:
            lr = 0.1
        elif epoch < 150:
            lr = 0.01
        elif epoch < 200:
            lr = 0.001
        net.fit(train_loader, lr=lr)
        net.validate(val_loader)

    path = "checkpoints/resnet20test.pt"
    net.save(path)

    netnew = ResNet20Wrapper()
    netnew.load(path)
    netnew.validate(val_loader)
# val prec: 89.290
