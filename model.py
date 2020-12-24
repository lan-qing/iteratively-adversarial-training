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

    def validate(self, test_loader):
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
    def __init__(self):
        super(ResNet20Wrapper).__init__()
        self.model = resnet20()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, start_epoch=0, epochs=200):
        assert epochs > start_epoch
        optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150],
                                                            last_epoch=start_epoch - 1)
        criterion = nn.CrossEntropyLoss().cuda()

        loss, prec = None, None
        for epoch in range(start_epoch, epochs):
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            loss, prec = train(train_loader, self.model, criterion, optimizer, epoch)
            lr_scheduler.step()

        return loss, prec

    def predict(self, test_loader):
        pass

    def validate(self, test_loader):
        pass
