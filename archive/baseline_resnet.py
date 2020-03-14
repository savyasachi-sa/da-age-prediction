import torch.nn as nn
import torchvision.models as models
from nntools import NeuralNetwork


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


class BaselineResnet(NeuralNetwork):
    def __init__(self, fine_tuning = True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.resnet.fc.apply(init_weights)

        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.resnet.forward(x)

    def criterion(self, y, d):
        return self.loss(y, d)
