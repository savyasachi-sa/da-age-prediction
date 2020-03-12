# from baseline_resnet import BaselineResnet
import torch.nn as nn
import torchvision.models as models
from nntools import NeuralNetwork
import torch
from utils import *


class ResnetAdaptive(NeuralNetwork):
    def __init__(self, feature_size=2048, fine_tuning=True):
        super().__init__()
        self.feature_size = feature_size
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning
        self.feature_layers = nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2,
                self.resnet.layer3,
                self.resnet.layer4,
                self.resnet.avgpool
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True)
        )

        self.fc.apply(init_weights)
        self.loss = nn.SmoothL1Loss()


    def forward(self, x):
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)
        y = self.fc(f)
        return f, y

    def feature_size(self):
        return self.feature_size

    def criterion(self, y, d):
        return self.loss(y, d)
