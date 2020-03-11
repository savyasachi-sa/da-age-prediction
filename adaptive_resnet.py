# from baseline_resnet import BaselineResnet
import torch.nn as nn
import torchvision.models as models
from nntools import NeuralNetwork
import torch
from utils import *


class ResnetAdaptive(NeuralNetwork):
    def __init__(self, feature_size=256, fine_tuning=True):
        super().__init__()
        self.feature_size = feature_size
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        self.bottleneck =  nn.Linear(2048, self.feature_size, bias=True)
        # self.bn_bottleneck =  nn.BatchNorm1d(self.feature_size)
        self.bottleneck.apply(init_weights)

        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc = nn.Linear(self.feature_size, 1)
        self.fc.apply(init_weights)


    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        # x = torch.flatten(x, 1)
        y = self.fc(x)
        return x, y

    def feature_size(self):
        return self.feature_size

    def criterion(self, y, d):
        return nn.MSELoss()(y, d)
