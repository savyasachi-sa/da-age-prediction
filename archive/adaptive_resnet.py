import torchvision.models as models
from nntools import NeuralNetwork
from utils import *
from config import *


class ResnetAdaptive(NeuralNetwork):
    def __init__(self, fine_tuning=True):
        super().__init__()
        self.adapt_conf = REGRESSOR_CONF['adaptive_layers_conf']
        self.fs = REGRESSOR_CONF['feature_sizes']
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

        self.fc = nn.ModuleList([
            nn.Linear(self.fs[0], self.fs[1], bias=True),
            nn.BatchNorm1d(self.fs[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.fs[1], self.fs[2], bias=True),
            nn.BatchNorm1d(self.fs[2]),
            nn.ReLU(inplace=True),
            nn.Linear(self.fs[2], self.fs[3], bias=True),
            nn.BatchNorm1d(self.fs[3]),
            nn.ReLU(inplace=True),
            nn.Linear(self.fs[3], self.fs[4], bias=True)
        ])

        self.fc.apply(init_weights)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x):

        y = self.feature_layers(x)
        y = y.squeeze()

        layers_adapt = ()

        if (self.adapt_conf['conv']):
            layers_adapt += (y,)

        fc_idx = 0
        for idx, layer in enumerate(self.fc):
            y = layer(y)
            if isinstance(layer, nn.Linear):
                fc_idx += 1
                if fc_idx in self.adapt_conf['n_fc']:
                    layers_adapt += (y,)

        f_all = torch.cat(layers_adapt, dim=1)
        return f_all, y

    def feature_size(self):
        return self.feature_size

    def criterion(self, y, d):
        return self.loss(y, d)
