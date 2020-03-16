import torch.nn as nn
import torchvision.models as models
from nntools import NeuralNetwork
from config import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


class FinalResnet(NeuralNetwork):
    def __init__(self, fine_tuning=True):
        super().__init__()

        print('Parameterised ResNet with options : Pairwise {} \t Rank loss : {}'.format(PAIRWISE, RANK))

        self.adapt_conf = REGRESSOR_CONF['adaptive_layers_conf']
        self.fs = REGRESSOR_CONF['feature_sizes']
        self.rank = RANK
        self.pairwise = PAIRWISE
        self.reg_loss_type = LOSS
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*[i for i in self.resnet.children()][:-1])

        if self.pairwise:
            # modify the first layer of the model to take 6 channel input
            self.resnet_list = []
            self.resnet_list.append(nn.Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), padding=(3, 3), bias=True))
            for j in [i for i in self.resnet.children()][1:]:  # remove the top classifier layer
                self.resnet_list.append(j)

            self.resnet = nn.Sequential(*[i for i in self.resnet_list])

        linear_out = nn.Linear(self.fs[3], 1, bias=True)
        if self.rank:
            # modify the output to give 2 outputs instead of the single regression output
            linear_out = nn.Linear(self.fs[3], 2, bias=True)
            self.rank_loss = nn.BCELoss()
            self.sig = nn.Sigmoid()

        if self.reg_loss_type == 'L1':
            self.reg_loss = nn.SmoothL1Loss()
        elif self.reg_loss_type == 'L2':
            self.reg_loss = nn.MSELoss()

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
            linear_out
        ])

        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning

        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.resnet.forward(x)
        y = torch.squeeze(x)
        for layer in self.fc:
            y = layer(y)

        if self.rank:
            y[:, 1] = self.sig(y[:, 1])

        return y

    def forward_adaptive(self, x):

        y = self.resnet(x)
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

        if self.rank:
            y[:, 1] = self.sig(y[:, 1])

        return f_all, y

    def criterion(self, y, d):

        if self.rank:
            d = d.view([len(d), 2])
            reg_l = self.reg_loss(y[:, 0], d[:, 0])
            rank_l = self.rank_loss(y[:, 1], d[:, 1])
            loss = 0.7 * reg_l + 0.3 * rank_l
        else:
            reg_l = self.reg_loss(y.squeeze(), d.squeeze())
            loss = reg_l

        return loss
