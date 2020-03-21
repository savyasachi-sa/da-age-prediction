from torch import nn

from nntools import NeuralNetwork
from utils import init_weights
from functions import ReverseLayerF
import numpy as np
from config import *


class AdverserialNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__()

        adv_in_feature = 0
        feature_sizes = REGRESSOR_CONF['feature_sizes']
        n_fc = REGRESSOR_CONF['adaptive_layers_conf']['n_fc']
        take_conv = REGRESSOR_CONF['adaptive_layers_conf']['conv']
        if (take_conv):
            adv_in_feature += feature_sizes[0]
        if (len(n_fc) > 0):
            adv_in_feature += sum(feature_sizes[n_fc[0]:n_fc[-1] + 1])

        hidden_size = ADV_CONF['hidden_size']

        self.ad_layer1 = nn.Linear(adv_in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

    def forward(self, x, iter_num):
        coefficient = self.calc_coeff(iter_num)
        reversed_x = ReverseLayerF.apply(x, coefficient)
        return self.forward_pass(reversed_x)

    def forward_pass(self, x):
        f = self.ad_layer1(x)
        f = self.relu1(f)
        f = self.dropout1(f)
        f = self.ad_layer2(f)
        f = self.relu2(f)
        f = self.dropout2(f)
        y = self.ad_layer3(f)
        y = self.sigmoid(y)
        return y

    def criterion(self, y, d):
        return nn.BCELoss()(y.squeeze(), d.squeeze())
