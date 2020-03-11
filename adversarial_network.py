from torch import nn

from nntools import NeuralNetwork
from utils import init_weights
from functions import ReverseLayerF
import numpy as np


class AdverserialNetwork(NeuralNetwork):
    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_features, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

    def forward(self, x, iter_num):
        coefficient = self.calc_coeff(iter_num)
        reversed_x = ReverseLayerF.apply(x, coefficient)
        f = self.ad_layer1(reversed_x)
        f = self.relu1(f)
        f = self.dropout1(f)
        f = self.ad_layer2(f)
        f = self.relu2(f)
        f = self.dropout2(f)
        y = self.ad_layer3(f)
        y = self.sigmoid(y)
        return y

    def criterion(self, y, d):
        print("shouldn't be calling this function")
