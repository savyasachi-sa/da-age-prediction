import torch.nn as nn
from baseline_resnet import BaselineResnet


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


class AdaptiveRegressor(BaselineResnet):
    def __init__(self, fine_tuning = True):
        super().__init__(fine_tuning)

    def forward(self, x):

        x_source = x['source']
        x_target = x['target']

        y_source = self.resnet(x_source)
        y_target = self.resnet(x_target)

        y = {
            'source': y_source,
            'target': y_target
        }

        return y

    ## Insert Loss Here
    def criterion(self, y, d):
        return self.loss(y['source']-y['target'], d['source']-d['target'])
