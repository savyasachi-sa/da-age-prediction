import torch
import numpy as np
from torch import nn
from config import *


def CDAN(features, ad_net, epoch):
    features_both = torch.cat((features['source'], features['target']), dim=0)
    adv_out = ad_net(features_both, epoch)
    batch_size = features_both.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(DEVICE)
    return nn.BCELoss()(adv_out,
                        dc_target)  # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
