import torch
import numpy as np
from config import *


def CDAN(features, ad_net, epoch):
    features_both = torch.cat((features['source'], features['target']), dim=0)
    adv_out = ad_net(features_both, epoch)
    dc_target = torch.from_numpy(
        np.array([[1]] * features['source'].size(0) + [[0]] * features['target'].size(0))).float().to(DEVICE)
    return ad_net.criterion(adv_out, dc_target)