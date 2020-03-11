import torch
import numpy as np
from torch import nn
from config import *


def CDAN(input_list, ad_net):
    features = input_list[0]
    outputs = input_list[1]
    features_both = torch.cat((features['source'], features['target']), dim=0)
    outputs_both = torch.cat((outputs['source'], outputs['target']), dim=0)
    softmax_out = nn.Softmax(dim=1)(outputs_both)
    op_out = torch.bmm(softmax_out.unsqueeze(2), features_both.unsqueeze(1))
    adv_out = ad_net(op_out.view(-1, softmax_out.size(1) * features_both.size(1)))
    batch_size = softmax_out.size(0) // 2

    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(DEVICE)
    return nn.BCELoss()(adv_out,
                        dc_target)  # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
