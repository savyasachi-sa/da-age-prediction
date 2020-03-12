# import torch
# import torch.nn as nn
# import numpy as np
#
# from adaptive_resnet import ResnetAdaptive
#
#
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight.data)
#         m.bias.data.zero_()
#
#
# class AdaptiveRegressor(ResnetAdaptive):
#     def __init__(self, fine_tuning=True):
#         super().__init__(fine_tuning)
#
#     # def forward(self, x):
#     #     x_source = x['source']
#     #     x_target = x['target']
#     #
#     #     feat_source, out_source = ResnetAdaptive.forward(x_source)
#     #     feat_target, out_target = ResnetAdaptive.forward(x_target)
#     #
#     #     feat = {
#     #         'source': y_source,
#     #         'target': y_target
#     #     }
#     #
#     #
#     #     return y
#
#     ## Insert Loss Here
#     def criterion(self, y, d):
#         # CDAN
#         return self.loss(y['source'] - y['target'], d['source'] - d['target'])
#
#     def CDAN(input_list, ad_net):
#         features = input_list[0]
#         outputs = input_list[1]
#         features_both = torch.cat((features['source'], features['target']), dim=0)
#         outputs_both = torch.cat((outputs['source'], outputs['target']), dim=0)
#         softmax_out = nn.Softmax(dim=1)(outputs_both)
#         op_out = torch.bmm(softmax_out.unsqueeze(2), features_both.unsqueeze(1))
#         adv_out = ad_net(op_out.view(-1, softmax_out.size(1) * features_both.size(1)))
#         batch_size = softmax_out.size(0) // 2
#         dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
#         return nn.BCELoss()(adv_out,
#                             dc_target)  # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
