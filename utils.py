from torch import nn


def get_experiment_name(name, pairwise, ranking, adaptive, loss):
    out = "./models/" + name

    if pairwise:
        out = out + "_Pairwise"

    if ranking:
        out = out + "_Ranking"

    if adaptive:
        out = out + "_Adaptive"

    out = out + "_" + loss

    return out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
