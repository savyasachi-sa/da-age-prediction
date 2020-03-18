from torch import nn
import json


def write_to_file(path, data):
    """
        This method will save the passed data in the specified file
        path: path to the file in which you wish to write the data.
                Example - './data/users'
        data: a python object you wish to write in the file.
                Example - In your case, it should be a list of dictionary objects corresponding to either User or Move.
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


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


def get_checkpoint_path(model_dir):
    return './models/' + model_dir + '/checkpoint.pth.tar'


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
