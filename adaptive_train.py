from adaptive_experiment import AdaptiveExperiment
from adversarial_network import AdverserialNetwork
from final_resnet import FinalResnet
from nntools import Experiment, StatsManager
from config import *
from utils import get_checkpoint_path


def report_loss(exp: Experiment):
    if exp.epoch > 0:
        print("Loss Value: ", exp.history[exp.epoch - 1])


def train(experiment_name, pretrained_model_name = None, stats_manager=StatsManager()):

    net = FinalResnet()

    if pretrained_model_name is not None:
        checkpoint_path = get_checkpoint_path(pretrained_model_name)
        data = torch.load(checkpoint_path, map_location=DEVICE)
    else:
        data = None

    adver_net = AdverserialNetwork()

    exp = AdaptiveExperiment(net, adver_net, stats_manager,
                             output_dir=experiment_name,
                             perform_validation_during_training=False, pretrained_data=data)

    exp.run(num_epochs=ROOT_CONFIG['num_epochs'], plot=lambda e: report_loss(e))
