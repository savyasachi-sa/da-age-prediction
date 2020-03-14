from adaptive_experiment import AdaptiveExperiment
from adversarial_network import AdverserialNetwork
from final_resnet import FinalResnet
from nntools import Experiment, StatsManager
from config import *


# Todo: Add Plots
def report_loss(exp: Experiment):
    if exp.epoch > 0:
        print("Loss Value: ", exp.history[exp.epoch - 1])


def train(experiment_name, stats_manager=StatsManager()):

    net = FinalResnet()
    adver_net = AdverserialNetwork()

    exp = AdaptiveExperiment(net, adver_net, stats_manager,
                             output_dir=experiment_name,
                             perform_validation_during_training=False)

    exp.run(num_epochs=ROOT_CONFIG['num_epochs'], plot=lambda e: report_loss(e))
