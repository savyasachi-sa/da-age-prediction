from final_resnet import FinalResnet
from nntools import Experiment
from stats_manager import AgeStatsManager
from config import *
import matplotlib.pyplot as plt
import os


# Todo: Add Plots
def report_loss(exp: Experiment, output_dir):
    if exp.epoch > 0:
        print("Training: Running Loss: {} \t Running Threshold-Based Acc: {}".format(
            exp.history[exp.epoch - 1][0]['loss'],
            exp.history[exp.epoch - 1][0]['accuracy']))
        print("Validation: Running Loss: {} \t Running Threshold-Based Acc: {}".format(
            exp.history[exp.epoch - 1][1]['loss'],
            exp.history[exp.epoch - 1][1]['accuracy']))
        plot_separate(exp, output_dir, plot_type='accuracy')
        plot_separate(exp, output_dir, plot_type='loss')


# separately plot and save figures for accuracy and loss:
def plot_separate(exp: Experiment, output_dir, plot_type='accuracy'):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Training ' + plot_type.upper()[0] + plot_type[1:])
    plt.plot([exp.history[k][0][plot_type] for k in range(exp.epoch)], label="training_{}".format(plot_type))
    plt.legend()
    plots_path = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    plt.savefig(plots_path + '/plot_training_{}.png'.format(plot_type))

    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Validation ' + plot_type.upper()[0] + plot_type[1:])
    plt.plot([exp.history[k][1][plot_type] for k in range(exp.epoch)], label="validation_{}".format(plot_type))
    plt.legend()
    plots_path = os.path.join(output_dir, "plots")

    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    plt.savefig(plots_path + '/plot_validation_{}.png'.format(plot_type))


def train(experiment_name, stats_manager=AgeStatsManager(WINDOW_THRESH)):
    net = FinalResnet()
    exp = Experiment(net, stats_manager, output_dir=experiment_name,
                     perform_validation_during_training=True)

    exp.run(num_epochs=ROOT_CONFIG['num_epochs'], plot=lambda e: report_loss(e, output_dir=experiment_name))
