from dataset import UTK
from dataset_design import DatasetDesign
from nntools import Experiment, StatsManager
from stats_manager import AgeStatsManager
from baseline_resnet import BaselineResnet
from config import *

import os
import matplotlib.pyplot as plt

# Todo: Add Plots
def report_loss(exp: Experiment, output_dir):
    if exp.epoch > 0:
        print("Running Loss: {} \t Running Threshold-Based Acc: {}".format(exp.history[exp.epoch-1][0]['loss'],exp.history[exp.epoch-1][0]['accuracy']))
        plot_separate(exp, output_dir, plot_type = 'accuracy')
        plot_separate(exp, output_dir, plot_type = 'loss')

# def plot_metrics(exp : Experiment, output_dir):
    
#     fig, axes = plt.subplots(2,2)
#     axes = prep_plots(axes)
#     axes[0][0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)],label="training loss")
#     axes[0][1].plot([exp.history[k][0]['accuracy'] for k in range(exp.epoch)],label="training accuracy")
#     axes[0][0].legend()
#     axes[0][1].legend()
#     plt.tight_layout(pad = 3)
#     fig.canvas.draw()
#     plt.savefig('./baseline/plots/plot.png')
#     plt.show()

# def prep_plots(axes):
#     axes[0][0].set_xlabel('Epoch')
#     axes[0][0].set_ylabel('Loss')
#     axes[0][1].set_xlabel('Epoch')
#     axes[0][1].set_ylabel('Threshold-Based Accuracy')
              
#     return axes
              
#separately plot and save figures for accuracy and loss:
def plot_separate(exp : Experiment, output_dir, plot_type = 'accuracy'):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(plot_type.upper()[0] + plot_type[1:])
    plt.plot([exp.history[k][0][plot_type] for k in range(exp.epoch)],label="training {}".format(plot_type))
    plt.legend()
    plt.savefig('./baseline/plots/plot_{}.png'.format(plot_type))
    
ethnicity = {
    "source": 0,
    "target": 1,
}

# Build Dataset
source_val_split_percentage = 80
_dataset = DatasetDesign(ethnicity, source_val_split_percentage)
train_dataset = UTK(SOURCE_TRAIN_PATH)
val_dataset = UTK(SOURCE_VAL_PATH)

# Setup Experiment
net = BaselineResnet()
net = net.to(DEVICE)
stats_manager = AgeStatsManager(WINDOW_THRESH)
optimizer = torch.optim.Adam(net.parameters(), lr=BASELINE_CONFIG['learning_rate'])
exp = Experiment(net, train_dataset, val_dataset, stats_manager, optimizer, BASELINE_CONFIG, output_dir="baseline",
                 perform_validation_during_training=True)

# Run Experiment

exp.run(num_epochs=100, plot=lambda e: report_loss(e, output_dir = './baseline'))