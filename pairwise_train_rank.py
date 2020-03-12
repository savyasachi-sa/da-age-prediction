from pairwise_rank_dataset import PairwiseUTK
from dataset import UTK
from dataset_design import DatasetDesign
from nntools2 import Experiment
from stats_manager import AgeStatsManager
from final_resnet import FinalResnet
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


ethnicity = {
    "source": 0,
    "target": 1,
}

# Build Dataset
source_val_split_percentage = 80
_dataset = DatasetDesign(ethnicity, source_val_split_percentage)
train_dataset = PairwiseUTK(SOURCE_TRAIN_PATH, random_flips=True)
val_dataset = PairwiseUTK(SOURCE_VAL_PATH, random_flips=False)

# Setup Experiment
OUTPUT_DIR = "../da_logs/finalres_rankpair" 
net = FinalResnet(rank = True, pairwise = True)
net = net.to(DEVICE)
stats_manager = AgeStatsManager(WINDOW_THRESH, rank = True)
optimizer = torch.optim.Adam(net.parameters(), lr = BASELINE_CONFIG['learning_rate'])
exp = Experiment(net, train_dataset, val_dataset, stats_manager, optimizer, BASELINE_CONFIG, output_dir=OUTPUT_DIR,
                 perform_validation_during_training=True)

# Run Experiment

exp.run(num_epochs=250, plot=lambda e: report_loss(e, output_dir=OUTPUT_DIR))
