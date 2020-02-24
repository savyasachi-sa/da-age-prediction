from adaptive_experiment import AdaptiveExperiment
from adaptive_regressor import AdaptiveRegressor
from dataset import UTK
from dataset_design import DatasetDesign
from nntools import Experiment, StatsManager
from baseline_resnet import BaselineResnet
from config import *


# Todo: Add Plots
def report_loss(exp: Experiment):
    if exp.epoch > 0:
        print("Loss Value: ", exp.history[exp.epoch - 1])


if __name__ == "__main__":
    ethnicity = {
        "source": 0,
        "target": 1,
    }

    # Build Dataset
    source_val_split_percentage = 80
    _dataset = DatasetDesign(ethnicity, source_val_split_percentage)
    train_dataset = UTK(SOURCE_TRAIN_PATH, random_flips=True)
    val_dataset = UTK(SOURCE_VAL_PATH, random_flips=False)
    target_dataset = UTK(TARGET_TRAIN_PATH, random_flips=True)

    # Setup Experiment
    print(DEVICE)
    net = AdaptiveRegressor()
    net = net.to(DEVICE)
    stats_manager = StatsManager()
    optimizer = torch.optim.Adam(net.parameters(), lr=BASELINE_CONFIG['learning_rate'])
    exp = AdaptiveExperiment(net, train_dataset, val_dataset, target_dataset, stats_manager, optimizer, BASELINE_CONFIG,
                             output_dir="adaptive",
                             perform_validation_during_training=False)

    # Run Experiment

    exp.run(num_epochs=10, plot=lambda e: report_loss(e))
