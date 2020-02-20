from dataset import UTK
from dataset_design import DatasetDesign
from nntools import Experiment, StatsManager
from baseline_resnet import BaselineResnet
from config import *

# Todo: Add Plots
def report_loss(exp: Experiment):
    if exp.epoch > 0:
        print("Loss Value: ", exp.history[exp.epoch-1])


if __name__ == "__main__":
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
    stats_manager = StatsManager()
    optimizer = torch.optim.Adam(net.parameters(), lr=BASELINE_CONFIG['learning_rate'])
    exp = Experiment(net, train_dataset, val_dataset, stats_manager, optimizer, BASELINE_CONFIG, output_dir="baseline",
                     perform_validation_during_training=True)

    # Run Experiment

    exp.run(num_epochs=35, plot=lambda e: report_loss(e))
