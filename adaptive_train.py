from adaptive_experiment import AdaptiveExperiment
# from adaptive_regressor import AdaptiveRegressor
from adaptive_resnet import ResnetAdaptive
from adversarial_network import AdverserialNetwork
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
    adv_in_feature = 0
    feature_sizes = REGRESSOR_CONF['feature_sizes']
    n_fc = REGRESSOR_CONF['adaptive_layers_conf']['n_fc']
    take_conv = REGRESSOR_CONF['adaptive_layers_conf']['conv']
    if (take_conv):
        adv_in_feature += feature_sizes[0]
    adv_in_feature += sum(feature_sizes[n_fc[0]:n_fc[-1]+1])
    net = ResnetAdaptive(REGRESSOR_CONF['feature_sizes'], REGRESSOR_CONF['finetune'])
    adver_net = AdverserialNetwork(adv_in_feature, ADV_CONF['hidden_size'])
    net = net.to(DEVICE)
    adver_net = adver_net.to(DEVICE)
    stats_manager = StatsManager()
    optimizer = torch.optim.Adam(net.parameters(), lr=BASELINE_CONFIG['learning_rate'])
    optimizer_adv = torch.optim.Adam(adver_net.parameters(), lr=BASELINE_CONFIG['learning_rate'])
    exp = AdaptiveExperiment(net, adver_net, train_dataset, val_dataset, target_dataset, stats_manager, optimizer,
                             optimizer_adv, BASELINE_CONFIG,
                             output_dir="adaptive_v2",
                             perform_validation_during_training=True)

    # Run Experiment
    exp.run(num_epochs=BASELINE_CONFIG['num_epochs'], plot=lambda e: report_loss(e))
