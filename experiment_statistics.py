import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from config import *  # USE Config only for datasets
from dataset_factory import *
import torch.utils.data as td

from final_resnet import FinalResnet
from stats_manager import *


def plot_save(output_dir, epochs, plot_name, x_name, y_s={}):
    plt.clf()
    plt.xlabel(x_name)
    x = [i + 1 for i in range(epochs)]
    for key in y_s:
        plt.plot(x, y_s[key], label = key)
    plt.legend()
    plots_path = os.path.join(output_dir, "")

    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    plt.savefig(plots_path + '/{}.png'.format(plot_name))


class GenStatsManager(object):

    def __init__(self):
        self.init(None, [])

    def __repr__(self):
        return self.__class__.__name__

    def init(self, file_path, metrics_names=[], win_thresh=3):
        self.number_update = 0
        self.win_thresh = win_thresh
        self.metrics = dict()
        self.file_path = file_path
        for key in metrics_names:
            self.metrics[key] = 0

    def calc_store(self, loss, x, y, d):
        metric_instance = dict()
        if 'loss' in self.metrics:
            metric_instance['loss'] = loss
        if 'accuracy' in self.metrics:
            metric_instance['accuracy'] = torch.mean((torch.abs(y - d) <= self.win_thresh).float())
        self._accumulate(metric_instance)
        return metric_instance

    def _accumulate(self, metrics={}):
        for key in metrics:
            self.metrics[key] += metrics[key]
        self.number_update += 1

    def summarize(self, save=True):
        for key in self.metrics:
            self.metrics[key] /= self.number_update
        if (save):
            self._save_file()
        return self.metrics

    def _save_file(self):
        with open(self.file_path, 'w') as f:
            f.write(str(self.metrics))


class ExperimentStatistics():

    def __init__(self, network, output_dir, checkpoint_path, is_adaptive=False):
        self.checkpoint = torch.load(checkpoint_path, None)
        self.output_dir = output_dir
        self.batch_size = 16  # TODO: any would work?
        self.is_adaptive = is_adaptive
        self.network = network
        self.train_losses = []
        self.val_losses = []  # only one value if no val values were stored
        self.target_loss = []  # will not be calculate for each iterations
        self.stats_manager = GenStatsManager()

    def optimizer_stats(self):
        optimizer_stats = self.checkpoint['Optimizer']
        stats = []
        for id, pg in enumerate(optimizer_stats['param_groups']):
            lr = pg['lr']
            betas = pg['betas']
            weight_decay = pg['weight_decay']
            stats.append("param_group: {}, lr: {}, betas: {}, weight_decay: {}".format(id, lr, betas, weight_decay))

        with open('./optimizer_stats.txt', 'w') as f:
            for content in stats:
                f.write(str(content) + '\n')

    def parent_epoch_losses(self):
        if (self.is_adaptive):
            try:
                parent_stats = self.checkpoint['Stats']
                all_stats = {}
                all_stats['train'] = np.array(parent_stats)[:, 0]
                all_stats['val'] = np.array(parent_stats)[:, 1]
                all_stats['target'] = np.array(parent_stats)[:, 2]
                plot_save(self.output_dir, len(parent_stats), 'all_parent_stats', 'epochs', all_stats)
            except BaseException as e:
                print('An exception occurred while ready parent epoch losses: {}'.format(e))

    def network_predictions(self, metric_names=['loss']):
        self.get_data_loader()
        self.network.load_state_dict(self.checkpoint['Net'])
        self.network.eval()
        with torch.no_grad():
            self.stats_manager.init('{}train_model_stats.txt'.format(self.output_dir), metric_names)
            for x, d in self.train_loader:
                x, d = x.to(self.network.device), d.to(self.network.device)
                d = d.view([len(d), 1])
                y = self.network.forward(x)
                loss = self.network.criterion(y, d)
                self.stats_manager.calc_store(loss, x, y, d)
                self.stats_manager.summarize()

            self.stats_manager.init('{}val_model_stats.txt'.format(self.output_dir), metric_names)
            for x, d in self.val_loader:
                x, d = x.to(self.network.device), d.to(self.network.device)
                d = d.view([len(d), 1])
                y = self.network.forward(x)
                loss = self.network.criterion(y, d)
                self.stats_manager.calc_store(loss, x, y, d)
                self.stats_manager.summarize()

            self.stats_manager.init('{}target_model_stats.txt'.format(self.output_dir), metric_names)
            if (self.is_adaptive):
                for x, d in self.target_loader:
                    x, d = x.to(self.network.device), d.to(self.network.device)
                    d = d.view([len(d), 1])
                    y = self.network.forward(x)
                    loss = self.network.criterion(y, d)
                    self.stats_manager.calc_store(loss, x, y, d)
                    self.stats_manager.summarize()

    def iteration_stats(self):
        history = self.checkpoint['History']
        losses = {}
        if (history == None or len(history) == 0):  # TODO: throw error
            return
        if (isinstance(history[0], tuple)):  # checks if has validation scores
            losses['train_losses'] = np.array(history)[:, 0]  # TODO: check history if np array
            losses['val_losses'] = np.array(history)[:, 1]  # TODO: in text file?
            plot_save(self.output_dir, len(history), 'iteration_losses', 'iteration', losses)
        else:
            losses['train_losses'] = np.array(history)
            plot_save(self.output_dir, len(history), 'iteration_losses_only_train', 'iteration', losses)

    def get_data_loader(self):
        # TODO: get val and target data loader
        train_set, val_set, target_set = get_datasets()
        self.train_loader = td.DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                          pin_memory=True)
        self.val_loader = td.DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                        pin_memory=True)
        self.target_loader = td.DataLoader(target_set, batch_size=self.batch_size, shuffle=True,
                                           pin_memory=True)





# Test
# net = FinalResnet()
# exp = ExperimentStatistics(net, './results/', './models/astuti_test1_Adaptive_L1/checkpoint.pth.tar', is_adaptive=True)
# exp.optimizer_stats()
# exp.parent_epoch_losses()
#exp.iteration_stats()
#exp.network_predictions()
