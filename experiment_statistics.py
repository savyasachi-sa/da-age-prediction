import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import *  # USE Config only for datasets
from dataset_factory import *
import torch.utils.data as td
from final_resnet import FinalResnet
from stats_manager import *
from utils import get_checkpoint_path


def plot_save(output_dir, epochs, plot_name, x_name, y_s={}):
    plt.clf()
    plt.xlabel(x_name)
    x = [i + 1 for i in range(epochs)]
    for key in y_s.keys():
        yplot = y_s[key]
        plt.plot(x, yplot, label=key)
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

    def init(self, file_path, metrics_names=[], win_thresh=WINDOW_THRESH):
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

    def __init__(self, network, output_dir, checkpoint_path, is_adaptive=ADAPTIVE):
        self.checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.output_dir = output_dir
        self.batch_size = ROOT_CONFIG['batch_size']
        self.is_adaptive = 'Stats' in self.checkpoint
        self.network = network
        self.stats_manager = GenStatsManager()

    def optimizer_stats(self):
        optimizer_stats = self.checkpoint['Optimizer']
        stats = []
        for id, pg in enumerate(optimizer_stats['param_groups']):
            lr = pg['lr']
            betas = pg['betas']
            weight_decay = pg['weight_decay']
            stats.append("param_group: {}, lr: {}, betas: {}, weight_decay: {}".format(id, lr, betas, weight_decay))

        optimizer_stats_path = os.path.join(self.output_dir, 'optimizer_stats.txt')
        with open(optimizer_stats_path, 'w') as f:
            for content in stats:
                f.write(str(content) + '\n')

    def parent_epoch_losses(self):
        if (self.is_adaptive):
            try:
                parent_stats = self.checkpoint['Stats']
                all_stats = {
                    'train': [],
                    'val': [],
                    'target': []
                }
                for stat in parent_stats:
                    all_stats['train'].append(stat['training_loss'])
                    all_stats['val'].append(stat['val_loss'].item())
                    all_stats['target'].append(stat['tar_loss'].item())
                    
                plot_save(self.output_dir, len(parent_stats), 'all_parent_stats', 'epochs', all_stats)
            except BaseException as e:
                print('An exception occurred while ready parent epoch losses: {}'.format(e))

    def network_performance(self, metric_names=['loss'], calc_train=False):
        self.get_data_loader()
        self.network.load_state_dict(self.checkpoint['Net'])
        self.network.eval()
        with torch.no_grad():
            if (calc_train):
                self.stats_manager.init('{}/train_pred_stats.txt'.format(self.output_dir), metric_names)
                for x, d in self.train_loader:
                    x, d = x.to(self.network.device), d.to(self.network.device)
                    d = d.view([len(d), 1])
                    y = self.network.forward(x)
                    loss = self.network.criterion(y, d)
                    self.stats_manager.calc_store(loss.item(), x, y, d)
                self.stats_manager.summarize()

            self.stats_manager.init('{}/val_pred_stats.txt'.format(self.output_dir), metric_names)
            for x, d in self.val_loader:
                x, d = x.to(self.network.device), d.to(self.network.device)
                d = d.view([len(d), 1])
                y = self.network.forward(x)
                loss = self.network.criterion(y, d)
                self.stats_manager.calc_store(loss.item(), x, y, d)
            self.stats_manager.summarize()

            if (self.is_adaptive):
                self.stats_manager.init('{}/target_pred_stats.txt'.format(self.output_dir), metric_names)
                for x, d in self.target_loader:
                    x, d = x.to(self.network.device), d.to(self.network.device)
                    d = d.view([len(d), 1])
                    y = self.network.forward(x)
                    loss = self.network.criterion(y, d)
                    self.stats_manager.calc_store(loss.item(), x, y, d)
                self.stats_manager.summarize()

    def iteration_stats(self):
        history = self.checkpoint['History']
        losses = {}
        if (history == None or len(history) == 0):
            raise Exception('No iteration history found')
        if (isinstance(history[0], tuple)):  # checks if has validation scores
            losses['train_losses'] = np.array(history)[:, 0]
            losses['train_losses'] = [l['loss'] for l in losses['train_losses']]
            losses['val_losses'] = np.array(history)[:, 1]
            losses['val_losses'] = [l['loss'] for l in losses['val_losses']]
                                      
            plot_save(self.output_dir, len(history), 'iteration_losses', 'iteration', losses)
        else:
            losses['train_losses'] = np.array(history)
            losses['train_losses'] = [l['loss'] for l in losses['train_losses']]
            plot_save(self.output_dir, len(history), 'iteration_losses_only_train', 'iteration', losses)

    def get_data_loader(self):

        train_set, val_set, target_set = get_datasets()
        self.train_loader = td.DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                          pin_memory=True)
        self.val_loader = td.DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                        pin_memory=True)
        self.target_loader = td.DataLoader(target_set, batch_size=self.batch_size, shuffle=False,
                                           pin_memory=True)


# Test
# TODO: output director acc to each model
# TODO: dataset acc to each model

if __name__ == "__main__":
    
    MODEL_NAME = 'baseline_L1'
    
    checkpoint_path = get_checkpoint_path(MODEL_NAME)
    output_dir = os.path.join(STATS_OUTPUT_DIR,MODEL_NAME)
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    net = FinalResnet()
    net = net.to(DEVICE)
    exp = ExperimentStatistics(net, output_dir, checkpoint_path, is_adaptive=ADAPTIVE)
    try:
        exp.optimizer_stats()
    except BaseException as e:
        print('Error in saving OPTIMIZER stats for {}: {}'.format(checkpoint_path, e))

    try:
        exp.parent_epoch_losses()
    except BaseException as e:
        print('Error saving PARENT EPOCH statistics for {}: {}'.format(checkpoint_path, e))

    try:
        exp.iteration_stats()
    except BaseException as e:
        print('Error saving ITERATION statistics for {}: {}'.format(checkpoint_path, e))

    try:
        exp.network_performance(metric_names=['loss'], calc_train=False)
    except BaseException as e:
        print('Error saving NETWORK PERFORMANCE statistics for {}: {}'.format(checkpoint_path, e))
