import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from config import * #USE Config only for datasets
from dataset import UTK
import torch.utils.data as td

from stats_manager import *

class GenStatsManager(object):

    def __init__(self):
        self.init()

    def __repr__(self):
        return self.__class__.__name__

    def init(self, metrics_names = [], win_thresh=3):
        self.number_update = 0
        self.win_thresh = win_thresh
        self.metrics = dict()
        for key in metrics_names:
            self.metrics[key] = 0

    def calc_store(self, loss, x, y, d):
        metric_instance = dict()
        if 'loss' in self.metrics:
            metric_instance['loss'] = loss
        if 'accuracy' in self.metrics:
            metric_instance['accuracy'] = torch.mean((torch.abs(y - d) <= self.win_thresh).float())
        self._accumulate(metric_instance)
        return  metric_instance

    def _accumulate(self, metrics= {}):
        for key, val in metrics:
            self.metrics[key] += val
        self.number_update += 1

    def summarize(self):
        for key, val in self.metrics:
            self.metrics[key] /= self.number_update
        return self.metrics

class ExperimentStatistics():

    def __init__(self, network, output_dir, checkpoint_path, is_adaptive=False):
        self.checkpoint = torch.load(checkpoint_path, None)
        self.output_dir = output_dir
        self.batch_size = 16 # TODO: any would work?
        self.is_adaptive = is_adaptive
        self.network = network
        self.parent_stats = self.checkpoint['Stats'] #TODO: check key
        self.train_losses = []
        self.val_losses = [] #only one value if no val values were stored
        self.target_loss = [] #will not be calculate for each iterations
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
        parent_stats = self.checkpoint['stats']
        self.parent_train = np.array(parent_stats)[:,0]
        self.parent_val = np.array(parent_stats)[:,1]
        self.parent_target = np.array(parent_stats)[:,2]
        #TODO: plot and save


    def network_predictions(self, metric_names = ['loss']):
        self.get_data_loader()
        self.network.load_state_dict(self.checkpoint['Net'])
        self.network.eval()
        self.stats_manager.init()
        with torch.no_grad():
            self.stats_manager.init(metric_names)
            for x, d in self.train_loader:
                x, d = x.to(self.network.device), d.to(self.network.device)
                d = d.view([len(d), 1])
                y = self.network.forward(x)
                loss = self.network.criterion(y, d)
                self.stats_manager.calc_store(loss, x, y, d)

            self.stats_manager.init(metric_names)
            for x, d in self.val_loader:
                x, d = x.to(self.network.device), d.to(self.network.device)
                d = d.view([len(d), 1])
                y = self.network.forward(x)
                loss = self.network.criterion(y, d)
                self.stats_manager.calc_store(loss, x, y, d)

            self.stats_manager.init(metric_names)
            if(self.is_adaptive):
                for x, d in self.target_loader:
                    x, d = x.to(self.network.device), d.to(self.network.device)
                    d = d.view([len(d), 1])
                    y = self.network.forward(x)
                    loss = self.network.criterion(y, d)
                    self.stats_manager.calc_store(loss, x, y, d)



    def iteration_stats(self):
        history = self.checkpoint['History']
        if (history==None or len(history) == 0):  #TODO: throw error
            return
        if (isinstance(history[0], tuple)):  # checks if has validation scores
            self.train_losses = np.array(history)[:, 0] #TODO: check history if np array
            self.val_losses = np.array(history)[:, 1] #TODO: in text file?
            #TODO: plot both

    def get_data_loader(self):
        #TODO: get val and target data loader
        train_set = UTK(SOURCE_TRAIN_PATH, random_flips=False)
        val_set = UTK(SOURCE_VAL_PATH, random_flips=False)
        target_set = UTK(TARGET_TRAIN_PATH, random_flips=False)
        self.train_loader = td.DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                        pin_memory=True)
        self.val_loader = td.DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                        pin_memory=True)
        self.target_loader = td.DataLoader(target_set, batch_size=self.batch_size, shuffle=True,
                                           pin_memory=True)


    def plot_save(self, epochs, name, y_s ={}):
        plt.clf()
        plt.xlabel('Epoch')
        x = [i+1 for i in range(epochs)]
        for key,y in y_s:
            plt.ylabel(key.upper())
            plt.plot(x, y[key])
        plt.legend()
        plots_path = os.path.join(self.output_dir, "plots")

        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        plt.savefig(plots_path + '/{}.png'.format(name))
