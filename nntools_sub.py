import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod
import nntools2

class PermuteExperiment(nntools2.Experiment):
    def __init__(self, net, train_set, val_set, stats_manager, optimizer, config,
                 output_dir=None, perform_validation_during_training=False):
        super(PermuteExperiment, self).__init__(net, train_set, val_set, stats_manager, optimizer, config,
                 output_dir=None, perform_validation_during_training=False)
    
    def run(self, num_epochs, plot=None):
        torch.cuda.empty_cache()
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()
            for x, d in self.train_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)   
                
                #placeholders for scope
                d_switch = []
                d_same = []
                
                if d.shape[1] == 1: #only regression
                    d_switch = -1 * d
                    d_same = torch.zeros(d.shape)
                    
                
                elif d.shape[1] == 2: #both rank and regression
                    d_switch = -1 * d
                    d_switch[:,1] = 1 + d[:,1] # i.e d_rev = 0 if d = -1 and d_rev = 1 if d = 0
                    d_same = torch.zeros(d.shape)
                    d_same[:,1] = 0.5 #same person so max entropy encouraged
                #d = d.view([len(d), 1])
                d_switch = d_switch.to(self.net.device)
                d_same = d_same.to(self.net.device)
                
                
                loss_val = 0
                
                self.optimizer.zero_grad()
                
                y = self.net.forward(x)
                loss = self.net.criterion(y, d)
                loss_val += loss.item()
                loss.backward()
                self.optimizer.step()
                del loss
                
                self.optimizer.zero_grad()
                #identity encouragement:
                x_perm = x[:,[0,1,2,0,1,2]]
                y = self.net.forward(x_perm)
                loss = self.net.criterion(y, d_same)
                loss_val += loss.item()
                loss.backward()
                self.optimizer.step()
                del loss
                
                self.optimizer.zero_grad()
                x_perm = x[:,[3,4,5,3,4,5]]
                y = self.net.forward(x_perm)
                loss = self.net.criterion(y, d_same)
                loss_val += loss.item()
                loss.backward()
                self.optimizer.step()
                del loss
                
                self.optimizer.zero_grad()
                #negativity encouragement:
                x_perm = x[:,[3,4,5,0,1,2]]
                y = self.net.forward(x_perm)
                loss = self.net.criterion(y, d_switch)
                loss_val += loss.item()
                loss.backward()
                self.optimizer.step()
                del loss

                with torch.no_grad():
                    self.stats_manager.accumulate(loss_val, x, y, d)
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append(
                    (self.stats_manager.summarize(), self.evaluate()))
            print("Epoch {} (Time: {:.2f}s)".format(
                self.epoch, time.time() - s))
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))
