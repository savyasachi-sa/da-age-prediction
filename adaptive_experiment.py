from nntools import Experiment
import time
import torch
import torch.utils.data as td


class AdaptiveExperiment(Experiment):
    def __init__(self, net, train_set, val_set, target_set, stats_manager, optimizer, config,
                 output_dir=None, perform_validation_during_training=False):
        super().__init__(net, train_set, val_set, stats_manager, optimizer, config,
                 output_dir=output_dir, perform_validation_during_training=perform_validation_during_training)

        self.target_loader = td.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                                     pin_memory=True)


    def run(self,num_epochs, plot=None):
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)

        if start_epoch != 0:
            iter_source = iter(self.train_loader)
            iter_target = iter(self.target_loader)

        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()

            len_train_source = len(self.train_loader)
            len_train_target = len(self.target_loader)

            if epoch % len_train_source == 0:
                iter_source = iter(self.train_loader)
            if epoch % len_train_target == 0:
                iter_target = iter(self.target_loader)

            x_source, d_source = iter_source.next()
            x_target, d_target = iter_target.next()

            x_source, d_source = x_source.to(self.net.device), d_source.to(self.net.device)
            x_target, d_target = x_target.to(self.net.device), d_target.to(self.net.device)

            d_source = d_source.view([len(d_source), 1])
            d_target = d_target.view([len(d_target), 1])

            x = {
                'source': x_source,
                'target': x_target
            }

            d = {
                'source': d_source,
                'target': d_target
            }

            self.optimizer.zero_grad()
            y = self.net.forward(x)
            loss = self.net.criterion(y, d)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.stats_manager.accumulate(loss.item(), x, y, d)

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
