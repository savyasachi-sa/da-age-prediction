from dataset_factory import get_datasets
from cdan import *

from config import *

import os
import time
import torch
import torch.utils.data as td
from utils import write_to_file


class AdaptiveExperiment(object):

    def __init__(self, net, adver_net, stats_manager,
                 output_dir=None, perform_validation_during_training=False):

        config = ROOT_CONFIG
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        num_workers = config['num_workers']

        net = net.to(DEVICE)
        adver_net = adver_net.to(DEVICE)

        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
        optimizer_adv = torch.optim.Adam(adver_net.parameters(), lr=config['learning_rate'])

        self.output_dir = output_dir
        self.config = config
        self.adv_net = adver_net
        self.adv_optimizer = optimizer_adv
        self.best_loss = 1e6

        train_dataset, val_dataset, target_dataset = get_datasets()

        # Define data loaders
        self.train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          pin_memory=True)
        self.val_loader = td.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                        pin_memory=True)
        self.target_loader = td.DataLoader(target_dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=True)
        # Initialize history
        history = []
        stats = []

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            # with open(config_path, 'r') as f:
            #     if f.read()[:-1] != repr(self):
            #         print(f.read()[:-1], repr(self))
            #         raise ValueError(
            #             "Cannot create this experiment: "
            #             "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        
        if ADVERSARIAL_FLAG:
            return {'Net': self.net,
                    "AdvNet": self.adv_net,
                    'TrainSet': self.train_dataset,
                    'ValSet': self.val_dataset,
                    'Optimizer': self.optimizer,
                    'AdvOptimizer': self.adv_optimizer,
                    'StatsManager': self.stats_manager,
                    'BatchSize': self.batch_size,
                    'PerformValidationDuringTraining': self.perform_validation_during_training}
        
        return {'Net': self.net,
#                 "AdvNet": self.adv_net,
                'TrainSet': self.train_dataset,
                'ValSet': self.val_dataset,
                'Optimizer': self.optimizer,
#                 'AdvOptimizer': self.adv_optimizer,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}
        

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        if ADVERSARIAL_FLAG:
            return {'Net': self.net.state_dict(),
                    'AdvNet': self.adv_net.state_dict(),
                    'Optimizer': self.optimizer.state_dict(),
                    'AdvOptimizer': self.adv_optimizer.state_dict(),
                    'History': self.history,
                    'Stats': self.stats}
        return {'Net': self.net.state_dict(),
                    'Optimizer': self.optimizer.state_dict(),
                    'History': self.history,
                    'Stats': self.stats}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        
        if ADVERSARIAL_FLAG:
            self.adv_net.load_state_dict(checkpoint['AdvNet'])
            self.adv_optimizer.load_state_dict(checkpoint['AdvOptimizer'])
        
        self.history = checkpoint['History']
        self.stats = checkpoint['Stats']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)
        if ADVERSARIAL_FLAG:
            for adv_state in self.adv_optimizer.state.values():
                for k, v in adv_state.items():
                    if isinstance(v, torch.Tensor):
                        adv_state[k] = v.to(self.adv_net.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.net.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, plot=None):
        self.net.train()
        if ADVERSARIAL_FLAG:
            self.adv_net.train()
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
                if (epoch - len_train_source >= 0):
                    epoch_loss = self.history[(epoch - len_train_source):epoch]
                    training_losses = [e for e in epoch_loss]
                    training_loss = sum(training_losses) / len(training_losses)
                    val_loss, tar_loss = self.evaluate()
                    print("**** Parent epoch now ***** ", training_loss, val_loss, tar_loss)
                    stats = {'training_loss': training_loss, 'val_loss': val_loss, 'tar_loss': tar_loss}
                    self.stats.append(stats)
                    writable_stats = [{'training': d['training_loss'], 'validation': d['val_loss'].item(),
                                       'target': d['tar_loss'].item()} for d in self.stats]
                    write_to_file(self.output_dir + '/stats.txt', writable_stats)

                iter_source = iter(self.train_loader)
            if epoch % len_train_target == 0:
                iter_target = iter(self.target_loader)
                
            id_loss = 0
            x_source = []
            x_target = []
            t_source = []
            t_target = []
            
            if PAIRWISE:
                x1_s, x2_s, t_source = iter_source.next()
                x1_t, x2_t, t_target = iter_target.next()
                
                x1_s, x2_s, t_source = x1_s.to(self.net.device), x2_s.to(self.net.device),t_source.to(self.net.device)
                x1_t, x2_t, t_target = x1_t.to(self.net.device), x2_t.to(self.net.device), t_target.to(self.net.device)
                
                x_source = torch.cat([x1_s, x2_s], dim=1)
                x_target = torch.cat([x1_t, x2_t], dim=1)
                
                if IDENTITY_FLAG:
                    d_switch = []
                    d_same = []
                    if not RANK:  # only regression
                        d_switch = -1 * t_source
                        d_same = torch.zeros(t_source.shape)


                    else:  # both rank and regression
                        d_switch = -1 * t_source
                        d_switch[:, 1] = 1 + t_source[:, 1]  # i.e d_rev = 0 if d = -1 and d_rev = 1 if d = 0
                        d_same = torch.zeros(t_source.shape)
                        d_same[:, 1] = 0.5  # same person so max entropy encouraged

                    d_switch = d_switch.to(self.net.device)
                    d_same = d_same.to(self.net.device)

                    self.optimizer.zero_grad()
                    y = self.net.forward(torch.cat((x1_s, x2_s), 1))
                    id_loss = self.net.criterion(y, t_source)

                    y = self.net.forward(torch.cat((x1_s, x1_s), 1))
                    id_loss += self.net.criterion(y, d_same)

                    y = self.net.forward(torch.cat((x2_s, x2_s), 1))
                    id_loss += self.net.criterion(y, d_same)

                    y = self.net.forward(torch.cat((x2_s, x1_s), 1))
                    id_loss += self.net.criterion(y, d_switch)

            else:
                x_source, t_source = iter_source.next()
                x_target, t_target = iter_target.next()
            
            x_source, t_source = x_source.to(self.net.device), t_source.to(self.net.device)
            x_target, t_target = x_target.to(self.net.device), t_target.to(self.net.device)

            t_source = t_source.unsqueeze(1)
            t_target = t_target.unsqueeze(1)

            x = {
                'source': x_source,
                'target': x_target
            }

            t = {
                'source': t_source,
                'target': t_target
            }

            self.optimizer.zero_grad()
            if ADVERSARIAL_FLAG:
                self.adv_optimizer.zero_grad()
            features_source, outputs_source = self.net.forward_adaptive(x['source'])
            features_target, outputs_target = self.net.forward_adaptive(x['target'])

            features = {
                'source': features_source,
                'target': features_target
            }
            
            mmd_size = min(features['source'].shape[0],features['target'].shape[0])
            outputs = {
                'source': outputs_source,
                'target': outputs_target
            }

            cdan_loss = 0
            if ADVERSARIAL_FLAG:
                cdan_loss = CDAN(features, self.adv_net, self.epoch)
            mmd_loss = 0
            smooth_loss = 0
            if MMD_FLAG:
                mmd_loss = self.net.mmd_criterion({'source':features['source'][:mmd_size,:],
                                                  'target' : features['target'][:mmd_size,:]})

            if SMOOTH_FLAG:
                smooth_loss = self.net.smoothing_criterion(features, outputs)

            
            loss = self.net.criterion(outputs['source'], t['source'])
            total_loss = loss + cdan_loss * self.config['cdan_hypara'] + mmd_loss * self.config['mmd_hypara'] + smooth_loss * self.config['smooth_hypara'] + id_loss * self.config['id_hypara']
            total_loss.backward()
            self.optimizer.step()
            if ADVERSARIAL_FLAG:
                self.adv_optimizer.step()

            print('Epoch: {}, TRAIN, rgre_loss: {}, total_loss: {}'.format(self.epoch, loss.item(), total_loss.item()))
#             print('Iteration Number = ', epoch)
            with torch.no_grad():
                self.stats_manager.accumulate(total_loss.item(), None, None,
                                              None)  # x,outputs, t are not used by stats manager

            self.history.append(self.stats_manager.summarize())

#             print("Epoch {} (Time: {:.2f}s)".format(
#                 self.epoch, time.time() - s))
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self):
        self.net.eval()
        val_loss = 0
        tar_loss = 0
        self.net.eval()
        with torch.no_grad():
            for x, d in self.val_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)
                f, y = self.net.forward_adaptive(x)
                loss = self.net.criterion(y, d)
                val_loss += loss

        val_loss = val_loss / len(self.val_loader)

        with torch.no_grad():
            for x, d in self.target_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)
                d = d.view([len(d), 1])
                f, y = self.net.forward_adaptive(x)
                loss = self.net.criterion(y, d)
                tar_loss += loss

        tar_loss = tar_loss / len(self.target_loader)

        self.net.train()

        print('Epoch: {}', self.epoch)
        print('VAL_rgre_loss: {}'.format(val_loss / len(self.val_loader)))
        print('TAR_rgre_loss: {}'.format(tar_loss / len(self.target_loader)))

        return val_loss, tar_loss
