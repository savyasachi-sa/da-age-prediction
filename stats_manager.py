import nntools as nt
import torch


class AgeStatsManager(nt.StatsManager):
    def __init__(self, win_thresh):
        super(AgeStatsManager, self).__init__()
        self.win_thresh = win_thresh

    def init(self):
        super(AgeStatsManager, self).init()
        self.running_win_acc = 0

    def accumulate(self, loss, x, y, d):
        super(AgeStatsManager, self).accumulate(loss, x, y, d)

        # compute the window_acc based on threshold on the prediction
        win_acc_per_batch = torch.mean((torch.abs(y - d) <= self.win_thresh).float())

        self.running_win_acc += win_acc_per_batch

    def summarize(self):
        loss = super(AgeStatsManager, self).summarize()
        thresh_acc = 100 * self.running_win_acc / self.number_update
        return {'loss': loss, 'accuracy': thresh_acc}

class AdaptiveStatsManager(nt.StatsManager):
    def __init__(self):
        super(AdaptiveStatsManager, self).__init__()
        self.running_target_loss = 0


    def init(self):
        super(AdaptiveStatsManager, self).init()
        self.running_target_loss = 0

    def accumulate(self, loss, target_domain_loss, x, y, d):
        super(AgeStatsManager, self).accumulate(loss, x, y, d)

        self.running_target_loss += target_domain_loss

    def summarize(self):
        loss = super(AgeStatsManager, self).summarize()
        thresh_acc = 100 * self.running_win_acc / self.number_update
        return {'loss': loss, 'accuracy': thresh_acc, 'target_loss': self.running_target_loss}
