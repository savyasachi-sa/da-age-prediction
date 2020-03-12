import nntools as nt
import torch


class AgeStatsManager(nt.StatsManager):
    def __init__(self, win_thresh, rank):
        super(AgeStatsManager, self).__init__()
        self.win_thresh = win_thresh
        self.rank = rank

    def init(self):
        super(AgeStatsManager, self).init()
        self.running_win_acc = 0

    def accumulate(self, loss, x, y, d):
        super(AgeStatsManager, self).accumulate(loss, x, y, d)
        if self.rank:
            # compute the window_acc based on threshold on the prediction
            win_acc_per_batch = torch.mean((torch.abs(y[0,:] - d[0,:]) <= self.win_thresh).float())
        else:
            win_acc_per_batch = torch.mean((torch.abs(y - d) <= self.win_thresh).float())
        self.running_win_acc += win_acc_per_batch

    def summarize(self):
        loss = super(AgeStatsManager, self).summarize()
        thresh_acc = 100 * self.running_win_acc / self.number_update
        return {'loss': loss, 'accuracy': thresh_acc}
