import torch
from dataset import UTK
import random


class PairwiseUTK(UTK):
    def __init__(self, csv_file, random_flips=True):
        super().__init__(csv_file, random_flips)
        random.seed(2)

    def __getitem__(self, idx):
        img1, label1 = self._get_item(idx)
        second_idx = random.randint(0, super().__len__() - 1)
        img2, label2 = self._get_item(second_idx)
        return torch.cat((img1, img2), 0), label1 - label2
