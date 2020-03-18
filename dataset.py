from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torch
import random

import numpy as np
from config import RANK, LABEL_NORM


class UTK(Dataset):
    def __init__(self, csv_file, random_flips=True):
        self.data = pd.read_csv(csv_file)
        self.random_flips = random_flips
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip()
            # transforms.RandomVerticalFlip()
        ])
        self.norm_factor = 1
        if LABEL_NORM:
            self.norm_factor = np.max(self.data['labels']) - np.min(self.data['labels'])

    def __len__(self):
        return len(self.data)
    
    def getNormFactor(self):
        return self.norm_factor
        
    def _get_item(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_full = Image.open(img_name).convert('RGB')
        label_age = self.data.iloc[idx, 1]

        if self.random_flips:
            self.flip(img_full)

        img_full = self.normalize(img_full)

        label_age = torch.tensor(label_age).type(torch.FloatTensor)
        
        return img_full, label_age * 1./ self.norm_factor

    def __getitem__(self, idx):
        return self._get_item(idx)


class PairwiseUTK(UTK):
    def __init__(self, csv_file, random_flips=True):
        super().__init__(csv_file, random_flips)
        random.seed(2)
        self.norm_factor = 1
        if LABEL_NORM:
            self.norm_factor = np.max(self.data['labels']) - np.min(self.data['labels'])


    def __getitem__(self, idx):
        img1, label1 = self._get_item(idx)
        second_idx = random.randint(0, super().__len__() - 1)
        img2, label2 = self._get_item(second_idx)

        if RANK:
            rank = 0
            if label1 - label2 > 0:
                rank = 1
            return torch.cat((img1, img2), 0), torch.Tensor(((label1 - label2) / self.norm_factor, rank))
        return img1, img2, (label1 - label2) * 1./ self.norm_factor

    def getNormFactor(self):
        return self.norm_factor
        