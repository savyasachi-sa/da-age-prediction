from torch.utils.data import Dataset, DataLoader # For custom data-sets
import tarfile

import numpy as np
from PIL import Image, ImageOps
import torch
from collections import namedtuple

class UTK(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img_full = Image.open(img_name).convert('RGB')
        label_age = self.data.iloc[idx, 1]

        img_full = np.asarray(img_full)

        # convert to tensor
        img_full = torch.from_numpy(np.array(img_full).copy()).float()
        label_age = torch.tensor(np.float(label_age))

        return img_full, label_age