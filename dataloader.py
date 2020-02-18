from torch.utils.data import Dataset, DataLoader# For custom data-sets
import tarfile

# import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import torch
# import pandas as pd
from collections import namedtuple
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as TF
# import sys

class UTK(Dataset):
    def __init__(self, file):
        self.data = tarfile.open(file)#read data here
        self.datum_names = self.data.getmembers()
        c = len(self.datum_names)

    def __len__(self):
        return len(self.datum_names)

    def __getitem__(self, idx):
        _img = self.datum_names[idx]
        img = self.data.extractfile(_img)
        img_full = Image.open(img).convert('RGB')
        # img_full.show()
        label_age = self.datum_names[idx].name.split('_')[0].split('/')[1]
        label_ethnicity = self.datum_names[idx].name.split('_')[2]

        img_full = np.asarray(img_full)

        # convert to tensor
        img_full = torch.from_numpy(np.array(img_full).copy()).float()
        label_age = torch.tensor(np.float(label_age))
        label_ethnicity = torch.tensor(np.float(label_ethnicity))

        return img_full, label_age, label_ethnicity


# utk_data = UTK(None)
# utk_data[0]