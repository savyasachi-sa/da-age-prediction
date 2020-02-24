from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torch


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

    def __len__(self):
        return len(self.data)

    def _get_item(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_full = Image.open(img_name).convert('RGB')
        label_age = self.data.iloc[idx, 1]

        if self.random_flips:
            self.flip(img_full)

        img_full = self.normalize(img_full)

        label_age = torch.tensor(label_age).type(torch.FloatTensor)

        return img_full, label_age

    def __getitem__(self, idx):
        return self._get_item(idx)
