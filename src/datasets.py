from pathlib import Path
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class FaceTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data.columns = self.data.columns.str.lower()
        assert {'image_path', 'gt'}.issubset(self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel = self.data.iloc[idx]['image_path']
        img = Image.open(self.root_dir / img_rel).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_rel


class FaceReIDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data.columns = self.data.columns.str.lower()
        assert {'image_path', 'gt'}.issubset(self.data.columns)
        self.labels = [int(gt.split('_')[1]) for gt in self.data['gt']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel = self.data.iloc[idx]['image_path']
        img = Image.open(self.root_dir / img_rel).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label