# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class BeaconDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)['data']  # shape: (N, 10, 6)
        self.x = data[:, :, :5].astype(np.float32)  # RSSI (10, 5)
        self.y = data[:, 0, 5].astype(np.int64) - 1  # Zone (1~24 → 0~23)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
