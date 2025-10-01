# data_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from glob import glob
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# Label utilities
# -----------------------------
def _normalize_labels_zero_based(y: np.ndarray, expect_classes: int = 24) -> np.ndarray:
    """
    라벨을 0-based로 표준화:
      - 1..expect_classes  -> 0..expect_classes-1
      - 0..expect_classes-1 -> 유지
      - 그 외: 최소값을 0으로 시프트 (경고)
    """
    y = y.astype(np.int64, copy=False)
    u = np.unique(y)
    umin, umax = int(u.min()), int(u.max())

    if umin == 1 and umax == expect_classes:
        y = y - 1
    elif umin == 0 and umax == expect_classes - 1:
        pass
    else:
        shift = umin
        print(f"[WARN] Labels are [{umin}..{umax}]. Auto-shifting by {shift} to start at 0.")
        y = y - shift
    return y

# -----------------------------
# Dataset
# -----------------------------
class RSSIWindowNPZDataset(Dataset):
    """
    지원 포맷 1) K-fold npz
      - key: 'data'  shape: (N, T=10, 6)  # 마지막 컬럼이 Zone
      - X = data[..., :5], y = data[:, 0, 5]

    지원 포맷 2) 이전 포맷
      - keys: 'X' (N,T,5), 'y' (N,)

    path:
      - 디렉터리 경로: 내부의 *.npz 모두 로드 후 concat
      - 단일 파일 경로: 해당 파일만 로드
    """
    def __init__(self, path: str, expect_classes: int = 24):
        X, y = self._load_path(path)
        X = X.astype(np.float32, copy=False)
        y = _normalize_labels_zero_based(y.astype(np.int64, copy=False), expect_classes)

        if X.ndim != 3 or X.shape[-1] != 5:
            raise ValueError(f"Expected X shape (N,T,5), got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Label shape mismatch: X={X.shape}, y={y.shape}")

        self.X, self.y = X, y

    # -------- internal loaders --------
    def _load_path(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.isdir(path):
            files = sorted(glob(os.path.join(path, "*.npz")))
            if not files:
                raise FileNotFoundError(f"No .npz files under directory: {path}")
            Xs: List[np.ndarray] = []
            Ys: List[np.ndarray] = []
            for f in files:
                x, y = self._load_one(f)
                Xs.append(x); Ys.append(y)
            return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)
        else:
            return self._load_one(path)

    def _load_one(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(npz_path)
        if "data" in data:  # K-fold 포맷
            arr = data["data"]  # (N, T, 6)
            if arr.ndim != 3 or arr.shape[-1] < 6:
                raise ValueError(f"'data' must have shape (N,T,6), got {arr.shape} in {npz_path}")
            X = arr[:, :, :5].copy()
            y = arr[:, 0, 5].copy()
        elif "X" in data and "y" in data:  # 이전 포맷
            X = data["X"].copy()
            y = data["y"].copy()
        else:
            raise KeyError(f"Unsupported npz keys in {npz_path}. Expect 'data' or ('X','y').")
        return X, y

    # -------- torch Dataset API --------
    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (T,5)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

# -----------------------------
# Dataloaders
# -----------------------------
def make_dataloaders_from_file(
    npz_path: str,
    batch_size: int = 256,
    val_ratio: float = 0.0,
    num_workers: int = 0,
    expect_classes: int = 24
):
    """
    단일 파일 경로 또는 디렉터리 경로(npz 모음)를 받아 DataLoader 생성.
    - val_ratio > 0: train 내 랜덤 분할
    """
    ds = RSSIWindowNPZDataset(npz_path, expect_classes=expect_classes)

    if val_ratio and val_ratio > 0:
        n_val = int(len(ds) * val_ratio)
        n_train = len(ds) - n_val
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                drop_last=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=num_workers, pin_memory=True)
        val_loader = None

    return train_loader, val_loader
