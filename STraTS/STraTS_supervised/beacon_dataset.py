#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Beacon RSSI → STraTS supervised 학습용 Dataset.
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch

from utils import CycleIndex


class BeaconDataset:
    def __init__(self, args):
        self.args = args

        # 1) .pt 파일 로드
        pt_path = os.path.join(
            args.beacon_data_root,
            f"strats_{args.pos_split}.pt"
        )
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"STraTS .pt 파일을 찾을 수 없습니다: {pt_path}")

        obj: Dict[str, Any] = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=False,  # 🔴 PyTorch 2.6 때문에 꼭 넣어야 함
        )
        self.pt_path = pt_path

        train_samples_raw: List[Dict[str, Any]] = obj["train"]
        test_samples_raw: List[Dict[str, Any]] = obj["test"]

        all_samples = train_samples_raw + test_samples_raw

        # 2) 내부 리스트로 정리
        self.values: List[List[float]] = []
        self.times: List[List[float]] = []
        self.varis: List[List[int]] = []
        self.labels_raw: List[int] = []

        for s in all_samples:
            self.times.append(s["times"].astype(np.float32).tolist())
            self.varis.append(s["features"].astype(np.int64).tolist())
            self.values.append(s["values"].astype(np.float32).tolist())
            self.labels_raw.append(int(s["label"]))

        # 3) Zone 라벨을 0 ~ C-1로 매핑
        uniq_labels = sorted(set(self.labels_raw))
        label2idx: Dict[int, int] = {lab: i for i, lab in enumerate(uniq_labels)}
        labels = np.array([label2idx[l] for l in self.labels_raw], dtype=np.int64)

        self.label2idx = label2idx
        self.labels = labels
        num_classes = len(uniq_labels)

        # 4) 메타 정보에서 beacon 개수(V) 가져오기
        if "meta" in obj and "beacon_cols" in obj["meta"]:
            V = len(obj["meta"]["beacon_cols"])
        else:
            V = 5  # 기본값 (B1~B5)

        D = 0  # demo feature 없음

        # 🔹 Dataset / args 양쪽에 모두 저장
        self.V = V
        self.D = D
        self.num_classes = num_classes

        args.V = V
        args.D = D
        args.num_classes = num_classes

        self.N = len(all_samples)

        # 5) train / val / test split 구성
        n_train_raw = len(train_samples_raw)
        train_indices_all = np.arange(n_train_raw)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(train_indices_all)

        # 🔹 val_frac=0 이면 val 없이 전부 train으로 사용
        if getattr(args, "val_frac", 0.0) > 0:
            n_val = int(round(args.val_frac * n_train_raw))
        else:
            n_val = 0

        val_idx = train_indices_all[:n_val].tolist()
        train_idx = train_indices_all[n_val:].tolist()

        test_offset = n_train_raw
        test_idx = list(range(test_offset,
                              test_offset + len(test_samples_raw)))

        self.splits: Dict[str, List[int]] = {
            "train": train_idx,
            "val": val_idx,   # 👉 main에서는 안 쓸 거라서 비어 있어도 OK
            "test": test_idx,
        }
        self.splits["eval_train"] = train_idx[:min(2000, len(train_idx))]

        # 6) train용 배치 샘플러
        self.train_cycler = CycleIndex(self.splits["train"],
                                       args.train_batch_size)

        # 로그 찍어주기
        args.logger.write(
            f"\n[BeaconDataset] loaded from {pt_path}\n"
            f"  pos_split = {args.pos_split}\n"
            f"  #train = {len(train_idx)}\n"
            f"  #val   = {len(val_idx)}\n"
            f"  #test  = {len(test_idx)}\n"
            f"  #classes = {num_classes}"
        )

    # ------------------------------------------------------------------
    # 배치 생성
    # ------------------------------------------------------------------
    def get_batch(self, ind: List[int] | None = None) -> Dict[str, torch.Tensor]:
        if ind is None:
            ind = self.train_cycler.get_batch_ind()

        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)

        values_batch = []
        times_batch = []
        varis_batch = []
        obs_mask_batch = []

        for idx, pad_len in zip(ind, pad_lens):
            v = self.values[idx]
            t = self.times[idx]
            f = self.varis[idx]

            values_batch.append(v + [0.0] * pad_len)
            times_batch.append(t + [0.0] * pad_len)
            varis_batch.append(f + [0] * pad_len)

            obs_mask_batch.append([1] * len(v) + [0] * pad_len)

        values = torch.FloatTensor(values_batch)
        times = torch.FloatTensor(times_batch)
        varis = torch.IntTensor(varis_batch)
        obs_mask = torch.IntTensor(obs_mask_batch)

        # demo feature 없음 → (B, 0)짜리 텐서
        demo = torch.zeros((len(ind), 0), dtype=torch.float32)

        labels = torch.LongTensor(self.labels[ind])

        return {
            "values": values,
            "times": times,
            "varis": varis,
            "obs_mask": obs_mask,
            "demo": demo,
            "labels": labels,
        }
