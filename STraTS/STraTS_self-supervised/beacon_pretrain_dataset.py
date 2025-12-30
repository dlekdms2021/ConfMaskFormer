#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Beacon RSSI → STraTS self-supervised pretrain용 Dataset.

- 기존 전처리에서 생성한 strats_pos_k.pt (train/test 샘플 리스트)를 사용
- label(Zone)은 무시하고, 이벤트 시계열만 가지고 self-supervised forecast 학습
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch

from utils import CycleIndex


class BeaconPretrainDataset:
    """
    self-supervised pretrain용 Dataset.

    - 입력: (values, times, varis, obs_mask, demo)
    - 타깃: (forecast_values, forecast_mask)
      * forecast_values: 각 변수(비콘)별 "미래 구간에서의 마지막 값"
      * forecast_mask: 그 변수가 미래에 한 번이라도 등장했는지(1/0)
    """

    def __init__(self, args):
        self.args = args

        # 1) .pt 파일 로드 (supervised용과 동일 경로)
        pt_path = os.path.join(
            args.beacon_data_root,
            f"strats_{args.pos_split}.pt"
        )
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"STraTS .pt 파일을 찾을 수 없습니다: {pt_path}")

        obj: Dict[str, Any] = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=False,
        )
        self.pt_path = pt_path

        train_samples_raw: List[Dict[str, Any]] = obj["train"]
        test_samples_raw: List[Dict[str, Any]] = obj["test"]

        # pretrain은 supervised 라벨 안 쓰니까 train+test 모두 unsupervised 데이터로 쓸 수 있음
        all_samples = train_samples_raw + test_samples_raw

        self.values: List[List[float]] = []
        self.times: List[List[float]] = []
        self.varis: List[List[int]] = []

        for s in all_samples:
            self.times.append(s["times"].astype(np.float32).tolist())
            self.varis.append(s["features"].astype(np.int64).tolist())
            self.values.append(s["values"].astype(np.float32).tolist())

        self.N = len(all_samples)

        # 메타 정보에서 beacon 개수(V) 가져오기
        if "meta" in obj and "beacon_cols" in obj["meta"]:
            V = len(obj["meta"]["beacon_cols"])
        else:
            V = 5  # 기본값 (B1~B5)

        self.V = V
        self.D = 0  # demo feature 없음

        # args에도 저장
        args.V = V
        args.D = self.D

        # train/val split (비지도라서 간단히 비율로 나눔)
        indices = np.arange(self.N)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(indices)

        val_frac = getattr(args, "pretrain_val_frac", 0.1)
        n_val = int(round(val_frac * self.N))
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()

        self.splits: Dict[str, List[int]] = {
            "train": train_idx,
            "val": val_idx,
        }

        # train용 배치 샘플러
        self.train_cycler = CycleIndex(self.splits["train"],
                                       args.train_batch_size)

        args.logger.write(
            f"\n[BeaconPretrainDataset] loaded from {pt_path}\n"
            f"  pos_split = {args.pos_split}\n"
            f"  #unsup train = {len(train_idx)}\n"
            f"  #unsup val   = {len(val_idx)}\n"
            f"  V (beacons)  = {self.V}"
        )

    # ------------------------------------------------------------------
    # 배치 생성 (self-supervised)
    # ------------------------------------------------------------------
    def get_batch(self, ind: List[int] | None = None) -> Dict[str, torch.Tensor]:
        # ind가 None이면 train_cycler에서 batch 인덱스 뽑기 (train용)
        if ind is None:
            ind = self.train_cycler.get_batch_ind()

        bsz = len(ind)

        input_values: List[List[float]] = []
        input_times: List[List[float]] = []
        input_varis: List[List[int]] = []

        forecast_values = torch.zeros((bsz, self.V), dtype=torch.float32)
        forecast_mask = torch.zeros((bsz, self.V), dtype=torch.int64)

        for b, i in enumerate(ind):
            vals = self.values[i]
            tms = self.times[i]
            vrs = self.varis[i]
            L = len(vals)

            if L < 2:
                # 샘플 길이가 너무 짧으면 그냥 전부 입력으로 쓰고, 타깃 없음
                cut_idx = L
            else:
                # 1 ~ L-1 사이에서 랜덤 cut (앞은 관측, 뒤는 예측 구간)
                cut_idx = np.random.randint(1, L)

            # 관측 구간
            input_values.append(vals[:cut_idx])
            input_times.append(tms[:cut_idx])
            input_varis.append(vrs[:cut_idx])

            # 예측 구간: cut_idx ~ L-1
            # 각 변수마다 "마지막 값"을 forecast target으로 사용
            last_seen: Dict[int, float] = {}
            for j in range(cut_idx, L):
                v_id = int(vrs[j])
                last_seen[v_id] = float(vals[j])

            for v_id, v_val in last_seen.items():
                if 0 <= v_id < self.V:
                    forecast_mask[b, v_id] = 1
                    forecast_values[b, v_id] = v_val

        # padding
        num_obs = [len(v) for v in input_values]
        max_obs = max(num_obs) if num_obs else 0
        pad_lens = max_obs - np.array(num_obs)

        values_batch = []
        times_batch = []
        varis_batch = []
        obs_mask_batch = []

        for v, t, f, pad_len in zip(input_values, input_times, input_varis, pad_lens):
            values_batch.append(v + [0.0] * pad_len)
            times_batch.append(t + [0.0] * pad_len)
            varis_batch.append(f + [0] * pad_len)
            obs_mask_batch.append([1] * len(v) + [0] * pad_len)

        if max_obs == 0:
            # 극단적인 경우 방어용 (거의 안 나올 거라 예상)
            values = torch.zeros((bsz, 1), dtype=torch.float32)
            times = torch.zeros((bsz, 1), dtype=torch.float32)
            varis = torch.zeros((bsz, 1), dtype=torch.int32)
            obs_mask = torch.zeros((bsz, 1), dtype=torch.int32)
        else:
            values = torch.FloatTensor(values_batch)
            times = torch.FloatTensor(times_batch)
            varis = torch.IntTensor(varis_batch)
            obs_mask = torch.IntTensor(obs_mask_batch)

        # demo feature 없음 → (B, 0)짜리 텐서
        demo = torch.zeros((bsz, 0), dtype=torch.float32)

        return {
            "values": values,
            "times": times,
            "varis": varis,
            "obs_mask": obs_mask,
            "demo": demo,
            "forecast_values": forecast_values,
            "forecast_mask": forecast_mask,
        }
