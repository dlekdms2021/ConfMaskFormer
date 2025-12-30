#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Beacon 데이터 (B1~B5, Zone)에 대해
- pos_0 ~ pos_4 5-fold 분할 (기존 코드와 동일)
- train/test만 사용 (val 없음)
- window_size/step_size 기반 슬라이딩 윈도우 (기존과 동일)
을 유지하면서

STraTS supervised 학습에 맞는 포맷으로 전처리하는 스크립트.

또한, 변환 과정을 보기 위해 다음 중간 결과를 모두 저장한다.

1) step1_raw   : pos split 이후 train/test raw CSV
2) step2_norm  : 비컨별 z-score 정규화된 train/test CSV
3) step3_windows: 슬라이딩 윈도우 적용 후 (window_size, 5+1) npz
4) 최종 STraTS .pt: (times, features, values, label) 샘플 리스트
"""

import os
from glob import glob
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


# 🔹 Split 이름 (기존과 동일)
split_names = ['pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4']

# 🔹 데이터 폴더 (CSV가 있는 곳)
base_dirs = ['../../../Git/data/in-motion']

# 🔹 RSSI 피처
beacon_cols = ['B1', 'B2', 'B3', 'B4', 'B5']

# 🔹 슬라이딩 윈도우 설정 (기존과 동일)
window_size = 10
step_size = 1

# 🔹 STraTS용 전처리 설정
min_events_per_sample = 5  # 한 샘플(윈도우) 안에 이벤트가 너무 적으면 버림

# 🔹 저장 루트
#   - step1_raw, step2_norm, step3_windows, 최종 .pt 모두 이 아래에 저장
save_root_strats = './data_split_beacon_strats'


def compute_beacon_norm_stats(df_all_train: pd.DataFrame,
                              beacon_cols: List[str]) -> Dict[int, Dict[str, float]]:
    """
    train 전체에서 비컨별 mean/std를 계산 (0은 결측으로 간주하여 제외).
    반환: {feature_id(int): {"mean": float, "std": float}}
    """
    stats: Dict[int, Dict[str, float]] = {}
    for f_idx, col in enumerate(beacon_cols):
        vals = df_all_train[col].replace(0, np.nan).dropna()
        if len(vals) == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(vals.mean())
            std = float(vals.std())
            if std == 0 or np.isnan(std):
                std = 1.0
        stats[f_idx] = {"mean": mean, "std": std}
    return stats


def normalize_beacons(df: pd.DataFrame,
                      beacon_cols: List[str],
                      stats: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    비컨별 z-score 정규화.
    - RSSI == 0 은 "결측"으로 간주 → 그대로 0 유지
    """
    df_norm = df.copy()
    for f_idx, col in enumerate(beacon_cols):
        mean = stats[f_idx]["mean"]
        std = stats[f_idx]["std"]

        def _norm(x):
            if x == 0:
                return 0.0
            return (x - mean) / std

        df_norm[col] = df_norm[col].apply(_norm)
    return df_norm


def window_to_strats_sample(window_values: np.ndarray,
                            zone_label: int,
                            beacon_cols: List[str]) -> Any:
    """
    (window_size, num_beacons) 배열을
    STraTS용 (times, features, values, label) 샘플로 변환.

    - time: 0 ~ window_size-1
    - feature: 0~4 (B1~B5)
    - values: 정규화된 RSSI (0은 결측 → 이벤트에서 제외)

    반환: sample dict 또는 None (이벤트 개수가 너무 적을 때)
    """
    T, F = window_values.shape
    assert F == len(beacon_cols)

    times_list = []
    feat_list = []
    val_list = []

    for t in range(T):
        for f_idx in range(F):
            v = float(window_values[t, f_idx])
            if v == 0.0:
                continue  # 결측은 이벤트로 쓰지 않음
            times_list.append(float(t))    # 단위 time step
            feat_list.append(int(f_idx))   # beacon index
            val_list.append(v)

    if len(times_list) < min_events_per_sample:
        return None

    sample = {
        "times": np.asarray(times_list, dtype=np.float32),
        "features": np.asarray(feat_list, dtype=np.int32),
        "values": np.asarray(val_list, dtype=np.float32),
        "label": int(zone_label)
    }
    return sample


def main():
    for base_dir in base_dirs:
        folder_name = os.path.basename(base_dir.rstrip('/'))
        csv_files = sorted(glob(os.path.join(base_dir, "*.csv")))

        for split_idx, split_name in enumerate(split_names):
            # ============================================================
            # 1) 기존 코드와 동일하게 train/test 분할 + step1_raw 저장
            # ============================================================
            all_train_parts: List[pd.DataFrame] = []
            file_split_info: Dict[str, Dict[str, pd.DataFrame]] = {}

            # step1_raw 저장 폴더
            raw_root = os.path.join(save_root_strats, 'step1_raw', folder_name, split_name)
            raw_train_dir = os.path.join(raw_root, 'train')
            raw_test_dir = os.path.join(raw_root, 'test')
            os.makedirs(raw_train_dir, exist_ok=True)
            os.makedirs(raw_test_dir, exist_ok=True)

            for csv_path in csv_files:
                filename = os.path.basename(csv_path)
                base_name = filename.replace('.csv', '')
                df = pd.read_csv(csv_path)
                total_len = len(df)

                if total_len < 5:
                    print(f"⚠️ Too short: {filename} (len={total_len})")
                    continue

                # 5등분 인덱스
                split_size = total_len // 5
                split_indices = [i * split_size for i in range(5)] + [total_len]

                # 현재 split을 test 부분으로 사용
                test_start = split_indices[split_idx]
                test_end = split_indices[split_idx + 1]

                df_test = df.iloc[test_start:test_end].copy()
                split_data: Dict[str, pd.DataFrame] = {'test': df_test}

                # train 부분 (앞+뒤) 처리
                if test_start == 0 or test_end == total_len:
                    df_train = pd.concat(
                        [df.iloc[:test_start], df.iloc[test_end:]],
                        axis=0
                    ).copy()
                    split_data['train'] = df_train
                    all_train_parts.append(df_train)

                    # step1_raw 저장
                    df_train.to_csv(os.path.join(raw_train_dir, filename), index=False)
                else:
                    df_train_0 = df.iloc[:test_start].copy()
                    df_train_1 = df.iloc[test_end:].copy()
                    split_data['train_0'] = df_train_0
                    split_data['train_1'] = df_train_1
                    all_train_parts.extend([df_train_0, df_train_1])

                    # step1_raw 저장 (train_0, train_1 구분)
                    df_train_0.to_csv(os.path.join(raw_train_dir, f"{base_name}_0.csv"), index=False)
                    df_train_1.to_csv(os.path.join(raw_train_dir, f"{base_name}_1.csv"), index=False)

                # test 저장
                df_test.to_csv(os.path.join(raw_test_dir, filename), index=False)
                file_split_info[filename] = split_data

            if not all_train_parts:
                print(f"❌ train 데이터가 없습니다: {folder_name} / {split_name}")
                continue

            # ============================================================
            # 2) train 전체에서 비컨별 mean/std 계산 (0은 결측으로 제외)
            # ============================================================
            df_all_train = pd.concat(all_train_parts, axis=0).reset_index(drop=True)
            norm_stats = compute_beacon_norm_stats(df_all_train, beacon_cols)

            print(f"\n📁 Dataset: {folder_name} / Split {split_name}")
            for f_idx, col in enumerate(beacon_cols):
                print(f"  ▪ {col}: mean={norm_stats[f_idx]['mean']:.3f}, "
                      f"std={norm_stats[f_idx]['std']:.3f}")

            # ============================================================
            # 3) 슬라이딩 윈도우 + STraTS 샘플 생성
            #    + step2_norm, step3_windows 저장
            # ============================================================
            train_samples: List[Dict[str, Any]] = []
            test_samples: List[Dict[str, Any]] = []

            # step2_norm / step3_windows 저장 폴더
            norm_root = os.path.join(save_root_strats, 'step2_norm', folder_name, split_name)
            norm_train_dir = os.path.join(norm_root, 'train')
            norm_test_dir = os.path.join(norm_root, 'test')
            os.makedirs(norm_train_dir, exist_ok=True)
            os.makedirs(norm_test_dir, exist_ok=True)

            windows_root = os.path.join(save_root_strats, 'step3_windows', folder_name)
            os.makedirs(windows_root, exist_ok=True)
            # phase별 윈도우 모으는 리스트
            all_windows_train: List[np.ndarray] = []
            all_windows_test: List[np.ndarray] = []

            for phase in ['train', 'test']:
                print(f"\n▶ Generating {phase} windows for {folder_name}/{split_name}")

                for filename, split_data in tqdm(file_split_info.items()):
                    base_name = filename.replace('.csv', '')

                    # 해당 phase에 해당하는 DataFrame 파트들 모으기
                    phase_parts: List[pd.DataFrame] = []
                    phase_part_names: List[str] = []  # 파일명 구분용

                    if phase == 'test':
                        phase_parts.append(split_data['test'])
                        phase_part_names.append(base_name)
                    else:  # train
                        if 'train' in split_data:
                            phase_parts.append(split_data['train'])
                            phase_part_names.append(base_name)
                        else:
                            phase_parts.append(split_data['train_0'])
                            phase_parts.append(split_data['train_1'])
                            phase_part_names.append(base_name + "_0")
                            phase_part_names.append(base_name + "_1")

                    for part_name, df_part in zip(phase_part_names, phase_parts):
                        if len(df_part) < window_size:
                            continue

                        # -------- step2_norm: 정규화된 CSV 저장 --------
                        df_norm = normalize_beacons(df_part, beacon_cols, norm_stats)
                        df_norm_with_zone = df_norm.copy()
                        df_norm_with_zone['Zone'] = df_part['Zone'].values

                        if phase == 'train':
                            norm_path = os.path.join(norm_train_dir, f"{part_name}.csv")
                        else:
                            norm_path = os.path.join(norm_test_dir, f"{part_name}.csv")
                        df_norm_with_zone.to_csv(norm_path, index=False)

                        # -------- 슬라이딩 윈도우 & STraTS 샘플 / step3_windows --------
                        values = df_norm[beacon_cols].values   # (T, 5)
                        zones = df_part['Zone'].values         # (T,)

                        for i in range(0, len(values) - window_size + 1, step_size):
                            window = values[i:i + window_size]       # (window_size, 5)
                            zone_label = int(zones[i + window_size // 2])

                            # step3_windows: grid + label 저장용
                            zone_column = np.full((window_size, 1), zone_label, dtype=np.int32)
                            window_with_label = np.concatenate([window, zone_column], axis=1)  # (T, 6)

                            # STraTS 샘플 생성
                            sample = window_to_strats_sample(
                                window_values=window,
                                zone_label=zone_label,
                                beacon_cols=beacon_cols
                            )
                            if sample is None:
                                continue

                            if phase == 'train':
                                train_samples.append(sample)
                                all_windows_train.append(window_with_label)
                            else:
                                test_samples.append(sample)
                                all_windows_test.append(window_with_label)

                print(f"  → {phase} samples: "
                      f"{len(train_samples) if phase=='train' else len(test_samples)}")

            # step3_windows: npz로 저장 (기존 all_windows 느낌)
            if all_windows_train:
                all_windows_train_arr = np.stack(all_windows_train, axis=0)  # (N, T, 6)
                np.savez(
                    os.path.join(windows_root, f"train_{split_name}.npz"),
                    data=all_windows_train_arr
                )
                print(f"  ✅ step3_windows train saved: "
                      f"{all_windows_train_arr.shape} -> {os.path.join(windows_root, f'train_{split_name}.npz')}")
            if all_windows_test:
                all_windows_test_arr = np.stack(all_windows_test, axis=0)  # (N, T, 6)
                np.savez(
                    os.path.join(windows_root, f"test_{split_name}.npz"),
                    data=all_windows_test_arr
                )
                print(f"  ✅ step3_windows test saved: "
                      f"{all_windows_test_arr.shape} -> {os.path.join(windows_root, f'test_{split_name}.npz')}")

            # ============================================================
            # 4) STraTS 포맷으로 최종 저장 (.pt)
            # ============================================================
            save_dir = os.path.join(save_root_strats, 'final_pt', folder_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"strats_{split_name}.pt")

            out_obj = {
                "train": train_samples,
                "test": test_samples,
                "norm_stats": norm_stats,
                "meta": {
                    "beacon_cols": beacon_cols,
                    "window_size": window_size,
                    "step_size": step_size,
                    "min_events": min_events_per_sample,
                    "split_name": split_name,
                    "folder_name": folder_name,
                },
            }
            torch.save(out_obj, save_path)
            print(f"\n✅ Saved final STraTS .pt: {save_path}")
            print(f"   train samples: {len(train_samples)}, "
                  f"test samples: {len(test_samples)}")


if __name__ == "__main__":
    main()
