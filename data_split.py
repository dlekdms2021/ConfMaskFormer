import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# 🔹 Split 이름
split_names = ['pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4']

# 🔹 데이터 폴더
base_dirs = ['./data/in-motion']

# 🔹 저장 경로
save_root_raw = './data/data_split/raw'
save_root_norm = './data/data_split/norm'
save_root_npz = './data/data_split/npz'

# 🔹 RSSI 피처
beacon_cols = ['B1', 'B2', 'B3', 'B4', 'B5']

# 🔹 슬라이딩 윈도우 설정
window_size = 10
step_size = 1

for base_dir in base_dirs:
    folder_name = os.path.basename(base_dir.rstrip('/'))
    csv_files = sorted(glob(os.path.join(base_dir, "*.csv")))

    for split_idx, split_name in enumerate(split_names):
        all_train_parts = []  # 전체 train 구간 누적
        file_split_info = {}  # {filename: dict with 'train_0', 'train_1', 'test'}

        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            df = pd.read_csv(csv_path)
            total_len = len(df)

            if total_len < 5:
                print(f"⚠️ Too short: {filename} (len={total_len})")
                continue

            split_size = total_len // 5
            split_indices = [i * split_size for i in range(5)] + [total_len]

            test_start = split_indices[split_idx]
            test_end = split_indices[split_idx + 1]

            df_test = df.iloc[test_start:test_end].copy()
            split_data = {'test': df_test}

            if test_start == 0 or test_end == total_len:
                df_train = pd.concat([df.iloc[:test_start], df.iloc[test_end:]], axis=0).copy()
                split_data['train'] = df_train
                all_train_parts.append(df_train)
            else:
                df_train_0 = df.iloc[:test_start].copy()
                df_train_1 = df.iloc[test_end:].copy()
                split_data['train_0'] = df_train_0
                split_data['train_1'] = df_train_1
                all_train_parts.extend([df_train_0, df_train_1])

            file_split_info[filename] = split_data

        # 🔹 전체 train에서 global min/max 계산 (0 제외)
        df_all_train = pd.concat(all_train_parts, axis=0).reset_index(drop=True)
        rssi_values = df_all_train[beacon_cols].replace(0, pd.NA).values.flatten()
        rssi_values = pd.Series(rssi_values).dropna()
        global_min = rssi_values.min()
        global_max = rssi_values.max()

        print(f"\n📁 Dataset: {folder_name} / Split {split_name}")
        print(f"    ▪ Global Min (RSSI): {global_min}")
        print(f"    ▪ Global Max (RSSI): {global_max}")

        # 🔸 정규화 함수 정의
        def normalize(df_sub):
            normed = df_sub[beacon_cols].copy()
            for col in beacon_cols:
                normed[col] = normed[col].apply(
                    lambda x: (x - global_min) / (global_max - global_min) if x != 0 else 0
                )
            return pd.concat([normed, df_sub[['Zone']]], axis=1)

        # 🔁 각 파일 저장
        for filename, split_data in file_split_info.items():
            base_name = filename.replace('.csv', '')

            # 저장 경로 준비
            save_dirs = {
                'raw': {
                    'train': os.path.join(save_root_raw, folder_name, split_name, 'train'),
                    'test': os.path.join(save_root_raw, folder_name, split_name, 'test'),
                },
                'norm': {
                    'train': os.path.join(save_root_norm, folder_name, split_name, 'train'),
                    'test': os.path.join(save_root_norm, folder_name, split_name, 'test'),
                }
            }

            for kind in ['raw', 'norm']:
                for part in ['train', 'test']:
                    os.makedirs(save_dirs[kind][part], exist_ok=True)

            # 🔹 test 저장
            df_test_raw = split_data['test']
            df_test_norm = normalize(df_test_raw)
            df_test_raw.to_csv(os.path.join(save_dirs['raw']['test'], filename), index=False)
            df_test_norm.to_csv(os.path.join(save_dirs['norm']['test'], filename), index=False)

            # 🔹 train 저장 (연속 / 분리 모두 대응)
            if 'train' in split_data:
                df_train_raw = split_data['train']
                df_train_norm = normalize(df_train_raw)
                df_train_raw.to_csv(os.path.join(save_dirs['raw']['train'], filename), index=False)
                df_train_norm.to_csv(os.path.join(save_dirs['norm']['train'], filename), index=False)
            else:
                for i in [0, 1]:
                    df_train_raw = split_data[f'train_{i}']
                    df_train_norm = normalize(df_train_raw)
                    df_train_raw.to_csv(os.path.join(save_dirs['raw']['train'], f"{base_name}_{i}.csv"), index=False)
                    df_train_norm.to_csv(os.path.join(save_dirs['norm']['train'], f"{base_name}_{i}.csv"), index=False)

        # 🔸 슬라이딩 윈도우 적용 및 .npz 저장
        for phase in ['train', 'test']:
            norm_dir = os.path.join(save_root_norm, folder_name, split_name, phase)
            npz_save_dir = os.path.join(save_root_npz, folder_name)
            os.makedirs(npz_save_dir, exist_ok=True)

            all_windows = []

            csv_paths = sorted(glob(os.path.join(norm_dir, '*.csv')))
            for csv_path in tqdm(csv_paths, desc=f"Sliding window: {folder_name}/{split_name}/{phase}"):
                df = pd.read_csv(csv_path)
                values = df[beacon_cols].values         # (T, 5)
                zones = df['Zone'].values               # (T,)

                for i in range(0, len(values) - window_size + 1, step_size):
                    window = values[i:i + window_size]                    # (10, 5)
                    zone_label = zones[i + window_size // 2]              # 중앙 행 기준 Zone
                    zone_column = np.full((window_size, 1), zone_label)   # (10, 1)
                    window_with_label = np.concatenate([window, zone_column], axis=1)  # (10, 6)
                    all_windows.append(window_with_label)

            if all_windows:
                all_windows = np.array(all_windows)  # shape: (N, 10, 6)
                save_path = os.path.join(npz_save_dir, f"{phase}_{split_name}.npz")
                np.savez(save_path, data=all_windows)
                print(f"✅ Saved {save_path} : {all_windows.shape[0]} samples")
                