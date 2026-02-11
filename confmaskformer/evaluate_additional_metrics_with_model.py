#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델을 재로드하여 Top-2 정확도와 평균 거리 오차를 정확하게 계산

results/ 폴더의 best_pos_X.pt 모델을 로드하여
테스트 데이터에 대한 softmax 확률을 계산하고 Top-2 정확도를 구합니다.

사용법:
    python evaluate_additional_metrics_with_model.py --results_dir ./results --npz_root ../data/data_split/npz/in-motion
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from glob import glob
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*nested tensors.*')

# ============================================================================
# Zone 좌표 정의 (24개 zone의 물리적 위치)
# 직선 배치: Zone 0~23이 일렬로 이어져 있음, 각 3m 간격
# ============================================================================
ZONE_COORDINATES = {i: (i * 3.0, 0) for i in range(24)}

UNIT_DISTANCE = 1.0  # 거리 단위 (이미 좌표에 3m이 포함됨)


def euclidean_distance(zone1: int, zone2: int) -> float:
    """두 zone 간의 유클리드 거리 계산"""
    x1, y1 = ZONE_COORDINATES[zone1]
    x2, y2 = ZONE_COORDINATES[zone2]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * UNIT_DISTANCE


def calculate_average_distance_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균 거리 오차 계산"""
    distances = [euclidean_distance(int(yt), int(yp)) for yt, yp in zip(y_true, y_pred)]
    return np.mean(distances)


def calculate_top2_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """확률 분포로부터 Top-2 정확도 계산"""
    # 상위 2개 예측 인덱스 추출
    top2_preds = np.argsort(probs, axis=1)[:, -2:]  # (N, 2)
    
    # 각 샘플에 대해 정답이 top2에 포함되는지 확인
    correct = np.array([yt in top2 for yt, top2 in zip(y_true, top2_preds)])
    return correct.mean()


@torch.no_grad()
def predict_with_probabilities(model, loader, device):
    """모델로부터 예측 확률과 레이블 추출"""
    model.eval()
    all_probs = []
    all_labels = []
    
    for x, y in loader:
        x = x.to(device)
        out = model(x, do_aux=False)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        all_probs.append(probs)
        all_labels.append(y.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics with model re-inference")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--npz_root", type=str, default="../data/data_split/npz/in-motion")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--zone_coords_file", type=str, default=None)
    args = parser.parse_args()
    
    # Zone 좌표 로드
    if args.zone_coords_file and os.path.exists(args.zone_coords_file):
        coords_df = pd.read_csv(args.zone_coords_file)
        global ZONE_COORDINATES
        ZONE_COORDINATES = {row['zone']: (row['x'], row['y']) for _, row in coords_df.iterrows()}
        print(f"✓ Zone 좌표를 {args.zone_coords_file}에서 로드했습니다.")
        
    # Import 필요 모듈
    from data_loader import make_dataloaders_from_file
    from transformer_model import TransformerClassifierImproved
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}\n")
    
    # pos_X 폴더들 찾기
    pos_folders = sorted(glob(os.path.join(args.results_dir, "pos_*")))
    if not pos_folders:
        print(f"❌ {args.results_dir}에서 pos_* 폴더를 찾을 수 없습니다.")
        return
    
    print(f"{'='*80}")
    print(f"추가 평가 지표 계산 (모델 재추론 포함)")
    print(f"{'='*80}\n")
    
    results = []
    
    for pos_folder in pos_folders:
        pos_name = os.path.basename(pos_folder)
        pos_idx = int(pos_name.split("_")[1])
        
        # 모델 로드
        model_path = os.path.join(pos_folder, f"best_{pos_name}.pt")
        if not os.path.exists(model_path):
            print(f"⚠ {pos_name}: 모델 파일을 찾을 수 없습니다. 스킵합니다.")
            continue
        
        print(f"[{pos_name}] 모델 로드 중...")
        
        # 데이터 로더 생성 (test split)
        try:
            test_npz = os.path.join(args.npz_root, f"test_{pos_name}.npz")
            if not os.path.exists(test_npz):
                print(f"⚠ {pos_name}: 테스트 파일을 찾을 수 없습니다: {test_npz}")
                continue
            
            test_loader, _ = make_dataloaders_from_file(
                test_npz,
                batch_size=args.batch_size,
                num_workers=0,
                val_ratio=0.0
            )
        except Exception as e:
            print(f"⚠ {pos_name}: 데이터 로더 생성 실패: {e}")
            continue
        
        # 모델 초기화 및 가중치 로드
        # Cfg 설정 (기본값 사용)
        class Cfg:
            window_size = 10
            embedding_dim = 96
            n_heads = 8
            n_layers = 2
            dropout = 0.3
            aux_mask_ratio = 0.30
            aux_loss_weight = 0.40
            beacon_dropout_p = 0.30
        
        n_classes = 24
        n_beacons = 5
        
        model = TransformerClassifierImproved(
            cfg=Cfg,
            n_classes=n_classes,
            n_beacons=n_beacons
        ).to(device)
        
        # 가중치 로드
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ 모델 로드 완료")
        
        # 예측 확률 계산
        print(f"  예측 확률 계산 중...")
        y_true, probs = predict_with_probabilities(model, test_loader, device)
        y_pred = np.argmax(probs, axis=1)
        
        # 지표 계산
        acc = accuracy_score(y_true, y_pred)
        avg_distance_error = calculate_average_distance_error(y_true, y_pred)
        top2_acc = calculate_top2_accuracy(y_true, probs)
        
        results.append({
            "Fold": pos_name,
            "Accuracy": acc,
            "Top-2 Accuracy": top2_acc,
            "Avg Distance Error (m)": avg_distance_error
        })
        
        print(f"  Accuracy:              {acc:.4f}")
        print(f"  Top-2 Accuracy:        {top2_acc:.4f}")
        print(f"  Avg Distance Error:    {avg_distance_error:.4f} m")
        print()
    
    # 전체 평균
    if results:
        avg_acc = np.mean([r["Accuracy"] for r in results])
        avg_top2 = np.mean([r["Top-2 Accuracy"] for r in results])
        avg_dist = np.mean([r["Avg Distance Error (m)"] for r in results])
        
        print(f"{'='*80}")
        print(f"전체 평균 (5-Fold Cross Validation)")
        print(f"{'='*80}")
        print(f"  Average Accuracy:           {avg_acc:.4f}")
        print(f"  Average Top-2 Accuracy:     {avg_top2:.4f}")
        print(f"  Average Distance Error:     {avg_dist:.4f} m")
        print()
        
        # CSV 저장
        df = pd.DataFrame(results)
        output_path = os.path.join(args.results_dir, "additional_metrics_full.csv")
        df.to_csv(output_path, index=False)
        print(f"✓ 결과가 {output_path}에 저장되었습니다.")
        
        # Summary 저장
        summary_path = os.path.join(args.results_dir, "additional_metrics_full_summary.txt")
        with open(summary_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("추가 평가 지표 요약 (모델 재추론)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Average Accuracy:           {avg_acc:.4f}\n")
            f.write(f"Average Top-2 Accuracy:     {avg_top2:.4f}\n")
            f.write(f"Average Distance Error:     {avg_dist:.4f} m\n\n")
            f.write("="*80 + "\n")
            f.write("개별 Fold 결과\n")
            f.write("="*80 + "\n\n")
            for r in results:
                f.write(f"{r['Fold']}:\n")
                f.write(f"  Accuracy:           {r['Accuracy']:.4f}\n")
                f.write(f"  Top-2 Accuracy:     {r['Top-2 Accuracy']:.4f}\n")
                f.write(f"  Distance Error:     {r['Avg Distance Error (m)']:.4f} m\n\n")
        print(f"✓ 요약이 {summary_path}에 저장되었습니다.\n")


if __name__ == "__main__":
    main()
