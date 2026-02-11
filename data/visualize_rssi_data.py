#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSSI 데이터 시각화: 그림3(결측률), 그림4(평균 RSSI)
논문 4.2.2, 4.2.3 섹션 용 그래프 생성
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_all_data(data_dir):
    """모든 zone의 CSV 데이터 로드"""
    data = {}
    for zone_id in range(1, 25):
        csv_path = os.path.join(data_dir, f"{zone_id}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data[zone_id] = df
        else:
            print(f"Warning: {csv_path} not found")
    return data

def compute_missing_ratio(data):
    """
    비컨별 결측률 계산
    Returns: Dict[beacon] = List[missing_ratio for zone 1-24]
    """
    beacons = ['B1', 'B2', 'B3', 'B4', 'B5']
    missing_ratios = {b: [] for b in beacons}
    
    for zone_id in range(1, 25):
        if zone_id in data:
            df = data[zone_id]
            for beacon in beacons:
                if beacon in df.columns:
                    # 0은 결측, 음수는 유효한 RSSI 값
                    missing_count = (df[beacon] == 0).sum()
                    missing_ratio = (missing_count / len(df)) * 100
                    missing_ratios[beacon].append(missing_ratio)
    
    return missing_ratios

def compute_average_rssi(data):
    """
    비컨별 평균 RSSI 계산 (결측은 제외)
    Returns: 2D array (zones x beacons)
    """
    zones = list(range(1, 25))
    beacons = ['B1', 'B2', 'B3', 'B4', 'B5']
    avg_rssi = np.zeros((len(zones), len(beacons)))
    
    for i, zone_id in enumerate(zones):
        if zone_id in data:
            df = data[zone_id]
            for j, beacon in enumerate(beacons):
                if beacon in df.columns:
                    # 결측(0)을 제외한 RSSI 평균
                    valid_rssi = df[df[beacon] != 0][beacon]
                    if len(valid_rssi) > 0:
                        avg_rssi[i, j] = valid_rssi.mean()
                    else:
                        avg_rssi[i, j] = np.nan
    
    return avg_rssi, zones, beacons

def plot_missing_ratio(missing_ratios, output_path):
    """그림3: Missing Ratio per Zone by Beacon"""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    zones = list(range(1, 25))
    beacons = ['B1', 'B2', 'B3', 'B4', 'B5']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for beacon, color, marker in zip(beacons, colors, markers):
        ratios = missing_ratios[beacon]
        ax.plot(zones, ratios, 
               marker=marker, markersize=9, linewidth=2.5,
               label=beacon, color=color, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Zone', fontsize=17)
    ax.set_ylabel('Missing Ratio (%)', fontsize=17)
    ax.set_title('Missing Ratio per Zone by Beacon', fontsize=17, pad=20)
    ax.set_xticks(zones)
    ax.set_xticklabels([f'{z}' for z in zones], fontsize=16)
    ax.set_ylim(55, 90)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.7)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, title='Beacon', title_fontsize=10)
    
    # y축 눈금 추가
    ax.set_yticks([55, 60, 65, 70, 75, 80, 85, 90])
    ax.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 그림3 저장: {output_path}")
    plt.close()

def plot_average_rssi_heatmap(avg_rssi, zones, beacons, output_path):
    """그림4: Average RSSI per Zone and Beacon (Heatmap)"""
    fig, ax = plt.subplots(figsize=(8, 8.5))
    
    # DataFrame 생성 (플롯 용이)
    df_rssi = pd.DataFrame(
        avg_rssi,
        index=[f'Zone {z}' for z in zones],
        columns=beacons
    )
    
    # 히트맵 (값 폰트 크기 증가)
    sns.heatmap(df_rssi, 
               annot=False,     # 값 표시 제거
               cmap='coolwarm', # 빨강-파랑 컬러맵
               cbar_kws={'label': 'RSSI (dB)', 'shrink': 0.95},
               ax=ax,
               linewidths=1,
               linecolor='white',
               vmin=-90,
               vmax=-50)
    
    ax.set_xlabel('Beacon', fontsize=13, labelpad=10)
    ax.set_ylabel('Zone', fontsize=13, labelpad=10)
    ax.set_title('Average RSSI per Zone and Beacon', fontsize=15, pad=20)
    
    # Tick 레이블 크기
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 그림4 저장: {output_path}")
    plt.close()

def compute_statistics(data, missing_ratios, avg_rssi):
    """통계 정보 출력"""
    print("\n" + "="*60)
    print("통계 정보")
    print("="*60)
    
    beacons = ['B1', 'B2', 'B3', 'B4', 'B5']
    
    print("\n[비컨별 평균 결측률]")
    for beacon in beacons:
        mean_missing = np.mean(missing_ratios[beacon])
        print(f"{beacon}: {mean_missing:.2f}%")
    
    print("\n[비컨별 평균 RSSI]")
    for j, beacon in enumerate(beacons):
        valid_rssi = avg_rssi[:, j][~np.isnan(avg_rssi[:, j])]
        if len(valid_rssi) > 0:
            mean_rssi = np.mean(valid_rssi)
            print(f"{beacon}: {mean_rssi:.2f} dB")
    
    # Zone 22-23 인접 RSSI 차이 (논문에서 언급)
    print("\n[Zone 22-23 인접 RSSI 차이]")
    if 22 in data and 23 in data:
        df22 = data[22]
        df23 = data[23]
        for i, beacon in enumerate(beacons):
            valid22 = df22[df22[beacon] != 0][beacon]
            valid23 = df23[df23[beacon] != 0][beacon]
            if len(valid22) > 0 and len(valid23) > 0:
                mean22 = valid22.mean()
                mean23 = valid23.mean()
                diff = mean22 - mean23
                print(f"{beacon}: Zone22={mean22:.1f} - Zone23={mean23:.1f} = {diff:+.1f} dB")

def main():
    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'in-motion')
    
    if not os.path.exists(data_dir):
        print(f"❌ 에러: {data_dir} 디렉토리를 찾을 수 없습니다")
        return
    
    print("데이터 로딩 중...")
    data = load_all_data(data_dir)
    print(f"✓ {len(data)}개 zone의 데이터 로드 완료\n")
    
    # 통계 계산
    print("통계 계산 중...")
    missing_ratios = compute_missing_ratio(data)
    avg_rssi, zones, beacons = compute_average_rssi(data)
    print("✓ 계산 완료\n")
    
    # 그래프 생성
    print("그래프 생성 중...")
    plot_missing_ratio(missing_ratios, os.path.join(script_dir, 'figure3_missing_ratio.png'))
    plot_average_rssi_heatmap(avg_rssi, zones, beacons, os.path.join(script_dir, 'figure4_average_rssi.png'))
    
    # 통계 출력
    compute_statistics(data, missing_ratios, avg_rssi)
    
    print("\n" + "="*60)
    print("✓ 시각화 완료!")
    print(f"📁 저장 위치: {script_dir}")
    print("  - figure3_missing_ratio.png")
    print("  - figure4_average_rssi.png")
    print("="*60)

if __name__ == "__main__":
    main()
