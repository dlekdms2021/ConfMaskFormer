#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study 결과 분석 스크립트
각 구성요소의 성능 기여도를 정량적으로 계산합니다.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def extract_results(results_dir):
    """결과 디렉토리에서 최종 성능 메트릭 추출"""
    results = {}
    
    for split_dir in Path(results_dir).glob("pos_*"):
        split_name = split_dir.name
        # metrics_pos_X.txt 파일 찾기
        metrics_file = split_dir / f"metrics_{split_name}.txt"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                content = f.read()
                # "Best TEST Acc" 라인 찾기
                for line in content.split('\n'):
                    if 'Best TEST Acc' in line:
                        try:
                            # "Best TEST Acc   : 0.4355" 형식에서 추출
                            acc = float(line.split(':')[-1].strip())
                            results[split_name] = acc
                            break
                        except:
                            pass
    
    return results

def analyze_ablation_study(results_root="./results_ablation"):
    """Ablation Study 결과 분석"""
    
    configurations = {
        "Baseline": {
            "confidence_gate": False,
            "auxiliary_loss": False,
            "beacon_dropout": False,
            "combined_input": False
        },
        "Confidence Gate": {
            "confidence_gate": True,
            "auxiliary_loss": False,
            "beacon_dropout": False,
            "combined_input": False
        },
        "Auxiliary Loss": {
            "confidence_gate": False,
            "auxiliary_loss": True,
            "beacon_dropout": False,
            "combined_input": False
        },
        "Beacon Dropout": {
            "confidence_gate": False,
            "auxiliary_loss": False,
            "beacon_dropout": True,
            "combined_input": False
        },
        "Combined Input": {
            "confidence_gate": False,
            "auxiliary_loss": False,
            "beacon_dropout": False,
            "combined_input": True
        },
        "CG + AL + BD": {
            "confidence_gate": True,
            "auxiliary_loss": True,
            "beacon_dropout": True,
            "combined_input": False
        },
        "CG + AL + CI": {
            "confidence_gate": True,
            "auxiliary_loss": True,
            "beacon_dropout": False,
            "combined_input": True
        },
        "CG + BD + CI": {
            "confidence_gate": True,
            "auxiliary_loss": False,
            "beacon_dropout": True,
            "combined_input": True
        },
        "AL + BD + CI": {
            "confidence_gate": False,
            "auxiliary_loss": True,
            "beacon_dropout": True,
            "combined_input": True
        },
        "Full Model": {
            "confidence_gate": True,
            "auxiliary_loss": True,
            "beacon_dropout": True,
            "combined_input": True
        }
    }
    
    # 결과 수집
    results_data = []
    
    for config_name, config_flags in configurations.items():
        # 폴더명 생성
        if config_name == "Baseline":
            folder_name = "_1_baseline"
        elif config_name == "Full Model":
            folder_name = "_10_full_model"
        elif config_name == "Confidence Gate":
            folder_name = "_2_conf_gate"
        elif config_name == "Auxiliary Loss":
            folder_name = "_3_aux_loss"
        elif config_name == "Beacon Dropout":
            folder_name = "_4_beacon_dropout"
        elif config_name == "Combined Input":
            folder_name = "_5_combined_input"
        elif config_name == "CG + AL + BD":
            folder_name = "_6_cg_al_bd"
        elif config_name == "CG + AL + CI":
            folder_name = "_7_cg_al_ci"
        elif config_name == "CG + BD + CI":
            folder_name = "_8_cg_bd_ci"
        elif config_name == "AL + BD + CI":
            folder_name = "_9_al_bd_ci"
        
        results_dir = os.path.join(results_root, folder_name)
        
        if os.path.exists(results_dir):
            accuracies = extract_results(results_dir)
            if accuracies:
                avg_acc = np.mean(list(accuracies.values()))
                std_acc = np.std(list(accuracies.values()))
                results_data.append({
                    "Configuration": config_name,
                    "Confidence Gate": config_flags["confidence_gate"],
                    "Auxiliary Loss": config_flags["auxiliary_loss"],
                    "Beacon Dropout": config_flags["beacon_dropout"],
                    "Combined Input": config_flags["combined_input"],
                    "Avg Accuracy": avg_acc,
                    "Std Accuracy": std_acc,
                    "Details": accuracies
                })
            else:
                print(f"⚠️  {folder_name}: 결과를 찾을 수 없습니다.")
        else:
            print(f"⚠️  {folder_name}: 디렉토리를 찾을 수 없습니다.")
    
    # DataFrame 생성
    df = pd.DataFrame(results_data)
    
    if len(df) > 0:
        # 정렬 (정확도 기준)
        df = df.sort_values("Avg Accuracy", ascending=False).reset_index(drop=True)
        
        # 결과 출력
        print("\n" + "="*80)
        print("Ablation Study 결과 분석")
        print("="*80)
        print("\n전체 결과 (정확도 내림차순):\n")
        print(df[["Configuration", "Confidence Gate", "Auxiliary Loss", 
                   "Beacon Dropout", "Combined Input", "Avg Accuracy", "Std Accuracy"]].to_string())
        
        # Baseline 및 Full Model 성능
        baseline = df[df["Configuration"] == "Baseline"]
        full_model = df[df["Configuration"] == "Full Model"]
        
        if len(baseline) > 0 and len(full_model) > 0:
            baseline_acc = baseline["Avg Accuracy"].values[0]
            full_acc = full_model["Avg Accuracy"].values[0]
            improvement = ((full_acc - baseline_acc) / baseline_acc) * 100
            
            print("\n" + "="*80)
            print("핵심 비교")
            print("="*80)
            print(f"Baseline (모든 요소 OFF):     {baseline_acc:.4f}")
            print(f"Full Model (모든 요소 ON):    {full_acc:.4f}")
            print(f"전체 성능 향상:               {improvement:+.2f}%")
        
        # 개별 요소별 기여도 계산
        print("\n" + "="*80)
        print("개별 구성요소 기여도 추정 (Baseline 대비)")
        print("="*80)
        
        if len(baseline) > 0:
            baseline_acc = baseline["Avg Accuracy"].values[0]
            
            # 각 단일 요소 기여도
            components = ["Confidence Gate", "Auxiliary Loss", "Beacon Dropout", "Combined Input"]
            contributions = {}
            
            for comp in components:
                single_config = df[df["Configuration"] == comp]
                if len(single_config) > 0:
                    acc = single_config["Avg Accuracy"].values[0]
                    contrib = ((acc - baseline_acc) / baseline_acc) * 100
                    contributions[comp] = contrib
                    print(f"{comp:20s}: {contrib:+7.2f}% (Acc: {acc:.4f})")
            
        
        # CSV로 저장
        output_file = os.path.join(results_root, "ablation_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✅ 결과가 {output_file}에 저장되었습니다.")
    
    else:
        print("❌ 실행된 실험 결과를 찾을 수 없습니다.")

if __name__ == "__main__":
    analyze_ablation_study()
