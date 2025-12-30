#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1) 각 pos_fold별 결과 디렉터리에서
   - 혼동행렬 confmat_pos_k.npy 로드
   - report_pos_k.txt 에서 Best TEST Acc / MacroF1 파싱

2) 평균 혼동행렬(average confusion matrix) 계산 후
   - confmat_avg.npy 저장
   - confmat_avg.png 그림 저장 (save_confusion_matrix 이용)

3) K-Fold Best-by-Fold Summary 텍스트 생성
   - pos_k별 acc / macroF1
   - 평균 및 표준편차까지
   => kfold_summary.txt 로 저장
"""

import os
import re
import numpy as np

from utils import save_confusion_matrix


# 🔧 실험 결과 폴더 기본 경로 (train 스크립트의 ./outputs 와 동일)
BASE_DIR = "./outputs"

# 🔧 output_dir 패턴: "./outputs/" + OUTPUT_DIR_PREFIX + MODEL_TYPE + f"|pos:{pos_split}"
# 예: "./outputs/strats|pos:pos_0"
OUTPUT_DIR_PREFIX = ""       # train에서 args.output_dir_prefix 쓴 값이 있으면 여기에
MODEL_TYPE = "strats"        # 'strats' 또는 'istrats'

# 사용할 pos 목록 (k-fold)
POS_LIST = [f"pos_{i}" for i in range(5)]

# report 파일에서 acc / macroF1 파싱용 정규식
RE_ACC = re.compile(r"Best TEST Acc\s*:\s*([0-9.]+)")
RE_F1 = re.compile(r"Best TEST MacroF1\s*:\s*([0-9.]+)")


def parse_metrics_from_report(report_path: str):
    """report_pos_k.txt에서 acc, macroF1 숫자를 파싱."""
    acc = None
    macro_f1 = None

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    m_acc = RE_ACC.search(text)
    m_f1 = RE_F1.search(text)

    if m_acc:
        acc = float(m_acc.group(1))
    if m_f1:
        macro_f1 = float(m_f1.group(1))

    return acc, macro_f1


def main():
    fold_stats = []   # [(pos, acc, macro_f1), ...]
    cms = []          # [cm_pos0, cm_pos1, ...]

    for pos in POS_LIST:
        # train 스크립트에서 만들어진 output_dir과 동일한 규칙으로 맞추기
        out_dir = os.path.join(
            BASE_DIR,
            f"{OUTPUT_DIR_PREFIX}{MODEL_TYPE}|pos:{pos}"
        )

        # 1) report_pos_k.txt에서 acc / macroF1 파싱
        report_path = os.path.join(out_dir, f"report_{pos}.txt")
        if not os.path.exists(report_path):
            print(f"[WARN] report not found for {pos}: {report_path}")
        else:
            acc, macro_f1 = parse_metrics_from_report(report_path)
            if acc is None or macro_f1 is None:
                print(f"[WARN] cannot parse metrics from {report_path}")
            else:
                fold_stats.append((pos, acc, macro_f1))

        # 2) confmat_pos_k.npy 로드
        cm_path = os.path.join(out_dir, f"confmat_{pos}.npy")
        if not os.path.exists(cm_path):
            print(f"[WARN] confusion matrix not found for {pos}: {cm_path}")
        else:
            cm = np.load(cm_path)
            cms.append(cm)

    # ---------------- 평균 혼동행렬 계산 & 저장 ----------------
    if not cms:
        print("[ERROR] No confusion matrices loaded. 경로/파일명을 확인해줘!")
    else:
        # float로 평균 내기
        avg_cm = sum(cms) / len(cms)

        # 평균 혼동행렬 저장 디렉터리
        avg_dir = os.path.join(
            BASE_DIR,
            f"{OUTPUT_DIR_PREFIX}{MODEL_TYPE}|pos:avg"
        )
        os.makedirs(avg_dir, exist_ok=True)

        # NPY 저장
        np.save(os.path.join(avg_dir, "confmat_avg.npy"), avg_cm)

        # PNG 그림 저장 (pos='avg' + matrix_override 사용)
        save_confusion_matrix(
            matrix_override=avg_cm,
            pos='avg',
            result_dir=avg_dir,
            num_classes=avg_cm.shape[0],
            filename="confmat_avg.png"
        )
        print(f"[✓] Average confusion matrix saved to '{avg_dir}'")

    # ---------------- K-Fold summary txt 생성 & 저장 ----------------
    if not fold_stats:
        print("[ERROR] No fold stats parsed. report_*.txt 경로를 확인해줘!")
        return

    accs = np.array([x[1] for x in fold_stats])
    f1s = np.array([x[2] for x in fold_stats])

    acc_mean = accs.mean()
    acc_std = accs.std(ddof=0)
    f1_mean = f1s.mean()
    f1_std = f1s.std(ddof=0)

    lines = []
    lines.append("K-Fold Best-by-Fold Summary")
    lines.append("===========================")
    for pos, acc, macro_f1 in fold_stats:
        lines.append(f"{pos}: acc={acc:.4f}, macroF1={macro_f1:.4f}")
    lines.append("")
    lines.append(f"Avg ACC     : {acc_mean:.4f}  (± {acc_std:.4f})")
    lines.append(f"Avg Macro-F1: {f1_mean:.4f}  (± {f1_std:.4f})")

    summary_text = "\n".join(lines)

    # 요약 txt 저장 위치: BASE_DIR 안에 kfold_summary.txt
    summary_path = os.path.join(BASE_DIR, "kfold_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\n[✓] K-fold summary saved to '{summary_path}'")


if __name__ == "__main__":
    main()
