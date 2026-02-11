# Ablation Study - 구성요소 On/Off 사용법

각 구성요소를 on/off하여 성능 기여도를 측정합니다.

## 🎯 4가지 구성요소

1. `--use_confidence_gate` - 신뢰도 게이트
2. `--use_auxiliary_loss` - 보조 복원 손실
3. `--use_beacon_dropout` - 비콘 드롭아웃
4. `--use_combined_input` - RSSI+Mask 결합 입력

## 💻 사용 방법

### 기본 형식
```bash
python main.py \
    --npz_root <데이터_경로> \
    --epochs 100 \
    --use_confidence_gate <true|false> \
    --use_auxiliary_loss <true|false> \
    --use_beacon_dropout <true|false> \
    --use_combined_input <true|false> \
    --results_dir <결과_저장_경로>
```

### 예시 1: Baseline (모든 요소 OFF)
```bash
python main.py \
    --npz_root "../experiment_daeun/iBeacon_JUIndoorLoc/npz/in-motion" \
    --epochs 100 \
    --use_confidence_gate false \
    --use_auxiliary_loss false \
    --use_beacon_dropout false \
    --use_combined_input false \
    --results_dir "./results_baseline"
```

### 예시 2: Full Model (모든 요소 ON)
```bash
python main.py \
    --npz_root "../experiment_daeun/iBeacon_JUIndoorLoc/npz/in-motion" \
    --epochs 100 \
    --use_confidence_gate true \
    --use_auxiliary_loss true \
    --use_beacon_dropout true \
    --use_combined_input true \
    --results_dir "./results_full"
```

### 예시 3: Confidence Gate만 ON
```bash
python main.py \
    --npz_root "../experiment_daeun/iBeacon_JUIndoorLoc/npz/in-motion" \
    --epochs 100 \
    --use_confidence_gate true \
    --use_auxiliary_loss false \
    --use_beacon_dropout false \
    --use_combined_input false \
    --results_dir "./results_conf_gate"
```

### 자동 테스트 (간단한 버전)
```bash
bash run_simple_test.sh
```

## 📝 Boolean 옵션
- **ON**: `--use_confidence_gate true` (소문자)
- **OFF**: `--use_auxiliary_loss false` (소문자)

## 📊 결과 확인
```bash
cat results_baseline/pos_0/classification_report.txt
cat results_full/pos_0/classification_report.txt
```
