#!/bin/bash
set -e
set -u

# 구성요소별 성능 기여도 측정 스크립트
# 10가지 조합 실행 (baseline + 개별 4개 + 3개 조합 4개 + full model)

echo "========== Ablation Study: 10가지 조합 실행 =========="
echo ""

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 설정값
NPZ_ROOT="../data/data_split/npz/in-motion"
EPOCHS=100
BATCH_SIZE=256
EMBED=96
HEADS=8
LAYERS=2
DROPOUT=0.3
LR=0.001
WEIGHT_DECAY=0.001
AUX_MASK_RATIO=0.30
AUX_LOSS_WEIGHT=0.40
BEACON_DROPOUT_P=0.30
ALPHA_NEIGHBOR=0.20

# 변수 확인
echo "변수 설정:"
echo "  NPZ_ROOT: $NPZ_ROOT"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo ""

# 데이터 경로 확인
if [ ! -d "$NPZ_ROOT" ]; then
    echo "❌ 에러: 데이터 경로를 찾을 수 없습니다: $NPZ_ROOT"
    echo "📝 NPZ_ROOT을 수정하여 다시 실행해주세요."
    exit 1
fi

echo "✓ 데이터 경로 확인됨"
echo ""

# 1. Baseline: 모든 요소 OFF
echo "[1/10] Baseline (모든 요소 OFF)"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate false \
    --use_auxiliary_loss false \
    --use_beacon_dropout false \
    --use_combined_input false \
    --results_dir "./results_ablation/_1_baseline"

# 2. Confidence Gate
echo "[2/10] Confidence Gate ON"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate true \
    --use_auxiliary_loss false \
    --use_beacon_dropout false \
    --use_combined_input false \
    --results_dir "./results_ablation/_2_conf_gate"

# 3. Auxiliary Loss
echo "[3/10] Auxiliary Loss ON"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate false \
    --use_auxiliary_loss true \
    --use_beacon_dropout false \
    --use_combined_input false \
    --results_dir "./results_ablation/_3_aux_loss"

# 4. Beacon Dropout
echo "[4/10] Beacon Dropout ON"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate false \
    --use_auxiliary_loss false \
    --use_beacon_dropout true \
    --use_combined_input false \
    --results_dir "./results_ablation/_4_beacon_dropout"

# 5. Combined Input
echo "[5/10] Combined Input ON"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate false \
    --use_auxiliary_loss false \
    --use_beacon_dropout false \
    --use_combined_input true \
    --results_dir "./results_ablation/_5_combined_input"

# 6. CG + AL + BD
echo "[6/10] Confidence Gate + Auxiliary Loss + Beacon Dropout"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate true \
    --use_auxiliary_loss true \
    --use_beacon_dropout true \
    --use_combined_input false \
    --results_dir "./results_ablation/_6_cg_al_bd"

# 7. CG + AL + CI
echo "[7/10] Confidence Gate + Auxiliary Loss + Combined Input"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate true \
    --use_auxiliary_loss true \
    --use_beacon_dropout false \
    --use_combined_input true \
    --results_dir "./results_ablation/_7_cg_al_ci"

# 8. CG + BD + CI
echo "[8/10] Confidence Gate + Beacon Dropout + Combined Input"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate true \
    --use_auxiliary_loss false \
    --use_beacon_dropout true \
    --use_combined_input true \
    --results_dir "./results_ablation/_8_cg_bd_ci"

# 9. AL + BD + CI
echo "[9/10] Auxiliary Loss + Beacon Dropout + Combined Input"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate false \
    --use_auxiliary_loss true \
    --use_beacon_dropout true \
    --use_combined_input true \
    --results_dir "./results_ablation/_9_al_bd_ci"

# 10. Full Model
echo "[10/10] Full Model (모든 요소 ON)"
python main.py \
    --npz_root "$NPZ_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embed "$EMBED" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --aux_mask_ratio "$AUX_MASK_RATIO" \
    --aux_loss_weight "$AUX_LOSS_WEIGHT" \
    --beacon_dropout_p "$BEACON_DROPOUT_P" \
    --alpha_neighbor "$ALPHA_NEIGHBOR" \
    --use_confidence_gate true \
    --use_auxiliary_loss true \
    --use_beacon_dropout true \
    --use_combined_input true \
    --results_dir "./results_ablation/_10_full_model"


echo ""
echo "========== Ablation Study 완료 =========="
echo "결과는 ./results_ablation 디렉토리에 저장되었습니다."
