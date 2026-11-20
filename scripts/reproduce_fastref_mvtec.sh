#!/usr/bin/env bash
set -euo pipefail

# Reproduce MVTec-AD 1/2/4-shot AnomalyDINO vs. AnomalyDINO+ (FastRef, CVPR 2026 Table 2).

DATA_ROOT="${DATA_ROOT:-/data/MVTecAD}"
MODEL_NAME="${MODEL_NAME:-dinov2_vits14}"
RESOLUTION="${RESOLUTION:-448}"
PREPROCESS="${PREPROCESS:-force_no_mask_no_rotation}"
SHOTS="${SHOTS:-1 2 4}"
NUM_SEEDS="${NUM_SEEDS:-3}"
DEVICE="${DEVICE:-cuda:0}"
ALPHA="${ALPHA:-0.5}"
LAMBDA="${LAMBDA:-0.1}"
ITERS="${ITERS:-2}"
OT_ITERS="${OT_ITERS:-10}"
OT_EPS="${OT_EPS:-0.1}"
SCORE_MODE="${SCORE_MODE:-reconstruction}"
OT_SCALE="${OT_SCALE:-row_normalized}"
TAG="${TAG:-}"

read -r -a SHOTS_ARR <<< "${SHOTS}"

COMMON_ARGS=(
  --dataset MVTec
  --data_root "${DATA_ROOT}"
  --model_name "${MODEL_NAME}"
  --resolution "${RESOLUTION}"
  --preprocess "${PREPROCESS}"
  --shots "${SHOTS_ARR[@]}"
  --num_seeds "${NUM_SEEDS}"
  --device "${DEVICE}"
  --eval_clf
  --eval_segm
  --no-save_examples
)

TAG_ARGS=()
if [[ -n "${TAG}" ]]; then
  TAG_ARGS=(--tag "${TAG}")
fi

echo "=== Running Baseline AnomalyDINO on MVTec-AD ==="
python3 run_anomalydino.py \
  "${COMMON_ARGS[@]}" \
  "${TAG_ARGS[@]}"

echo "=== Running AnomalyDINO+ (FastRef) on MVTec-AD ==="
python3 run_anomalydino.py \
  "${COMMON_ARGS[@]}" \
  "${TAG_ARGS[@]}" \
  --use_fastref \
  --fastref_alpha "${ALPHA}" \
  --fastref_lambda "${LAMBDA}" \
  --fastref_iters "${ITERS}" \
  --fastref_ot_iters "${OT_ITERS}" \
  --fastref_ot_eps "${OT_EPS}" \
  --fastref_score_mode "${SCORE_MODE}" \
  --fastref_ot_scale "${OT_SCALE}"

echo "=== Summary vs. CVPR 2026 Table 2 ==="
python3 scripts/summarize_fastref_mvtec.py \
  --results_root results_MVTec \
  --model_dir "${MODEL_NAME}_${RESOLUTION}" \
  --preprocess "${PREPROCESS}" \
  --shots "${SHOTS_ARR[@]}" \
  "${TAG_ARGS[@]}"

# Optional ablation examples:
# 1) Nearest-neighbor scoring ablation:
#    SCORE_MODE=nn TAG=ablation_nn bash scripts/reproduce_fastref_mvtec.sh
#    (passes --fastref_score_mode nn)
# 2) Unscaled optimal transport plan ablation:
#    OT_SCALE=unscaled TAG=ablation_unscaled bash scripts/reproduce_fastref_mvtec.sh
#    (passes --fastref_ot_scale unscaled)
# 3) Agnostic preprocessing (masking + rotation) ablation:
#    PREPROCESS=agnostic TAG=ablation_agnostic bash scripts/reproduce_fastref_mvtec.sh
#    (passes --preprocess agnostic)
