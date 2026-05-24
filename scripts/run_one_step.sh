#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DATASET_REPO="${DATASET_REPO:-Dangindev/viet-cultural-vqa}"
VJEPA_REPO="${VJEPA_REPO:-facebook/vjepa2-vitl-fpc64-256}"
QENCODER_REPO="${QENCODER_REPO:-Qwen/Qwen3-0.6B}"
YENCODER_REPO="${YENCODER_REPO:-google/embeddinggemma-300m}"
MAX_QUERY_LEN="${MAX_QUERY_LEN:-512}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20}"
VAL_SAMPLES="${VAL_SAMPLES:-8}"
TEST_SAMPLES="${TEST_SAMPLES:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
BASE_LR="${BASE_LR:-5e-5}"
DEVICE="${DEVICE:-auto}"
MAX_STEPS="${MAX_STEPS:-}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-}"

COMMON_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --dataset-repo "$DATASET_REPO"
  --vjepa-repo "$VJEPA_REPO"
  --qencoder-repo "$QENCODER_REPO"
  --yencoder-repo "$YENCODER_REPO"
  --max-query-len "$MAX_QUERY_LEN"
  --train-samples "$TRAIN_SAMPLES"
  --val-samples "$VAL_SAMPLES"
  --test-samples "$TEST_SAMPLES"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --epochs "$EPOCHS"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --base-lr "$BASE_LR"
  --device "$DEVICE"
)

if [[ -n "$MAX_STEPS" ]]; then
  COMMON_ARGS+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "$EVAL_MAX_STEPS" ]]; then
  COMMON_ARGS+=(--eval-max-steps "$EVAL_MAX_STEPS")
fi
"$PYTHON" "$REPO_ROOT/src/train.py" \
  --scenario one_step \
  "${COMMON_ARGS[@]}"
