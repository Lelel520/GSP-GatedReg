#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
BENCHMARK="${1:-3DMatch}"
export GPU_ID="${GPU_ID:-0}"
export EPOCH_PAIRS="${EPOCH_PAIRS:-0}"
export LR="${LR:-0.0005}"
export START_EPOCH="${START_EPOCH:-1}"
export EPOCHS="${EPOCHS:-400}"
export SAVE_AFTER_EPOCH="${SAVE_AFTER_EPOCH:-0}"
export SAVE_EPOCH_INTERVAL="${SAVE_EPOCH_INTERVAL:-50}"
export LAMBDA_DIST="${LAMBDA_DIST:-1.0}"
export LAMBDA_VAR="${LAMBDA_VAR:-200}"
export VAR_GAMMA="${VAR_GAMMA:-0.06}"
export TAU_LOW="${TAU_LOW:-0.03}"
export TAU_HIGH="${TAU_HIGH:-0.20}"
case "$BENCHMARK" in
  3DMatch)
    export LAMBDA_DEG="${LAMBDA_DEG:-0.05}"
    export SAFE_RADIUS="${SAFE_RADIUS:-0.10}"
    export CIRCLE_POS_MARGIN="${CIRCLE_POS_MARGIN:-0.10}"
    ENTRY=gsp_reg/run_pre_softcls.py
    ;;
  KITTI)
    export LAMBDA_DEG="${LAMBDA_DEG:-0.10}"
    export SAFE_RADIUS="${SAFE_RADIUS:-1.0}"
    export CIRCLE_POS_MARGIN="${CIRCLE_POS_MARGIN:-0.5}"
    export PREPROC_DIR="${PREPROC_DIR:-data/kitti/preprocessed}"
    ENTRY=gsp_reg/run_pre_softcls_kitti.py
    ;;
  *) echo "Expected 3DMatch or KITTI. Never train on 3DLoMatch test pairs." >&2; exit 2 ;;
esac
"$PYTHON" -u "$ENTRY"
