#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
BENCHMARK="${1:-3DMatch}"
shift "$(( $# > 0 ? 1 : 0 ))"
N_DESCS="${N_DESCS:-5000}"
RANSAC_ITERS="${RANSAC_ITERS:-50000}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
GPU="${GPU:-0}"
export GPU_ID="$GPU"
mkdir -p logs
case "$BENCHMARK" in
  3DMatch|3DLoMatch)
    CKPT="${CKPT:-checkpoints/gsp_gatedreg_3dmatch.pth}"
    ENTRY=gsp_reg/val_eval_checkpoint.py
    SPLIT=(--benchmark "$BENCHMARK")
    ;;
  KITTI)
    CKPT="${CKPT:-checkpoints/gsp_gatedreg_kitti.pth}"
    ENTRY=gsp_reg/val_eval_kitti_checkpoint.py
    SPLIT=(--split test)
    ;;
  *) echo "Expected 3DMatch, 3DLoMatch, or KITTI" >&2; exit 2 ;;
esac
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT (see checkpoints/README.md)" >&2
  exit 2
fi
TAG="$(date +%Y%m%d_%H%M%S)"
for N in $N_DESCS; do
  EXTRA=()
  if [[ -n "${CACHE_DIR:-}" ]]; then
    [[ "$BENCHMARK" != KITTI ]] || { echo "KITTI cache mode is not implemented" >&2; exit 2; }
    EXTRA+=(--preprocessed_dir "$CACHE_DIR")
  fi
  if [[ "$SAMPLE_MODE" == predator_prob ]]; then
    DEFAULT_PREDATOR=third_party/OverlapPredator/weights/indoor.pth
    [[ "$BENCHMARK" != KITTI ]] || DEFAULT_PREDATOR=third_party/OverlapPredator/weights/kitti.pth
    EXTRA+=(--predator_ckpt "${PREDATOR_CKPT:-$DEFAULT_PREDATOR}")
  fi
  "$PYTHON" -u "$ENTRY" "$CKPT" "${SPLIT[@]}" --gpu "$GPU" \
    --sample_mode "$SAMPLE_MODE" --n_desc "$N" --ransac_iters "$RANSAC_ITERS" \
    --log "logs/eval_${BENCHMARK}_${SAMPLE_MODE}_n${N}_r${RANSAC_ITERS}_${TAG}.log" \
    "${EXTRA[@]}" "$@"
done
