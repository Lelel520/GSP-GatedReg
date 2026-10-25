#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
for extension in cpp_subsampling cpp_neighbors; do
  (cd "$ROOT/gsp_reg/cpp_wrappers/$extension" && "$PYTHON" setup.py build_ext --inplace)
done
echo "Built grid-subsampling and radius-neighbor extensions."
