#!/usr/bin/env bash
# Build experiments/ascent/fmha_ctx/bench.cu against a cpp/kernels/fmha_v2/src tree.
#   build.sh <fmha_v2_src_dir> <out_binary> [extra nvcc args...]
# Same compile definitions as the generated sm120 fmha_v2 TUs (USE_DEMO_BERT_PARAMS=1 GENERATE_CUBIN=1,
# --expt-extended-lambda --expt-relaxed-constexpr, -O3, sm_120 family target). Prints ptxas resource usage.
set -euo pipefail
SRC_DIR="$1"
OUT="$2"
shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="-gencode arch=compute_120f,code=sm_120f"
if ! nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120f; then
  ARCH="-gencode arch=compute_120a,code=sm_120a"
fi
t0=$(date +%s)
nvcc -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr \
  -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 $ARCH \
  -Xptxas -v -I"$SRC_DIR" "$@" "$HERE/bench.cu" -o "$OUT" -lcuda 2>&1 \
  | grep -E "error|warning: .*(spill|stack)|fmha_v2_flash_attention.*|Used [0-9]+ registers|spill" || true
test -x "$OUT"
echo "[build] $OUT from $SRC_DIR in $(( $(date +%s) - t0 ))s"
