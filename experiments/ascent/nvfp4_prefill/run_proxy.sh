#!/bin/bash
# 2-arm accuracy proxy: A = kill switch (TRTLLM_FP8_PREFILL_NVFP4_GDN=0, main
# behaviour), C = NVFP4 prefill copy of GDN in_proj_qkvz. One process per arm.
# Usage: run_proxy.sh <python> <ckpt>
set -u
PY=$1
CKPT=$2
OUT=${ASCENT_OUT:-/tmp/nvfp4_prefill_proxy}
mkdir -p "$OUT"
FILT='^\[[0-9]{2}:|^\{|Error|Traceback|fp8_prefill_nvfp4|Autotun|autotun|KV cache|kv cache|max_tokens'
echo "=== arm A (kill switch) ==="
TRTLLM_FP8_PREFILL_NVFP4_GDN=0 "$PY" -u experiments/ascent/nvfp4_prefill/proxy.py \
  --ckpt "$CKPT" --arm A --out "$OUT/A.json" 2>&1 \
  | grep --line-buffered -E "$FILT" | cut -c1-400
echo "=== arm C (in_proj_qkvz NVFP4 prefill) ==="
env -u TRTLLM_FP8_PREFILL_NVFP4_GDN "$PY" -u experiments/ascent/nvfp4_prefill/proxy.py \
  --ckpt "$CKPT" --arm C --out "$OUT/C.json" 2>&1 \
  | grep --line-buffered -E "$FILT" | cut -c1-400
echo "=== compare ==="
"$PY" experiments/ascent/nvfp4_prefill/compare.py "$OUT/A.json" "$OUT/C.json"
