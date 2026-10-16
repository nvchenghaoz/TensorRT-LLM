#!/bin/bash
# Idea 86 accuracy proxy: run two arms (first = reference) in one job, then
# compare the second against the first. Usage: run_arms.sh <python> <ckpt> <ref> <cand>
# Arms (GDN in_proj_qkvz prefill copy stays on, as on main):
#   A = new groups off (main), B = all on (attn qkv opt-in =1),
#   C = attn o_proj + GDN out_proj on, attn qkv off (branch default).
# Watchdog (pitfall 480): python -u, grep --line-buffered, no cut, 60 s heartbeat.
set -u
PY=$1
CKPT=$2
REF=$3
CAND=$4
OUT=${ASCENT_ARTIFACTS_DIR:-${ARTIFACTS_DIR:-/tmp/nvfp4_prefill_proxy39}}
mkdir -p "$OUT"
FILT='^\[[0-9]{2}:|^\{|Error|Traceback|fp8_prefill_nvfp4|nvfp4_decode_copy|KV cache|kv cache|max_tokens|^hb'
(while sleep 60; do echo "hb $(date +%T)"; done) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

arm_env() {
  case $1 in
    A) echo "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_QG=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_O=0 TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT=0" ;;
    B) echo "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV=1 TRTLLM_FP8_PREFILL_NVFP4_ATTN_QG=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_O=1 TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT=1" ;;
    C) echo "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_QG=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_O=1 TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT=1" ;;
    # Q = C + attn qkv_proj Q+gate columns on W4A4 (K/V stay FP8), idea 141
    Q) echo "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV=0 TRTLLM_FP8_PREFILL_NVFP4_ATTN_QG=1 TRTLLM_FP8_PREFILL_NVFP4_ATTN_O=1 TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT=1" ;;
  esac
}

echo "TREE_COMMIT $(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null)"
for ARM in "$REF" "$CAND"; do
  echo "=== arm $ARM: $(arm_env $ARM) ==="
  env $(arm_env $ARM) "$PY" -u experiments/ascent/nvfp4_prefill/proxy.py \
    --ckpt "$CKPT" --arm "$ARM" --out "$OUT/$ARM.json" 2>&1 \
    | grep --line-buffered -E "$FILT"
done
kill $HB 2>/dev/null
wait $HB 2>/dev/null
echo "=== compare $CAND vs $REF ==="
"$PY" experiments/ascent/nvfp4_prefill/compare.py "$OUT/$REF.json" "$OUT/$CAND.json"
