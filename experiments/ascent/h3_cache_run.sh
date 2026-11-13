#!/bin/bash
# Usage: h3_cache_run.sh <python> <checkpoint> <H> <W> <F> <arm>...
#   arm = "tag|cache_json_or_none|seed seed ..."
# Each arm is its own 2-GPU VisualGen instance (untimed 2-step warmup, then its seeds back
# to back). Afterwards every arm's first output is compared with the arm tagged "base"
# (if present), an arm's repeated seed is compared with its first output, and per-rank
# Cache-DiT decisions are checked. Logs go to $AUTO_DEV_ARTIFACTS.
set -o pipefail
PY=$1; CKPT=$2; H=$3; W=$4; F=$5; shift 5
ART=${AUTO_DEV_ARTIFACTS:-/tmp/h3art}
OUT=/tmp/h3_cache_out_$$
mkdir -p "$ART" "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
export TLLM_LOG_LEVEL=INFO
CROSS=${H3_CROSS:-}
TAGS=()
for ARM in "$@"; do
  IFS='|' read -r TAG CACHE SEEDS <<< "$ARM"
  echo "=== arm $TAG: cache=$CACHE seeds=$SEEDS"
  $PY "$HERE/h3_cache_check.py" --model "$CKPT" --tag "$TAG" --cache "$CACHE" --out "$OUT" \
    --height $H --width $W --frames $F --seeds $SEEDS 2>&1 | tee "$ART/${TAG}_arm.log"
  echo "arm $TAG rc=$?"
  TAGS+=("$TAG")
done
$PY "$HERE/h3_cache_compare.py" --dir "$OUT" --logdir "$ART" --tags "${TAGS[@]}" --cross $CROSS
