#!/bin/bash
# Idea 96 / branch 199: first 4-GPU job = preamble (plan 3241 (9)) + DIRTY SLOT guard + 12-pair peer assert
# + lp_check STEP 1 (timeout -s KILL 300) + the Stage 0 table (stage0.py).
# usage: bash run_stage0.sh <python> <checkpoint> [stage0|<other script.py> args...]
PY="$1"; CKPT="$2"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
LPT="$(dirname "$HERE")/lp_ar_trace"
{ echo "=== PREAMBLE $(date -Is)"; hostname; nvidia-smi topo -m; nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; cat /proc/cmdline;
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv;
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv;
  echo "HEAD $(git -C "$HERE" rev-parse HEAD 2>&1)"; } 1>&2
bash "$LPT/guard.sh" true || exit 3
# pitfall 3271: receipt of the native library this job actually loads (path, mtime, landing-61 mtile_v3 symbol count)
LIB=$("$PY" -c 'import os, tensorrt_llm; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libtensorrt_llm.so"))' 2>/dev/null | tail -1)
{ echo "[lib] path=$LIB"; ls -l --time-style=full-iso "$LIB" 2>&1;
  for f in "$(dirname "$LIB")"/*.so; do
    c1=$(grep -c -a fused_pmap_gemm_fma_mtile_v3 "$f" 2>/dev/null); c2=$(grep -c -a lowPrecisionTwoShotAllReduceKernel "$f" 2>/dev/null)
    [ "${c1:-0}" != "0" -o "${c2:-0}" != "0" ] && echo "[lib] $(ls -l --time-style=full-iso "$f") mtile_v3=$c1 lpTwoShot=$c2"
  done; } 1>&2
"$PY" -c '
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", file=sys.stderr, flush=True)
sys.exit(0 if n == 4 and not bad else 3)
' || { echo '{"passed": false, "error": "peer-access assert failed"}'; exit 3; }
echo "=== STEP 1 $(date -Is) lp_check (timeout -s KILL 300)" 1>&2
MARK="lpcheck_$$_${RANDOM}"
LP_CHECK_MARK="$MARK" timeout -s KILL 300 "$PY" "$LPT/lp_check.py" 1>&2
RC=$?
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  if tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -qx "LP_CHECK_MARK=${MARK}"; then kill -9 "$pid" 2>/dev/null; fi
done
sleep 2
echo "LP-STEP1 rc=${RC}" 1>&2
if [ "$RC" -ne 0 ]; then echo "{\"passed\": false, \"error\": \"lp_check STEP 1 failed or hung\", \"rc\": ${RC}}"; exit 3; fi
WHAT="${1:-stage0}"; [ $# -gt 0 ] && shift
if [ "$WHAT" = "stage0" ]; then
  echo "=== STAGE 0 $(date -Is)" 1>&2
  exec "$PY" "$HERE/stage0.py" "$CKPT"
fi
exec "$PY" "$HERE/$WHAT" "$CKPT" "$@"
