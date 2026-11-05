#!/bin/bash
# Branch 181 / idea 226 JOB A: ONE 4-GPU job, steps in order (brief rev 5n/5o).
#   STEP 0  node receipt to stderr (hostname, topo -m / -p2p r / -p2p w), then the 12-pair peer-access assert.
#   STEP 1  lp_check.py under timeout -s KILL 300 at the committed default grid: [2048,5120] LP, every call vs fp32 sum.
#   STEP 3/4 width.py under timeout -s KILL 1500: width sweep timing (NCCL, LP 16..128), bitwise vs env=16 on every
#           element and call, 1000-call robust run at the default with IPC flag inspection, routing/kernels last.
#   STEP 5  trace_graph.py <checkpoint> --with-check on the default (no TRTLLM_LP_AR_TWOSHOT_CTAS in env).
# Any failure or hang stops the job before the next step. Last stdout line: combined JSON with "passed".
# usage: bash run_job_a.sh <python> <checkpoint>
PY="$1"; CKPT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
LPAR="$(dirname "$HERE")/lp_ar"
ART="${AUTO_DEV_ARTIFACTS:-/tmp/lp_job_a_$$}"
mkdir -p "$ART"
unset TRTLLM_LP_AR_TWOSHOT_CTAS
T0=$(date +%s)
echo "=== STEP 0 $(date -Is)"
{ hostname; nvidia-smi topo -m; nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; } 1>&2
nvidia-smi --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv 1>&2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} hostname=$(hostname)"
env | grep -E '^(NCCL_|TRTLLM_ENABLE_PDL|TRTLLM_LP_)' || echo "no NCCL_/TRTLLM_ENABLE_PDL/TRTLLM_LP_ env"

"$PY" -c '
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", flush=True)
sys.exit(0 if n == 4 and not bad else 3)
' || { echo '{"passed": false, "error": "STEP 0 peer-access assert failed"}'; exit 3; }

kill_marked() {
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    if tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -qx "LP_CHECK_MARK=$1"; then
      echo "[cleanup] kill -9 ${pid} $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-160)"
      kill -9 "$pid" 2>/dev/null
    fi
  done
  sleep 2
}

echo "=== STEP 1 $(date -Is) lp_check (timeout -s KILL 300, default grid)"
MARK="lpcheck_$$_${RANDOM}"
T1=$(date +%s)
LP_CHECK_MARK="$MARK" timeout -s KILL 300 "$PY" "$HERE/lp_check.py" | tee "$ART/lp_check.log"
RC1=${PIPESTATUS[0]}
kill_marked "$MARK"
echo "LP-STEP1 rc=${RC1} wall=$(( $(date +%s) - T1 ))s"
tail -n 1 "$ART/lp_check.log" > "$ART/lp_check_last.json"
if [ "$RC1" -ne 0 ]; then
  echo "{\"passed\": false, \"error\": \"STEP 1 lp_check failed or hung\", \"step1_rc\": ${RC1}}"
  exit 3
fi

echo "=== STEP 3/4 $(date -Is) width.py (timeout -s KILL 1500)"
MARK="lpwidth_$$_${RANDOM}"
T3=$(date +%s)
LP_CHECK_MARK="$MARK" timeout -s KILL 1500 "$PY" "$LPAR/width.py" | tee "$ART/width.log"
RC3=${PIPESTATUS[0]}
kill_marked "$MARK"
echo "LP-STEP3 rc=${RC3} wall=$(( $(date +%s) - T3 ))s"
tail -n 1 "$ART/width.log" > "$ART/width_last.json"
if [ "$RC3" -ne 0 ]; then
  echo "{\"passed\": false, \"error\": \"STEP 3/4 width sweep failed or hung\", \"step3_rc\": ${RC3}, \"width\": $(cat "$ART/width_last.json" | head -c 20000 | "$PY" -c 'import sys,json
try: print(json.dumps(json.loads(sys.stdin.read())))
except Exception: print("null")')}"
  exit 3
fi

echo "=== STEP 5 $(date -Is) trace_graph.py --with-check (after $(( $(date +%s) - T0 ))s)"
"$PY" "$HERE/trace_graph.py" "$CKPT" --with-check | tee "$ART/trace_graph.log"
RC5=${PIPESTATUS[0]}
tail -n 1 "$ART/trace_graph.log" > "$ART/trace_last.json"
echo "LP-STEP5 rc=${RC5} total=$(( $(date +%s) - T0 ))s"
"$PY" - "$ART" <<'EOF'
import json, os, sys
art = sys.argv[1]
def load(n):
    try:
        return json.load(open(os.path.join(art, n)))
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "unparsed": repr(e)[:200]}
lc, wd, tr = load("lp_check_last.json"), load("width_last.json"), load("trace_last.json")
tr_small = {k: tr.get(k) for k in ("passed", "decode_lamport_only", "routing_ok", "tail_graph_engaged",
                                   "captured_pairs", "capture_failed_lines", "rank0_prefill", "rank0_decode",
                                   "check", "watchdog_fired", "trace_error")}
tr_small["per_rank_census"] = {r: {k: v.get(k) for k in ("prefill_lp_calls", "prefill_nccl_ar", "prefill_lp_us_median",
                                                          "prefill_lp_twoshot_grid", "decode")}
                               for r, v in (tr.get("per_rank_census") or {}).items()}
print(json.dumps({"passed": bool(lc.get("passed") and wd.get("passed") and tr.get("passed")),
                  "lp_check": lc, "width": wd, "trace": tr_small}))
EOF
