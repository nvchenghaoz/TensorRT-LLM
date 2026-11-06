#!/bin/bash
# Branch 158 / idea 293: ONE 4-GPU job, steps in order (area lead 13:0x PST 11-05).
#   STEP 0  read-only node receipt to stderr: hostname, topo -m / -p2p r / -p2p w, /proc/cmdline, each GPU's
#           iommu group + type, GPU idle util; then the 12-pair peer-access assert.
#   STEP 1  kill-bounded (timeout -s KILL 300), no model: lp_check.py = LOWPRECISION two-shot (peer READ) on
#           [2048,5120], 220 calls each checked vs the fp32 sum. Any timeout/hang/WRONG -> exit 3, no model trace.
#   STEP 2  trace_graph.py <checkpoint> --with-check: model trace (watchdog) + bench.py sweep (1000-call robust run).
# usage: bash run_lp_trace.sh <python> <checkpoint>
PY="$1"; CKPT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
T0=$(date +%s)
{
  echo "=== STEP 0 $(date -Is)"
  hostname
  nvidia-smi topo -m
  nvidia-smi topo -p2p r
  nvidia-smi topo -p2p w
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "--- /proc/cmdline"; cat /proc/cmdline
  echo "--- iommu groups"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader | while IFS=', ' read -r idx uuid bus; do
    bdf=$(echo "${bus: -12}" | tr 'A-Z' 'a-z')
    g=$(readlink "/sys/bus/pci/devices/${bdf}/iommu_group" 2>&1)
    t=$(cat "/sys/bus/pci/devices/${bdf}/iommu_group/type" 2>&1)
    echo "gpu${idx} ${uuid} ${bus} sysfs=${bdf} iommu_group=${g##*/} type=${t}"
  done
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,power.draw,clocks.sm --format=csv
} 1>&2

"$PY" -c '
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", file=sys.stderr, flush=True)
sys.exit(0 if n == 4 and not bad else 3)
' || { echo '{"passed": false, "error": "STEP 0 peer-access assert failed"}'; exit 3; }

echo "=== STEP 1 $(date -Is) lp_check (timeout -s KILL 300)" 1>&2
MARK="lpcheck_$$_${RANDOM}"
T1=$(date +%s)
LP_CHECK_MARK="$MARK" timeout -s KILL 300 "$PY" "$HERE/lp_check.py"
RC=$?
W1=$(( $(date +%s) - T1 ))
# Kill only processes started by STEP 1 (they carry LP_CHECK_MARK in their environment), never other jobs'.
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  if tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -qx "LP_CHECK_MARK=${MARK}"; then
    echo "[cleanup] kill -9 ${pid} $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-160)" 1>&2
    kill -9 "$pid" 2>/dev/null
  fi
done
sleep 2
echo "LP-STEP1 rc=${RC} wall=${W1}s" 1>&2
if [ "$RC" -ne 0 ]; then
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,power.draw --format=csv 1>&2
  echo "{\"passed\": false, \"error\": \"STEP 1 lp_check failed or hung\", \"step1_rc\": ${RC}, \"step1_wall_s\": ${W1}}"
  exit 3
fi

echo "=== STEP 2 $(date -Is) trace_graph.py --with-check (after $(( $(date +%s) - T0 ))s)" 1>&2
exec "$PY" "$HERE/trace_graph.py" "$CKPT" --with-check
