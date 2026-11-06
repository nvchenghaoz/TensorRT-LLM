#!/bin/bash
# Idea 244 / branch 201: ONE 4-GPU job driving experiments/ascent/pcie_latency_probe.py.
#   preamble (plan 3241 (9)) to stderr, HEAD, 12-pair peer assert, wait-only DIRTY SLOT guard (guard.sh, method 3088),
#   then build -> (i) lat -> (ii) floor -> (iv-a) bwread -> (iv-b/c) bwrest -> (iii) served, each under its own
#   `timeout -s KILL`. A failed or killed part is recorded and the next part still runs, except a peer HANG in (i)
#   or (iv-a) stops the job (tell the area lead). Last stdout line: combined compact JSON.
# usage: bash pcie_probe_driver.sh <python>
PY="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="${AUTO_DEV_ARTIFACTS:-/tmp/pcie_probe_$$}"
mkdir -p "$ART"
export AUTO_DEV_ARTIFACTS="$ART"
export PCIE_PROBE_EXT_DIR="/tmp/pcie_probe_ext_$$"
unset TRTLLM_ENABLE_PDL
T0=$(date +%s)
echo "=== PREAMBLE $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
{
  hostname; nvidia-smi topo -m; nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; cat /proc/cmdline
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} 1>&2
# the same receipt on stdout (kept in the job log)
hostname; nvidia-smi topo -m; nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; cat /proc/cmdline
grep -q "iommu=pt" /proc/cmdline && echo "[preamble] /proc/cmdline has iommu=pt" || echo "[preamble] /proc/cmdline LACKS iommu=pt"
echo "[preamble] visible GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) clocks by UUID:"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used,utilization.gpu --format=csv
cat /proc/loadavg
echo "[preamble] HEAD $(git -C "$HERE" rev-parse HEAD 2>/dev/null) (detached snapshot)"
git -C "$HERE" log --oneline -3 2>/dev/null
env | grep -E '^(NCCL_|TRTLLM_ENABLE_PDL|TLLM_AR|TRTLLM_DSV4)' || echo "[preamble] no NCCL_/TRTLLM_ENABLE_PDL/TLLM_AR/TRTLLM_DSV4 env"

"$PY" -c '
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", flush=True)
sys.exit(0 if n == 4 and not bad else 3)
' || { echo '{"passed": false, "error": "peer-access assert failed"}'; exit 3; }

bash "$HERE/lp_ar_trace/guard.sh" true
RCG=$?
if [ "$RCG" -ne 0 ]; then echo '{"passed": false, "error": "DIRTY SLOT (guard.sh exit 3); receipts on stderr"}'; exit 3; fi

run_part() {  # name timeout
  local name="$1" to="$2"
  echo "=== PART ${name} $(date -Is) (timeout -s KILL ${to}) t+$(( $(date +%s) - T0 ))s"
  timeout -s KILL "$to" "$PY" "$HERE/pcie_latency_probe.py" "$name" > "$ART/part_${name}.log" 2> "$ART/part_${name}.err"
  local rc=$?
  grep -v '^{' "$ART/part_${name}.log" | tail -n 80
  tail -n 30 "$ART/part_${name}.err" | grep -vE 'Warning|warn' | tail -n 15
  tail -n 1 "$ART/part_${name}.log" > "$ART/last_${name}.json"
  echo "PART ${name} rc=${rc} wall=$(( $(date +%s) - T0 ))s"
  return $rc
}

run_part build 900 || { echo '{"passed": false, "error": "build failed"}'; exit 3; }
export PCIE_PROBE_PREBUILT=1
run_part lat 300; RC_LAT=$?
if [ "$RC_LAT" -eq 137 ]; then
  echo '{"passed": false, "error": "HANG in (i) lat (KILL 300): tell the area lead"}'; exit 3
fi
run_part floor 300
run_part bwread 300; RC_BR=$?
if [ "$RC_BR" -eq 137 ]; then
  echo '{"passed": false, "error": "HANG in (iv-a) peer reads (KILL 300): tell the area lead"}'; exit 3
fi
run_part bwrest 300
run_part served 900
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --query-gpu=index,uuid,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv
echo "=== DONE t+$(( $(date +%s) - T0 ))s"
"$PY" "$HERE/pcie_probe_summary.py" "$ART"
