#!/bin/bash
# Idea 360 / branch 215: ONE 4-GPU job driving experiments/ascent/lp_push/dev.py <mode>.
#   preamble (plan 3347 (9)) to stderr and stdout, HEAD, lib receipt (push kernel present in the loaded .so),
#   12-pair peer assert, wait-only DIRTY SLOT guard (lp_ar_trace/guard.sh, method 3088), then dev.py under
#   `timeout -s KILL` with a 60 s heartbeat. Last stdout line: dev.py's JSON.
# usage: bash run.sh <python> <dev|pair>
PY="$1"
MODE="${2:-dev}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="${AUTO_DEV_ARTIFACTS:-/tmp/lp_push_$$}"
mkdir -p "$ART"
export AUTO_DEV_ARTIFACTS="$ART"
export PYTHONUNBUFFERED=1
unset TRTLLM_ENABLE_PDL TRTLLM_LP_AR_PUSH TRTLLM_LP_AR_TWOSHOT_CTAS
T0=$(date +%s)
echo "=== PREAMBLE mode=${MODE} $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
{
  hostname; nvidia-smi topo -m; nvidia-smi topo -p2p r; nvidia-smi topo -p2p w; cat /proc/cmdline
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} 1>&2
if [ "$MODE" = "dev" ]; then
  hostname; nvidia-smi topo -m; cat /proc/cmdline
  grep -q "iommu=pt" /proc/cmdline && echo "[preamble] /proc/cmdline has iommu=pt" || echo "[preamble] /proc/cmdline LACKS iommu=pt"
fi
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used,utilization.gpu --format=csv
cat /proc/loadavg
echo "[preamble] HEAD $(git -C "$HERE" rev-parse HEAD 2>/dev/null) tree $(git -C "$HERE" rev-parse 'HEAD^{tree}' 2>/dev/null) dirty=$(git -C "$HERE" status --porcelain 2>/dev/null | wc -l) (detached snapshot)"
git -C "$HERE" log --oneline -3 2>/dev/null
env | grep -E '^(NCCL_|TRTLLM_ENABLE_PDL|TRTLLM_LP_AR|TLLM_AR|TRTLLM_DSV4)' || echo "[preamble] no NCCL_/TRTLLM_ENABLE_PDL/TRTLLM_LP_AR/TLLM_AR/TRTLLM_DSV4 env"

echo "[preamble] host=$(hostname) slot GPUs=${CUDA_VISIBLE_DEVICES}"
nvidia-smi topo -p2p r
nvidia-smi --query-gpu=index,uuid,clocks.sm,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
# Kernel/library receipt: dev.py records it IN-PROCESS (pitfall 3620: a grep of tensorrt_llm/libs/libtensorrt_llm.so
# found 0 even for strings certainly in the base .cu). pair mode: CUPTI kernel names per rank at M=2048 plus a
# /proc/self/maps scan of every mapped tensorrt_llm .so; dev mode: CUPTI dispatch table.

"$PY" -c '
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", flush=True)
sys.exit(0 if n == 4 and not bad else 3)
' || { echo '{"passed": false, "error": "peer-access assert failed"}'; exit 3; }

bash "$HERE/../lp_ar_trace/guard.sh" true
RCG=$?
if [ "$RCG" -ne 0 ]; then echo '{"passed": false, "error": "DIRTY SLOT (guard.sh exit 3); receipts on stderr"}'; exit 3; fi

echo "=== RUN dev.py ${MODE} $(date -Is) t+$(( $(date +%s) - T0 ))s"
timeout -s KILL 1500 "$PY" -u "$HERE/dev.py" "$MODE" > "$ART/${MODE}.log" 2> "$ART/${MODE}.err" &
PID=$!
LAST=0
while kill -0 $PID 2>/dev/null; do
  sleep 10
  N=$(wc -l < "$ART/${MODE}.log")
  if [ "$N" -gt "$LAST" ]; then sed -n "$((LAST + 1)),${N}p" "$ART/${MODE}.log" | grep -v '^{' | cut -c1-600; LAST=$N; fi
  if [ $(( ($(date +%s) - T0) % 60 )) -lt 10 ]; then echo "hb $(date +%T) t+$(( $(date +%s) - T0 ))s"; fi
done
wait $PID
RC=$?
N=$(wc -l < "$ART/${MODE}.log")
if [ "$N" -gt "$LAST" ]; then sed -n "$((LAST + 1)),${N}p" "$ART/${MODE}.log" | grep -v '^{' | cut -c1-600; fi
grep -vE 'Warning|warn' "$ART/${MODE}.err" | tail -n 25
echo "=== DONE rc=${RC} t+$(( $(date +%s) - T0 ))s $(date -Is)"
if [ "$RC" -eq 137 ]; then echo '{"passed": false, "error": "HANG: dev.py killed after 1500 s (tell the area lead)"}'; exit 3; fi
tail -n 1 "$ART/${MODE}.log"
exit $RC
