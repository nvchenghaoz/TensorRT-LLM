#!/bin/bash
# Idea 345 / branch 192: node receipt + wait-only DIRTY SLOT guard (method 3088), then exec "$@".
# All receipts go to stderr; the wrapped command's stdout (last line = JSON) is untouched.
# usage: bash guard.sh <cmd> [args...]
{
  echo "=== GUARD $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} load=$(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu,clocks.sm --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} 1>&2
T0=$(date +%s)
while true; do
  DIRTY=""
  VIS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  for g in ${VIS//,/ }; do
    mu=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$mu" ] && [ "$mu" -gt 4096 ]; then DIRTY="$DIRTY gpu$g=${mu}MiB"; fi
  done
  if [ -z "$DIRTY" ]; then echo "[guard] slot clean after $(( $(date +%s) - T0 ))s" 1>&2; break; fi
  if [ $(( $(date +%s) - T0 )) -ge 180 ]; then
    {
      echo "DIRTY SLOT:$DIRTY"
      nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
    } 1>&2
    echo "{\"passed\": false, \"error\": \"DIRTY SLOT\", \"dirty\": \"${DIRTY}\"}"
    exit 3
  fi
  sleep 5
done
exec "$@"
