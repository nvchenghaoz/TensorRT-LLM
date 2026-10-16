#!/bin/bash
# Driver: FP8-KV arm and BF16-KV arm, each under nsys with only the
# benchmark-shape prefix-reuse request captured (2048 new tokens over 32768
# cached + 256 greedy decode tokens); attention-kernel and cudaGraphLaunch
# summaries per arm; then BF16 vs FP8 output comparison + determinism verdict.
# Usage: fp8kv_run.sh <python> <ckpt> [llm_kwargs.json]
set -u
PY=$1
CKPT=$2
KW=${3:-}
OUT=${ASCENT_OUT:-/tmp/fp8kv}
mkdir -p "$OUT"
KWARG=()
if [ -n "$KW" ]; then KWARG=(--llm-kwargs "$KW"); fi
echo "nsys: $(which nsys 2>/dev/null) $(nsys --version 2>/dev/null)"

summ() {  # $1 = arm name
  nsys stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o "$OUT/$1" "$OUT/$1.nsys-rep" >/dev/null 2>&1
  "$PY" - "$OUT/${1}_cuda_gpu_kern_sum.csv" "$OUT/${1}_cuda_api_sum.csv" "$1" <<'EOF'
import csv, sys
arm = sys.argv[3]
for r in csv.DictReader(open(sys.argv[1])):
    n = r.get("Name", "")
    if any(s in n.lower() for s in ("fmha", "mha", "xqa", "attention", "attn", "unfused", "softmax", "bmm")):
        print(f"[{arm}] KERNEL {float(r['Total Time (ns)'])/1e6:10.2f} ms  n={r['Instances']:>6}  avg={float(r['Avg (ns)'])/1e3:9.1f} us  med={float(r['Med (ns)'])/1e3:9.1f} us  {n[:160]}")
for r in csv.DictReader(open(sys.argv[2])):
    n = r.get("Name", "")
    if "Graph" in n:
        print(f"[{arm}] API {n:30s} n={r['Num Calls']}")
EOF
}

for ARM in fp8 auto; do
  echo "=== KV dtype $ARM (nsys) ==="
  nsys profile -o "$OUT/$ARM" -f true -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    "$PY" experiments/ascent/fp8kv_diag.py --ckpt "$CKPT" --kv-dtype "$ARM" \
    --profile --out "$OUT/$ARM.json" "${KWARG[@]}" 2>&1 \
    | grep -E '^\[[0-9]{2}:|^\{|Error|error|Traceback|fallback|Capture' | tail -n 40
  echo "=== kernel summary ($ARM) ==="
  summ "$ARM"
done

echo "=== compare ==="
"$PY" experiments/ascent/fp8kv_compare.py "$OUT/auto.json" "$OUT/fp8.json"
