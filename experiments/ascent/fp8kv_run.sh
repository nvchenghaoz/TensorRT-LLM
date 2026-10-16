#!/bin/bash
# Driver: BF16-KV arm, FP8-KV arm under nsys (prefix-reuse request captured),
# kernel summary, and output comparison. Usage: fp8kv_run.sh <python> <ckpt> [llm_kwargs.json]
set -u
PY=$1
CKPT=$2
KW=${3:-}
OUT=${ASCENT_OUT:-/tmp/fp8kv}
mkdir -p "$OUT"
ART=${ARTIFACTS_DIR:-$OUT}
KWARG=()
if [ -n "$KW" ]; then KWARG=(--llm-kwargs "$KW"); fi
echo "nsys: $(which nsys 2>/dev/null) $(nsys --version 2>/dev/null)"

echo "=== FP8 KV arm (nsys) ==="
nsys profile -o "$OUT/fp8kv" -f true -t cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  "$PY" experiments/ascent/fp8kv_diag.py --ckpt "$CKPT" --kv-dtype fp8 \
  --profile --out "$OUT/fp8.json" "${KWARG[@]}" 2>&1 | grep -v 'Warning\|warn' | tail -n 60
echo "=== kernel summary (fp8) ==="
nsys stats -r cuda_gpu_kern_sum -f csv -o "$OUT/fp8kern" "$OUT/fp8kv.nsys-rep" >/dev/null 2>&1
"$PY" - "$OUT/fp8kern_cuda_gpu_kern_sum.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    n = r.get("Name", "")
    if any(s in n.lower() for s in ("fmha", "mha", "xqa", "attention", "attn", "flash")):
        print(f"{float(r['Total Time (ns)'])/1e6:10.2f} ms  n={r['Instances']:>6}  avg={float(r['Avg (ns)'])/1e3:9.1f} us  {n[:200]}")
EOF

echo "=== BF16 KV arm ==="
"$PY" experiments/ascent/fp8kv_diag.py --ckpt "$CKPT" --kv-dtype auto \
  --out "$OUT/bf16.json" "${KWARG[@]}" 2>&1 | grep -v 'Warning\|warn' | tail -n 30

cp "$OUT"/*.json "$ART"/ 2>/dev/null || true
cp "$OUT"/fp8kern_cuda_gpu_kern_sum.csv "$ART"/ 2>/dev/null || true
echo "=== compare ==="
"$PY" experiments/ascent/fp8kv_compare.py "$OUT/bf16.json" "$OUT/fp8.json"
