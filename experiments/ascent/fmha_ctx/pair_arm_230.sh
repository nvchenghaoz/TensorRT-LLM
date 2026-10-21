#!/usr/bin/env bash
# One arm of the idea-230 kernel pair (SM120 e4m3 tiled context FMHA + last-q-tile split, --split -1, kMaxBatch 3).
# The baseline arm runs this same text inline (main has no copy of this file). Extra nvcc flags come from $ARM_NVCC
# (the tree default is FMHA_SM120_EXP2=1 since idea 244; -DFMHA_SM120_EXP2=0 is the kill switch). Builds bench.cu against the arm's own
# cpp/kernels/fmha_v2/src, runs 2 discarded warm-up processes, then c1,c2,c3,c4 with 10 warmup + 100 timed calls
# (each: NaN-poisoned O and split workspace, 384 MiB L2 flush, own event pair; output hash vs call 1).
# Last stdout line: the evidence JSON (class from $ARM_CLASS, default exact).
set -euo pipefail
W=/tmp/fmha230_arm
mkdir -p $W
echo "TREE=$PWD COMMIT=$(git rev-parse HEAD 2>/dev/null || echo none) ARM_NVCC='${ARM_NVCC:-}'"
bash experiments/ascent/fmha_ctx/build.sh cpp/kernels/fmha_v2/src $W/bench -DBENCH_MAX_BATCH=3 ${ARM_NVCC:-}
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader || true
for i in 1 2; do $W/bench --cases c1,c3 --warmup 10 --iters 40 --split -1 > /dev/null; echo "[arm] warm-up $i done"; done
$W/bench --cases c1,c2,c3,c4 --warmup 10 --iters 100 --split -1 | tee $W/log
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader || true
ARM_CLASS="${ARM_CLASS:-exact}" python3 - $W/log <<'PY'
import json, os, sys
res = {}
for line in open(sys.argv[1]):
    t = line.split()
    if t and t[0] == 'RESULT':
        res[t[1]] = {'b': int(t[2]), 'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]),
                     'noise': float(t[8]), 'repeat_mismatches': int(t[9]), 'calls': int(t[10]),
                     'nonfinite': int(t[11]), 'inputs_unchanged': bool(int(t[12])), 'out_hash': t[13]}
batch = {}
for c, b in (('c1', '1'), ('c2', '2'), ('c3', '3'), ('c4', '4')):
    r = res[c]
    batch[b] = {'kernel_ms': r['median_ms'], 'noise_floor': r['noise'], 'p10_ms': r['p10_ms'], 'p90_ms': r['p90_ms'],
                'out_hash': r['out_hash']}
ok = all(r['repeat_mismatches'] == 0 and r['nonfinite'] == 0 and r['inputs_unchanged'] for r in res.values())
print(json.dumps({'batch': batch, 'output': {'class': os.environ['ARM_CLASS']}, 'repeat_ok': ok, 'cases': res}))
PY
