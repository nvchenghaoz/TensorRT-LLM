#!/usr/bin/env bash
# One arm of the branch-31 kernel pair (idea 76: last q-tile KV split of the sm_120 e4m3 tiled context FMHA).
# The same sequence runs in both arms (the baseline arm gets this text inline, since the base tree does not have
# this file): build experiments/ascent/fmha_ctx/bench.cu against the arm's own cpp/kernels/fmha_v2/src, warm the
# GPU up (40 untimed c1 calls), then time c1/c2/c4 with 10 warmup + 100 timed calls, each preceded by a NaN-poison
# of O (and of the split workspace) and a 384 MiB L2 read-flush, each call's output hash compared with call 1.
# --split -1 is the TRT-LLM dispatch (split for b <= 2 when it saves a wave); a tree without the split header (main)
# ignores it and runs the generated kernel alone. Last stdout line: the evidence JSON.
set -euo pipefail
W=/tmp/fhs_arm
mkdir -p $W
if nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120f; then ARCH='-gencode arch=compute_120f,code=sm_120f'; else ARCH='-gencode arch=compute_120a,code=sm_120a'; fi
nvcc -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 $ARCH -Xptxas -v -I$PWD/cpp/kernels/fmha_v2/src experiments/ascent/fmha_ctx/bench.cu -o $W/bench -lcuda > $W/build.log 2>&1 || { grep -E 'error' $W/build.log | head -20; exit 1; }
grep -E 'Compiling entry|spill|Used [0-9]+ registers' $W/build.log | sed 's/ptxas info    ://' | cut -c1-200 || true
echo "[arm] built in $PWD"
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true
$W/bench --cases c1 --warmup 40 --iters 1 --split -1 > /dev/null
echo "[arm] warmed up"
$W/bench --cases c1,c2,c4 --warmup 10 --iters 100 --split -1 | tee $W/log
python3 - $W/log <<'PY'
import json, sys
res, split, kinfo = {}, {}, None
for line in open(sys.argv[1]):
    t = line.split()
    if t and t[0] == 'KINFO':
        kinfo = {'regs': int(t[1]), 'local_bytes': int(t[2]), 'dyn_smem': int(t[3]), 'occupancy_ctas_per_sm': int(t[4]), 'store_o_from_regs': int(t[5])}
    if t and t[0] == 'SPLIT':
        split[t[1]] = {'pieces': int(t[3]), 'q_loop_last': int(t[5]), 'ctas_a': int(t[7]), 'ctas_b': int(t[9])}
    if t and t[0] == 'RESULT':
        res[t[1]] = {'b': int(t[2]), 'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]), 'min_ms': float(t[7]), 'noise': float(t[8]), 'repeat_mismatches': int(t[9]), 'calls': int(t[10]), 'nonfinite': int(t[11]), 'inputs_unchanged': bool(int(t[12])), 'out_hash': t[13]}
batch = {}
for c, b in (('c1', '1'), ('c2', '2'), ('c4', '4')):
    r = res[c]
    batch[b] = {'kernel_ms': r['median_ms'], 'noise_floor': r['noise'], 'p10_ms': r['p10_ms'], 'p90_ms': r['p90_ms'], 'out_hash': r['out_hash'], 'split_pieces': split.get(c, {}).get('pieces', 0)}
ok = all(r['repeat_mismatches'] == 0 and r['nonfinite'] == 0 and r['inputs_unchanged'] for r in res.values())
cls = 'reordered' if any(v.get('pieces', 0) > 0 for v in split.values()) else 'exact'
print(json.dumps({'batch': batch, 'output': {'class': cls}, 'kernel': kinfo, 'split': split, 'repeat_ok': ok, 'cases': res}))
PY
