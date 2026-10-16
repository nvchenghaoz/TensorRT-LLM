#!/usr/bin/env bash
# One arm of the run_gpu_pair kernel comparison (the pair passes bench.cu inline to both arms, since the base
# tree does not have this directory; this script is the same sequence for reference / reruns).
#   pair_arm.sh [bench.cu]
# Builds bench.cu against ./cpp/kernels/fmha_v2/src (the arm's own tree), times c1/c2/c4 (+ c1_1984 tail probe)
# with 10 warmup + 100 timed calls (NaN-poison + 384 MiB L2 read-flush before every call), prints evidence JSON.
set -euo pipefail
BENCH="${1:-experiments/ascent/fmha_ctx/bench.cu}"
W=/tmp/fhp
mkdir -p $W
if nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120f; then ARCH='-gencode arch=compute_120f,code=sm_120f'; else ARCH='-gencode arch=compute_120a,code=sm_120a'; fi
nvcc -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 $ARCH -Xptxas -v -I$PWD/cpp/kernels/fmha_v2/src $BENCH -o $W/bench -lcuda 2>&1 | grep -E 'error|Function properties for fmha|spill stores|Used 2[0-9][0-9] registers' || true
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true
$W/bench --cases c1,c2,c4,c1_1984 --warmup 10 --iters 100 | tee $W/log
python3 - $W/log <<'PY'
import json, sys
res, kinfo = {}, None
for line in open(sys.argv[1]):
    t = line.split()
    if t and t[0] == 'KINFO':
        kinfo = {'regs': int(t[1]), 'local_bytes': int(t[2]), 'dyn_smem': int(t[3]), 'occupancy_ctas_per_sm': int(t[4]), 'store_o_from_regs': int(t[5])}
    if t and t[0] == 'RESULT':
        res[t[1]] = {'b': int(t[2]), 'ctas': int(t[3]), 'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]), 'min_ms': float(t[7]), 'noise': float(t[8]), 'repeat_mismatches': int(t[9]), 'calls': int(t[10]), 'nonfinite': int(t[11]), 'inputs_unchanged': bool(int(t[12])), 'out_hash': t[13]}
batch = {}
for c, b in (('c1', '1'), ('c2', '2'), ('c4', '4')):
    r = res[c]
    batch[b] = {'kernel_ms': r['median_ms'], 'noise_floor': r['noise'], 'p10_ms': r['p10_ms'], 'p90_ms': r['p90_ms'], 'ctas': r['ctas'], 'out_hash': r['out_hash']}
ok = all(r['repeat_mismatches'] == 0 and r['nonfinite'] == 0 and r['inputs_unchanged'] for r in res.values())
out = {'batch': batch, 'output': {'class': 'exact'}, 'kernel': kinfo, 'repeat_ok': ok, 'tail_probe': {'c1_1984_ms': res['c1_1984']['median_ms'], 'c1_1984_p90_ms': res['c1_1984']['p90_ms'], 't2048_minus_t1984_ms': res['c1']['median_ms'] - res['c1_1984']['median_ms']}, 'cases': res}
print(json.dumps(out))
PY
