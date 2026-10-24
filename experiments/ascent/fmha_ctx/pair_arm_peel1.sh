#!/usr/bin/env bash
# Idea 35 (branch 30) kernel pair arm: FMHA_SM120_PEEL_SYNC_OPT in the PEEL_UNMASKED_KV_STEPS copy of the SM120 e4m3
# tiled context FMHA (fused_multihead_flash_attention_kernel_noloop_tiled.h): BMM1/BMM2 main chunk loops unrolled and
# the 2 redundant "done reading" barriers per step dropped (14 -> 12). Exact (bitwise) change.
#   bash pair_arm_peel1.sh <base|cand> <python> [procs (5)] [iters (100)]
# The baseline arm runs this same text inline (main has no copy). Both arms: copy the arm's own bench.cu, add the
# declared mixed-step cases K0/K1/K2 if the tree's copy lacks them (same lines as branch 30's bench.cu), build it
# against the arm's own cpp/kernels/fmha_v2/src with the tree's split dispatch constant (kMaxBatch), 2 discarded warm
# processes, then PROCS processes, each timing K1,K2,K0,c1,c2,c3 with 10 warmup + ITERS timed calls (--split -1 = the
# TRT-LLM dispatch; NaN-poisoned O and split workspace, 384 MiB L2 read-flush before every call, own event pair,
# output hash vs call 1, inputs checked unchanged). Then the BUILT TensorRT-LLM libraries: the base arm copies its
# tensorrt_llm/libs to /tmp/fmha_peel1_sass/baselibs; the candidate arm runs lib_sass_cmp.py compare against it.
# Last stdout line: the evidence JSON. kernel_ms = mean of the per-process medians; noise_floor = max of the
# per-process (p90-p10)/median and the spread (max-min)/mean of the per-process medians.
set -uo pipefail
MODE="$1"
PY="$2"
PROCS="${3:-5}"
ITERS="${4:-100}"
CASES=K1,K2,K0,c1,c2,c3
W=/tmp/fmha_peel1_arm_$MODE
rm -rf $W && mkdir -p $W
echo "ARM=$MODE TREE_COMMIT=$(git rev-parse HEAD 2>/dev/null) HEAD1=$(git rev-parse HEAD~1 2>/dev/null) DIRTY=$(git status --porcelain 2>/dev/null | wc -l) LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg) DATE=$(date +%T)"
echo "[arm] PEEL_SYNC_OPT mentions in kernel header: $(grep -c 'PEEL_SYNC_OPT' cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled.h || true)"
cp experiments/ascent/fmha_ctx/bench.cu $W/bench.cu
python3 - $W/bench.cu <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if '"K1"' not in s:
    add = '''    else if (name == "K0")
    {
        p.q_lens = {2048, 2048, 2048, 1984}; // b = 4 mixed-step context part
        p.past_lens = {P, P, P, P};
    }
    else if (name == "K1")
    {
        p.q_lens = {2048, 2048, 2048, 1920, 64}; // c64 mixed step (t = 8128)
        p.past_lens = {P, P, P, P, P};
    }
    else if (name == "K2")
    {
        p.q_lens = {2048, 2048, 2048, 1856, 64}; // c128 mixed step (t = 8064)
        p.past_lens = {P, P, P, P, P};
    }
'''
    anchor = '    else if (name == "c1_1984")\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, add + anchor)
    open(p, 'w').write(s)
    print('[arm] K0/K1/K2 cases added to the bench copy')
else:
    print('[arm] bench copy already has K0/K1/K2')
PY
CU=cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/last_tile_split_sm120/fmha_v2_e4m3_last_tile_split_sm120.cu
MB=$(grep -oE 'constexpr int kMaxBatch = [0-9]+;' $CU | grep -oE '[0-9]+' || echo 2)
echo "[arm] kMaxBatch $MB"
if nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120f; then ARCH='-gencode arch=compute_120f,code=sm_120f'; else ARCH='-gencode arch=compute_120a,code=sm_120a'; fi
nvcc -O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 $ARCH \
  -Xptxas -v -I$PWD/cpp/kernels/fmha_v2/src -DBENCH_MAX_BATCH=$MB $W/bench.cu -o $W/bench -lcuda > $W/build.log 2>&1 \
  || { grep -E 'error' $W/build.log | head -20; exit 2; }
grep -E 'Compiling entry|spill|Used [0-9]+ registers' $W/build.log | sed 's/ptxas info    ://' | cut -c1-220 || true
q() { nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null || true; }
q
for i in 1 2; do $W/bench --cases K1,c1 --warmup 10 --iters 30 --split -1 > /dev/null; echo "[arm] warm-up $i done"; done
for i in $(seq 1 $PROCS); do
  echo "[arm] process $i $(q)"
  $W/bench --cases $CASES --warmup 10 --iters $ITERS --split -1 > $W/p$i.log
  grep -E 'RESULT|SPLIT|KINFO' $W/p$i.log
done
q
echo "LOADAVG_END=$(cut -d' ' -f1-3 /proc/loadavg) DATE=$(date +%T)"
LIBS=$("$PY" -c "import importlib.util,os;print(os.path.join(list(importlib.util.find_spec('tensorrt_llm').submodule_search_locations)[0],'libs'))")
echo "LIBS=$LIBS"
: > $W/sass.json
if [ "$MODE" = base ]; then
  rm -rf /tmp/fmha_peel1_sass/baselibs; mkdir -p /tmp/fmha_peel1_sass
  cp -rL "$LIBS" /tmp/fmha_peel1_sass/baselibs && echo "[sass] base libs saved" && echo '{"saved": true}' > $W/sass.json
else
  if [ -d /tmp/fmha_peel1_sass/baselibs ]; then
    "$PY" experiments/ascent/fmha_ctx/lib_sass_cmp.py compare /tmp/fmha_peel1_sass/baselibs | tee $W/sass.log
    tail -n 1 $W/sass.log > $W/sass.json
  else
    echo '{"error": "no base libs in /tmp/fmha_peel1_sass/baselibs"}' > $W/sass.json
  fi
fi
python3 - $W $PROCS $W/sass.json <<'PY'
import json, sys
w, procs, sass_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
bmap = {'K1': '5', 'K2': '5', 'K0': '4', 'c1': '1', 'c2': '2', 'c3': '3'}
runs, kinfo, split = [], None, {}
for i in range(1, procs + 1):
    res = {}
    for line in open(f'{w}/p{i}.log'):
        t = line.split()
        if t and t[0] == 'KINFO':
            kinfo = {'regs': int(t[1]), 'local_bytes': int(t[2]), 'dyn_smem': int(t[3]), 'occupancy_ctas_per_sm': int(t[4])}
        if t and t[0] == 'SPLIT':
            split[t[1]] = int(t[3])
        if t and t[0] == 'RESULT':
            res[t[1]] = {'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]), 'noise': float(t[8]),
                         'repeat_mismatches': int(t[9]), 'calls': int(t[10]), 'nonfinite': int(t[11]),
                         'inputs_unchanged': bool(int(t[12])), 'out_hash': t[13]}
    runs.append(res)
cases = {}
ok = True
for c in bmap:
    meds = [r[c]['median_ms'] for r in runs]
    mean = sum(meds) / len(meds)
    spread = (max(meds) - min(meds)) / mean
    within = max(r[c]['noise'] for r in runs)
    hashes = sorted({r[c]['out_hash'] for r in runs})
    clean = all(r[c]['repeat_mismatches'] == 0 and r[c]['nonfinite'] == 0 and r[c]['inputs_unchanged'] for r in runs)
    ok = ok and clean and len(hashes) == 1
    cases[c] = {'b': int(bmap[c]), 'kernel_ms': mean, 'proc_medians_ms': meds, 'proc_median_spread': spread,
                'within_proc_p90_p10_max': within, 'noise_floor': max(spread, within), 'out_hashes': hashes,
                'repeat_ok': clean, 'split_pieces': split.get(c, 0)}
    print(c, json.dumps(cases[c]), flush=True)
# The evidence batch keys: one entry per batch size; b = 5 is K1 (c64 mixed step), K2 is reported in cases.
batch = {}
for c in ('K1', 'K0', 'c1', 'c2', 'c3'):
    e = cases[c]
    batch[str(e['b'])] = {'kernel_ms': e['kernel_ms'], 'noise_floor': e['noise_floor'], 'case': c,
                          'proc_median_spread': e['proc_median_spread'], 'within_proc_p90_p10_max': e['within_proc_p90_p10_max']}
try:
    sass = json.loads(open(sass_path).read() or '{}')
except Exception as ex:
    sass = {'error': repr(ex)}
print(json.dumps({'batch': batch, 'output': {'class': 'exact'}, 'kernel': kinfo, 'repeat_ok': ok, 'lib_sass': sass,
                  'cases': cases}))
PY
