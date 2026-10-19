#!/usr/bin/env bash
# Idea 165 (branch 99): kernel A/B of the sm_120 e4m3 last-q-tile KV split at kMaxBatch 2 (main) vs 3 (candidate),
# with arms rotated per round inside one process and an A/A arm (a second main arm), in two processes with opposite
# arm lists. The kernel code (cpp/kernels/fmha_v2/src) is the same in main and this branch (checked below against
# AB_BASE); only the TRT-LLM dispatch constant differs, which --arms reproduces per arm: name:max_batch, --split -1.
# Every call: NaN-poison of O (+ split workspace) and a 384 MiB L2 read-flush first, own CUDA event pair, output
# hash compared with the arm's call 1; Q / KV pool hashes checked unchanged per case.
# env: AB_BASE (main sha), AB_ITERS (300), AB_WARMUP (10), AB_CASES (c3,c1,c2,c4). Last stdout line: evidence JSON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${AB_BASE:-66c3ff64b97af2aafa395ed53e3d74ca40446262}"
ITERS="${AB_ITERS:-300}"
WARMUP="${AB_WARMUP:-10}"
CASES="${AB_CASES:-c3,c1,c2,c4}"
W=/tmp/fmha_ab_rotated
mkdir -p $W
echo "[ab] HEAD $(git rev-parse HEAD 2>/dev/null || echo ?) base $BASE"
if git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  if git diff --quiet "$BASE" -- cpp/kernels/fmha_v2/src; then SRC_SAME=1; else SRC_SAME=0; fi
else
  SRC_SAME=-1
fi
echo "[ab] fmha_v2/src identical to base: $SRC_SAME"
CU=cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/last_tile_split_sm120/fmha_v2_e4m3_last_tile_split_sm120.cu
MB_TREE=$(grep -oE 'constexpr int kMaxBatch = [0-9]+;' $CU | grep -oE '[0-9]+')
MB_BASE=$(git show "$BASE:$CU" 2>/dev/null | grep -oE 'constexpr int kMaxBatch = [0-9]+;' | grep -oE '[0-9]+' || echo 2)
echo "[ab] kMaxBatch tree $MB_TREE base $MB_BASE"
bash "$HERE/build.sh" "$PWD/cpp/kernels/fmha_v2/src" $W/bench -DBENCH_MAX_BATCH=$MB_TREE
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true
$W/bench --cases c1 --warmup 40 --iters 1 --split -1 > /dev/null
echo "[ab] warmed up"
$W/bench --cases "$CASES" --warmup $WARMUP --iters $ITERS --split -1 --arms "main:$MB_BASE,cand:$MB_TREE,main_aa:$MB_BASE" | tee $W/p1.log
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true
$W/bench --cases "$CASES" --warmup $WARMUP --iters $ITERS --split -1 --arms "main_aa:$MB_BASE,cand:$MB_TREE,main:$MB_BASE" | tee $W/p2.log
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true
python3 - $W/p1.log $W/p2.log "$SRC_SAME" "$MB_BASE" "$MB_TREE" <<'PY'
import json, sys
logs, src_same, mb_base, mb_tree = sys.argv[1:3], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
procs = []
for path in logs:
    res, split = {}, {}
    for line in open(path):
        t = line.split()
        if t and t[0] == 'SPLIT' and len(t) >= 11:
            split[(t[1], t[10])] = {'pieces': int(t[3]), 'ctas_a': int(t[7]), 'ctas_b': int(t[9])}
        if t and t[0] == 'RESULT' and len(t) >= 15:
            res[(t[1], t[14])] = {'b': int(t[2]), 'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]), 'noise': float(t[8]), 'repeat_mismatches': int(t[9]), 'calls': int(t[10]), 'nonfinite': int(t[11]), 'inputs_unchanged': bool(int(t[12])), 'out_hash': t[13]}
    procs.append((res, split))
cases = sorted({c for (c, a) in procs[0][0]}, key=lambda c: int(c[1:]))
out = {'cases': {}, 'src_same_as_base': src_same, 'kmaxbatch_base': mb_base, 'kmaxbatch_cand': mb_tree}
batch, ok = {}, src_same == 1
for c in cases:
    e = {'procs': []}
    meds = {'main': [], 'cand': [], 'main_aa': []}
    for res, split in procs:
        pe = {a: dict(res[(c, a)], pieces=split.get((c, a), {}).get('pieces', 0)) for a in ('main', 'cand', 'main_aa')}
        for a in meds:
            meds[a].append(pe[a]['median_ms'])
        pe['cand_vs_main'] = pe['cand']['median_ms'] / pe['main']['median_ms'] - 1
        pe['aa_vs_main'] = pe['main_aa']['median_ms'] / pe['main']['median_ms'] - 1
        pe['cand_p90_below_main_p10'] = pe['cand']['p90_ms'] < min(pe['main']['p10_ms'], pe['main_aa']['p10_ms'])
        e['procs'].append(pe)
        ok = ok and all(pe[a]['repeat_mismatches'] == 0 and pe[a]['nonfinite'] == 0 and pe[a]['inputs_unchanged'] for a in meds)
        ok = ok and pe['main']['out_hash'] == pe['main_aa']['out_hash']
    m = {a: sum(v) / len(v) for a, v in meds.items()}
    e['mean_median_ms'] = m
    e['cand_vs_main'] = m['cand'] / m['main'] - 1
    e['aa_floor'] = abs(m['main_aa'] / m['main'] - 1)
    e['aa_floor_max_proc'] = max(abs(p['aa_vs_main']) for p in e['procs'])
    e['arm_spread_max'] = max(p[a]['noise'] for p in e['procs'] for a in ('main', 'cand', 'main_aa'))
    e['same_dispatch'] = all(p['main']['pieces'] == p['cand']['pieces'] for p in e['procs'])
    e['out_hash_cand_equals_main'] = all(p['main']['out_hash'] == p['cand']['out_hash'] for p in e['procs'])
    out['cases'][c] = e
    b = str(procs[0][0][(c, 'main')]['b'])
    batch[b] = {'kernel_ms': m['cand'], 'main_kernel_ms': m['main'], 'main_aa_kernel_ms': m['main_aa'],
                'noise_floor': max(e['arm_spread_max'], e['aa_floor_max_proc']), 'cand_vs_main': e['cand_vs_main'],
                'aa_floor': e['aa_floor'], 'pieces_main': e['procs'][0]['main']['pieces'],
                'pieces_cand': e['procs'][0]['cand']['pieces']}
c3 = out['cases'].get('c3')
if c3:
    gain = -c3['cand_vs_main']
    out['gate_c3'] = {'gain': gain, 'ge_2p5pct': gain >= 0.025, 'beyond_aa_floor': gain > c3['aa_floor_max_proc'],
                      'beyond_arm_spread': gain > c3['arm_spread_max'],
                      'cand_p90_below_main_p10_all_procs': all(p['cand_p90_below_main_p10'] for p in c3['procs'])}
unchanged = {c: out['cases'][c]['same_dispatch'] and out['cases'][c]['out_hash_cand_equals_main'] for c in out['cases'] if c != 'c3'}
out['c1_c2_c4_same_dispatch_and_bitwise'] = unchanged
out['repeat_ok'] = ok
print(json.dumps({'batch': batch, 'output': {'class': 'reordered'}, 'ab': out}))
PY
