#!/usr/bin/env bash
# STEP 0' of idea 157 (branch 98): masked-KV-step peel of the SM120 e4m3 tiled context FMHA vs its base, palindrome.
#   peel_step0.sh <base_sha> [cases (c1,c2,c3,c4)] [iters (100)]
# Builds experiments/ascent/fmha_ctx/bench.cu (this tree's copy, which has the c3 case) against BASE's
# cpp/kernels/fmha_v2/src (git archive) and against this tree's. One discarded warm process first, then one process
# per position in the order base,peel,peel,base,base,peel,peel,base; each process times every case with 10 warmup +
# ITERS timed calls, --split -1 (the TRT-LLM dispatch), NaN-poison + 384 MiB L2 read-flush before every call, each
# call's output hash compared with call 1, inputs checked unchanged. Summary (last stdout line, JSON): per case the
# palindrome means (base positions 1,4,5,8 vs peel 2,3,6,7), the gain, the A/A floors in the same drift-cancelled
# form (base {1,8} vs {4,5}, peel {2,7} vs {3,6}), the out_hash of every run, and pass/fail of the >= 4% and
# >= 2x A/A gates for c1 and c3.
set -euo pipefail
BASE="${1:?usage: peel_step0.sh <base_sha> [cases] [iters]}"
CASES="${2:-c1,c2,c3,c4}"
ITERS="${3:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/tmp/fmha_peel_step0
rm -rf $W && mkdir -p $W/base
git archive "$BASE" cpp/kernels/fmha_v2/src | tar -x -C $W/base
bash $HERE/build.sh $W/base/cpp/kernels/fmha_v2/src $W/bench_base > $W/build_base.log 2>&1 &
pb=$!
bash $HERE/build.sh $PWD/cpp/kernels/fmha_v2/src $W/bench_peel > $W/build_peel.log 2>&1 &
pp=$!
wait $pb
wait $pp
echo "===== build base ($BASE)"; cat $W/build_base.log
echo "===== build peel ($PWD)"; cat $W/build_peel.log
q() { nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -i 0 || true; }
q
echo "[step0] discarded warm process"
$W/bench_base --cases "$CASES" --warmup 20 --iters 20 --split -1 > $W/warm.log
grep RESULT $W/warm.log || true
pos=0
for v in base peel peel base base peel peel base; do
  pos=$((pos + 1))
  echo "[step0] position $pos: $v $(q)"
  $W/bench_$v --cases "$CASES" --warmup 10 --iters "$ITERS" --split -1 > $W/pos${pos}_$v.log
  grep -E 'RESULT|SPLIT|KINFO' $W/pos${pos}_$v.log
done
q
python3 - $W "$CASES" <<'PY'
import glob, json, os, sys
w, cases = sys.argv[1], sys.argv[2].split(',')
runs = {}
for f in sorted(glob.glob(f'{w}/pos*_*.log')):
    pos, var = os.path.basename(f)[3:-4].split('_')
    pos = int(pos)
    res, kinfo = {}, None
    for line in open(f):
        t = line.split()
        if t and t[0] == 'KINFO':
            kinfo = {'regs': int(t[1]), 'local_bytes': int(t[2]), 'dyn_smem': int(t[3]), 'occ': int(t[4])}
        if t and t[0] == 'RESULT':
            res[t[1]] = {'median_ms': float(t[4]), 'p10_ms': float(t[5]), 'p90_ms': float(t[6]), 'noise': float(t[8]),
                         'mismatches': int(t[9]), 'nonfinite': int(t[11]), 'inputs_ok': bool(int(t[12])),
                         'out_hash': t[13]}
    runs[pos] = {'variant': var, 'kinfo': kinfo, 'res': res}
mean = lambda xs: sum(xs) / len(xs)
out = {'positions': {p: runs[p]['variant'] for p in sorted(runs)}, 'kinfo': {runs[p]['variant']: runs[p]['kinfo'] for p in runs}, 'cases': {}}
passed = True
for c in cases:
    med = {p: runs[p]['res'][c]['median_ms'] for p in runs}
    base = mean([med[p] for p in (1, 4, 5, 8)])
    peel = mean([med[p] for p in (2, 3, 6, 7)])
    gain = 100.0 * (1.0 - peel / base)
    aa_base = 100.0 * abs(mean([med[1], med[8]]) - mean([med[4], med[5]])) / base
    aa_peel = 100.0 * abs(mean([med[2], med[7]]) - mean([med[3], med[6]])) / peel
    aa = max(aa_base, aa_peel)
    hashes = sorted({runs[p]['res'][c]['out_hash'] for p in runs})
    clean = all(runs[p]['res'][c]['mismatches'] == 0 and runs[p]['res'][c]['nonfinite'] == 0 and runs[p]['res'][c]['inputs_ok'] for p in runs)
    entry = {'medians_by_position': [med[p] for p in sorted(runs)], 'base_mean_ms': base, 'peel_mean_ms': peel,
             'gain_pct': gain, 'aa_base_pct': aa_base, 'aa_peel_pct': aa_peel, 'aa_floor_pct': aa,
             'gain_over_aa': gain / aa if aa > 0 else None, 'out_hashes': hashes, 'out_hash_equal': len(hashes) == 1,
             'repeat_ok': clean, 'max_noise': max(runs[p]['res'][c]['noise'] for p in runs)}
    if c in ('c1', 'c3'):
        entry['gate'] = bool(gain >= 4.0 and gain >= 2.0 * aa and len(hashes) == 1 and clean)
        passed = passed and entry['gate']
    else:
        passed = passed and len(hashes) == 1 and clean
    out['cases'][c] = entry
    print(c, json.dumps(entry), flush=True)
out['passed'] = bool(passed)
print(json.dumps(out))
PY
