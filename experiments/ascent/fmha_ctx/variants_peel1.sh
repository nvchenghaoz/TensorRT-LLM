#!/usr/bin/env bash
# Branch 30 variant sweep: build bench.cu against BASE's fmha_v2/src and against this tree's with several -D flag sets,
# print ptxas resources of the main kernel per build, and time all builds in the order fwd,rev,fwd,rev (one process per
# position; 10 warmup + ITERS calls per case, --split -1, cold L2 + NaN-poison per call, hash vs call 1).
#   bash variants_peel1.sh <base_sha> <cases> <iters> name=flags ...   (flags: comma-separated -D options, or none)
set -euo pipefail
BASE="$1"; CASES="$2"; ITERS="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/tmp/fmha_variants_peel1
rm -rf $W && mkdir -p $W/base
git archive "$BASE" cpp/kernels/fmha_v2/src | tar -x -C $W/base
names=(base)
bash $HERE/build.sh $W/base/cpp/kernels/fmha_v2/src $W/bench_base -DBENCH_MAX_BATCH=3 > $W/build_base.log 2>&1 &
for spec in "$@"; do
  n="${spec%%=*}"; f="${spec#*=}"; names+=("$n")
  fl=""; if [ "$f" != none ]; then fl="${f//,/ }"; fi
  bash $HERE/build.sh $PWD/cpp/kernels/fmha_v2/src $W/bench_$n -DBENCH_MAX_BATCH=3 $fl > $W/build_$n.log 2>&1 &
done
wait
for n in "${names[@]}"; do echo "== $n: $(grep -A2 'kernel_nl_tiled' $W/build_$n.log | grep -E 'spill|registers' | tr '\n' ' ')"; done
$W/bench_base --cases "$CASES" --warmup 20 --iters 20 --split -1 > /dev/null
order=("${names[@]}"); rev=(); for ((i=${#names[@]}-1; i>=0; i--)); do rev+=("${names[$i]}"); done
pos=0
for round in fwd rev fwd rev; do
  if [ $round = fwd ]; then seq=("${order[@]}"); else seq=("${rev[@]}"); fi
  for n in "${seq[@]}"; do
    pos=$((pos+1))
    $W/bench_$n --cases "$CASES" --warmup 10 --iters "$ITERS" --split -1 > $W/pos${pos}_$n.log
    echo "[pos $pos $n] $(grep RESULT $W/pos${pos}_$n.log | awk '{printf "%s %s %s | ", $2, $5, $14}')"
  done
done
python3 - $W "$CASES" "${names[@]}" <<'PY'
import glob, json, os, sys
w, cases, names = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
med = {n: {c: [] for c in cases} for n in names}
hashes = {c: set() for c in cases}
clean = True
for f in glob.glob(f'{w}/pos*_*.log'):
    n = os.path.basename(f)[:-4].split('_', 1)[1]
    for line in open(f):
        t = line.split()
        if t and t[0] == 'RESULT':
            med[n][t[1]].append(float(t[4])); hashes[t[1]].add(t[13])
            clean = clean and int(t[9]) == 0 and int(t[11]) == 0 and t[12] == '1'
out = {'cases': {}, 'repeat_ok': clean}
for c in cases:
    b = sum(med['base'][c]) / len(med['base'][c])
    out['cases'][c] = {n: {'mean_ms': sum(med[n][c]) / len(med[n][c]), 'gain_pct': 100 * (1 - sum(med[n][c]) / len(med[n][c]) / b),
                           'spread_pct': 100 * (max(med[n][c]) - min(med[n][c])) / b} for n in names}
    out['cases'][c]['hash_equal_all'] = len(hashes[c]) == 1
    print(c, json.dumps(out['cases'][c]))
print(json.dumps(out))
PY
