#!/usr/bin/env bash
# Idea 35 (branch 30): SASS checks of the peel-loop one-sync change.
#   bash sass_peel1.sh <base_sha>
# Compiles experiments/ascent/fmha_ctx/bench.cu (the generated e4m3 64x128 tiled paged-KV d256 causal kernel plus the
# last-tile split piece / merge kernels) to cubin three times: against BASE's cpp/kernels/fmha_v2/src (git archive),
# against this tree's, and against this tree's with the kill switch -DFMHA_SM120_PEEL_SYNC_OPT=0. Per kernel function:
# SASS instruction hash, BAR.SYNC count, ptxas registers / spills / stack. Then sass_same.sh BASE (BF16/FP16 tiled and
# non-tiled e4m3 instantiations of the same headers must be identical). Last stdout line: JSON.
set -euo pipefail
BASE="${1:?usage: sass_peel1.sh <base_sha>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/tmp/fmha_sass_peel1
rm -rf $W && mkdir -p $W/base
git archive "$BASE" cpp/kernels/fmha_v2/src | tar -x -C $W/base
FLAGS="-O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 -DBENCH_MAX_BATCH=3 -cubin -arch=sm_120a -Xptxas -v"
nvcc $FLAGS -I$W/base/cpp/kernels/fmha_v2/src $HERE/bench.cu -o $W/base.cubin > $W/base.log 2>&1 &
nvcc $FLAGS -I$PWD/cpp/kernels/fmha_v2/src $HERE/bench.cu -o $W/cand.cubin > $W/cand.log 2>&1 &
nvcc $FLAGS -DFMHA_SM120_PEEL_SYNC_OPT=0 -I$PWD/cpp/kernels/fmha_v2/src $HERE/bench.cu -o $W/kill.cubin > $W/kill.log 2>&1 &
wait
for v in base cand kill; do
  echo "===== ptxas $v"; grep -E 'Compiling entry|spill|Used [0-9]+ registers|error' $W/$v.log | sed 's/ptxas info    ://' | cut -c1-220 || true
  cuobjdump -sass $W/$v.cubin > $W/$v.sass
done
python3 - $W <<'PY'
import hashlib, json, re, sys
w = sys.argv[1]
def funcs(path):
    out, cur, lines = {}, None, []
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if cur: out[cur] = lines
            cur, lines = m.group(1), []
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m and cur is not None:
            lines.append(m.group(1))
    if cur: out[cur] = lines
    return out
def ptxas(path):
    res, cur = {}, None
    for line in open(path):
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m: cur = m.group(1)
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur: res.setdefault(cur, {}).update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m and cur: res.setdefault(cur, {})['regs'] = int(m.group(1))
    return res
F = {v: funcs(f'{w}/{v}.sass') for v in ('base', 'cand', 'kill')}
P = {v: ptxas(f'{w}/{v}.log') for v in ('base', 'cand', 'kill')}
out = {'functions': {}}
for name in sorted(F['base']):
    short = 'main' if 'kernel_nl_tiled' in name and 'last_tile' not in name else ('piece' if 'piece' in name else ('merge' if 'merge' in name else name[:60]))
    e = {}
    for v in ('base', 'cand', 'kill'):
        L = F[v].get(name, [])
        e[v] = {'sass_lines': len(L), 'sha': hashlib.sha256('\n'.join(L).encode()).hexdigest()[:16],
                'bar_sync': sum(1 for x in L if x.startswith('BAR.SYNC') or ' BAR.SYNC' in x or x.startswith('BAR ')),
                'ptxas': P[v].get(name, {})}
    e['kill_identical_to_base'] = F['kill'].get(name) == F['base'].get(name)
    e['cand_identical_to_base'] = F['cand'].get(name) == F['base'].get(name)
    out['functions'][short] = e
    print(short, json.dumps(e), flush=True)
fm = out['functions']
no_new_spills = all(fm[k]['cand']['ptxas'].get('spill_st', 0) <= fm[k]['base']['ptxas'].get('spill_st', 0)
                    and fm[k]['cand']['ptxas'].get('spill_ld', 0) <= fm[k]['base']['ptxas'].get('spill_ld', 0) for k in fm)
out['kill_switch_identical_all'] = all(fm[k]['kill_identical_to_base'] for k in fm)
out['piece_merge_identical'] = all(fm[k]['cand_identical_to_base'] for k in fm if k in ('piece', 'merge'))
out['main_changed'] = not fm['main']['cand_identical_to_base']
out['no_new_spills'] = no_new_spills
json.dump(out, open(f'{w}/summary.json', 'w'))
PY
echo "===== sass_same.sh $BASE"
bash $HERE/sass_same.sh "$BASE" | tee $W/sass_same.log
python3 - $W <<'PY'
import json, sys
w = sys.argv[1]
s = json.load(open(f'{w}/summary.json'))
same = json.loads(open(f'{w}/sass_same.log').read().strip().splitlines()[-1])
s['sass_same_other_instantiations'] = same
s['passed'] = bool(s['kill_switch_identical_all'] and s['piece_merge_identical'] and s['main_changed'] and s['no_new_spills'] and same['passed'])
print(json.dumps(s))
PY
