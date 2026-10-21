#!/usr/bin/env bash
# Idea 244 kernel pair arm (SM120 e4m3 tiled context FMHA + last-q-tile split, --split -1, kMaxBatch 3).
#   bash pair_arm_244.sh <base|cand> <python>
# The baseline arm runs this same text inline (main has no copy). Both arms: build bench.cu against the arm's own
# cpp/kernels/fmha_v2/src with the tree defaults (main: stock; candidate: FMHA_SM120_EXP2=1 default), 2 discarded
# warm-up processes, then c1,c2,c3,c4 with 10 warmup + 100 timed calls (NaN-poisoned O and split workspace, 384 MiB
# L2 flush, own event pair; output hash vs call 1). Then the BUILT TensorRT-LLM libraries: the base arm copies its
# tensorrt_llm/libs to /tmp/fmha244_sass/baselibs; the candidate arm runs lib_sass_cmp.py compare against it.
# Last stdout line: the evidence JSON (class precision).
set -uo pipefail
MODE="$1"
PY="$2"
W=/tmp/fmha244_arm_$MODE
mkdir -p $W
echo "ARM=$MODE TREE_COMMIT=$(git rev-parse HEAD 2>/dev/null) HEAD_TREE=$(git rev-parse 'HEAD^{tree}' 2>/dev/null) DIRTY=$(git status --porcelain 2>/dev/null | wc -l) LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg) DATE=$(date +%T)"
bash experiments/ascent/fmha_ctx/build.sh cpp/kernels/fmha_v2/src $W/bench -DBENCH_MAX_BATCH=3 || exit 2
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader || true
for i in 1 2; do $W/bench --cases c1,c3 --warmup 10 --iters 40 --split -1 > /dev/null; echo "[arm] warm-up $i done"; done
$W/bench --cases c1,c2,c3,c4 --warmup 10 --iters 100 --split -1 | tee $W/log
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader || true
echo "LOADAVG_END=$(cut -d' ' -f1-3 /proc/loadavg) DATE=$(date +%T)"
LIBS=$("$PY" -c "import importlib.util,os;print(os.path.join(list(importlib.util.find_spec('tensorrt_llm').submodule_search_locations)[0],'libs'))")
echo "LIBS=$LIBS"; ls -la "$LIBS" | head -40
: > $W/sass.json
if [ "$MODE" = base ]; then
  rm -rf /tmp/fmha244_sass/baselibs; mkdir -p /tmp/fmha244_sass
  cp -rL "$LIBS" /tmp/fmha244_sass/baselibs && echo "[sass] base libs saved" && echo '{"saved": true}' > $W/sass.json
else
  if [ -d /tmp/fmha244_sass/baselibs ]; then
    "$PY" experiments/ascent/fmha_ctx/lib_sass_cmp.py compare /tmp/fmha244_sass/baselibs | tee $W/sass.log
    tail -n 1 $W/sass.log > $W/sass.json
  else
    echo '{"error": "no base libs in /tmp/fmha244_sass/baselibs"}' > $W/sass.json
  fi
fi
python3 - $W/log $W/sass.json <<'PY'
import json, sys
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
try:
    sass = json.loads(open(sys.argv[2]).read() or '{}')
except Exception as e:
    sass = {'error': repr(e)}
ok = all(r['repeat_mismatches'] == 0 and r['nonfinite'] == 0 and r['inputs_unchanged'] for r in res.values())
print(json.dumps({'batch': batch, 'output': {'class': 'precision'}, 'repeat_ok': ok, 'lib_sass': sass, 'cases': res}))
PY
