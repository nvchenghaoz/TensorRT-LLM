#!/usr/bin/env bash
# Compile sass_same.cu against main's fmha_v2/src (git archive BASE) and this tree's; diff the SASS of each kernel and
# the smem-size table. Last line: {"passed": ..., ...}.
set -euo pipefail
BASE="${1:-87bcb21958a58c5bf22327e50aa0fdfee07db213}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/tmp/fmha_sass_same
rm -rf $W && mkdir -p $W/base
git archive "$BASE" cpp/kernels/fmha_v2/src | tar -x -C $W/base
FLAGS="-O3 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -DUSE_DEMO_BERT_PARAMS=1 -DGENERATE_CUBIN=1 -cubin -arch=sm_120a"
nvcc $FLAGS -I$W/base/cpp/kernels/fmha_v2/src $HERE/sass_same.cu -o $W/base.cubin &
nvcc $FLAGS -I$PWD/cpp/kernels/fmha_v2/src $HERE/sass_same.cu -o $W/cand.cubin &
wait
res=""
ok=1
for k in k_bf16_tiled k_fp16_tiled k_e4m3_nl; do
  cuobjdump -sass -fun $k $W/base.cubin | grep -E '^\s+/\*[0-9a-f]{4}\*/' > $W/base_$k.sass
  cuobjdump -sass -fun $k $W/cand.cubin | grep -E '^\s+/\*[0-9a-f]{4}\*/' > $W/cand_$k.sass
  n=$(wc -l < $W/base_$k.sass)
  if cmp -s $W/base_$k.sass $W/cand_$k.sass && [ "$n" -gt 100 ]; then same=true; else same=false; ok=0; diff $W/base_$k.sass $W/cand_$k.sass | head -10 || true; fi
  echo "$k: sass_lines=$n identical=$same"
  res="$res\"$k\": {\"sass_lines\": $n, \"identical\": $same}, "
done
nb=$(cuobjdump -sass $W/base.cubin | md5sum | cut -c1-16); nc=$(cuobjdump -sass $W/cand.cubin | md5sum | cut -c1-16)
if cmp -s $W/base.cubin $W/cand.cubin; then cubin_same=true; else cubin_same=false; fi
echo "whole cubin (incl. the smem-size table data) byte-identical: $cubin_same"
echo "{${res}\"whole_cubin_sass_md5\": {\"base\": \"$nb\", \"cand\": \"$nc\"}, \"whole_cubin_identical\": $cubin_same, \"passed\": $([ $ok = 1 ] && echo true || echo false)}"
