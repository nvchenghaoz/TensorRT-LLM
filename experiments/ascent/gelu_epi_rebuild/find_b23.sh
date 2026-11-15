#!/bin/bash
# Locate branch 23's tree (marker gate_up_gelu_fp4_permuted) in any runner repo and dump its diff.
R=/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048
OUT=${AUTO_DEV_ARTIFACTS:-/tmp}
F=tensorrt_llm/_torch/models/modeling_gemma4.py
ls $R/runner/repos 2>/dev/null
REPOS=$(ls -d $R/runner/slots/*/workspace/trtllm $R/runner/repos/* $R/tensorrt_llm 2>/dev/null)
FOUND=""
for D in $REPOS; do
  git -C $D rev-parse --git-dir >/dev/null 2>&1 || continue
  echo "== $D"
  git -C $D cat-file -t fbc5e8d663d1 2>&1 | head -1
  # snapshot commits in the reflog / all refs
  for sha in $( (git -C $D log -g --all --format='%H %s' 2>/dev/null; git -C $D log --all --format='%H %s' 2>/dev/null) | grep -i 'snapshot' | awk '{print $1}' | sort -u); do
    if git -C $D grep -q gate_up_gelu_fp4_permuted $sha -- $F 2>/dev/null; then
      echo "MARKER $D $sha parent=$(git -C $D rev-parse $sha~1) date=$(git -C $D log -1 --format=%ci $sha) msg=$(git -C $D log -1 --format=%s $sha)"
      FOUND="$FOUND $D:$sha"
    fi
  done
  git -C $D log --all --reflog -S gate_up_gelu_fp4_permuted --format='%H %P %ci %s' -- $F 2>/dev/null | head -5
done
echo "FOUND=$FOUND"
# Pick the newest found commit (by commit date) and dump the diff against its merge-base with 8c2c2cb5 and with main 4f541f15
best=""; bestt=0
for x in $FOUND; do D=${x%%:*}; s=${x##*:}; t=$(git -C $D log -1 --format=%ct $s); if [ $t -gt $bestt ]; then bestt=$t; best=$x; fi; done
if [ -n "$best" ]; then
  D=${best%%:*}; s=${best##*:}
  echo "BEST $D $s"
  P=$(git -C $D rev-parse $s~1)
  echo "parent $P"; git -C $D log --oneline -12 $P
  for base in 8c2c2cb5 4f541f15; do
    mb=$(git -C $D merge-base $s $base 2>/dev/null); echo "merge-base with $base: $mb"
  done
  git -C $D diff --stat 8c2c2cb5 $s
  git -C $D diff 8c2c2cb5 $s > $OUT/b23_vs_8c2c2cb5.diff
  git -C $D diff 8c2c2cb5 $P > $OUT/b23parent_vs_8c2c2cb5.diff
  git -C $D diff --stat $P $s
  mkdir -p $OUT/b23files $OUT/base8c
  for f in $(git -C $D diff --name-only 8c2c2cb5 $s); do
    mkdir -p $OUT/b23files/$(dirname $f) $OUT/base8c/$(dirname $f)
    git -C $D show $s:$f > $OUT/b23files/$f 2>/dev/null
    git -C $D show 8c2c2cb5:$f > $OUT/base8c/$f 2>/dev/null
  done
  wc -c $OUT/*.diff
fi
echo '{"passed": true}'
