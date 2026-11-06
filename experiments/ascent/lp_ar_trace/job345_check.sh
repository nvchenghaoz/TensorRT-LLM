#!/bin/bash
# Idea 345 / branch 192 check job: git receipts (served bytes == parent head dce96ec1; comm-file drift on main since
# faef77c2), then guard.sh (DIRTY SLOT, wait-only) -> run_lp_trace.sh (STEP 0 topo/peer, STEP 1 lp_check under
# timeout -s KILL 300, STEP 2 trace_graph.py --with-check = model trace + bench.py sweep incl. 1000-call robust run).
# usage: bash job345_check.sh <python> <checkpoint>
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
{
  echo "=== GIT $(date -Is) root=$ROOT"
  git -C "$ROOT" rev-parse HEAD 2>&1
  echo "--- git diff --stat dce96ec1 HEAD -- . ':(exclude)experiments/ascent' (must be empty)"
  git -C "$ROOT" diff --stat dce96ec1 HEAD -- . ':(exclude)experiments/ascent' 2>&1
  for ref in main origin/main; do
    if git -C "$ROOT" rev-parse --verify -q "$ref" >/dev/null 2>&1; then
      echo "--- $ref = $(git -C "$ROOT" rev-parse "$ref"); git diff --stat faef77c2 $ref on comm files + engine.json"
      git -C "$ROOT" diff --stat faef77c2 "$ref" -- \
        cpp/tensorrt_llm/kernels/communicationKernels cpp/tensorrt_llm/thop/allreduceOp.cpp \
        tensorrt_llm/_torch/distributed experiments/ascent-config/engine.json 2>&1
    fi
  done
} 1>&2
exec bash "$HERE/guard.sh" bash "$HERE/run_lp_trace.sh" "$1" "$2"
