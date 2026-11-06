#!/bin/bash
# Idea 96 / branch 199 kernel_time pair arm that also carries the untimed acceptance checks (one 4-GPU queue slot):
#   1. pair.py (timed; the boundary as this tree runs it)   2. chain_check.py (1000 calls x 3 shapes, IPC flags)
#   3. e2e_check.py off,on 8 (LLM, env0 vs pipelined)      -> last stdout line = pair.py's JSON + "checks".
# With arg "pair-only" only step 1 runs (baseline arm). usage: bash pair_then_checks.sh <python> <ckpt> [pair-only]
PY="$1"; CKPT="$2"; MODE="${3:-all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="${AUTO_DEV_ARTIFACTS:-.}"; mkdir -p "$ART"
{ echo "=== PAIR ARM $(date -Is) host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES} HEAD=$(git -C "$HERE" rev-parse HEAD 2>&1) mode=$MODE";
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm --format=csv;
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; } 1>&2
bash "$(dirname "$HERE")/lp_ar_trace/guard.sh" true || exit 3
"$PY" "$HERE/pair.py" "$CKPT" 2>&1 | tee "$ART/pair_stdout.log" | cut -c1-1500 1>&2
PAIR=$(tail -n 1 "$ART/pair_stdout.log")
if [ "$MODE" = "pair-only" ]; then echo "$PAIR"; exit 0; fi
echo "=== CHAIN CHECK $(date -Is)" 1>&2
"$PY" "$HERE/chain_check.py" "$CKPT" 2>&1 | tee "$ART/chain_stdout.log" | cut -c1-1500 1>&2
CRC=${PIPESTATUS[0]}
CHAIN=$(tail -n 1 "$ART/chain_stdout.log")
echo "=== E2E CHECK $(date -Is)" 1>&2
"$PY" "$HERE/e2e_check.py" "$CKPT" off,on 8 2>&1 | tee "$ART/e2e_stdout.log" | cut -c1-1500 1>&2
ERC=${PIPESTATUS[0]}
E2E=$(tail -n 1 "$ART/e2e_stdout.log")
exec "$PY" -c '
import json, sys
def parse(s):
    try:
        return json.loads(s)
    except Exception:
        return {"passed": False, "unparsed": s[:500]}
p, c, e = parse(sys.argv[1]), parse(sys.argv[2]), parse(sys.argv[3])
cmp_ = e.get("compare") or {}
p["checks"] = {"passed": bool(c.get("passed")) and bool(e.get("passed")),
               "chain_passed": c.get("passed"), "chain_summary": c.get("summary"),
               "e2e_passed": e.get("passed"), "e2e_compare": {k: cmp_.get(k) for k in ("n", "passed", "identical",
                                                                                     "max_dlp_before_div")},
               "e2e_tail_graph": e.get("tail_graph"), "e2e_decode_lamport_only_0_LP": e.get("decode_lamport_only_0_LP"),
               "chain_rc": int(sys.argv[4]), "e2e_rc": int(sys.argv[5])}
print(json.dumps(p))
' "$PAIR" "$CHAIN" "$E2E" "$CRC" "$ERC"
