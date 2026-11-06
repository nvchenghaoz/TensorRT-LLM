#!/bin/bash
# Idea 374 / branch 226 final kernel pair arm (top-lead ruling 12550), check folded in (one 4-GPU queue slot per arm):
#   1. pair2.py (timed + untimed repeated-execution sequence + bitwise vs 77's run() forced + ABAB D vs B at 6145)
#   2. gate_check.py (stream roles and bitwise vs main's run() at 4096/6144/6145/8192; env 0 => no plan)
#   3. chain_check.py at 4096, 4099, 6145, 8192 (1000 calls each, LP IPC flags inspected before every reset)
# -> last stdout line = pair2's JSON + "checks". With arg "pair-only" only step 1 runs (baseline arm).
# usage: bash pair2_then_checks.sh <python> <ckpt> [pair-only]
PY="$1"; CKPT="$2"; MODE="${3:-all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="${AUTO_DEV_ARTIFACTS:-.}"; mkdir -p "$ART"
{ echo "=== PAIR2 ARM $(date -Is) host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES} HEAD=$(git -C "$HERE" rev-parse HEAD 2>&1) mode=$MODE";
  nvidia-smi topo -p2p r;
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm --format=csv;
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv;
  for g in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    b=$(echo "${g:4}" | tr 'A-Z' 'a-z'); echo "numa $g: $(cat /sys/bus/pci/devices/$b/numa_node 2>/dev/null)"; done; } 1>&2
bash "$(dirname "$HERE")/lp_ar_trace/guard.sh" true || exit 3
"$PY" "$HERE/pair2.py" "$CKPT" 2>&1 | tee "$ART/pair2_stdout.log" | cut -c1-2000 1>&2
PAIR=$(tail -n 1 "$ART/pair2_stdout.log")
{ echo "=== PAIR2 END $(date -Is)"; nvidia-smi --query-gpu=index,uuid,clocks.sm,temperature.gpu,power.draw --format=csv; } 1>&2
if [ "$MODE" = "pair-only" ]; then echo "$PAIR"; exit 0; fi
echo "=== GATE CHECK $(date -Is)" 1>&2
"$PY" "$HERE/gate_check.py" "$CKPT" 2>&1 | tee "$ART/gate_stdout.log" | cut -c1-1500 1>&2
GRC=${PIPESTATUS[0]}
GATE=$(tail -n 1 "$ART/gate_stdout.log")
echo "=== CHAIN CHECK $(date -Is)" 1>&2
CHAIN_SHAPES=4096,4099,6145,8192 "$PY" "$HERE/chain_check.py" "$CKPT" 2>&1 | tee "$ART/chain_stdout.log" | cut -c1-1500 1>&2
CRC=${PIPESTATUS[0]}
CHAIN=$(tail -n 1 "$ART/chain_stdout.log")
exec "$PY" -c '
import json, sys
def parse(s):
    try:
        return json.loads(s)
    except Exception:
        return {"passed": False, "unparsed": s[:500]}
p, g, c = parse(sys.argv[1]), parse(sys.argv[2]), parse(sys.argv[3])
out = p.get("output", {})
pair_ok = bool(out.get("bitwise_vs_77_forced_all")) and bool(out.get("finite_all"))
p["checks"] = {"passed": pair_ok and bool(g.get("passed")) and bool(c.get("passed")),
               "pair_untimed_seq_bitwise_vs_77_and_finite": pair_ok,
               "gate_passed": g.get("passed"), "gate_shapes": g.get("shapes"), "gate_env0": g.get("env0"),
               "gate_plan_equal_main": g.get("plan_equal_main_T1_16384"),
               "chain_passed": c.get("passed"), "chain_summary": {k: {kk: v.get(kk) for kk in (
                   "bounds", "calls_per_rank", "ipc_flags_ok_all_ranks", "rerun_mismatch", "nonfinite_calls",
                   "max_li_rel_vs_ref_piped", "max_li_rel_vs_ref_serial", "lp_calls_per_pipelined_boundary",
                   "passed")} for k, v in (c.get("summary") or {}).items()},
               "gate_rc": int(sys.argv[4]), "chain_rc": int(sys.argv[5])}
p["passed"] = p["checks"]["passed"]
print(json.dumps(p))
' "$PAIR" "$GATE" "$CHAIN" "$GRC" "$CRC"
