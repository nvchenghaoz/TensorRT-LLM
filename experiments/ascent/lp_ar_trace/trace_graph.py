"""Waiver 2(b) trace for idea 242 on a main that includes landing 44 (L20-39 tail MoE CUDA graph).
Runs experiments/ascent/lp_ar_trace/trace.py <checkpoint> in a fresh process with TLLM_LOG_LEVEL=INFO (so the
'[TAIL_MOE_GRAPH] rank R layer L: captured 128-row MoE window' lines from _dsv41_tail_moe_graph.py are printed),
streams its output, counts captured (rank, layer) pairs and 'capture failed' lines, and judges:
  rank 0 EXTEND: 40 LP two-shot calls and 40 NCCL ARs; DECODE: 0 LP kernels;
  tail graph ENGAGED: 20 layers x 4 ranks captured, 0 capture failed.
NOT timing evidence. Last stdout line: JSON summary with "passed".
"""
import json
import os
import re
import subprocess
import sys
import time

CAP_RE = re.compile(r"\[TAIL_MOE_GRAPH\] rank (\d+) layer (\d+): captured")
FAIL_RE = re.compile(r"\[TAIL_MOE_GRAPH\][^\n]*capture failed")


def main():
    ckpt = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(root)
    env = dict(os.environ)
    env.setdefault("TLLM_LOG_LEVEL", "INFO")
    for k in list(env):
        if k.startswith("TRTLLM_DSV4_TAIL_MOE_GRAPH"):
            print(f"[wrapper] WARNING env {k}={env[k]} set; removing", flush=True)
            env.pop(k)
    t0 = time.time()
    argv = [sys.executable, os.path.join(here, "trace.py"), ckpt]
    print(f"[wrapper] start {argv} TLLM_LOG_LEVEL={env['TLLM_LOG_LEVEL']}", flush=True)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env=env)
    last, captured, failed = "", set(), []
    for line in p.stdout:
        m = CAP_RE.search(line)
        if m:
            captured.add((int(m.group(1)), int(m.group(2))))
            sys.stdout.write(line)
        elif FAIL_RE.search(line):
            failed.append(line.strip()[:400])
            sys.stdout.write(line)
        elif len(line) < 4000:
            sys.stdout.write(line)
        else:
            sys.stdout.write(line[:2000] + " ...[truncated]\n")
        sys.stdout.flush()
        if line.strip():
            last = line.strip()
    p.wait()
    print(f"[wrapper] trace rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    try:
        trace = json.loads(last)
    except Exception:  # noqa: BLE001
        trace = {"passed": False, "unparsed_last_line": last[:500]}
    per_rank_layers = {r: sorted(l for rr, l in captured if rr == r) for r in sorted({r for r, _ in captured})}
    pw = trace.get("rank0_prefill") or {}
    dw = trace.get("rank0_decode") or {}
    engaged = len(captured) == 80 and len(per_rank_layers) == 4 and not failed
    routing_ok = pw.get("lp_calls") == 40 and pw.get("nccl_allreduce") == 40 and dw.get("lowPrecision", 1) == 0
    out = {
        "passed": bool(routing_ok and engaged),
        "routing_ok": routing_ok,
        "tail_graph_engaged": engaged,
        "captured_pairs": len(captured),
        "captured_layers_per_rank": per_rank_layers,
        "capture_failed_lines": failed[:8],
        "rank0_prefill": pw,
        "rank0_decode": dw,
        "rank0_steps": trace.get("rank0_steps"),
        "layout": trace.get("layout"),
        "per_rank_census": trace.get("per_rank_census"),
        "profiled_request": trace.get("profiled_request"),
        "trace_error": trace.get("error"),
        "trace_rc": p.returncode,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
