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
import threading
import time

CAP_RE = re.compile(r"\[TAIL_MOE_GRAPH\] rank (\d+) layer (\d+): captured")
FAIL_RE = re.compile(r"\[TAIL_MOE_GRAPH\][^\n]*capture failed")


PEER_CHECK = r"""
import sys, torch
n = torch.cuda.device_count()
bad = [(i, j) for i in range(n) for j in range(n) if i != j and not torch.cuda.can_device_access_peer(i, j)]
print(f"[peer] devices={n} pairs_checked={n*(n-1)} bad={bad}", flush=True)
sys.exit(0 if n == 4 and not bad else 3)
"""
WATCHDOG_S = float(os.environ.get("LP_TRACE_WATCHDOG_S", "600"))


def preamble():
    """Node receipt (hostname, topology, idle util) to stderr, then the 12-pair peer-access assert."""
    for cmd in (["hostname"], ["nvidia-smi", "topo", "-m"], ["nvidia-smi", "topo", "-p2p", "r"],
                ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,power.draw,clocks.sm",
                 "--format=csv"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            sys.stderr.write(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"$ {' '.join(cmd)} failed: {e}\n")
    sys.stderr.flush()
    r = subprocess.run([sys.executable, "-c", PEER_CHECK], capture_output=True, text=True, timeout=300)
    sys.stderr.write(r.stdout + r.stderr)
    sys.stderr.flush()
    return r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    ckpt = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    os.chdir(root)
    ok, peer_line = preamble()
    if not ok:
        print(json.dumps({"passed": False, "error": "peer-access assert failed", "peer": peer_line}))
        return
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
                         env=env, start_new_session=True)
    last, captured, failed = "", set(), []
    progress = [time.time(), ""]
    watchdog_fired = []

    def watchdog():
        # Heartbeat lines do not count as progress; kill the whole process group after WATCHDOG_S of silence.
        while p.poll() is None:
            time.sleep(15)
            if time.time() - progress[0] > WATCHDOG_S and p.poll() is None:
                watchdog_fired.append(progress[1][:400])
                print(f"[wrapper] WATCHDOG: no log line for {WATCHDOG_S:.0f}s; last: {progress[1][:400]}", flush=True)
                try:
                    subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,power.draw",
                                    "--format=csv"], timeout=60)
                    os.killpg(p.pid, 9)
                except Exception as e:  # noqa: BLE001
                    print(f"[wrapper] kill failed: {e}", flush=True)
                return

    threading.Thread(target=watchdog, daemon=True).start()
    for line in p.stdout:
        if line.strip() and "heartbeat" not in line:
            progress[0], progress[1] = time.time(), line.strip()
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
    decode_lamport_only = (dw.get("lowPrecision", 1) == 0 and dw.get("ncclDevKernel_AllReduce", 1) == 0
                           and (dw.get("oneshot_lamport") or 0) > 0)
    check = None
    if "--with-check" in sys.argv[2:] and not watchdog_fired:
        # Repeated-call correctness: bench.py sweep (every-iteration error vs the fp32 sum, the 1000-call robust run
        # with IPC flag inspection before every reset barrier, routing bitwise vs AUTO), fresh process, bounded.
        t1 = time.time()
        cargv = [sys.executable, os.path.join(root, "experiments", "ascent", "lp_ar", "bench.py"), "sweep"]
        print(f"[wrapper] start {cargv}", flush=True)
        cp = subprocess.Popen(cargv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                              start_new_session=True)
        timer = threading.Timer(900, lambda: os.killpg(cp.pid, 9))
        timer.start()
        clast = ""
        for line in cp.stdout:
            sys.stdout.write(line if len(line) < 4000 else line[:2000] + " ...[truncated]\n")
            sys.stdout.flush()
            if line.strip():
                clast = line.strip()
        cp.wait()
        timer.cancel()
        print(f"[wrapper] check rc={cp.returncode} in {time.time()-t1:.0f}s", flush=True)
        try:
            c = json.loads(clast)
        except Exception:  # noqa: BLE001
            c = {"passed": False, "unparsed_last_line": clast[:500]}
        rob = c.get("robust_LP_2048", {})
        check = {"passed": c.get("passed"), "reasons": c.get("reasons"),
                 "robust": {k: rob.get(k) for k in ("calls_per_rank", "reset_barrier_every", "nonfinite_calls",
                                                    "max_rel_rms", "max_e", "ipc_flags_ok_all_ranks",
                                                    "rerun_bitwise_mismatch", "flag_states_bad")},
                 "err_2048": (c.get("2048") or {}).get("err"),
                 "routing": {m: v.get("LP_bitwise_equal_AUTO_all_ranks")
                             for m, v in (c.get("routing") or {}).items()},
                 "unparsed": c.get("unparsed_last_line"), "rc": cp.returncode}
    out = {
        "passed": bool(routing_ok and engaged and decode_lamport_only and (check is None or check.get("passed"))),
        "decode_lamport_only": decode_lamport_only,
        "check": check,
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
        "watchdog_fired": watchdog_fired,
        "peer": peer_line,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
