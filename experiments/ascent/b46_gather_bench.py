# Branch 46 microbench: steady-decode next-token gather (runner.py _apply_steady_gen_fast_prepare).
# BRACE-FREE on purpose: run_gpu_pair inlines it via a bash heredoc in both trees (the base tree lacks this file).
# The arm's op is picked from the tree it runs in: if runner.py caches "previous_slots_i64" the arm is the
# candidate (torch.gather with the cached int64 (n,1) index), else main (torch.index_select with the int32 slots).
# Mode argument: auto (tree op), both (time main and cand ops in this process).
#   src = new_tokens[0, :, :1] of a (1, 33, 1) int32 buffer (max_num_sequences + 1 = 33), dst = input_ids_cuda[:n].
# Each op is timed as one CUDA graph of REP back-to-back launches (GPU time per op incl. graph node overhead),
# WARM warmup replays then ITERS timed replays, median/p10/p90; then an A/A re-run of the same arm (fresh graph).
# Every iteration sees identical inputs (pure copy, src/index never written). Untimed repeated-execution check:
# same warmup/iteration count, after every replay dst compared with iteration 1 and with index_select (bitwise),
# plus a random-slot bitwise sweep incl. repeated slots at n = 1,2,4,8,12,16,24,32.
import json
import os
import sys

import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
NS = [1, 2, 4, 8, 12, 16, 24, 32]
REP = 100
ITERS = 60
WARM = 10
S = 33
dev = "cuda"
RUNNER = "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py"
TREE_OP = "cand" if "previous_slots_i64" in open(RUNNER).read() else "main"

torch.manual_seed(0)
new_tokens = torch.randint(0, 150000, (1, S, 1), dtype=torch.int32, device=dev)
src = new_tokens[0, :, :1]
input_ids = torch.zeros(8192, dtype=torch.int32, device=dev)


def op_main(n, idx32, idx64):
    torch.index_select(src, 0, idx32, out=input_ids[:n].view(n, 1))


def op_cand(n, idx32, idx64):
    torch.gather(src, 0, idx64, out=input_ids[:n].view(n, 1))


OPS = dict(main=op_main, cand=op_cand)


def rand_idx(n, g, repeated):
    if repeated or n > S - 1:
        idx = torch.randint(0, S - 1, (n,), generator=g)
    else:
        idx = torch.randperm(S - 1, generator=g)[:n]
    idx32 = idx.to(torch.int32).to(dev)
    idx64 = idx32.to(torch.int64).view(n, 1)
    return idx32, idx64


def sweep_check(arm):
    g = torch.Generator().manual_seed(1)
    bad = 0
    total = 0
    for n in NS:
        for trial in range(20):
            idx32, idx64 = rand_idx(n, g, trial % 2 == 1)
            ref = torch.index_select(src, 0, idx32).clone()
            for it in range(3):
                input_ids.fill_(-7)
                OPS[arm](n, idx32, idx64)
                total += 1
                if not torch.equal(input_ids[:n].view(n, 1), ref) or not bool((input_ids[n:n + 8] == -7).all()):
                    bad += 1
    torch.cuda.synchronize()
    return bad, total


def build_graph(arm, n, idx32, idx64):
    fn = OPS[arm]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(n, idx32, idx64)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(REP):
            fn(n, idx32, idx64)
    return graph


def repeated_check(arm, n):
    g = torch.Generator().manual_seed(100 + n)
    idx32, idx64 = rand_idx(n, g, False)
    ref = torch.index_select(src, 0, idx32).clone()
    graph = build_graph(arm, n, idx32, idx64)
    first = None
    max_err = 0
    for it in range(WARM + ITERS):
        input_ids[:n].fill_(-7)
        graph.replay()
        out = input_ids[:n].view(n, 1).clone()
        if first is None:
            first = out
        e1 = int((out.long() - first.long()).abs().max().item())
        e2 = int((out.long() - ref.long()).abs().max().item())
        max_err = max(max_err, e1, e2)
    return max_err


def time_arm(arm, n):
    g = torch.Generator().manual_seed(100 + n)
    idx32, idx64 = rand_idx(n, g, False)
    graph = build_graph(arm, n, idx32, idx64)
    for _ in range(WARM):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / REP)
    times.sort()
    return times[len(times) // 2], times[int(0.1 * len(times))], times[int(0.9 * len(times))]


def main():
    arms = ["main", "cand"] if MODE == "both" else [TREE_OP]
    print("tree op: %s; timing arms %s; device %s" % (TREE_OP, arms, torch.cuda.get_device_name()), flush=True)
    checks = dict()
    for arm in arms:
        bad, total = sweep_check(arm)
        rep_err = max(repeated_check(arm, n) for n in NS)
        checks[arm] = dict(sweep_bad=bad, sweep_total=total, repeated_max_err=rep_err)
        print("check %s: sweep bad %d/%d vs index_select, repeated max err %d" % (arm, bad, total, rep_err), flush=True)
    res = dict()
    for n in NS:
        row = dict()
        for arm in arms:
            m1 = time_arm(arm, n)
            m2 = time_arm(arm, n)
            row[arm] = dict(med_us=m1[0], p10_us=m1[1], p90_us=m1[2], aa_med_us=m2[0], aa_p10_us=m2[1],
                            aa_p90_us=m2[2])
        res[str(n)] = row
        print("n=%d " % n + " ".join("%s: med %.3f us p10 %.3f p90 %.3f | A/A med %.3f" % (
            a, v["med_us"], v["p10_us"], v["p90_us"], v["aa_med_us"]) for a, v in row.items()), flush=True)
    arm = "cand" if MODE == "both" else TREE_OP
    batch = dict()
    for n in (8, 16, 32):
        v = res[str(n)][arm]
        noise = (v["p90_us"] - v["p10_us"]) / v["med_us"]
        aa = abs(v["aa_med_us"] - v["med_us"]) / v["med_us"]
        batch[str(n)] = dict(step_ms=v["med_us"] / 1000.0, noise_floor=max(noise, aa))
    passed = all(c["sweep_bad"] == 0 and c["repeated_max_err"] == 0 for c in checks.values())
    out = dict(batch=batch, output=dict([("class", "exact")]), passed=passed, arm=arm, checks=checks, all=res,
               measure="GPU us per gather op, CUDA graph of %d launches, median of %d replays; step_ms = that op's "
               "time in ms (the only change in the decode step)" % (REP, ITERS))
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    if art:
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "b46_gather_bench.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
