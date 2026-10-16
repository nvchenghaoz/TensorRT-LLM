#!/usr/bin/env python3
"""Correctness for branch 16 (idea 45): main's vs this tree's e4m3 tiled context FMHA, bitwise.

Builds experiments/ascent/fmha_ctx/bench.cu against main's fmha_v2/src (git archive of BASE) and against this
tree's, runs both on the same deterministic inputs with the timed loop's protocol (warmup 10 + 50 calls, every
call preceded by NaN-poisoning O and an L2 flush; every call's output hash compared with call 1; Q / KV pool
hashes checked unchanged), dumps the last output, and compares main vs candidate byte for byte. Also reports the
error of both against an FP32 torch reference (dequantized e4m3 Q/K/V, FP32 softmax) for a subset of heads
(advisory for class exact), and ncu occupancy of both kernels. Last stdout line: {"passed": ..., ...}.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

BASE = sys.argv[1] if len(sys.argv) > 1 else "87bcb21958a58c5bf22327e50aa0fdfee07db213"
CASES = ["c1", "c4", "ragged", "empty"]
WARMUP, ITERS = 10, 50
W = "/tmp/fmha_ctx_check"
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.getcwd()


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def parse(stdout):
    res, meta, kinfo = {}, {}, None
    for line in stdout.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "KINFO":
            kinfo = dict(regs=int(t[1]), local_bytes=int(t[2]), dyn_smem=int(t[3]),
                         occupancy_ctas_per_sm=int(t[4]), store_o_from_regs=int(t[5]))
        elif t[0] == "RESULT":
            res[t[1]] = dict(b=int(t[2]), ctas=int(t[3]), median_ms=float(t[4]), p90_ms=float(t[6]),
                             repeat_mismatches=int(t[9]), calls=int(t[10]), nonfinite=int(t[11]),
                             inputs_unchanged=bool(int(t[12])), out_hash=t[13])
        elif t[0] == "META":
            b = int(t[2])
            meta[t[1]] = dict(b=b, max_blocks=int(t[3]), q_lens=[int(x) for x in t[4:4 + b]],
                              past_lens=[int(x) for x in t[4 + b:4 + 2 * b]])
    return res, meta, kinfo


def ref_errors(d, case, m, heads=(0, 11, 23)):
    """Max/mean abs error of the dumped O vs an FP32 reference, for the first and last sequence."""
    h, h_kv, dim, tpb = 24, 4, 256, 32
    dev = "cuda"
    q = torch.from_numpy(np.fromfile(f"{d}/{case}_q.bin", dtype=np.uint8)).to(dev)
    q = q.view(torch.float8_e4m3fn).float().view(-1, h, dim)
    pool = torch.from_numpy(np.fromfile(f"{d}/{case}_pool.bin", dtype=np.uint8)).to(dev).view(-1, h_kv, tpb, dim)
    offs = torch.from_numpy(np.fromfile(f"{d}/{case}_offs.bin", dtype=np.int32)).view(m["b"], 2, m["max_blocks"])
    o = torch.from_numpy(np.fromfile(f"{d}/{case}_o.bin", dtype=np.uint16).astype(np.int16)).to(dev)
    o = o.view(torch.bfloat16).float().view(-1, h, dim)
    cu = np.concatenate([[0], np.cumsum(m["q_lens"])])
    out = {"max_abs": 0.0, "mean_abs": 0.0, "n": 0}
    for i in sorted({0, m["b"] - 1}):
        ql, past = m["q_lens"][i], m["past_lens"][i]
        kvl = past + ql
        nblk = (kvl + tpb - 1) // tpb
        kb = offs[i, 0, :nblk].long().to(dev)
        vb = offs[i, 1, :nblk].long().to(dev)
        k = pool[kb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(h_kv, -1, dim)[:, :kvl]
        v = pool[vb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(h_kv, -1, dim)[:, :kvl]
        qi = q[cu[i]:cu[i + 1]]
        oi = o[cu[i]:cu[i + 1]]
        qpos = torch.arange(ql, device=dev)[:, None] + past
        kpos = torch.arange(kvl, device=dev)[None, :]
        mask = kpos > qpos
        for hq in heads:
            g = hq // (h // h_kv)
            s = (qi[:, hq] @ k[g].T) * (1.0 / 16.0)
            s.masked_fill_(mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            r = p @ v[g]
            e = (oi[:, hq] - r).abs()
            out["max_abs"] = max(out["max_abs"], float(e.max()))
            out["mean_abs"] += float(e.sum())
            out["n"] += e.numel()
            del s, p, r, e
    out["mean_abs"] /= max(out["n"], 1)
    return out


def ncu_occupancy(binary):
    metrics = ",".join([
        "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers", "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__maximum_warps_per_active_cycle_pct", "gpu__time_duration.sum"
    ])
    cmd = (f"ncu --kernel-name regex:fmha_v2_flash --launch-count 1 --metrics {metrics} --csv "
           f"{binary} --cases c1 --warmup 0 --iters 1")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    vals = {}
    for line in r.stdout.splitlines():
        cols = [c.strip('"') for c in line.split('","')]
        if len(cols) >= 3 and cols[-3] in metrics.split(","):
            vals[cols[-3]] = cols[-1]
    if not vals:
        vals["error"] = (r.stdout[-400:] + r.stderr[-400:]).strip()
    return vals


def main():
    os.makedirs(f"{W}/base", exist_ok=True)
    sh(f"git archive {BASE} cpp/kernels/fmha_v2/src | tar -x -C {W}/base")
    sh(f"bash {HERE}/build.sh {W}/base/cpp/kernels/fmha_v2/src {W}/bench_base")
    sh(f"bash {HERE}/build.sh {TREE}/cpp/kernels/fmha_v2/src {W}/bench_cand")
    runs = {}
    for arm in ("base", "cand"):
        d = f"{W}/dump_{arm}"
        os.makedirs(d, exist_ok=True)
        r = subprocess.run([f"{W}/bench_{arm}", "--cases", ",".join(CASES), "--warmup", str(WARMUP), "--iters",
                            str(ITERS), "--dump", d], capture_output=True, text=True)
        print(f"===== {arm} (rc {r.returncode}) =====\n{r.stdout}{r.stderr}", flush=True)
        runs[arm] = (r.returncode,) + parse(r.stdout)

    summary = {"base_commit": BASE, "kernel": {a: runs[a][3] for a in runs}, "cases": {}}
    passed = runs["base"][0] == 0 and runs["cand"][0] == 0
    for case in CASES:
        rb, rc = runs["base"][1][case], runs["cand"][1][case]
        m = runs["cand"][2][case]
        same_inputs = all(
            np.array_equal(np.fromfile(f"{W}/dump_base/{case}_{n}.bin", dtype=np.uint8),
                           np.fromfile(f"{W}/dump_cand/{case}_{n}.bin", dtype=np.uint8)) for n in ("q", "pool", "offs"))
        ob = np.fromfile(f"{W}/dump_base/{case}_o.bin", dtype=np.uint8)
        oc = np.fromfile(f"{W}/dump_cand/{case}_o.bin", dtype=np.uint8)
        diff_bytes = int((ob != oc).sum()) if ob.size == oc.size else -1
        err_b = ref_errors(f"{W}/dump_base", case, m)
        err_c = ref_errors(f"{W}/dump_cand", case, m)
        ok = (same_inputs and diff_bytes == 0 and rb["out_hash"] == rc["out_hash"] and rc["repeat_mismatches"] == 0
              and rb["repeat_mismatches"] == 0 and rc["nonfinite"] == 0 and rc["inputs_unchanged"])
        passed = passed and ok
        summary["cases"][case] = {
            "q_lens": m["q_lens"], "past_lens": m["past_lens"], "out_bytes": int(oc.size),
            "bitwise_equal_to_main": diff_bytes == 0, "diff_bytes": diff_bytes, "same_inputs": same_inputs,
            "calls": rc["calls"], "cand_repeat_mismatches_vs_call1": rc["repeat_mismatches"],
            "main_repeat_mismatches_vs_call1": rb["repeat_mismatches"], "cand_nonfinite": rc["nonfinite"],
            "inputs_unchanged": rc["inputs_unchanged"], "out_hash_main": rb["out_hash"], "out_hash_cand": rc["out_hash"],
            "fp32_ref_main": err_b, "fp32_ref_cand": err_c, "ok": ok}
        print(case, json.dumps(summary["cases"][case]), flush=True)
    summary["ncu"] = {a: ncu_occupancy(f"{W}/bench_{a}") for a in ("base", "cand")}
    print("ncu", json.dumps(summary["ncu"]), flush=True)
    summary["passed"] = bool(passed)
    summary["compared_against"] = ("main's own kernel (fmha_v2/src at base commit, same template instantiation and "
                                   "inputs), bitwise; FP32 torch reference advisory")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
