"""Branch 57 (idea 96) untimed graph-level repeat sequence (acceptance item 4, plan 916(4)).

Per arm (cand = routed G3 + v_cf; base = the TRTLLM_FMHA_V2_FLUX3_VCF=0 kill switch, main's served
modules), two CUDA graphs of the served per-step attention sets, every call with its OWN output:
  t2i  1 BF16-out 16384 + 4 nfo 16384 + 27 nfo 16464 + 1 nfo q_rows(16384) at 16464
  i2i  1 BF16-out 16384 + 9 nfo 16384 + 27 nfo 32848 + 1 nfo q_rows(16384) at 32848
Order per round: t2i -> i2i -> t2i -> i2i. Warmup rounds, then --rounds measured rounds. Before
each round the Q/K/V buffers are restored from pristine copies; before each replay every output
is poisoned (BF16 NaN, uint8 0x11 / 0xEE alternating). Every output of every replay is compared
bitwise with the arm's round-0 output and with the base arm's round-0 output (q_rows outputs on
their written prefix). Inputs are checked unchanged before each restore. The G3 kernels hold no
global counters / flags / epochs (mbarriers live in shared memory and are re-initialised per CTA).
Also checks that TRTLLM_FMHA_V2_FLUX3_VCF=0 drops v_cf from ROUTED_LEVERS (fresh interpreter).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402

b = c.b
log = c.log
SETS = {"t2i": (1, 4, 16384, 16464), "i2i": (1, 9, 16384, 32848)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    lib = c.lib_mod()
    routed, levs = c.arm_levers(lib, b.make_qkv(4176, 7, False))
    mods = c.build(lib, levs, ("base", "cand"))
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    H, D, W = b.H, b.D, b.W
    cols = (W // 16 + 3) // 4 * 4
    nrow = lib.q_prefix_rows_written(c.QR)
    nsf = (nrow + 127) // 128 * 128 * cols
    failures = []

    env = dict(os.environ, TRTLLM_FMHA_V2_FLUX3_VCF="0")
    ks = subprocess.run([sys.executable, "-c",
                         "from tensorrt_llm._torch.visual_gen.attention.backends import "
                         "fmha_v2_flux3 as f; print(','.join(f.ROUTED_LEVERS))"],
                        capture_output=True, text=True, env=env, timeout=600)
    ks_levers = ks.stdout.strip().splitlines()[-1].split(",") if ks.stdout.strip() else []
    kill_switch = {"routed_with_VCF0": ks_levers, "rc": ks.returncode,
                   "equals_base": tuple(x for x in ks_levers if x != "load_trim") ==
                   tuple(x for x in levs["base"] if x != "load_trim")}
    if not kill_switch["equals_base"]:
        failures.append("kill_switch_levers_ne_base")
    log(f"[kill switch] {json.dumps(kill_switch)}")

    inps = {S: b.make_qkv(S, 4000 + S, False) for S in (16384, 16464, 32848)}
    graphs, outs = {}, {}
    for a in ("base", "cand"):
        for sname, (n_bf16, n_img, s_img, s_full) in SETS.items():
            calls = []  # (fn, [tensors], view)
            for _ in range(n_bf16):
                o = torch.empty((1, s_img, H, D), device="cuda", dtype=torch.bfloat16)
                calls.append((lambda m=mods[a]["bf16"], oo=o: b.call_bf16(lib, m, inps[s_img], oo),
                              [o], None))
            for S, n in ((s_img, n_img), (s_full, 27)):
                for _ in range(n):
                    no = b.NfoOut(S, post=4096)
                    calls.append((lambda m=mods[a]["nfo"], oo=no, S=S:
                                  b.call_nfo(lib, m, inps[S], oo, gscale), [no.fp4, no.sf], None))
            no = b.NfoOut(s_full, post=4096)
            calls.append((lambda m=mods[a]["nfo"], oo=no: c.call_nfo_q(lib, m, inps[s_full], oo,
                                                                       gscale),
                          [no.fp4, no.sf], lambda ts: [ts[0][:, :nrow], ts[1][:nsf]]))
            for f, _, _ in calls:
                f()
            torch.cuda.synchronize()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for f, _, _ in calls:
                    f()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f, _, _ in calls:
                    f()
            graphs[(a, sname)] = g
            outs[(a, sname)] = calls
        log(f"[capture] {a} done")

    def poison(calls, k):
        for _, ts, _ in calls:
            for t in ts:
                if t.dtype == torch.uint8:
                    t.fill_(0x11 if k % 2 == 0 else 0xEE)
                else:
                    t.fill_(float("nan"))

    def snap(calls):
        res = []
        for _, ts, view in calls:
            vs = view(ts) if view else ts
            res.append([v.view(torch.uint8).clone() if v.dtype != torch.uint8 else v.clone()
                        for v in vs])
        return res

    def cmp(calls, ref):
        bad = 0
        for (_, ts, view), r in zip(calls, ref):
            vs = view(ts) if view else ts
            for v, rr in zip(vs, r):
                vv = v.view(torch.uint8) if v.dtype != torch.uint8 else v
                bad += int(not torch.equal(vv, rr))
        return bad

    order = ["t2i", "i2i", "t2i", "i2i"]
    stats = {}
    ref0 = {}
    for a in ("base", "cand"):
        mism_self = mism_base = 0
        k = 0
        t0 = time.time()
        for r in range(args.warmup + args.rounds):
            for S, inp in inps.items():
                if not torch.equal(inp["guard"].payload, inp["pristine"]):
                    failures.append(f"{a}_inputs_changed_round{r}_S{S}")
                inp["guard"].payload.copy_(inp["pristine"])
            for sname in order:
                calls = outs[(a, sname)]
                poison(calls, k)
                k += 1
                graphs[(a, sname)].replay()
                torch.cuda.synchronize()
                if r < args.warmup:
                    continue
                if (a, sname) not in ref0:
                    ref0[(a, sname)] = snap(calls)
                mism_self += cmp(calls, ref0[(a, sname)])
                mism_base += cmp(calls, ref0[("base", sname)])
            if r >= args.warmup and (r - args.warmup) % 10 == 0:
                log(f"[{a}] round {r - args.warmup} self_mismatch={mism_self} "
                    f"vs_base_mismatch={mism_base} {time.time() - t0:.0f}s")
        stats[a] = dict(rounds=args.rounds, warmup=args.warmup, replays=4 * args.rounds,
                        outputs_compared_per_replay={s: sum(len(ts) for _, ts, _ in outs[(a, s)])
                                                     for s in SETS},
                        mismatch_vs_round0=mism_self, mismatch_vs_kill_switch=mism_base)
        if mism_self or mism_base:
            failures.append(f"{a}_mismatch")
    log(f"[stats] {json.dumps(stats)}")
    res = dict(passed=not failures, failures=failures, stats=stats, kill_switch=kill_switch,
               levers={k2: list(v) for k2, v in levs.items()},
               compared=("graph replays t2i->i2i->t2i->i2i, poisoned outputs, pristine inputs "
                         "restored per round; every output bitwise vs round 0 and vs the kill-switch "
                         "(base) arm's round 0"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
