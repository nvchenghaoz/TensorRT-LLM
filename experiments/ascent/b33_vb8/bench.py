"""Branch 33 (idea 39) timing: per-step FP8 attention sets, graph-timed, arms interleaved in-process.

The same script runs in both trees of run_gpu_pair. Arms built in one process (common.py):
base (main's served G3 levers), cand (base + v_b8t) and base2 (base + a no-op macro: own .so,
the A/A copy). In the baseline tree the G3 header has no v_b8t code, so cand == base there.
kernel_ms of a label = median replay of the tree's ROUTED arm (cand in the branch, base in main).

Labels (served 2048^2, T=80, knowledge 384 / 686, plan 724):
  t2i-2048  graph of 1 BF16-out G3 call at 16384 + 4 nvfp4_out at 16384 + 28 nvfp4_out at 16464
  i2i-2048  graph of 1 BF16-out G3 call at 16384 + 9 nvfp4_out at 16384 + 28 nvfp4_out at 32848
  i2i-2048-preL19 (optional) 6 BF16-out + 4 nvfp4_out at 16384 + 28 nvfp4_out at 32848
Each round times one block of replays per arm (both cyclic arm orders, rotated, so predecessors
are balanced over 12 rounds: pitfall 731) and samples SM MHz and board
W every 10 ms in a thread. Reported per round: CAND/BASE and BASE2/BASE (A/A) median ratios.
Inputs: fixed seeds, pristine copies restored before timing and compared after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as c  # noqa: E402

b = c.b
log = c.log
# label -> (BF16-out calls at S_img, nvfp4_out calls at S_img, S_img, S_full); + 28 nfo at S_full.
# i2i-2048 is the served set on main 8cc26728 (landing 19, COND_ATTN_NFO=1: plan 724);
# i2i-2048-preL19 is the older set (evidence only).
CHAINS = {"t2i-2048": (1, 4, 16384, 16464), "i2i-2048": (1, 9, 16384, 32848),
          "i2i-2048-preL19": (6, 4, 16384, 32848)}


class Sampler:
    def __init__(self):
        self.tag = None
        self.samples = []
        self.stop = False
        self.ok = True
        try:
            torch.cuda.power_draw()
            torch.cuda.clock_rate()
        except Exception as exc:  # noqa: BLE001
            log(f"[nvml] unavailable: {exc!r}")
            self.ok = False
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        while not self.stop and self.ok:
            t = self.tag
            if t is not None:
                try:
                    self.samples.append((t, time.time(), torch.cuda.power_draw() / 1000.0,
                                         torch.cuda.clock_rate()))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.01)

    def window(self, tag, t0, t1):
        xs = [s for s in self.samples if s[0] == tag and t0 <= s[1] <= t1]
        if not xs:
            return None, None
        return (round(sum(x[2] for x in xs) / len(xs), 1), round(sum(x[3] for x in xs) / len(xs)))


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="t2i-2048,i2i-2048")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--block_s", type=float, default=1.5)
    p.add_argument("--drop_s", type=float, default=0.3)
    args = p.parse_args()
    lib = c.lib_mod()
    routed, levs = c.arm_levers(lib, b.make_qkv(4176, 7, False))
    routed_arm = "cand" if "v_b8t" in routed else "base"
    arms = ("base", "cand", "base2")
    log(f"[levers] routed={routed} routed_arm={routed_arm}")
    mods = c.build(lib, levs, arms)
    gscale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device="cuda")
    H, D = b.H, b.D
    sampler = Sampler()
    out = {"batch": {}, "inproc": {}, "routed": list(routed), "routed_arm": routed_arm,
           "levers": {k: list(v) for k, v in levs.items()}}
    for lab in [x for x in args.labels.split(",") if x]:
        n_bf16, n_img, s_img, s_full = CHAINS[lab]
        inps = {S: b.make_qkv(S, 900 + S, False) for S in (s_img, s_full)}
        graphs, keep = {}, []
        for a in arms:
            nouts = {S: b.NfoOut(S, post=4096) for S in (s_img, s_full)}
            obf = torch.empty((1, s_img, H, D), device="cuda", dtype=torch.bfloat16)
            keep += [nouts, obf]

            def body(a=a, nouts=nouts, obf=obf):
                for _ in range(n_bf16):
                    b.call_bf16(lib, mods[a]["bf16"], inps[s_img], obf)
                for _ in range(n_img):
                    b.call_nfo(lib, mods[a]["nfo"], inps[s_img], nouts[s_img], gscale)
                for _ in range(28):
                    b.call_nfo(lib, mods[a]["nfo"], inps[s_full], nouts[s_full], gscale)

            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                body()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            graphs[a] = g
        for S in inps:  # restore pristine inputs before timing
            inps[S]["guard"].payload.copy_(inps[S]["pristine"])
        # warm + replay length
        for a in arms:
            graphs[a].replay()
        torch.cuda.synchronize()
        t0 = time.time()
        graphs["base"].replay()
        torch.cuda.synchronize()
        rep_s = time.time() - t0
        nrep = max(3, int(args.block_s / rep_s))
        log(f"[{lab}] replay ~{rep_s * 1e3:.1f} ms, {nrep} replays per block")
        per = {a: [] for a in arms}
        rounds = []
        for r in range(args.rounds):
            # both cyclic orders (base->cand->base2 and base->base2->cand), each rotated, so every
            # arm has every predecessor equally often over 12 rounds (pitfall 731)
            cyc = arms if (r // 3) % 2 == 0 else (arms[0], arms[2], arms[1])
            order = [cyc[(i + r) % 3] for i in range(3)]
            row = {"order": order}
            for a in order:
                sampler.tag = a
                tb = time.time()
                evs = []
                for _ in range(nrep):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[a].replay()
                    e1.record()
                    evs.append((e0, e1))
                torch.cuda.synchronize()
                te = time.time()
                sampler.tag = None
                ts = [x.elapsed_time(y) for x, y in evs]
                # drop the replays of the first drop_s seconds of the block (clock settles)
                k = min(len(ts) - 1, int(args.drop_s / rep_s))
                ts = ts[k:]
                per[a] += ts
                w, mhz = sampler.window(a, tb + args.drop_s, te)
                row[a] = {"ms": round(med(ts), 4), "W": w, "MHz": mhz}
            row["cand/base"] = round(row["cand"]["ms"] / row["base"]["ms"], 5)
            row["base2/base"] = round(row["base2"]["ms"] / row["base"]["ms"], 5)
            rounds.append(row)
            log(f"[{lab}] round {r} {json.dumps(row)}")
        cb = [x["cand/base"] for x in rounds]
        aa = [x["base2/base"] for x in rounds]
        lo, hi = min(aa), max(aa)
        outside = sum(1 for x in cb if x < lo or x > hi)
        summ = {"cand/base_median": round(med(cb), 5), "cand/base_range": [min(cb), max(cb)],
                "aa_median": round(med(aa), 5), "aa_range": [lo, hi],
                "rounds_outside_aa": outside, "rounds": rounds,
                "arm_ms": {a: round(med(per[a]), 4) for a in arms}}
        out["inproc"][lab] = summ
        xs = per[routed_arm]
        m = med(xs)
        out["batch"][lab] = {"kernel_ms": round(m, 4),
                             "noise_floor": round((pctl(xs, 0.9) - pctl(xs, 0.1)) / m, 4),
                             "calls": n_bf16 + n_img + 28, "S_img": s_img, "S_full": s_full}
        log(f"[{lab}] summary {json.dumps({k: v for k, v in summ.items() if k != 'rounds'})}")
        for S in inps:
            if not torch.equal(inps[S]["guard"].payload, inps[S]["pristine"]):
                out.setdefault("inputs_changed", []).append(f"{lab}_S{S}")
        del graphs, keep, inps
        torch.cuda.empty_cache()
    sampler.stop = True
    out["output"] = {"class": "exact"}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
