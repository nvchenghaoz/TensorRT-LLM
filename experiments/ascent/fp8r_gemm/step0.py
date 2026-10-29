"""STEP 0(a2) of idea 122: baseline split of image.0's FP8R torch._scaled_mm GEMMs at M=4096.

Torch only, synthetic tensors at the real shapes, no model load.
Per shape: WARM (one buffer set) and EVICTED (A, W, out rotated over >= 6 sets, > 128 MB),
graph of 10 calls, >= 100 rounds, median/p10/p90 per call, an A/A row, kernel name + grid.
Also the real trtllm::flux3_sm120_fp8r_out_pair_q wall and down/attn_out alone in the same form.
"""

import json
import os
import subprocess
import sys
import time

import torch

M = int(os.environ.get("STEP0_M", "4096"))
ROUNDS = int(os.environ.get("STEP0_ROUNDS", "120"))
CALLS = 10
MAIN_AT_START = os.environ.get("MAIN_AT_START", "0761b30b")
SHAPES = {  # name: (N, K)
    "gate_up": (18432, 3072),
    "qkv": (9216, 3072),
    "down": (3072, 9216),
    "attn_out": (3072, 3072),
}
FLOOR_US = {"gate_up": 573.6, "qkv": 286.8, "down": 286.8, "attn_out": 95.6}
L2_BYTES = 128 << 20


def log(*a):
    print(*a, flush=True)


def header():
    wt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def git(*args):
        try:
            return subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"err {e}"

    head = git("rev-parse", "HEAD")
    rc = subprocess.run(["git", "-C", wt, "merge-base", "--is-ancestor", MAIN_AT_START, "HEAD"]).returncode
    import tensorrt_llm

    log(json.dumps({"HEAD": head, "MAIN_AT_START": MAIN_AT_START, "trtllm": tensorrt_llm.__file__,
                    "is_ancestor_rc": rc, "gpu": torch.cuda.get_device_name()}))


def fp8(shape, gen):
    return (torch.randn(shape, device="cuda", generator=gen) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)


def make_sets(n, k, nsets, gen):
    sets = []
    for _ in range(nsets):
        a = fp8((M, k), gen)
        w = fp8((n, k), gen)
        sa = torch.rand(M, device="cuda", generator=gen) * 0.01 + 1e-3
        sb = torch.rand(n, device="cuda", generator=gen) * 0.01 + 1e-3
        sets.append((a, w, sa, sb))
    return sets


def nsets_for(n, k):
    per = M * k + n * k + 2 * M * n
    return max(6, -(-int(1.25 * L2_BYTES) // per) + 1)


def scaled_mm(a, w, sa, sb):
    return torch._scaled_mm(a, w.t(), scale_a=sa.reshape(-1, 1), scale_b=sb.reshape(1, -1),
                            out_dtype=torch.bfloat16)


def time_graph(fns, rounds=ROUNDS):
    """fns: list of CALLS closures captured into one graph; returns per-call us stats."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [f() for f in fns]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    for i in range(rounds):
        st[i].record()
        g.replay()
        en[i].record()
    torch.cuda.synchronize()
    t = sorted(st[i].elapsed_time(en[i]) * 1000.0 / len(fns) for i in range(rounds))
    del outs, g
    return {"med": t[len(t) // 2], "p10": t[len(t) // 10], "p90": t[(9 * len(t)) // 10]}


def kernel_info(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = f"/tmp/step0_trace_{os.getpid()}.json"
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    ks = []
    for e in tr.get("traceEvents", []):
        if e.get("cat") == "kernel":
            args = e.get("args", {})
            ks.append({"name": e["name"][:160], "grid": args.get("grid"), "block": args.get("block"),
                       "regs": args.get("registers per thread"), "smem": args.get("shared memory"),
                       "dur_us": e.get("dur")})
    return ks


def main():
    header()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    rows = {}
    for name, (n, k) in SHAPES.items():
        ns = nsets_for(n, k)
        sets = make_sets(n, k, ns, gen)
        log(f"[{name}] N={n} K={k} M={M} evict_sets={ns} "
            f"bytes/set={(M * k + n * k + 2 * M * n) / 1e6:.1f}MB")
        ki = kernel_info(lambda: scaled_mm(*sets[0]))
        warm = time_graph([(lambda s=sets[0]: scaled_mm(*s)) for _ in range(CALLS)])
        warm_aa = time_graph([(lambda s=sets[0]: scaled_mm(*s)) for _ in range(CALLS)])
        ev = time_graph([(lambda s=sets[i % ns]: scaled_mm(*s)) for i in range(CALLS)])
        ev_aa = time_graph([(lambda s=sets[(i + 3) % ns]: scaled_mm(*s)) for i in range(CALLS)])
        rows[name] = {"warm": warm, "warm_aa": warm_aa, "evicted": ev, "evicted_aa": ev_aa,
                      "kernels": ki, "floor_us": FLOOR_US[name]}
        log(f"[{name}] kernel={ki} ")
        log(f"[{name}] WARM {warm} A/A {warm_aa}")
        log(f"[{name}] EVICTED {ev} A/A {ev_aa}")
        del sets
        torch.cuda.empty_cache()

    # The real tail-pair op (down on current stream || attn_out with its quantize on a side stream).
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

    pair_op = torch.ops.trtllm.flux3_sm120_fp8r_out_pair_q
    nd, kd = SHAPES["down"]
    na, ka = SHAPES["attn_out"]
    per = M * kd + nd * kd + M * ka * 2 + na * ka + 2 * M * (nd + na)
    ns = max(6, -(-int(1.25 * L2_BYTES) // per) + 1)
    psets = []
    for _ in range(ns):
        dd = fp8((M, kd), gen)
        ds = torch.rand(M, device="cuda", generator=gen) * 0.01 + 1e-3
        dw = fp8((nd, kd), gen)
        dws = torch.rand(nd, device="cuda", generator=gen) * 0.01 + 1e-3
        at = torch.randn((M, ka), device="cuda", generator=gen).to(torch.bfloat16)
        aw = fp8((na, ka), gen)
        aws = torch.rand(na, device="cuda", generator=gen) * 0.01 + 1e-3
        psets.append((dd, ds, dw, dws, at, aw, aws))

    def pair(s):
        return pair_op(*s)

    def down_alone(s):
        return scaled_mm(s[0], s[2], s[1], s[3])

    def attn_alone(s):
        d, sc = quantize_fp8r(s[4])
        return scaled_mm(d.contiguous(), s[5], sc.float().reshape(-1).contiguous(), s[6])

    prow = {}
    for label, fn in (("pair", pair), ("down_alone", down_alone), ("attn_out_q_alone", attn_alone)):
        try:
            warm = time_graph([(lambda s=psets[0]: fn(s)) for _ in range(CALLS)])
            ev = time_graph([(lambda s=psets[i % ns]: fn(s)) for i in range(CALLS)])
            ev_aa = time_graph([(lambda s=psets[(i + 3) % ns]: fn(s)) for i in range(CALLS)])
        except Exception as e:  # noqa: BLE001
            log(f"[{label}] graph failed: {e!r}")
            warm = ev = ev_aa = None
        prow[label] = {"warm": warm, "evicted": ev, "evicted_aa": ev_aa}
        log(f"[{label}] WARM {warm} EVICTED {ev} A/A {ev_aa}")
    log("[pair] kernels " + json.dumps(kernel_info(lambda: pair(psets[0]))))

    # Table: gate input = EVICTED (in-model rows of profile 1305 are not reachable from this branch).
    gaps = {}
    for name, r in rows.items():
        gate = r["evicted"]["med"]
        gaps[name] = gate - FLOOR_US[name]
        log(f"| {name} | gate(EVICTED) {gate:.1f} | WARM {r['warm']['med']:.1f} | EVICTED "
            f"{r['evicted']['med']:.1f} (p10 {r['evicted']['p10']:.1f} p90 {r['evicted']['p90']:.1f}) | "
            f"in-model n/a | floor {FLOOR_US[name]} | gap {gaps[name]:.1f} | source torch._scaled_mm graph x10 |")
    gap_main = 16 * sum(gaps.values()) / 1000.0
    pw = prow["pair"]["evicted"]["med"] if prow["pair"]["evicted"] else None
    da = prow["down_alone"]["evicted"]["med"] if prow["down_alone"]["evicted"] else None
    hide = (pw - da) if (pw is not None and da is not None) else gaps["attn_out"]
    countable = 16 * (gaps["gate_up"] + gaps["qkv"] + min(gaps["attn_out"], hide)) / 1000.0
    res = {"M": M, "gaps_us": gaps, "gap_main_ms_per_req": gap_main,
           "pair_wall_us": pw, "down_alone_us": da, "pair_minus_down_us": hide,
           "countable_ms_per_req": countable, "stop": gap_main <= 1.85,
           "flag": gap_main > 1.85 and countable <= 0.75}
    log(json.dumps({"rows": rows, "pair": prow}, default=str))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"done {time.time() - t0:.1f}s")
