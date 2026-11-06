"""Idea 211 claim B, gate G1: the FFN-input boundary chain with the K1 fold OFF (decode M = 3/4/8).

One GPU, one MoE rank (EP4 rank 0: 96 x 2304, slot 0), 40 layers in one CUDA graph, cold L2 (512 MB flush read
before every replay; events bracket the replay only). Per layer distinct router weights / biases / K1 inputs:

  stand-in attention AR (router_prewait/pdl_ext.py producer, 4 CTAs x 256, spins W ns from its start, then writes
  x_attn; triggers its dependents at its start when the tree's attention AR triggers early, M <= 4, else at its end)
  -> hc_ffn K1 (mhc_lagged_layer_input, PDL) -> dsv4_fused_router (Stage 1, M > 2) -> Sm12xW4a8DecodeState.run

This is main's production schedule at M = 3..8 (K1_FOLD_MAX_TOKENS = 2). W = W_crit 2.9 us (fact 3069, job-1581
trace, c4 last-arriver AR minus the stand-in overhead; plan 3225 names it for 211-B), override K1G1_SPIN_NS.
Arms: DEFAULT (the tree's K1: fast on the branch, main's K1 on the base) and, when the tree has the fast K1,
REF (fast=False, main's kernel) ABBA-interleaved in the same job. kernel_ms = DEFAULT.
Checks (every arm, after warmup and after the last timed replay): every layer's layer_input bitwise == eager main
K1 (fast=False, no PDL) on the same inputs; router ids/weights/logits bitwise == eager router on that reference.
"""
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fused_router"))
sys.path.insert(0, os.path.join(HERE, "..", "router_prewait"))
from common import HIDDEN, LAYERS, TOPK, biases, log, make_decode_state, router_weights  # noqa: E402

import torch  # noqa: E402

import pdl_ext  # noqa: E402

K1_P = int(os.environ.get("K1G1_P", "224"))
SPIN_NS = int(os.environ.get("K1G1_SPIN_NS", "2900"))
EPS = 1e-6
SCALE = 1.5
WARMUP = 5
MS = [int(v) for v in os.environ.get("K1G1_MS", "3,4,8").split(",")]
SAMPLES, REPS = int(os.environ.get("K1G1_SAMPLES", "40")), 3
EARLY_MAX_M = 4  # TRTLLM_DSV4_ATTN_AR_EARLY_TRIGGER: the attention AR triggers early at M <= 4


def make_k1_inputs(m, gen, dev):
    x = torch.randn(m, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    res = torch.randn(m, 4, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    post = (2.0 * torch.rand(m, 4, generator=gen, device=dev)).float().contiguous()
    comb = torch.rand(m, 4, 4, generator=gen, device=dev)
    comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
    pre = (torch.rand(m, 4, generator=gen, device=dev) + 1e-6).float().contiguous()
    nw = (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen, device=dev)).to(torch.bfloat16)
    return x, res, post, comb, pre, nw


def bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else (
        t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32))


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, out


def stats(v):
    v = sorted(v)
    med = v[len(v) // 2]
    p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
    return med, p10, p90, (p90 - p10) / med


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as S

    ops = torch.ops.trtllm
    has_fast = "fast" in inspect.signature(S.mhc_lagged_layer_input).parameters
    tree = "CAND" if has_fast else "BASE"
    dev = "cuda"
    ext = pdl_ext.ext()
    log("TREE", tree, torch.cuda.get_device_name(), "K1 P", K1_P, "W ns", SPIN_NS, "Ms", MS,
        "K1_FAST" if has_fast and S.K1_FAST else "main K1")
    gen = torch.Generator(device=dev).manual_seed(21101)
    rws = router_weights(gen, LAYERS)
    bs = biases(gen, LAYERS)
    counters = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(LAYERS)]
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev).fill_(1.0)
    state, _dec = make_decode_state(96, 2304, 0)
    batch, secondary = {}, {}
    all_ok = True
    for m in MS:
        early = m <= EARLY_MAX_M
        x0 = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        i0 = (torch.arange(m * TOPK, device=dev, dtype=torch.int32) % 96).view(m, TOPK)
        state.run(x0, i0, torch.full((m, TOPK), 1.0 / TOPK, device=dev))
        k1in = [make_k1_inputs(m, gen, dev) for _ in range(LAYERS)]
        srcs = [t[0] for t in k1in]
        xattn = [torch.empty_like(s) for s in srcs]
        empty = pdl_ext.empty()
        arms = [("DEFAULT", {})]
        if has_fast:
            arms.append(("REF", {"fast": False}))
        li_by = {n: [torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
                 for n, _ in arms}

        def make_step(name, kw):
            lis = li_by[name]

            def step():
                outs = []
                for li in range(LAYERS):
                    ext.producer(srcs[li], xattn[li], SPIN_NS, 4, 256, empty, early)
                    _, res, post, comb, pre, nw = k1in[li]
                    x = lis[li]
                    S.mhc_lagged_layer_input(xattn[li], res, post, comb, pre, nw, EPS, K1_P, enable_pdl=True,
                                             out=x, **kw)
                    ids, wts, lg = ops.dsv4_fused_router(x, rws[li], bs[li], counters[li], SCALE)
                    o = state.run(x, ids, wts)
                    outs.append((lg, ids, wts, o))
                return outs
            return step

        graphs = {n: capture(make_step(n, kw)) for n, kw in arms}

        def check(tag):
            torch.cuda.synchronize()
            bad = {}
            for n, _ in arms:
                for xa in xattn:
                    xa.fill_(float("nan"))
                for t in li_by[n]:
                    t.fill_(float("nan"))
                graphs[n][0].replay()
                torch.cuda.synchronize()
                b = {"li": 0, "logits": 0, "ids": 0, "weights": 0, "moe_nonfinite": 0}
                for li in range(LAYERS):
                    kw = {"fast": False} if has_fast else {}
                    ref = S.mhc_lagged_layer_input(srcs[li], *k1in[li][1:], EPS, K1_P, enable_pdl=False, **kw)
                    rids, rwts, rlg = ops.dsv4_fused_router(ref, rws[li], bs[li],
                                                            torch.zeros(1, dtype=torch.int32, device=dev), SCALE)
                    lg, ids, wts, o = graphs[n][1][li]
                    b["li"] += int(not same(li_by[n][li], ref))
                    b["logits"] += int(not same(lg, rlg))
                    b["ids"] += int(not torch.equal(ids, rids))
                    b["weights"] += int(not same(wts, rwts))
                    b["moe_nonfinite"] += int(not bool(torch.isfinite(o.float()).all()))
                b["counters"] = sum(int(c.item() != 0) for c in counters)
                bad[n] = b
            ok = not any(v for b in bad.values() for v in b.values())
            log(f"CHECK M={m} {tag}: {bad} -> {'OK' if ok else 'FAIL'}")
            return ok, bad

        def timed(name):
            flush.sum()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[name][0].replay()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)

        names = [a[0] for a in arms]
        for _ in range(WARMUP):
            for n in names:
                timed(n)
        ok_w, bad_w = check(f"after {WARMUP} warmup replays")
        ts = {n: [] for n in names}
        for it in range(SAMPLES):
            order = names if it % 2 == 0 else names[::-1]  # ABBA
            for n in order:
                ts[n].append(sum(timed(n) for _ in range(REPS)) / REPS)
        ok_e, bad_e = check(f"after last of {SAMPLES}x{REPS} timed replays")
        all_ok = all_ok and ok_w and ok_e
        res = {}
        for n in names:
            med, p10, p90, noise = stats(ts[n])
            res[n] = {"kernel_ms": med, "noise_floor": noise, "us_layer": med * 1000.0 / LAYERS,
                      "p10_us_layer": p10 * 1000.0 / LAYERS, "p90_us_layer": p90 * 1000.0 / LAYERS}
            log(f"{tree} G1 M={m} {n}: median {med:.4f} ms = {med * 1000.0 / LAYERS:.3f} us/layer "
                f"[p10 {p10 * 1000.0 / LAYERS:.3f}, p90 {p90 * 1000.0 / LAYERS:.3f}] noise {noise:.4%}")
        batch[f"G1_M{m}"] = {"kernel_ms": res["DEFAULT"]["kernel_ms"], "noise_floor": res["DEFAULT"]["noise_floor"],
                             "us_layer": res["DEFAULT"]["us_layer"], "p10_us_layer": res["DEFAULT"]["p10_us_layer"],
                             "p90_us_layer": res["DEFAULT"]["p90_us_layer"], "W_us": SPIN_NS / 1000.0,
                             "ar_trigger": "early" if early else "end"}
        if "REF" in res:
            batch[f"G1_M{m}"]["in_job_ref_us_layer"] = res["REF"]["us_layer"]
            batch[f"G1_M{m}"]["in_job_gain_vs_ref"] = 1.0 - res["DEFAULT"]["us_layer"] / res["REF"]["us_layer"]
        secondary[str(m)] = {"arms": res, "check_after_warmup": bad_w, "check_after_timed": bad_e}
        graphs.clear()
        torch.cuda.empty_cache()
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "tree": tree, "checks_passed": all_ok,
                      "k1_p": K1_P, "w_ns": SPIN_NS, "secondary": secondary}))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
