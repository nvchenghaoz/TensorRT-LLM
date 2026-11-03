"""Branch 42 check: one-launch query replication in the SM120 split-key decode indexer.

Runs in the candidate tree. The reference is main's path, kept here verbatim
(``ref_split_key_decode_logits``: three ``expand(...).contiguous()`` replica
copies, DeepGEMM, as_strided assembly). For ratio-2, ratio-1 and candidate
layers at m = 1/2/4/8 (last row is a padding row, token_requests = -1):

1. replica bytes: new kernel (NaN/0xFF-prefilled destinations, and its own
   allocation, and non-contiguous inputs) vs the expand copies, bitwise;
2. raw logits: new ``_split_key_decode_logits`` vs reference, bitwise;
3. eager select_split_decode new (x2) vs reference: hook-masked scores bitwise,
   selections torch.equal (tie-aware only on exact K-boundary ties);
4. CUDA graphs of the reference path and of the new path, both captured at a
   warmup state, then a 10-step decode window with visible lengths advancing and
   q / q_scale / weights / key pages changing in place; at every step both graphs
   replay (with the timing loop's reset point) and are compared with each other
   and with the eager reference; then warmup+iters replays at the final state;
5. launch counts and per-kernel times of one layer, reference vs new.
"""

import argparse
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csa2_split_decode_bench as sdb  # noqa: E402
from csa2_prefill_consumer_bench import tie_aware  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.split_decode_replicate import (  # noqa: E402
    replicate_split_queries,
)

log = sdb.log


def ref_replicas(q_data, q_scale, weights, reps):
    rows, heads = q_data.shape[0], q_data.shape[1]
    q = q_data.view(torch.int8)
    q = q[:, None].expand(rows, reps, heads, q.shape[2]).contiguous()
    q = q.view(rows * reps, heads, q.shape[3])
    scale = q_scale.contiguous().view(torch.int32).reshape(rows, heads)
    scale = scale[:, None].expand(rows, reps, heads).contiguous().view(rows * reps, heads)
    w = weights[:, None].expand(rows, reps, heads).contiguous().view(rows * reps, heads)
    return q, scale, w


def ref_split_key_decode_logits(self, q_data, q_scale, weights, k_data, k_scale, width):
    """main's _split_key_decode_logits, verbatim."""
    rows, heads = q_data.shape[0], q_data.shape[1]
    splits = indexer_mod._split_decode_count(rows, width)
    block = indexer_mod._SPLIT_DECODE_BLOCK_Q
    chunk = width // splits
    reps = splits * block
    bounds = self._split_decode_bounds(rows, width, splits, q_data.device)
    q, scale, w = ref_replicas(q_data, q_scale, weights, reps)
    out = indexer_mod._fp8_fp4_mqa_logits(
        (q, scale),
        (k_data.view(torch.int8), k_scale.contiguous().view(torch.int32).reshape(-1)),
        w, bounds[0], bounds[1], clean_logits=False, max_seqlen_k=chunk,
    )
    stride = out.stride(0)
    return (out.as_strided((rows, splits, chunk), (reps * stride, block * stride, 1))
            .contiguous().view(rows, width))


def bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride() and \
        torch.equal(bits(a), bits(b))


def use_ref(c, on):
    if on:
        c.indexer._split_key_decode_logits = types.MethodType(ref_split_key_decode_logits,
                                                              c.indexer)
    else:
        c.indexer.__dict__.pop("_split_key_decode_logits", None)


def sel_ok(scores, mapped, a, b):
    if torch.equal(a, b):
        return True, None
    t = tie_aware(scores, mapped, a, b, sdb.TOPK)
    return bool(t["ok"]), t


def capture(c, ref):
    use_ref(c, ref)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        c.reset()
        c.new()
        stream.synchronize()
        c.hooked = []
        with torch.cuda.graph(graph, stream=stream):
            c.new(record=True)
    torch.cuda.synchronize()
    scores = c.hooked[0]
    use_ref(c, False)
    return graph, scores


def mutate(c, g):
    inp = c.inp
    m = c.m
    inp["visible"].add_(1)
    inp["q"].copy_(torch.randint(-128, 128, inp["q"].shape, generator=g, dtype=torch.int32)
                   .to(torch.int8))
    qs = torch.randint(122, 129, (m, inp["q_scale"].shape[1], 4), generator=g,
                       dtype=torch.int32).to(torch.uint8).view(torch.int32).reshape(m, -1)
    inp["q_scale"].copy_(qs)
    inp["weights"].copy_(torch.randn(inp["weights"].shape, generator=g))
    # perturb a few key bytes (data region) in place
    pages = inp["pages"].reshape(inp["pages"].shape[0], -1)  # [pages, 4096 data + 256 scale]
    npages, per = pages.shape
    pages = pages.view(-1)
    idx = torch.randint(0, npages, (4096,), generator=g) * per + torch.randint(
        0, 4096, (4096,), generator=g)
    pages.index_copy_(0, idx.to(pages.device),
                      torch.randint(0, 256, (4096,), generator=g, dtype=torch.int32)
                      .to(torch.uint8).to(pages.device))


def run_case(kind, m, args, device, failures):
    c = sdb.Case(kind, m, args, device)
    inp = c.inp
    rows, width = m, c.width
    splits = indexer_mod._split_decode_count(rows, width)
    reps = splits * indexer_mod._SPLIT_DECODE_BLOCK_Q
    info = dict(kind=kind, m=m, width=width, splits=splits, reps=reps)
    # 1. replica bytes
    rq, rs, rw = ref_replicas(inp["q"], inp["q_scale"], inp["weights"], reps)
    pre = (torch.full_like(rq, -1), torch.full_like(rs, -1), torch.full_like(rw, float("nan")))
    nq, ns, nw = replicate_split_queries(inp["q"], inp["q_scale"], inp["weights"], reps, out=pre)
    aq, as_, aw = indexer_mod.CSA2Indexer._split_decode_replicas(
        inp["q"], inp["q_scale"], inp["weights"], reps)
    # non-contiguous inputs
    q_nc = torch.empty((m, inp["q"].shape[1], 2 * inp["q"].shape[2]), dtype=torch.int8,
                       device=device)[..., ::2]
    q_nc.copy_(inp["q"])
    w_nc = torch.empty((m, 2 * inp["weights"].shape[1]), device=device)[:, 1::2]
    w_nc.copy_(inp["weights"])
    s_nc = torch.empty((inp["q_scale"].shape[1], m), dtype=torch.int32, device=device).t()
    s_nc.copy_(inp["q_scale"])
    cq, cs, cw = replicate_split_queries(q_nc, s_nc, w_nc, reps)
    rep_ok = all(same(a, b) for a, b in ((nq, rq), (ns, rs), (nw, rw), (aq, rq), (as_, rs),
                                         (aw, rw), (cq, rq), (cs, rs), (cw, rw)))
    info["replica_bitwise"] = rep_ok
    if not rep_ok:
        failures.append(f"replicas {kind} m{m}")
    # 2. raw logits
    keys, scales, mapped = c.new_keys()
    new_l = c.indexer._split_key_decode_logits(inp["q"], inp["q_scale"], inp["weights"], keys,
                                               scales, width)
    ref_l = ref_split_key_decode_logits(c.indexer, inp["q"], inp["q_scale"], inp["weights"],
                                        keys, scales, width)
    raw_ok = same(new_l, ref_l)
    info["raw_logits_bitwise"] = raw_ok
    if not raw_ok:
        failures.append(f"raw logits {kind} m{m}")

    def eager(ref):
        use_ref(c, ref)
        c.hooked = []
        c.reset()
        out = c.new(record=True).clone()
        sc = c.hooked[0].clone()
        use_ref(c, False)
        return out, sc

    # 3. eager
    r_out, r_sc = eager(True)
    n_out, n_sc = eager(False)
    n2_out, n2_sc = eager(False)
    ok1, t1 = sel_ok(r_sc, mapped, r_out, n_out)
    ok2, _ = sel_ok(r_sc, mapped, n_out, n2_out)
    eager_ok = ok1 and ok2 and same(n_sc, r_sc) and same(n2_sc, n_sc)
    info["eager_ok"] = eager_ok
    info["eager_sel_torch_equal"] = bool(torch.equal(r_out, n_out))
    if t1:
        info["eager_tie"] = t1
    if not eager_ok:
        failures.append(f"eager {kind} m{m}")
    # 4. graphs captured at the warmup state, replayed on an advancing decode window
    g_ref, g_ref_sc = capture(c, True)
    g_new, g_new_sc = capture(c, False)
    gen = torch.Generator(device="cpu").manual_seed(9000 + 10 * m + len(kind))
    step_bad, ties = [], 0
    steps = 10
    for step in range(steps):
        mutate(c, gen)
        _, _, mp = c.new_keys()
        e_out, e_sc = eager(True)
        c.reset()
        g_ref.replay()
        torch.cuda.synchronize()
        gr_out, gr_sc = c.out.clone(), g_ref_sc.clone()
        c.reset()
        g_new.replay()
        torch.cuda.synchronize()
        gn_out, gn_sc = c.out.clone(), g_new_sc.clone()
        a, ta = sel_ok(e_sc, mp, e_out, gn_out)
        b, tb = sel_ok(e_sc, mp, gr_out, gn_out)
        ties += int(not torch.equal(e_out, gn_out))
        if not (a and b and same(gn_sc, e_sc) and same(gn_sc, gr_sc)):
            step_bad.append(dict(step=step, sel_vs_eager=a, sel_vs_refgraph=b,
                                 sc_vs_eager=same(gn_sc, e_sc), sc_vs_refgraph=same(gn_sc, gr_sc),
                                 refgraph_vs_eager_sc=same(gr_sc, e_sc)))
    info["window_steps"] = steps
    info["window_bad_steps"] = step_bad
    info["window_tie_rows_steps"] = ties
    if step_bad:
        failures.append(f"window {kind} m{m}: {step_bad[:2]}")
    # replays at the final state, timing loop's warmup/iteration/reset points
    e_out, e_sc = eager(True)
    _, _, mp = c.new_keys()
    bad, total = 0, args.warmup + args.iters
    for _ in range(total):
        c.reset()
        g_new.replay()
        torch.cuda.synchronize()
        ok, _ = sel_ok(e_sc, mp, e_out, c.out)
        if not (ok and same(g_new_sc, e_sc)):
            bad += 1
    info["graph_bad_replays"] = bad
    info["graph_replays"] = total
    if bad:
        failures.append(f"graph {kind} m{m}: {bad}/{total}")
    del g_ref, g_new
    # 5. launch counts of one layer, reference vs new (warm eager)
    from torch.profiler import ProfilerActivity, profile
    counts = {}
    for name, ref in (("main", True), ("new", False)):
        use_ref(c, ref)
        for _ in range(2):
            c.reset()
            c.new()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            c.new()
            torch.cuda.synchronize()
        use_ref(c, False)
        kernels = {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA and "memset" not in ev.name.lower():
                k = ev.name[:70]
                n, t = kernels.get(k, (0, 0.0))
                kernels[k] = (n + 1, t + float(getattr(ev, "device_time_total", 0.0)))
        counts[name] = dict(launches=sum(n for n, _ in kernels.values()),
                            gpu_us=round(sum(t for _, t in kernels.values()), 1),
                            kernels={k: [n, round(t / n, 2)] for k, (n, t) in kernels.items()})
    info["counts"] = counts
    log(json.dumps({k: v for k, v in info.items() if k != "counts"}))
    log(f"  launches main={counts['main']['launches']} new={counts['new']['launches']} "
        f"gpu_us main={counts['main']['gpu_us']} new={counts['new']['gpu_us']}")
    log("  main: " + json.dumps(counts["main"]["kernels"]))
    log("  new:  " + json.dumps(counts["new"]["kernels"]))
    del c
    torch.cuda.empty_cache()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=36864)
    p.add_argument("--context", type=int, default=34816)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)}")
    failures, cases = [], []
    for kind in ("r2", "r1", "cand"):
        for m in [int(x) for x in args.batches.split(",")]:
            cases.append(run_case(kind, m, args, device, failures))
    steps = {}
    for m in [int(x) for x in args.batches.split(",")]:
        by = {c["kind"]: c["counts"] for c in cases if c["m"] == m}
        steps[str(m)] = {p_: 3 * by["r2"][p_]["launches"] + by["r1"][p_]["launches"]
                         + 4 * by["cand"][p_]["launches"] for p_ in ("main", "new")}
    print(json.dumps({
        "passed": not failures, "failures": failures[:20],
        "compared": "branch replication kernel vs main's expand(...).contiguous() replicas "
                    "(bytes), main's _split_key_decode_logits (raw logits), main's "
                    "select_split_decode path (hook-masked scores bitwise, int32 selections "
                    "torch.equal / tie-aware on K-boundary ties), eager x2, 10-step advancing "
                    "decode window on graphs captured at warmup state, warmup+iters replays",
        "launches_per_step_8_layers": steps,
        "cases": [{k: v for k, v in c.items() if k != "counts"} for c in cases]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
