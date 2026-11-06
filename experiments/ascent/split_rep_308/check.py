"""Idea 308 check: SM120 CSA2 split-key decode replicas in one Triton launch under capture.

Candidate tree only (needs csa2/split_decode_replicate.py). The reference is main's
code path in the same tree: the three ATen expand().contiguous() copies, which every
eager call runs and which a capture runs when replicate_split_queries is patched to
return None (``graph_ref``). Per layer kind (r2 / r1 / cand) and rows m:

 1. census: kernels of one replay of graph_new and graph_ref, and of one warm eager
    call; Triton cache size before/after capture (no JIT under capture); sync-debug
    warnings during capture; allocator segments created during capture.
 2. harness-matched sequence (warmup + iters replays, reset at every iteration as the
    timing loop does): before each replay the replica buffers and the recorded
    logits / scores clones are poisoned with FINITE sentinels; after it the replicas
    are compared with the ATen replicas, raw DeepGEMM logits, masked scores and int32
    selections with the eager reference, all bitwise.
 3. >= 8 advancing decode steps (visibility grows, q / scale / weights / key pages
    change in place), graph captured at warmup state: graph_new vs graph_ref bitwise
    (replicas, logits, scores, selections) and vs the eager reference (selections
    tie-aware, knowledge 627).
"""

import argparse
import json
import os
import sys
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import csa2_index_gather_bench as gb  # noqa: E402
import csa2_split_decode_bench as sdb  # noqa: E402
from b57_check import mutate, ref_replicas  # noqa: E402
from csa2_prefill_consumer_bench import tie_aware  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import (  # noqa: E402
    split_decode_replicate as sdr,
)

log = sdb.log
REC = {"reps": [], "logits": []}
FLAGS = {"disable": False, "record": False}
_orig_rep = indexer_mod.replicate_split_queries
_orig_mqa = indexer_mod._fp8_fp4_mqa_logits


def _rep(*a, **k):
    if FLAGS["disable"]:
        return None
    r = _orig_rep(*a, **k)
    if r is not None and FLAGS["record"]:
        REC["reps"].append(r)
    return r


def _mqa(*a, **k):
    out = _orig_mqa(*a, **k)
    if FLAGS["record"]:
        REC["logits"].append(out.clone())
    return out


indexer_mod.replicate_split_queries = _rep
indexer_mod._fp8_fp4_mqa_logits = _mqa


def bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def triton_cache_size():
    try:
        return sum(len(v[0]) for v in sdr._replicate_split_queries_kernel.device_caches.values())
    except Exception:  # noqa: BLE001
        return -1


def run_recorded(c):
    """One eager or captured call; returns (masked scores, raw logits) clones.

    The raw DeepGEMM output (clean_logits=False) is only defined on each replica
    block's scored sub-range (lane 0, ``chunk`` keys); the rest is never written, so
    the recorded logits are that region assembled to ``[rows, width]``, exactly as
    main's split-key assembly reads it.
    """
    REC["logits"].clear()
    c.hooked = []
    FLAGS["record"] = True
    try:
        c.new(record=True)
    finally:
        FLAGS["record"] = False
    raw = REC["logits"][0]
    m, width = c.m, c.width
    splits = indexer_mod._split_decode_count(m, width)
    block = indexer_mod._SPLIT_DECODE_BLOCK_Q
    chunk = width // splits
    stride = raw.stride(0)
    assembled = raw.as_strided((m, splits, chunk), (splits * block * stride, block * stride, 1))
    return c.hooked[0], assembled.contiguous().view(m, width)


def eager_ref(c):
    c.reset()
    sc, lg = run_recorded(c)
    torch.cuda.synchronize()
    return c.out.clone(), sc.clone(), lg.clone()


def capture(c, disable):
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    REC["reps"].clear()
    info = {}
    with torch.cuda.stream(stream):
        c.reset()
        c.new()
        stream.synchronize()
        n0 = triton_cache_size()
        seg0 = torch.cuda.memory_stats().get("segment.all.allocated", 0)
        FLAGS["disable"] = disable
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with torch.cuda.graph(graph, stream=stream):
                    sc, lg = run_recorded(c)
            finally:
                FLAGS["disable"] = False
                torch.cuda.set_sync_debug_mode("default")
        info["sync_warnings"] = [str(w.message)[:120] for w in caught
                                 if "sync" in str(w.message).lower()]
        info["triton_cache_before_after"] = [n0, triton_cache_size()]
        info["segments_during_capture"] = (
            torch.cuda.memory_stats().get("segment.all.allocated", 0) - seg0)
    torch.cuda.synchronize()
    reps = list(REC["reps"])
    REC["reps"].clear()
    return graph, sc, lg, reps, info


def poison(tensors):
    for t in tensors:
        if t.dtype == torch.int8:
            t.fill_(0x5A)
        elif t.dtype == torch.int32:
            t.fill_(0x1234567)
        else:
            t.fill_(-12345.5)


def reps_ok(c, rep):
    if rep is None:
        return False
    inp = c.inp
    m = c.m
    splits = indexer_mod._split_decode_count(m, c.width)
    reps = splits * indexer_mod._SPLIT_DECODE_BLOCK_Q
    rq, rs, rw = ref_replicas(inp["q"], inp["q_scale"], inp["weights"], reps)
    q, s, w = rep
    return bool(same(q, rq) and same(s, rs) and same(w, rw))


def kernels_of(fn):
    """Full kernel names -> launch counts of one call (torch.profiler CUDA events)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "memset" not in ev.name.lower() \
                and "memcpy" not in ev.name.lower():
            names[ev.name] = names.get(ev.name, 0) + 1
    return sum(names.values()), names


def classify(names):
    rep = sum(n for k, n in names.items() if "replicate_split_queries" in k)
    copies = sum(n for k, n in names.items() if "direct_copy" in k)
    return rep, copies


def run_case(kind, m, args, device, failures):
    c = sdb.Case(kind, m, args, device)
    info = dict(kind=kind, m=m, width=c.width,
                splits=indexer_mod._split_decode_count(m, c.width))
    for _ in range(3):
        c.reset()
        c.new()
    torch.cuda.synchronize()
    info["warm"] = sorted(list(k) for k in sdr._WARM)
    ref = eager_ref(c)
    g_new, sc_new, lg_new, reps_new, cap_new = capture(c, disable=False)
    g_ref, sc_ref, lg_ref, reps_ref, cap_ref = capture(c, disable=True)
    info["capture_new"] = cap_new
    info["capture_ref"] = cap_ref
    info["replicate_calls_captured"] = [len(reps_new), len(reps_ref)]
    rep = reps_new[0] if len(reps_new) == 1 else None
    if rep is None or reps_ref:
        failures.append(f"capture {kind} m{m}: replicate calls {info['replicate_calls_captured']}")
    if cap_new["triton_cache_before_after"][0] != cap_new["triton_cache_before_after"][1]:
        failures.append(f"JIT under capture {kind} m{m}")
    if cap_new["sync_warnings"]:
        failures.append(f"sync under capture {kind} m{m}")

    def replay(graph, sc, lg, rep_t):
        poison([sc, lg] + (list(rep_t) if rep_t else []))
        c.reset()
        graph.replay()
        torch.cuda.synchronize()
        return c.out.clone()

    def cmp(r, out, sc, lg):
        r_out, r_sc, r_lg = r
        return dict(sel=bool(torch.equal(out, r_out)), scores=same(sc, r_sc),
                    logits=same(lg, r_lg))

    # 2. harness-matched sequence: warmup + iters, reset at every iteration
    total = args.warmup + args.iters
    bad = []
    for it in range(total):
        out = replay(g_new, sc_new, lg_new, rep)
        r = cmp(ref, out, sc_new, lg_new)
        r["reps"] = reps_ok(c, rep)
        if not all(r.values()):
            bad.append(dict(it=it, **r))
    out_ref = replay(g_ref, sc_ref, lg_ref, None)
    ref_graph_ok = all(cmp(ref, out_ref, sc_ref, lg_ref).values())
    info["seq_iterations"] = total
    info["seq_bad"] = bad[:5]
    info["seq_bad_count"] = len(bad)
    info["ref_graph_matches_eager"] = ref_graph_ok
    if bad or not ref_graph_ok:
        failures.append(f"sequence {kind} m{m}: bad={len(bad)} ref_graph={ref_graph_ok}")
    # 3. advancing decode window
    gen = torch.Generator(device="cpu").manual_seed(7700 + 10 * m + len(kind))
    steps, ties = [], 0
    for step in range(args.steps):
        mutate(c, gen)
        er = eager_ref(c)
        out_n = replay(g_new, sc_new, lg_new, rep)
        rp = reps_ok(c, rep)
        sn, ln = sc_new.clone(), lg_new.clone()
        out_r = replay(g_ref, sc_ref, lg_ref, None)
        vs_ref = dict(sel=bool(torch.equal(out_n, out_r)), scores=same(sn, sc_ref),
                      logits=same(ln, lg_ref), reps=rp)
        sel_eager = bool(torch.equal(out_n, er[0]))
        if not sel_eager:
            _, _, mapped = c.new_keys()
            sel_eager = bool(tie_aware(er[1], mapped, er[0], out_n, sdb.TOPK)["ok"])
            ties += 1
        vs_eager = dict(sel=sel_eager, scores=same(sn, er[1]), logits=same(ln, er[2]))
        ok = all(vs_ref.values()) and all(vs_eager.values())
        if not ok:
            steps.append(dict(step=step, vs_graph_ref=vs_ref, vs_eager=vs_eager))
    info["window_steps"] = args.steps
    info["window_bad"] = steps[:4]
    info["window_eager_tie_steps"] = ties
    if steps:
        failures.append(f"window {kind} m{m}: {len(steps)} bad steps")
    # 1. census: one replay of each graph, one warm eager call
    n_new, k_new = kernels_of(g_new.replay)
    n_ref, k_ref = kernels_of(g_ref.replay)
    n_eag, k_eag = kernels_of(lambda: c.new())
    info["census"] = {
        "graph_new": {"launches": n_new, "replicate_copies": classify(k_new), "kernels": k_new},
        "graph_ref": {"launches": n_ref, "replicate_copies": classify(k_ref), "kernels": k_ref},
        "eager": {"launches": n_eag, "replicate_copies": classify(k_eag), "kernels": k_eag},
    }
    rn, cn = classify(k_new)
    rr, cr = classify(k_ref)
    re_, ce = classify(k_eag)
    # Eager also runs the assembly copy (another direct_copy) and mask_positions_.
    if not (rn == 1 and cr - cn == 3 and rr == 0 and re_ == 0 and ce >= 3 and n_new == n_ref - 2):
        failures.append(f"census {kind} m{m}: new {rn}/{cn}/{n_new} ref {rr}/{cr}/{n_ref} "
                        f"eager {re_}/{ce}/{n_eag}")
    log(json.dumps({k: v for k, v in info.items() if k != "census"}))
    log("census " + json.dumps({k: [v["launches"], v["replicate_copies"]]
                                for k, v in info["census"].items()}))
    del g_new, g_ref, c
    torch.cuda.empty_cache()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=36864)
    p.add_argument("--context", type=int, default=34816)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--heads", default="32")
    args = p.parse_args()
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)}")
    failures, cases = [], []
    for kind in ("r2", "r1", "cand"):
        for m in [int(x) for x in args.batches.split(",")]:
            cases.append(run_case(kind, m, args, device, failures))
    # Direct replica unit check at every head count / rows bucket, under capture.
    unit = []
    for heads in (16, 32, 64):
        for m in (1, 2, 4, 8, 16):
            for width in (16384, 18432, 36864):
                reps = indexer_mod._split_decode_count(m, width) * 4
                g = torch.Generator(device="cpu").manual_seed(heads * 100 + m)
                q = torch.randint(-128, 128, (m, heads, 64), generator=g, dtype=torch.int32
                                  ).to(torch.int8).to(device)
                s = torch.randint(0, 2**31 - 1, (m, heads), generator=g, dtype=torch.int32).to(device)
                w = torch.randn((m, heads), generator=g).to(device)
                sdr.warm(q, s, w, reps)
                gr = torch.cuda.CUDAGraph()
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    with torch.cuda.graph(gr, stream=st):
                        got = sdr.replicate_split_queries(q, s, w, reps)
                ok = got is not None
                if ok:
                    poison(list(got))
                    gr.replay()
                    torch.cuda.synchronize()
                    ok = all(same(a, b) for a, b in zip(got, ref_replicas(q, s, w, reps)))
                unit.append(dict(heads=heads, m=m, width=width, reps=reps, ok=ok))
                if not ok:
                    failures.append(f"unit heads{heads} m{m} w{width}")
                del gr
    log("unit " + json.dumps(unit))
    steps = {}
    for m in [int(x) for x in args.batches.split(",")]:
        by = {c["kind"]: c["census"] for c in cases if c["m"] == m}
        steps[str(m)] = {
            arm: {
                "launches": 3 * by["r2"][arm]["launches"] + by["r1"][arm]["launches"]
                + 4 * by["cand"][arm]["launches"],
                "replicate": 3 * by["r2"][arm]["replicate_copies"][0]
                + by["r1"][arm]["replicate_copies"][0] + 4 * by["cand"][arm]["replicate_copies"][0],
                "copies": 3 * by["r2"][arm]["replicate_copies"][1]
                + by["r1"][arm]["replicate_copies"][1] + 4 * by["cand"][arm]["replicate_copies"][1],
            } for arm in ("graph_new", "graph_ref", "eager")}
    print(json.dumps({
        "passed": not failures,
        "failures": failures[:20],
        "compared": "graph replay with the Triton replicate kernel vs main's ATen-copy path "
                    "(eager reference and the same capture with the kernel disabled): replica "
                    "buffers, raw DeepGEMM logits, masked scores, int32 selections bitwise, every "
                    "iteration of warmup+iters with finite-sentinel poisoning, plus an advancing "
                    "decode window; per-layer kernel census of graph/eager",
        "launches_per_step_8_layers": steps,
        "unit": unit,
        "cases": [{k: v for k, v in c.items() if k != "census"} for c in cases],
    }), flush=True)


if __name__ == "__main__":
    sys.exit(main())
