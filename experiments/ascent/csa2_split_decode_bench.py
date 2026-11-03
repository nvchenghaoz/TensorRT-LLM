"""CSA2 SM120 decode indexer: per-request chain vs one split-key pass over all rows.

Runs in either tree. The base tree has only the per-request path production runs
there (one fused key gather + ``CSA2Indexer.forward_prepared`` per decode row:
DeepGEMM fp8_fp4_mqa_logits, mask, Top-K, finalize). The candidate tree adds
``CSA2Indexer.select_split_decode`` (one gather for all rows, one split-key
logits launch, one assembly copy, one mask / Top-K / finalize), which
``sparse_attn_indexer`` now calls for SM120 decode rows.

Timing (default): one TP4 rank's decode indexer per step at batch m (V4.1
layout: Full layers 2/8/14 at ratio 2 (18432 keys), Full layer 20 at ratio 1
(36864 keys, candidate source), Reindex consumers 24/28/32/36 over 16384
candidates), captured in one CUDA graph, 256 MB L2 flush before every replay,
Top-K scratch and outputs reset from pristine copies at every reset point.
Components are timed the same way. A prefill call (2048 rows) checks the
prefill path is unchanged.

``--check`` (candidate tree): for ratio 1, ratio 2 and candidate tiles at
m = 1/2/4/8, compares bitwise against the per-request production path:
raw logits over each row's keys, masked scores, int32 selections; then replays
the new path in a CUDA graph with the timing loop's warmup/iteration/reset
points and compares every replay.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csa2_index_gather_bench as gb  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as kernel_mod  # noqa: E402

NEW = hasattr(indexer_mod.CSA2Indexer, "select_split_decode")
TOPK = gb.TOPK


def log(*a):
    print(*a, flush=True)


class Case:
    """One layer kind at batch m: inputs, scratch and both paths."""

    def __init__(self, kind, m, args, device):
        self.kind, self.m = kind, m
        ratio = 2 if kind == "r2" else 1
        self.inp = gb.make_inputs(
            {"r2": 11, "r1": 22, "cand": 33}[kind] * 100 + m, ratio, args.tokens_per_block,
            args.max_seq_len, m, args.context, device,
        )
        self.indexer = gb.make_indexer(ratio)
        self.cand = None
        if kind == "cand":
            g = torch.Generator(device="cpu").manual_seed(4000 + m)
            cwidth = min(2048, -(-self.inp["max_positions"] // 8)) * 8
            cand = torch.randint(0, args.context + 1, (m, cwidth), generator=g, dtype=torch.int32)
            cand[:, ::7] = -1
            self.cand = cand.to(device)
        self.width = self.cand.shape[1] if self.cand is not None else self.inp["width"]
        self.out = torch.empty((m, TOPK), dtype=torch.int32, device=device)
        self.radix_i = torch.empty((m, 10, TOPK), dtype=torch.int32, device=device)
        self.radix_l = torch.empty((m, 10, TOPK), dtype=torch.float32, device=device)
        self.hooked = []

    def reset(self):
        self.out.fill_(-7)
        self.radix_i.fill_(-3)
        self.radix_l.fill_(1.5)

    def hook_fn(self, record):
        if not record:
            return None

        def hook(scores):
            self.hooked.append(scores.clone())

        return hook

    def old(self, record=False):
        """The base tree's per-request decode chain (production _run_csa2_chunks)."""
        inp = self.inp
        for r in range(self.m):
            if self.cand is None:
                keys, scales, starts, ends, logical = gb.fused_shared_keys(inp, r, r + 1)
                shared = True
            else:
                keys, scales, starts, ends, logical = gb.fused_candidate_tile(inp, self.cand, r,
                                                                              r + 1)
                shared = False
            self.indexer.forward_prepared(
                inp["q"][r : r + 1], keys, scales, inp["weights"][r : r + 1], starts, ends,
                self.out[r : r + 1], inp["q_scale"][r : r + 1],
                logical_positions=logical, visible_lengths=inp["visible"][r : r + 1],
                is_prefill=False, radix_aux_indices=self.radix_i[r : r + 1],
                radix_aux_logits=self.radix_l[r : r + 1], shared_columns=shared,
                score_hook=self.hook_fn(record),
            )
        return self.out

    def new_keys(self):
        inp = self.inp
        keys, scales, mapped, _, _ = kernel_mod.gather_owner_index_keys(
            inp["pages"], inp["table"], inp["token_requests"][: self.m], inp["page_size"],
            inp["max_positions"], self.width, 0, positions=self.cand,
        )
        return keys, scales, mapped

    def new(self, record=False):
        inp = self.inp
        keys, scales, mapped = self.new_keys()
        self.indexer.select_split_decode(
            inp["q"], inp["q_scale"], inp["weights"], keys, scales, mapped, inp["visible"],
            self.out, self.hook_fn(record), self.radix_i, self.radix_l, None,
        )
        return self.out

    def step(self, record=False):
        return self.new(record) if NEW else self.old(record)


FLUSH = None


def time_graph(fn, reset, warmup, iters):
    global FLUSH
    if FLUSH is None:
        FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        reset()
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        reset()
        fn()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        reset()
        graph.replay()
    times = []
    for _ in range(iters):
        reset()  # reset point: outputs and Top-K scratch restored
        FLUSH.zero_()  # cold L2
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    med = times[len(times) // 2]
    noise = (times[int(0.9 * (len(times) - 1))] - times[int(0.1 * (len(times) - 1))]) / med
    del graph
    return med, noise


def run_bench(args, device):
    result = {"batch": {}, "output": {"class": "exact"}, "new_path": NEW, "components": {}}
    for m in [int(x) for x in args.batches.split(",")]:
        cases = {k: Case(k, m, args, device) for k in ("r2", "r1", "cand")}

        def reset():
            for c in cases.values():
                c.reset()

        def step():
            for _ in range(3):
                cases["r2"].step()
            cases["r1"].step()
            for _ in range(4):
                cases["cand"].step()

        med, noise = time_graph(step, reset, args.warmup, args.iters)
        result["batch"][str(m)] = {"kernel_ms": med, "noise_floor": noise}
        comp = {}
        for k, c in cases.items():
            cmed, cnoise = time_graph(c.step, c.reset, args.warmup, args.iters)
            comp[k] = {"ms": cmed, "noise": cnoise}
        result["components"][str(m)] = comp
        log(f"m={m} step={med * 1000:.1f}us noise={noise:.3f} components="
            + json.dumps({k: round(v["ms"] * 1000, 1) for k, v in comp.items()}))
        del cases
        torch.cuda.empty_cache()
    # Prefill: one 2048-row chunk over a ratio-2 prefix (unchanged code path).
    if args.prefill:
        inp = gb.make_inputs(77, 2, args.tokens_per_block, args.max_seq_len, 1, args.context, device)
        indexer = gb.make_indexer(2)
        rows = 2048
        width = inp["width"]
        keys, scales, starts, ends, logical = gb.fused_shared_keys(inp, 0, rows)
        g = torch.Generator(device="cpu").manual_seed(5)
        q = torch.randint(-128, 128, (rows, gb.HEADS, 64), generator=g, dtype=torch.int32).to(
            torch.int8).to(device)
        qs = inp["q_scale"][:1].expand(rows, -1).contiguous()
        w = torch.randn((rows, gb.HEADS), generator=g).to(device)
        vis = (torch.arange(rows, device=device) + (args.context // 2) - rows + 1).clamp(min=1)
        ends_c = vis.to(torch.int32)
        out = torch.empty((rows, TOPK), dtype=torch.int32, device=device)

        def pre():
            indexer.forward_prepared(q, keys, scales, w, starts, ends_c, out, qs,
                                     logical_positions=logical, visible_lengths=vis,
                                     is_prefill=True, shared_columns=True)

        pmed, pnoise = time_graph(pre, lambda: out.fill_(-7), 2, max(5, args.iters // 3))
        result["components"]["prefill_2048"] = {"ms": pmed, "noise": pnoise, "width": width}
        log(f"prefill 2048 rows x {width} keys: {pmed * 1000:.1f}us noise={pnoise:.3f}")
    print(json.dumps(result), flush=True)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.int32) if a.dtype == torch.float32 else a, 
        b.contiguous().view(torch.int32) if b.dtype == torch.float32 else b)


def run_check(args, device):
    assert NEW, "check runs in the candidate tree"
    failures, cases_out = [], []
    for kind in ("r2", "r1", "cand"):
        for m in [int(x) for x in args.batches.split(",")]:
            c = Case(kind, m, args, device)
            inp = c.inp
            # 1. raw logits over each row's own keys (unmasked, full width)
            keys, scales, mapped = c.new_keys()
            new_logits = c.indexer._split_key_decode_logits(
                inp["q"], inp["q_scale"], inp["weights"], keys, scales, c.width)
            raw_ok = True
            for r in range(m):
                k_r = keys[r * c.width : (r + 1) * c.width]
                s_r = scales[r * c.width : (r + 1) * c.width]
                zero = torch.zeros(1, dtype=torch.int32, device=device)
                ref = c.indexer._call_mqa_logits(
                    inp["q"][r : r + 1], k_r, s_r, inp["weights"][r : r + 1], zero,
                    zero + c.width, inp["q_scale"][r : r + 1], clean_logits=True)
                if not same(new_logits[r : r + 1], ref[:, : c.width]):
                    raw_ok = False
            if not raw_ok:
                failures.append(f"raw logits {kind} m{m}")
            # 2. reference: per-request production chain, eager
            c.reset()
            c.hooked = []
            ref_out = c.old(record=True).clone()
            ref_scores = torch.cat(c.hooked)
            # 3. new path eager, twice (determinism)
            c.hooked = []
            c.reset()
            got = c.new(record=True).clone()
            got_scores = c.hooked[0]
            c.reset()
            again = c.new().clone()
            eager_ok = same(got, ref_out) and same(got_scores, ref_scores) and same(again, got)
            if not eager_ok:
                failures.append(f"eager {kind} m{m}: sel_eq={same(got, ref_out)} "
                                f"scores_eq={same(got_scores, ref_scores)}")
            # 4. graph replays with the timing loop's warmup/iteration/reset points
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                c.reset()
                c.new()
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    c.new()
            torch.cuda.synchronize()
            bad, total = 0, args.warmup + args.iters
            for it in range(total):
                c.reset()  # reset point, as in timing
                graph.replay()
                torch.cuda.synchronize()
                if not same(c.out, ref_out):
                    bad += 1
            del graph
            if bad:
                failures.append(f"graph {kind} m{m}: {bad}/{total} replays differ")
            sel = (ref_out >= 0).sum(1)
            case = dict(kind=kind, m=m, width=c.width,
                        splits=indexer_mod._split_decode_count(m, c.width),
                        raw_logits_bitwise=raw_ok, eager_bitwise=eager_ok,
                        graph_bad_replays=bad, graph_replays=total,
                        selected_min=int(sel.min()), selected_max=int(sel.max()),
                        finite_scores=int(torch.isfinite(ref_scores).sum()))
            cases_out.append(case)
            log(json.dumps(case))
            del c
            torch.cuda.empty_cache()
    print(json.dumps({
        "passed": not failures, "failures": failures[:20],
        "compared": "select_split_decode (one gather, split-key DeepGEMM launch, one mask/"
                    "Top-K/finalize) vs the per-request production chain (fused gather + "
                    "forward_prepared per row): raw logits, masked scores, int32 selections, "
                    "bitwise, eager x2 and every CUDA-graph replay",
        "cases": cases_out}), flush=True)


def run_count(args, device):
    """Kernel launches and per-kernel GPU time of one decode layer, old vs new.

    Both paths run in the candidate tree (the old one is the unchanged per-request
    production chain). torch.profiler CUDA kernel events of one warm eager call.
    """
    from torch.profiler import ProfilerActivity, profile

    report = {}
    for m in [int(x) for x in args.batches.split(",")]:
        for kind in ("r2", "r1", "cand"):
            c = Case(kind, m, args, device)
            entry = {}
            for name, fn in (("old", c.old), ("new", c.new)):
                for _ in range(2):
                    c.reset()
                    fn()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                kernels = {}
                for ev in prof.events():
                    if ev.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in ev.name.lower() \
                            and "memset" not in ev.name.lower():
                        k = ev.name[:60]
                        n, t = kernels.get(k, (0, 0.0))
                        kernels[k] = (n + 1, t + float(getattr(ev, "device_time_total", 0.0)))
                entry[name] = {
                    "launches": sum(n for n, _ in kernels.values()),
                    "gpu_us": round(sum(t for _, t in kernels.values()), 1),
                    "kernels": {k: [n, round(t / n, 1)] for k, (n, t) in
                                sorted(kernels.items(), key=lambda kv: -kv[1][1])[:12]},
                }
            report[f"{kind}_m{m}"] = entry
            log(f"{kind} m={m}: old {entry['old']['launches']} launches {entry['old']['gpu_us']}us | "
                f"new {entry['new']['launches']} launches {entry['new']['gpu_us']}us")
            log("  old: " + json.dumps(entry["old"]["kernels"]))
            log("  new: " + json.dumps(entry["new"]["kernels"]))
            del c
    steps = {}
    for m in [int(x) for x in args.batches.split(",")]:
        steps[str(m)] = {
            p: 3 * report[f"r2_m{m}"][p]["launches"] + report[f"r1_m{m}"][p]["launches"]
            + 4 * report[f"cand_m{m}"][p]["launches"] for p in ("old", "new")}
    print(json.dumps({"passed": True, "launches_per_step_8_layers": steps, "layers": report}),
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--count", action="store_true")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=36864)
    p.add_argument("--context", type=int, default=34816)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--prefill", type=int, default=1)
    args = p.parse_args()
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} new_path={NEW} check={args.check}")
    if args.check:
        run_check(args, device)
    elif args.count:
        run_count(args, device)
    else:
        run_bench(args, device)


if __name__ == "__main__":
    sys.exit(main())
