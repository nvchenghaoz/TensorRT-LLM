"""CSA2 SM120 decode-indexer key-gather glue: microbenchmark and bitwise check.

Runs in either tree. The baseline tree has only the ATen chain (production
``CSA2TrtllmMetadata._map_global_slots`` + ``CSA2CacheManager.gather_indexer_keys``
+ ``torch.where``/``expand`` + ``masked_fill_`` mask); the candidate tree adds
``kernel.gather_owner_index_keys`` / ``kernel.mask_positions_``, which the
production indexer now calls. ``--check`` compares the fused path with the ATen
chain bitwise (keys, scales, logical positions, row bounds, valid mask, masked
logits, final int32 Top-K via ``CSA2Indexer.forward_prepared``) on every
iteration of an untimed sequence with the timing loop's warmup/iteration/reset
points, under CUDA-graph capture for decode batches 1/2/4/8.

Shapes: one TP4 rank, 32 index heads x 128 (FP4 packed 64 B + 4 B UE8M0 scales
in native [pages, 64, 1, 68] footer pages); graph-mode width = max_seq_len //
ratio; page tables with unallocated (-1) pages and padded (-1) request rows.
"""

import argparse
import hashlib
import json
import sys

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_mod
from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as kernel_mod
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout

FUSED = hasattr(kernel_mod, "gather_owner_index_keys")
HEADS, HEAD_DIM = 32, 128
TOPK = 512


def log(*args):
    print(*args, flush=True)


class _PagesStub:
    """Minimal ``self`` for the unbound production ``gather_indexer_keys``."""

    def __init__(self, pages):
        self.pages = pages

    def get_index_pages(self, layer_idx):
        return self.pages


def make_inputs(seed, ratio, tokens_per_block, max_seq_len, requests, context, device):
    """Index pages, owner page table, token rows and visibility for one decode batch."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    page_size = tokens_per_block // ratio
    max_positions = max_seq_len // ratio
    cols = -(-max_positions // page_size)
    native_per_global = page_size // 64
    table_rows = requests + 1  # one spare request row
    physical_pages = table_rows * cols + 8
    perm = torch.randperm(physical_pages, generator=g)[: table_rows * cols]
    table = perm.reshape(table_rows, cols).to(torch.int32)
    # Pages past each request's current length are unallocated, plus random holes.
    for r in range(table_rows):
        length = (context + r * 37) // ratio + 1
        allocated = -(-length // page_size)
        table[r, allocated:] = -1
    holes = torch.rand((table_rows, cols), generator=g) < 0.03
    table[holes] = -1
    native = physical_pages * native_per_global
    pages = torch.randint(0, 256, (native, 64 * 68), generator=g, dtype=torch.int32).to(torch.uint8)
    pages[:, 4096:] = torch.randint(118, 131, (native, 256), generator=g, dtype=torch.int32).to(
        torch.uint8
    )
    pages = pages.view(native, 64, 1, 68)
    # Padded batch: the last request row is padding (-1) when requests > 1.
    token_requests = torch.arange(requests, dtype=torch.int64)
    if requests > 1:
        token_requests[-1] = -1
    visible = torch.tensor(
        [(context + r * 37 + 1) // ratio for r in range(requests)], dtype=torch.int64
    )
    q = torch.randint(-128, 128, (requests, HEADS, HEAD_DIM // 2), generator=g, dtype=torch.int32)
    q_scale_bytes = torch.randint(122, 129, (requests, HEADS, 4), generator=g, dtype=torch.int32)
    weights = torch.randn((requests, HEADS), generator=g, dtype=torch.float32)
    return dict(
        pages=pages.to(device),
        table=table.to(device),
        token_requests=token_requests.to(device),
        visible=visible.to(device),
        q=q.to(torch.int8).to(device),
        q_scale=q_scale_bytes.to(torch.uint8).view(torch.int32).reshape(requests, HEADS).to(device),
        weights=weights.to(device),
        page_size=page_size,
        max_positions=max_positions,
        width=max_positions,
    )


# ----------------------------------------------------------------- reference


def ref_shared_keys(inp, start, end):
    """The original ``shared_keys`` closure body (ATen chain)."""
    width = inp["width"]
    device = inp["pages"].device
    positions = torch.arange(width, device=device)
    requests = inp["token_requests"][start : start + 1].long()
    slots = CSA2TrtllmMetadata._map_global_slots(
        positions[None, :], requests, inp["table"], None, inp["page_size"], inp["max_positions"]
    )[0]
    keys, scales = CSA2CacheManager.gather_indexer_keys(_PagesStub(inp["pages"]), 0, slots)
    starts = torch.zeros(end - start, dtype=torch.int32, device=device)
    logical = torch.where(slots >= 0, positions, -1).expand(end - start, -1)
    return keys, scales, starts, torch.full_like(starts, width), logical


def ref_candidate_tile(inp, candidates, start, end):
    """The original ``candidate_tile`` closure body (ATen chain)."""
    positions = candidates[start:end]
    requests = inp["token_requests"][start:end].long()
    slots = CSA2TrtllmMetadata._map_global_slots(
        positions, requests, inp["table"], None, inp["page_size"], inp["max_positions"]
    )
    keys, scales = CSA2CacheManager.gather_indexer_keys(_PagesStub(inp["pages"]), 0, slots.flatten())
    width = positions.shape[1]
    starts = torch.arange(end - start, dtype=torch.int32, device=positions.device) * width
    return keys, scales, starts, starts + width, torch.where(slots >= 0, positions, -1)


def ref_mask(scores, logical, visible):
    valid = logical >= 0
    valid &= logical < visible[:, None]
    scores.masked_fill_(~valid, -torch.inf)
    return valid


def fused_shared_keys(inp, start, end):
    keys, scales, mapped, starts, ends = kernel_mod.gather_owner_index_keys(
        inp["pages"],
        inp["table"],
        inp["token_requests"][start : start + 1],
        inp["page_size"],
        inp["max_positions"],
        inp["width"],
        end - start,
    )
    return keys, scales, starts, ends, mapped.expand(end - start, -1)


def fused_candidate_tile(inp, candidates, start, end):
    positions = candidates[start:end]
    width = positions.shape[1]
    keys, scales, mapped, starts, ends = kernel_mod.gather_owner_index_keys(
        inp["pages"],
        inp["table"],
        inp["token_requests"][start:end],
        inp["page_size"],
        inp["max_positions"],
        width,
        end - start,
        positions=positions,
        start_step=width,
    )
    return keys, scales, starts, ends, mapped


def production_shared_keys(inp, start, end):
    return fused_shared_keys(inp, start, end) if FUSED else ref_shared_keys(inp, start, end)


def production_mask(scores, logical, visible):
    if FUSED:
        return kernel_mod.mask_positions_(scores, logical, visible)
    return ref_mask(scores, logical, visible)


def glue_step(inp, requests, logits, queries=1):
    """One indexer layer's decode glue for a padded batch: per request one shared
    key gather (graph width) plus the selection mask over its logits row."""
    outs = []
    for r in range(requests):
        keys, scales, starts, ends, logical = production_shared_keys(inp, r, r + 1)
        scores = logits[r : r + 1]
        valid = production_mask(scores, logical, inp["visible"][r : r + 1])
        outs.append((keys, scales, starts, ends, logical, valid))
    return outs


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


# ----------------------------------------------------------------- top-k


def make_indexer(ratio):
    layout = CSA2Layout(
        compress_ratios=(ratio,),
        kv_source_layer_ids=(0,),
        index_source_layer_ids=(0,),
        index_topk=TOPK,
    )
    return indexer_mod.CSA2Indexer(layout, 0, HEADS, HEAD_DIM)


def topk_for(indexer, inp, r, chunk, queries=1):
    keys, scales, starts, ends, logical = chunk
    count = queries
    out = torch.empty((count, TOPK), dtype=torch.int32, device=keys.device)
    radix_i = torch.empty((count, 10, TOPK), dtype=torch.int32, device=keys.device)
    radix_l = torch.empty_like(radix_i, dtype=torch.float32)
    q = inp["q"][r : r + 1].expand(count, -1, -1).contiguous()
    qs = inp["q_scale"][r : r + 1].expand(count, -1).contiguous()
    w = inp["weights"][r : r + 1].expand(count, -1).contiguous()
    vis = inp["visible"][r : r + 1].expand(count).contiguous()
    indexer.forward_prepared(
        q,
        keys,
        scales,
        w,
        starts,
        ends,
        out,
        qs,
        logical_positions=logical,
        visible_lengths=vis,
        is_prefill=False,
        radix_aux_indices=radix_i,
        radix_aux_logits=radix_l,
        shared_columns=True,
    )
    return out


# ----------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--ratios", default="1,2")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=36864)
    p.add_argument("--context", type=int, default=34816)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--replays", type=int, default=20)
    args = p.parse_args()
    device = torch.device("cuda")
    ratios = [int(x) for x in args.ratios.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    log(f"device={torch.cuda.get_device_name(0)} fused={FUSED} check={args.check}")
    if args.check:
        run_check(args, ratios, batches, device)
    else:
        run_bench(args, ratios, batches, device)


def production_candidate_tile(inp, candidates, start, end):
    if FUSED:
        return fused_candidate_tile(inp, candidates, start, end)
    return ref_candidate_tile(inp, candidates, start, end)


def time_graph(args, step, reset):
    """Capture ``step`` once (after eager warmup that compiles the CuTe kernels)
    and time graph replays with CUDA events; ``reset`` restores mutable inputs
    at each reset point. Returns (median ms, noise floor, outputs)."""
    for _ in range(3):
        reset()
        step()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        reset()
        step()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            outs = step()
    torch.cuda.synchronize()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup):
        reset()
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        reset()  # reset point: masked logits restored from the pristine copy
        torch.cuda.synchronize()
        start_e.record()
        for _ in range(args.replays):
            graph.replay()
        end_e.record()
        torch.cuda.synchronize()
        times.append(start_e.elapsed_time(end_e) / args.replays)
    times.sort()
    med = times[len(times) // 2]
    p10 = times[max(0, int(0.1 * len(times)))]
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    # post-timing replay from a reset for the output digest
    reset()
    graph.replay()
    torch.cuda.synchronize()
    return med, (p90 - p10) / med, outs, graph


def run_bench(args, ratios, batches, device):
    """Per decode step of one TP4 rank (V4.1 layout: index layers 2/8/14 Full at
    ratio 2, 20 Full at ratio 1 (candidate source), 24/28/32/36 Reindex
    consumers of 20): per request one shared key gather + selection mask in each
    Full layer, one candidate tile gather ([M, 16384] candidates) in each
    consumer. Everything else in the indexer is unchanged and not timed."""
    result = {"batch": {}, "output": {"class": "exact"}, "fused": FUSED, "digests": {},
              "components": {}}
    for m in batches:
        r2 = make_inputs(1000 + m, 2, args.tokens_per_block, args.max_seq_len, m, args.context,
                         device)
        r1 = make_inputs(2000 + m, 1, args.tokens_per_block, args.max_seq_len, m, args.context,
                         device)
        g = torch.Generator(device="cpu").manual_seed(3000 + m)
        cwidth = min(2048, -(-r1["max_positions"] // 8)) * 8
        cand = torch.randint(0, (args.context + 1), (m, cwidth), generator=g, dtype=torch.int32)
        cand[:, ::7] = -1
        cand = cand.to(device)
        logits2_src = torch.randn((3, m, r2["width"]), generator=g).to(device)
        logits1_src = torch.randn((m, r1["width"]), generator=g).to(device)
        input_digest = digest([r2["pages"], r2["table"], r2["token_requests"], r2["visible"],
                               r1["pages"], r1["table"], r1["token_requests"], r1["visible"],
                               cand, logits2_src, logits1_src])
        logits2 = logits2_src.clone()
        logits1 = logits1_src.clone()

        def reset():
            logits2.copy_(logits2_src)
            logits1.copy_(logits1_src)

        def step():
            outs = []
            for layer in range(3):
                outs += glue_step(r2, m, logits2[layer])
            outs += glue_step(r1, m, logits1)
            cands = [production_candidate_tile(r1, cand, 0, m) for _ in range(4)]
            return outs, cands

        def shared_only():
            return glue_step(r1, m, logits1), []

        def cand_only():
            return [], [production_candidate_tile(r1, cand, 0, m)]

        med, noise, (outs, cands), graph = time_graph(args, step, reset)
        flat = []
        for keys, scales, starts, ends, logical, valid in outs:
            flat += [keys, scales, starts, ends, logical.contiguous(), valid]
        for chunk in cands:
            flat += [t.contiguous() for t in chunk]
        flat += [logits2, logits1]
        label = str(m)
        result["batch"][label] = {"kernel_ms": med, "noise_floor": noise}
        result["digests"][label] = digest(flat)
        result.setdefault("input_digests", {})[label] = input_digest
        result.setdefault("output_digest_parts", {})[label] = {
            "keys_scales_bounds_logical_valid": digest(flat[: len(flat) - 2]),
            "masked_logits": digest(flat[-2:]),
        }
        del graph
        comp = {}
        for name, fn in (("shared_r1_per_layer", shared_only), ("candidate_per_layer", cand_only)):
            cmed, cnoise, _, cgraph = time_graph(args, fn, reset)
            comp[name] = {"ms": cmed, "noise": cnoise}
            del cgraph
        result["components"][label] = comp
        log(f"m={m} step_glue={med*1000:.1f}us noise={noise:.3f} digest={result['digests'][label]} "
            f"components={json.dumps(comp)}")
    print(json.dumps(result), flush=True)


def run_check(args, ratios, batches, device):
    assert FUSED, "check runs in the candidate tree"
    failures = []
    report = {"cases": []}

    def compare(tag, got, ref):
        names = ["keys", "scales", "starts", "ends", "logical"]
        for name, a, b in zip(names, got, ref):
            if not same(a, b):
                failures.append(f"{tag}:{name}")
                return False
        return True

    for ratio in ratios:
        indexer = make_indexer(ratio)
        for m in batches:
            inp = make_inputs(1000 + m, ratio, args.tokens_per_block, args.max_seq_len, m,
                              args.context, device)
            width = inp["width"]
            # 1. shared keys, eager, for every request row (incl. padded -1 row),
            #    also MTP-style multi-query tiles (stride-0 logical rows).
            for r in range(m):
                for queries in (1, 2, 3):
                    ref = ref_shared_keys(inp, r, r + queries)
                    got = fused_shared_keys(inp, r, r + queries)
                    compare(f"shared r{ratio} m{m} row{r} q{queries}", got, ref)
            # 2. candidate tiles: int32 candidates with -1, out-of-range, beyond max.
            g = torch.Generator(device="cpu").manual_seed(77 + m)
            cwidth = min(2048, -(-inp["max_positions"] // 8)) * 8
            cand = torch.randint(-1, inp["max_positions"] + 64, (m, cwidth), generator=g,
                                 dtype=torch.int32)
            cand[:, ::5] = -1
            cand = cand.to(device)
            for dtype in (torch.int32, torch.int64):
                c = cand.to(dtype)
                for s, e in ((0, m), (0, 1), (m - 1, m)):
                    compare(f"cand r{ratio} m{m} {dtype} [{s},{e})",
                            fused_candidate_tile(inp, c, s, e), ref_candidate_tile(inp, c, s, e))
            # 3. mask over logits: shared stride-0 and contiguous int32/int64 positions.
            logits_src = torch.randn((m, width), generator=g).to(device)
            logits_src[:, ::97] = -torch.inf
            for r in range(m):
                _, _, _, _, logical = ref_shared_keys(inp, r, r + 2)
                vis = inp["visible"][r : r + 1].expand(2).contiguous()
                for pos in (logical, logical.contiguous(), logical.to(torch.int32)):
                    a = logits_src[r : r + 1].expand(2, -1).clone()
                    b = a.clone()
                    va = kernel_mod.mask_positions_(a, pos, vis)
                    vb = ref_mask(b, pos, vis)
                    if not (same(va, vb) and torch.equal(a.view(torch.int32), b.view(torch.int32))):
                        failures.append(f"mask r{ratio} m{m} row{r} {pos.dtype} {pos.stride()}")
            # 4. graph-captured decode step, repeated with the timing loop's reset
            #    points; compare every iteration against the eager ATen reference.
            ref_outs = []
            ref_logits = logits_src.clone()
            for r in range(m):
                chunk = ref_shared_keys(inp, r, r + 1)
                valid = ref_mask(ref_logits[r : r + 1], chunk[4], inp["visible"][r : r + 1])
                ref_outs.append((*chunk, valid))
            logits = logits_src.clone()
            for _ in range(3):
                logits.copy_(logits_src)
                glue_step(inp, m, logits)
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                logits.copy_(logits_src)
                glue_step(inp, m, logits)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    outs = glue_step(inp, m, logits)
            torch.cuda.synchronize()
            bad_iters = 0
            total_iters = args.warmup + args.iters * args.replays
            for it in range(total_iters):
                if it < args.warmup or (it - args.warmup) % args.replays == 0:
                    logits.copy_(logits_src)  # reset point
                graph.replay()
                torch.cuda.synchronize()
                ok = torch.equal(logits.view(torch.int32), ref_logits.view(torch.int32))
                for got, ref in zip(outs, ref_outs):
                    ok = ok and all(same(a.contiguous(), b.contiguous()) for a, b in zip(got, ref))
                if not ok:
                    bad_iters += 1
            if bad_iters:
                failures.append(f"graph r{ratio} m{m}: {bad_iters}/{total_iters} iterations differ")
            del graph
            # 5. final Top-K through CSA2Indexer.forward_prepared: fused gather +
            #    fused mask (production) vs ATen chain + original mask.
            topk_equal = True
            selected_counts = []
            for r in range(m):
                for queries in (1, 2):
                    got = topk_for(indexer, inp, r, fused_shared_keys(inp, r, r + queries), queries)
                    saved = indexer_mod.mask_positions_
                    indexer_mod.mask_positions_ = ref_mask
                    try:
                        ref = topk_for(indexer, inp, r, ref_shared_keys(inp, r, r + queries),
                                       queries)
                    finally:
                        indexer_mod.mask_positions_ = saved
                    again = topk_for(indexer, inp, r, fused_shared_keys(inp, r, r + queries),
                                     queries)
                    selected_counts.append(int((got >= 0).sum()))
                    if not (same(got, ref) and same(got, again)):
                        topk_equal = False
                        failures.append(f"topk r{ratio} m{m} row{r} q{queries}")
            # 6. Top-K over candidate tiles (non-shared columns path).
            cand_ok = True
            for r in range(m):
                chunk_f = fused_candidate_tile(inp, cand, r, r + 1)
                chunk_r = ref_candidate_tile(inp, cand, r, r + 1)
                outs_tk = []
                for chunk in (chunk_f, chunk_r):
                    keys, scales, starts, ends, logical = chunk
                    out = torch.empty((1, TOPK), dtype=torch.int32, device=device)
                    radix_i = torch.empty((1, 10, TOPK), dtype=torch.int32, device=device)
                    indexer.forward_prepared(
                        inp["q"][r : r + 1], keys, scales, inp["weights"][r : r + 1], starts,
                        ends, out, inp["q_scale"][r : r + 1], logical_positions=logical,
                        visible_lengths=inp["visible"][r : r + 1], is_prefill=False,
                        radix_aux_indices=radix_i,
                        radix_aux_logits=torch.empty_like(radix_i, dtype=torch.float32),
                    )
                    outs_tk.append(out)
                if not same(outs_tk[0], outs_tk[1]):
                    cand_ok = False
                    failures.append(f"cand-topk r{ratio} m{m} row{r}")
            case = dict(ratio=ratio, m=m, width=width, graph_bad_iters=bad_iters,
                        graph_iters=total_iters, topk_equal=topk_equal,
                        candidate_topk_equal=cand_ok,
                        selected_min=min(selected_counts), selected_max=max(selected_counts),
                        unbacked_columns=int(
                            (ref_shared_keys(inp, 0, 1)[4] < 0).sum()))
            report["cases"].append(case)
            log(json.dumps(case))
    passed = not failures
    print(json.dumps({"passed": passed, "failures": failures[:20],
                      "compared": "fused gather/mask vs original ATen chain (production "
                      "_map_global_slots + gather_indexer_keys + where/expand + masked_fill_): "
                      "keys, scales, logical positions, row bounds, valid mask, masked logits, "
                      "final int32 Top-K (forward_prepared), bitwise, every graph replay",
                      **report}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
