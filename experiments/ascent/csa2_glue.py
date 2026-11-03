"""CSA2 packed-cache publication + decode staging: fused CuTe DSL vs PyTorch reference.

Synthetic tensors at one TP4 rank's shapes (DeepSeek-V4.1-Flash): head_dim 512
(448 nope + 64 rope), SWA window 128, index_topk 512 (num_sparse_topk 640),
index 32 heads x 128, main row 288 B, index row 68 B in [pages,64,1,68] footer
pages, SWA row 528 B, 32768..34816-token context (main ratio 4, page 64).

Modes:
  check : byte-exact fused vs reference (gates forced False in-process), for
          decode M=1,2,4,8 and a 2048-row prefill chunk, with the harness's
          warmup/iteration/reset schedule, an 8-step advancing decode window,
          and CUDA-graph capture/replay of the fused step.
  bench : CUDA-graph replay time of one Full-layer step (publish SWA+main+
          index rows + pack index queries + stage 128 SWA + 512 main rows).
          Whatever path the tree's gates choose (base: reference on SM120).
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import metadata as M
from tensorrt_llm._torch.attention.backends.sparse.csa2 import quantization as Q
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2BackendForwardArgs,
    RowTransform,
)

DEV = torch.device("cuda", 0)
HEAD = 512
ROPE = 64
WINDOW = 128
TOPK = 512
IDX_HEADS = 32
IDX_DIM = 128
PAGE = 64
RATIO = 4
CTX = 34816
MAX_POS = CTX // RATIO  # 8704 compressed positions per request
PAGES_PER_REQ = MAX_POS // PAGE  # 136
WARMUP, ITERS = 3, 10

_ORIG = (Q._fused_gather_supported, Q._fused_store_supported)


def force_reference(flag: bool) -> None:
    if flag:
        Q._fused_gather_supported = lambda index: False
        Q._fused_store_supported = lambda index: False
    else:
        Q._fused_gather_supported, Q._fused_store_supported = _ORIG


def log(*args):
    print(*args, flush=True)


class FakeMeta:
    """Just the CSA2TrtllmMetadata state stage_selected reads and writes."""

    stage_selected = M.CSA2TrtllmMetadata.stage_selected
    _stage_scratch = M.CSA2TrtllmMetadata._stage_scratch
    _map_global_slots = staticmethod(M.CSA2TrtllmMetadata._map_global_slots)

    def __init__(self, capacity: int):
        extra = TOPK
        self.num_tokens = capacity
        self.num_sparse_topk = WINDOW + extra
        self.swa_pool = torch.zeros((capacity, WINDOW, HEAD), dtype=torch.bfloat16, device=DEV)
        self.extra_pool = torch.zeros((capacity, extra, HEAD), dtype=torch.bfloat16, device=DEV)
        self.prepared_indices = torch.full(
            (capacity, self.num_sparse_topk), -1, dtype=torch.int32, device=DEV
        )
        self.prepared_lens = torch.zeros(capacity, dtype=torch.int32, device=DEV)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=DEV)
        self.shared_plan = None
        self.is_cuda_graph = False
        self.kv_cache_manager = None
        self._csa2_stage_epoch = 0


def make_case(count: int, requests: int, seed: int, step: int = 0):
    """Inputs for one layer-step; pools are created separately (make_pools)."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed * 1000 + step)
    padded = count >= 4 and requests == count  # pad the last two decode rows
    real = count - 2 if padded else count
    req_of_tok = (
        torch.arange(count, device=DEV) % requests
        if requests == count
        else torch.zeros(count, dtype=torch.long, device=DEV)
    )
    # positions: decode at the end of each request's context; prefill 32768..34815
    if requests == count:
        pos = torch.full((count,), CTX - 16 + step, dtype=torch.long, device=DEV) - req_of_tok
    else:
        pos = 32768 + torch.arange(count, device=DEV)
    pos32 = pos.int()

    def bf(shape, scale=1.0):
        x = torch.randn(shape, generator=g, device=DEV) * scale
        return x.to(torch.bfloat16)

    swa_vals = bf((count, HEAD), 2.0)
    swa_vals[0, :7] = torch.tensor([0.0, -0.0, 1e-30, 6e4, -6e4, 3.0, -1.5], device=DEV).bfloat16()
    main_n = count if requests == count else count // RATIO
    main_vals = bf((main_n, HEAD), 3.0)
    index_vals = bf((main_n, IDX_DIM), 1.5)
    index_q = bf((count, IDX_HEADS, IDX_DIM), 1.0)
    # SWA write slots: per request ring of WINDOW*2 rows; padding -> -1
    swa_rows_per_req = 2 * WINDOW + (count if requests == 1 else 0)
    swa_slots = req_of_tok * swa_rows_per_req + (pos % swa_rows_per_req)
    if padded:
        swa_slots[real:] = -1
    # global (main/index) write slots: through the page table, compressed position
    cpos = (pos // RATIO)[:main_n] if requests == count else (32768 // RATIO + torch.arange(main_n, device=DEV))
    greq = req_of_tok[:main_n] if requests == count else torch.zeros(main_n, dtype=torch.long, device=DEV)
    gslots = greq * PAGES_PER_REQ * PAGE + cpos  # physical identity before permutation
    if requests == count:
        gslots = gslots.clone()
        if padded:
            gslots[real:] = -1
    else:
        gslots = gslots.clone()
        gslots[-3:] = -1
    cos_sin = make_table()
    swa_tf = RowTransform(norm=(NORM_W, 1e-6), rope=(pos32, cos_sin, ROPE))
    main_tf = RowTransform(norm=None, rope=(cpos.int(), cos_sin, ROPE))
    index_tf = RowTransform(norm=(INDEX_NORM_W, 1e-6), rope=(cpos.int(), cos_sin, ROPE))
    # staging selections
    swa_idx = torch.randint(
        0, swa_rows_per_req, (count, WINDOW), generator=g, device=DEV, dtype=torch.long
    ) + (req_of_tok * swa_rows_per_req)[:, None]
    swa_idx[:, -5:] = -1  # short windows / padding
    swa_idx[0, 3] = 10**9  # out-of-range slot
    if padded:
        swa_idx[real:] = -1
    logical = torch.randint(0, MAX_POS, (count, TOPK), generator=g, device=DEV, dtype=torch.int32)
    logical[:, -20:] = -1
    logical[:, 0] = MAX_POS + 5  # beyond max positions
    if requests == 1:
        visible = (pos // RATIO + 1).long()
    else:
        visible = torch.full((count,), MAX_POS, dtype=torch.long, device=DEV)
        visible[-1] = MAX_POS - 1000  # some selections not yet visible
    if padded:
        logical[real:] = -1
    return dict(
        count=count,
        swa_vals=swa_vals,
        main_vals=main_vals,
        index_vals=index_vals,
        index_q=index_q,
        swa_slots=swa_slots,
        gslots=gslots,
        swa_tf=swa_tf,
        main_tf=main_tf,
        index_tf=index_tf,
        swa_idx=swa_idx,
        logical=logical,
        visible=visible,
        requests=req_of_tok.int(),
    )


_TABLE = None
NORM_W = None
INDEX_NORM_W = None


def make_table():
    global _TABLE
    if _TABLE is None:
        positions = torch.arange(CTX + 64, device=DEV, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEV).float() / ROPE))
        ang = positions[:, None] * inv[None, :]
        _TABLE = torch.cat((ang.cos(), ang.sin()), -1).contiguous()
    return _TABLE


def make_pools(requests: int, seed: int, swa_rows: int):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    total_pages = requests * PAGES_PER_REQ + 8
    swa_pool = torch.randint(0, 256, (swa_rows, Q.row_bytes(HEAD, "swa")), generator=g, device=DEV, dtype=torch.uint8)
    # keep stored SWA scales sane (exponent bytes 120..130) and avoid NaN payloads
    swa_pool[:, HEAD:] = torch.randint(118, 132, (swa_rows, HEAD // 32), generator=g, device=DEV, dtype=torch.uint8)
    swa_pool[:, :HEAD] &= 0x77
    main_pool = torch.randint(0, 256, (total_pages * PAGE, Q.row_bytes(HEAD, "main")), generator=g, device=DEV, dtype=torch.uint8)
    main_pool[:, HEAD // 2 :] = torch.randint(0x20, 0x50, (total_pages * PAGE, HEAD // 16), generator=g, device=DEV, dtype=torch.uint8)
    index_pages = torch.randint(0, 256, (total_pages, PAGE, 1, 68), generator=g, device=DEV, dtype=torch.uint8)
    # page table: request -> permuted physical pages, some unallocated (-1)
    perm = torch.randperm(total_pages, generator=g, device=DEV)[: requests * PAGES_PER_REQ]
    table = perm.view(requests, PAGES_PER_REQ).int().contiguous()
    table[:, 50:52] = -1
    return dict(swa_pool=swa_pool, main_pool=main_pool, index_pages=index_pages, table=table)


def global_write_slots(case, pools):
    """Map identity global slots through the page table (what csa2_main_write_slots holds)."""
    slots = case["gslots"]
    table = pools["table"]
    valid = slots >= 0
    s = slots.clamp_min(0)
    req = s // (PAGES_PER_REQ * PAGE)
    page = (s % (PAGES_PER_REQ * PAGE)) // PAGE
    phys = table[req.clamp(max=table.shape[0] - 1), page]
    out = phys.long() * PAGE + s % PAGE
    return torch.where(valid & (phys >= 0), out, -1)


def run_step(case, pools, meta, *, layer_kind="full", group=None, mapped=None):
    """One layer's publication + staging; returns its outputs (views into live buffers)."""
    count = case["count"]
    gw = case["_gw"]
    main = index = None
    if layer_kind == "full":
        main = (pools["main_pool"], gw, case["main_vals"], case["main_tf"])
        index = (pools["index_pages"], gw, case["index_vals"], case["index_tf"])
    index_q = case["index_q"] if layer_kind in ("full", "reindex") else None
    packed = Q.store_layer_rows(
        pools["swa_pool"], case["swa_slots"], case["swa_vals"], case["swa_tf"],
        main=main, index=index, index_q=index_q,
    )
    meta.num_tokens = count
    native = Q._fused_gather_supported(DEV.index) if mapped is None else mapped
    mapping = None
    if native:
        topk = case["logical"]
        mapping = (pools["table"], PAGE, MAX_POS, case["requests"], case["visible"])
    else:
        topk = FakeMeta._map_global_slots(
            case["logical"], case["requests"].long(), pools["table"], case["visible"], PAGE, MAX_POS
        )
    inputs = CSA2BackendForwardArgs(
        swa_pool=pools["swa_pool"],
        swa_indices=case["swa_idx"],
        main_pool=pools["main_pool"],
        topk_indices=topk,
        main_logical_indices=case["logical"],
        main_mapping=mapping,
        stage_group=group,
        state=None,
    )
    meta.stage_selected(inputs, main_logical=case["logical"])
    return packed


def outputs(meta, count, packed):
    lens = meta.prepared_lens[:count].clone()
    kv = torch.cat((meta.swa_pool[:count], meta.extra_pool[:count]), 1)
    valid = torch.arange(meta.num_sparse_topk, device=DEV)[None, :] < lens[:, None]
    rows = torch.where(valid[..., None], kv, 0)
    out = dict(lens=lens, rows=rows.view(torch.int16).clone())
    out["indices"] = meta.prepared_indices[:count].clone()
    if packed is not None:
        out["q_data"] = packed[0].clone()
        out["q_scales"] = packed[1].clone()
    return out


def pool_state(pools):
    return {k: pools[k].clone() for k in ("swa_pool", "main_pool", "index_pages")}


def compare(a: dict, b: dict, tag: str, report: list) -> bool:
    ok = True
    for key in a:
        x, y = a[key], b[key]
        same = x.shape == y.shape and torch.equal(x, y)
        if not same:
            ok = False
            diff = (x != y).sum().item() if x.shape == y.shape else -1
            report.append(f"{tag}:{key} mismatch ({diff} elements)")
    return ok


CASES = [("1", 1, 1), ("2", 2, 2), ("4", 4, 4), ("8", 8, 8), ("prefill2048", 2048, 1)]


def swa_rows_for(count, requests):
    return requests * (2 * WINDOW + (count if requests == 1 else 0)) + 16


def init_weights():
    global NORM_W, INDEX_NORM_W
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    NORM_W = (1 + 0.1 * torch.randn(HEAD, generator=g, device=DEV)).bfloat16()
    INDEX_NORM_W = (1 + 0.1 * torch.randn(IDX_DIM, generator=g, device=DEV)).bfloat16()


def check(args) -> dict:
    report: list[str] = []
    passed = True
    fused_active = _ORIG[0](DEV.index) and _ORIG[1](DEV.index)
    log("gates fused gather/store:", _ORIG[0](DEV.index), _ORIG[1](DEV.index))
    summary = {}
    for label, count, requests in CASES:
        for kind in ("full", "reindex", "swa"):
            case = make_case(count, requests, seed=1)
            pristine = make_pools(requests, 11, swa_rows_for(count, requests))
            case["_gw"] = global_write_slots(case, pristine)
            per_iter = []
            results = {}
            for mode in ("ref", "fused", "fused_premapped"):
                force_reference(mode == "ref")
                meta = FakeMeta(count)
                iters_out = []
                for it in range(WARMUP + ITERS):
                    pools = {k: v.clone() for k, v in pristine.items()}  # reset point
                    packed = run_step(
                        case, pools, meta, layer_kind=kind,
                        mapped=False if mode != "fused" else None,
                    )
                    o = outputs(meta, count, packed)
                    o.update(pool_state(pools))
                    iters_out.append(o)
                force_reference(False)
                results[mode] = iters_out
            ok = True
            for it in range(WARMUP + ITERS):
                ok &= compare(results["ref"][it], results["fused"][it], f"{label}/{kind}/it{it}/fused", report)
                ok &= compare(results["ref"][it], results["fused_premapped"][it], f"{label}/{kind}/it{it}/premapped", report)
            lens = results["ref"][0]["lens"]
            summary[f"{label}/{kind}"] = {
                "exact": bool(ok),
                "lens_min": int(lens.min()),
                "lens_max": int(lens.max()),
            }
            log(label, kind, "exact" if ok else "MISMATCH", summary[f"{label}/{kind}"])
            passed &= ok
    # advancing decode window: 10 steps, cache evolves, 2 layers share a staging group
    for label, count, requests in CASES[:4]:
        pristine = make_pools(requests, 21, swa_rows_for(count, requests))
        evo = {}
        for mode in ("ref", "fused", "fused2"):
            force_reference(mode == "ref")
            pools = {k: v.clone() for k, v in pristine.items()}
            pools_b = {k: v.clone() for k, v in pristine.items()}
            meta = FakeMeta(count)
            steps = []
            for step in range(10):
                case = make_case(count, requests, seed=3, step=step)
                case["_gw"] = global_write_slots(case, pools)
                meta._csa2_stage_epoch += 1  # new forward
                pa = run_step(case, pools, meta, layer_kind="full", group=("L0", "L0", 0))
                oa = outputs(meta, count, pa)
                # second layer (Reuse of same owner/index source): own SWA pool, same selections
                pb = run_step(case, pools_b, meta, layer_kind="swa", group=("L0", "L0", 0))
                ob = outputs(meta, count, pb)
                steps.append(({f"a_{k}": v for k, v in oa.items()} | {f"b_{k}": v for k, v in ob.items()}))
            force_reference(False)
            steps.append(pool_state(pools) | {f"b_{k}": v for k, v in pool_state(pools_b).items()})
            evo[mode] = steps
        ok = True
        for s in range(len(evo["ref"])):
            ok &= compare(evo["ref"][s], evo["fused"][s], f"window{label}/step{s}", report)
            ok &= compare(evo["fused"][s], evo["fused2"][s], f"window{label}/step{s}/repeat", report)
        summary[f"window{label}"] = {"exact": bool(ok), "steps": 10}
        log("decode window", label, "exact" if ok else "MISMATCH")
        passed &= ok
    # CUDA graph capture + replay of the (fused) step at padded decode batches
    for label, count, requests in CASES[:4]:
        ok = True
        case = make_case(count, requests, seed=5)
        pristine = make_pools(requests, 31, swa_rows_for(count, requests))
        case["_gw"] = global_write_slots(case, pristine)
        force_reference(True)
        meta_r = FakeMeta(count)
        pools_r = {k: v.clone() for k, v in pristine.items()}
        ref = outputs(meta_r, count, run_step(case, pools_r, meta_r)) | pool_state(pools_r)
        force_reference(False)
        meta = FakeMeta(count)
        pools = {k: v.clone() for k, v in pristine.items()}
        s = torch.cuda.Stream(DEV)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warmup compiles CuTe kernels outside capture
                run_step(case, pools, meta, group=("G", "G", 0))
                meta._csa2_stage_epoch += 1
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                packed = run_step(case, pools, meta, group=("G", "G", 0))
        except Exception as exc:  # noqa: BLE001
            report.append(f"graph{label}: capture failed: {exc!r}")
            ok = False
        if ok:
            for rep in range(3):
                for k, v in pristine.items():
                    pools[k].copy_(v)
                meta.prepared_lens.zero_()
                meta.swa_pool.fill_(1.0)
                meta.extra_pool.fill_(1.0)
                graph.replay()
                torch.cuda.synchronize()
                got = outputs(meta, count, packed) | pool_state(pools)
                ok &= compare(ref, got, f"graph{label}/replay{rep}", report)
        summary[f"graph{label}"] = {"exact": bool(ok)}
        log("graph", label, "exact" if ok else "MISMATCH")
        passed &= ok
    return {
        "passed": bool(passed),
        "fused_active": bool(fused_active),
        "compared": "fused CuTe publish/stage vs PyTorch reference (gates forced False): SWA/main/index pool bytes, packed index-Q data+scales, staged BF16 rows under the prepared_lens valid prefix, prepared_lens, prepared_indices; every warmup+timed iteration, 10-step advancing decode window, graph replays",
        "cases": summary,
        "mismatches": report[:40],
    }


def bench(args) -> dict:
    result = {}
    for label, count, requests in CASES:
        case = make_case(count, requests, seed=1)
        pristine = make_pools(requests, 11, swa_rows_for(count, requests))
        case["_gw"] = global_write_slots(case, pristine)
        pools = {k: v.clone() for k, v in pristine.items()}
        meta = FakeMeta(count)
        s = torch.cuda.Stream(DEV)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # compile CuTe kernels and create staging scratch outside capture
                run_step(case, pools, meta, layer_kind=args.kind, group=None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run_step(case, pools, meta, layer_kind=args.kind, group=None)
            torch.cuda.synchronize()
        kernels = [
            e for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in e.name.lower()
        ]
        kernel_count = len(kernels)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.layers):
                run_step(case, pools, meta, layer_kind=args.kind, group=None)
        times = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for it in range(5 + 30):
            for k, v in pristine.items():
                pools[k].copy_(v)  # reset mutable pools before every timed replay
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if it >= 5:
                times.append(start.elapsed_time(end) / args.layers)
        times.sort()
        med = times[len(times) // 2]
        p10, p90 = times[int(0.1 * len(times))], times[int(0.9 * len(times)) - 1]
        result[label] = {
            "kernel_ms": round(med, 5),
            "noise_floor": round((p90 - p10) / med, 4),
            "kernels_per_layer_step": kernel_count,
        }
        log(label, result[label])
    return {"batch": result, "output": {"class": "exact"}, "kind": args.kind,
            "fused_active": bool(Q._fused_store_supported(DEV.index))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "bench"))
    parser.add_argument("--kind", default="full", choices=("full", "reindex", "swa"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--force-ref", action="store_true", help="bench the PyTorch reference path")
    parser.add_argument("--cases", default="", help="comma-separated case labels to bench")
    args = parser.parse_args()
    if args.cases:
        CASES[:] = [c for c in CASES if c[0] in args.cases.split(",")]
    if args.force_ref:
        force_reference(True)
    torch.cuda.set_device(DEV)
    log("device", torch.cuda.get_device_name(DEV), torch.cuda.get_device_capability(DEV))
    init_weights()
    out = check(args) if args.mode == "check" else bench(args)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
