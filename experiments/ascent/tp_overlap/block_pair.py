"""2-GPU block-level benchmark/check of one MiniMax-H3 transformer block at TP2.

Builds the real MiniMaxH3TransformerBlock (checkpoint config, synthetic weights from a fixed
seed, per-rank shards), compiles it the way the pipeline does (torch.compile per block,
mode=default, dynamic=None) and times one block forward at the per-rank shape
(M = 37966 packed rows, BF16, VANILLA attention, no padding mask).

On main the block has no overlap path, so the same script times the serialized all-reduces;
on the branch it takes the chunked overlap (TLLM_H3_TP_OVERLAP_CHUNKS, default 4).

  --mode time : 5 warmup + 20 timed block forwards (CUDA events, rank 0), prints
                {"batch": {"1": {"step_ms", "noise_floor"}}, "output": {"class": "exact"}}.
  --mode check: eager serialized reference (overlap disabled) vs compiled block over the same
                5 warmup + 20 iterations, bitwise per iteration, inputs checked read-only,
                dynamo graph/recompile counters after warmup; prints {"passed": ...}.
Usage: python block_pair.py <checkpoint> --mode time|check [--chunks 2,4,8]
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M = 37966
block_vgm = []  # the VisualGenMapping build() made (one per process)
WARMUP = 5
ITERS = int(os.environ.get("BLOCK_PAIR_ITERS", "20"))


def log(*a):
    print(*a, flush=True)


def build(ckpt, rank):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    vgm = VisualGenMapping(2, rank, tp_size=2)
    block_vgm[:] = [vgm]
    mc = DiffusionModelConfig(
        pretrained_config=None, mapping=vgm.to_llm_mapping(), visual_gen_mapping=vgm
    )
    try:
        mc.attention.backend = "VANILLA"
    except Exception as e:  # default backend is VANILLA
        log(f"attention backend left at default {mc.attention.backend}: {e}")
    block = h3.MiniMaxH3TransformerBlock(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        ffn_dim=cfg["ffn_dim"],
        time_embed_dim=cfg["time_embed_dim"],
        norm_eps=float(cfg.get("norm_eps", 1e-6)),
        qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-6)),
        model_config=mc,
        layer_idx=0,
        fuse_bf16_gate_up_swiglu=True,
    ).cuda()
    if hasattr(block, "setup_tp_overlap"):  # the branch; main has no overlap path
        block.setup_tp_overlap(mc)
    g = torch.Generator(device="cuda")
    # TP-sharded Linear weights differ per rank; replicated params (norms, AdaLN table, biases of
    # replicated modules) are identical on both ranks, as in the real model, so the replicated
    # block output can be compared across ranks.
    from tensorrt_llm._torch.modules.linear import Linear as _Lin

    sharded = set()
    for mname, mod in block.named_modules():
        if isinstance(mod, _Lin) and getattr(mod, "tp_size", 1) > 1 and getattr(mod, "tp_mode", None) is not None:
            row = "ROW" in str(mod.tp_mode)
            for pname, _ in mod.named_parameters(recurse=False):
                if pname == "weight" or not row:  # a ROW-parallel bias is replicated
                    sharded.add(f"{mname}.{pname}" if mname else pname)
    log(f"rank {rank}: {len(sharded)} TP-sharded params seeded per rank: {sorted(sharded)}")
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(block.named_parameters())):
            g.manual_seed(1000 * i + 17 + (rank if name in sharded else 0))
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, device="cuda", generator=g) / p.shape[1] ** 0.5)
            elif "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, device="cuda", generator=g))
            else:
                p.copy_(0.02 * torch.randn(p.shape, device="cuda", generator=g))
    for mod in block.modules():
        if hasattr(mod, "post_load_weights") and mod is not block:
            try:
                mod.post_load_weights()
            except Exception:
                pass
    rope = h3.MiniMaxH3RotaryPosEmbed(int(cfg.get("rope_freq_dim", 16)), float(cfg.get("rope_theta", 10000.0))).cuda()
    g.manual_seed(42)  # shared activations
    D = cfg["hidden_size"]
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16, generator=g)
    temb = torch.randn(2, cfg["time_embed_dim"], device="cuda", generator=g)
    idx = torch.randint(0, 6, (M,), device="cuda", generator=g)
    pos = torch.randint(0, 64, (M, 3), device="cuda", generator=g)
    rot = rope(pos)
    return block, (x, temb, idx, rot), cfg


def worker(rank, ckpt, mode, chunks, q):
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29711")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    block, (x, temb, idx, rot), cfg = build(ckpt, rank)
    has_overlap = hasattr(block, "_tp_overlap_chunks")
    default_chunks = getattr(block, "_tp_overlap_chunks", 0)
    if rank == 0:
        log(f"overlap path present: {has_overlap}, default chunks {default_chunks}")
    pristine = [x.clone(), temb.clone(), idx.clone(), rot[0].clone(), rot[1].clone()]

    blocks_list = [int(b) for b in os.environ.get("BLOCK_PAIR_BLOCKS", "1").split(",") if b]

    def call(fn):
        with torch.inference_mode():
            return fn(x, temb, idx, rot, None, None, valid_seq_len=None)

    def call_chain(fn, nb):
        """nb chained block forwards (block i's output feeds block i+1)."""
        with torch.inference_mode():
            y = x
            for _ in range(nb):
                y = fn(y, temb, idx, rot, None, None, valid_seq_len=None)
            return y

    if mode == "time" and blocks_list != [1]:
        nb_time = blocks_list[0]
        call = lambda fn: call_chain(fn, nb_time)  # noqa: E731

    def timed(fn):
        for _ in range(WARMUP):
            call(fn)
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            dist.barrier()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            call(fn)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med = statistics.median(ts)
        lo, hi = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1) + 0.5)]
        return {"median_ms": med, "p10": lo, "p90": hi, "noise": (hi - lo) / med, "n": len(ts)}

    out = {"rank": rank, "has_overlap": has_overlap, "default_chunks": default_chunks}
    compiled = torch.compile(block, mode="default", dynamic=None)
    if mode == "time":
        out["timing"] = timed(compiled)
        if rank == 0:
            log(f"block forward: {out['timing']}")
    elif mode == "trace":
        # Serialized vs overlapped, 10 block forwards each inside an NVTX range.
        out["trace"] = {}
        for k in [0] + chunks:
            if has_overlap:
                block._tp_overlap_chunks = k
            torch._dynamo.reset()
            fn = torch.compile(block, mode="default", dynamic=None)
            for _ in range(WARMUP):
                call(fn)
            torch.cuda.synchronize()
            dist.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"blocks_k{k}")
            s.record()
            for _ in range(10):
                call(fn)
            e.record()
            torch.cuda.nvtx.range_pop()
            e.synchronize()
            out["trace"][f"k{k}"] = s.elapsed_time(e) / 10
            if rank == 0:
                log(f"trace k={k}: {out['trace'][f'k{k}']:.3f} ms/block")
    elif mode == "check":
        from torch._dynamo.utils import counters

        ok = True
        spg = getattr(block, "_tp_overlap_pg", None)
        info = {"dedicated": spg is not None}
        if spg is not None:
            try:
                be = spg._get_backend(torch.device("cuda"))
                info["is_high_priority_stream"] = bool(be.options.is_high_priority_stream)
                info["min_ctas"] = int(be.options.config.min_ctas)
                info["max_ctas"] = int(be.options.config.max_ctas)
            except Exception as ex:
                info["error"] = repr(ex)
        out["shipped_pg"] = info
        if rank == 0:
            log(f"shipped TP overlap group: {info}")
        # The serving loader builds the model with skip_create_weights_in_init=True; the
        # overlap setup must work (and enable) before the weights exist.
        if has_overlap:
            from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

            with open(os.path.join(ckpt, "transformer", "config.json")) as f:
                cfg = json.load(f)
            from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

            vgm = block_vgm[0]
            mc2 = DiffusionModelConfig(
                pretrained_config=None,
                mapping=vgm.to_llm_mapping(),
                visual_gen_mapping=vgm,
                skip_create_weights_in_init=True,
            )
            lazy = h3.MiniMaxH3TransformerBlock(
                hidden_size=cfg["hidden_size"],
                num_attention_heads=cfg["num_attention_heads"],
                attention_head_dim=cfg["attention_head_dim"],
                ffn_dim=cfg["ffn_dim"],
                time_embed_dim=cfg["time_embed_dim"],
                norm_eps=float(cfg.get("norm_eps", 1e-6)),
                qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-6)),
                model_config=mc2,
                layer_idx=1,
                fuse_bf16_gate_up_swiglu=True,
            )
            lazy.setup_tp_overlap(mc2)
            out["lazy_weights_setup_chunks"] = lazy._tp_overlap_chunks
            ok = ok and lazy._tp_overlap_chunks == default_chunks > 1
            if rank == 0:
                log(f"skip_create_weights_in_init block: overlap chunks {lazy._tp_overlap_chunks}")
            del lazy
        # Reference: the serialized block as main compiles it (overlap off; its all-reduces
        # go through allreduce_pg_by_name -> c10d NCCL all_reduce), first call.
        if has_overlap:
            block._tp_overlap_chunks = 0
        ref = None
        eager_serial = call(block).clone()
        torch.cuda.synchronize()
        settings = [0] + (chunks if has_overlap else [])
        out["check"] = {}
        for k in settings:
            if has_overlap:
                block._tp_overlap_chunks = k
            torch._dynamo.reset()
            counters.clear()
            fn = torch.compile(block, mode="default", dynamic=None)
            bad = []
            maxd = 0.0
            graphs_after_warmup = None
            for it in range(WARMUP + ITERS):
                y = call(fn)
                torch.cuda.synchronize()
                if ref is None:
                    ref = y.clone()
                    d = (eager_serial.float() - ref.float())
                    out["eager_serial_vs_compiled_serial"] = {
                        "bitwise": bool(torch.equal(eager_serial, ref)),
                        "max_abs": d.abs().max().item(),
                        "rel_l2": (d.norm() / ref.float().norm()).item(),
                    }
                    if rank == 0:
                        log(f"eager vs compiled serial: {out['eager_serial_vs_compiled_serial']}")
                if it == WARMUP - 1:
                    graphs_after_warmup = dict(counters["stats"])
                if not torch.equal(y, ref):
                    bad.append(it)
                    maxd = max(maxd, (y.float() - ref.float()).abs().max().item())
                for t, p in zip((x, temb, idx, rot[0], rot[1]), pristine):
                    if not torch.equal(t, p):
                        bad.append(f"input_changed@{it}")
            graphs_end = dict(counters["stats"])
            gb = {k2: v for k2, v in counters["graph_break"].items()}
            recompiled = graphs_after_warmup != graphs_end
            out["check"][f"k{k}"] = {
                "bitwise_all_iters": not bad,
                "bad": bad[:10],
                "max_abs": maxd,
                "dynamo_stats_after_warmup": graphs_after_warmup,
                "dynamo_stats_end": graphs_end,
                "recompiled_after_warmup": recompiled,
                "graph_breaks": len(gb),
            }
            ok = ok and not bad and not recompiled
            if rank == 0:
                log(f"check chunks={k}: {out['check'][f'k{k}']}")
        out["passed"] = ok
    elif mode == "arms":
        out.update(arms_phase(rank, block, compiled, call, (x, temb, idx, rot), pristine))
    elif mode == "stage1":
        out.update(stage1_phase(rank, block, call_chain, (x, temb, idx, rot), pristine, blocks_list, ITERS))
    q.put(out)
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------------------------
# Idea 72 stage 0b: dedicated TP PG (CTAs, stream priority) and tapered splits, all in-process.
# ---------------------------------------------------------------------------------------------
SPLITS = {
    "u4": None,  # main: ceil-aligned uniform 4 (9600/9600/9600/9166)
    "u6": (1, 1, 1, 1, 1, 1),
    "u8": (1,) * 8,
    "t1331": (1, 3, 3, 1),
    "t1555": (1, 5, 5, 5),
    "k6s": (2, 2, 2, 2, 2, 1),
}
PGS = {  # name -> (ctas, high_priority); "shared" = main's TP group
    "shared": None,
    "ded": (0, False),
    "ded_hp": (0, True),
    "c8": (8, False),
    "c8_hp": (8, True),
    "c6": (6, False),
    "c6_hp": (6, True),
    "c16_hp": (16, True),
}
# Set "1" (after job 427): high-priority stream combined with tapered splits.
ARMS_1 = [
    ("main", "shared", "u4"), ("main_aa", "shared", "u4"),
    ("ded_hp", "ded_hp", "u4"), ("c8_hp", "c8_hp", "u4"), ("c16_hp", "c16_hp", "u4"),
    ("c8", "c8", "u4"),
    ("ded_hp_t1331", "ded_hp", "t1331"), ("ded_hp_t1555", "ded_hp", "t1555"),
    ("c8_hp_t1331", "c8_hp", "t1331"), ("c8_hp_t1555", "c8_hp", "t1555"),
    ("c8_hp_u6", "c8_hp", "u6"), ("c8_t1555", "c8", "t1555"),
]
PROFILE_ARMS_1 = ["main", "ded_hp", "c8_hp", "c8", "c8_hp_t1555", "ded_hp_t1555", "c8_hp_t1331"]
ARMS = [
    ("main", "shared", "u4"), ("main_aa", "shared", "u4"),
    ("ded", "ded", "u4"), ("ded_hp", "ded_hp", "u4"),
    ("c8", "c8", "u4"), ("c8_hp", "c8_hp", "u4"),
    ("c6", "c6", "u4"), ("c6_hp", "c6_hp", "u4"),
    ("c8_t1331", "c8", "t1331"), ("c8_t1555", "c8", "t1555"),
    ("c8_u6", "c8", "u6"), ("c8_u8", "c8", "u8"), ("c8_k6s", "c8", "k6s"),
    ("shared_t1331", "shared", "t1331"),
]
PROFILE_ARMS = ["main", "c8", "c8_hp", "ded_hp", "c6", "c8_t1331", "c8_u6", "c8_k6s"]


def _union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _measure(u):
    return sum(e - s for s, e in u)


def _minus(a, b):
    """|a \\ b| for unions a, b."""
    tot = 0.0
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                tot += b[k][0] - cur
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            tot += e - cur
    return tot


def _is_gemm(n):
    n = n.lower()
    return ("nvjet" in n or "gemm" in n or "cutlass" in n) and "flash" not in n and "fmha" not in n


def _analyze_trace(path):
    """Per-iteration exposure metrics from a kineto chrome trace (times in ms)."""
    with open(path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern, launch = [], {}
    for e in evs:
        cat = e.get("cat", "")
        a = e.get("args", {}) or {}
        if cat == "kernel":
            kern.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", ""),
                         a.get("correlation"), a.get("stream"), tuple(a.get("grid", []) or [])))
        elif cat in ("cuda_runtime", "cuda_driver") and "aunch" in e.get("name", ""):
            launch[a.get("correlation")] = float(e["ts"])
    kern.sort()
    iters, cur = [], []
    for k in kern:
        if cur and k[0] - max(c[1] for c in cur[-50:]) > 5000:
            iters.append(cur)
            cur = []
        cur.append(k)
    if cur:
        iters.append(cur)
    res = []
    for ks in iters:
        # drop the 1-element f32 barrier all-reduce (dist.barrier before each profiled call)
        ks = [k for k in ks if not ("nccl" in k[2].lower() and "f32" in k[2].lower())]
        nc = [k for k in ks if "nccl" in k[2].lower()]
        cp = [k for k in ks if "nccl" not in k[2].lower()]
        if not cp:
            continue
        uc = _union([(k[0], k[1]) for k in cp])
        un = _union([(k[0], k[1]) for k in nc])
        wall = max(k[1] for k in ks) - min(k[0] for k in ks)
        # drain: compute idle between the last down GEMM work and the end of the last AR (the
        # final gated residual waits for it)
        if nc:
            last_ar_end = max(k[1] for k in nc)
            before = [k[1] for k in cp if k[1] <= last_ar_end]
            drain = max(0.0, last_ar_end - max(before)) if before else 0.0
            last_gemm_end = max((k[1] for k in cp if _is_gemm(k[2])), default=None)
            res_tail = (last_ar_end - last_gemm_end) if last_gemm_end else None
        else:
            drain, res_tail = 0.0, None
        busy = _measure(uc)
        # NCCL launch-to-start delay: start - max(CPU launch, previous AR end, producer end)
        delays = []
        comp_by_launch = sorted((launch.get(k[3], k[0]), k[1]) for k in cp)
        prev_end = None
        for k in sorted(nc):
            lt = launch.get(k[3])
            ready = lt if lt is not None else k[0]
            # producer: the last compute kernel the CPU launched before this AR (its input GEMM)
            prod = [ce for (cl, ce) in comp_by_launch if lt is not None and cl < lt]
            if prod:
                ready = max(ready, prod[-1])
            if prev_end is not None:
                ready = max(ready, prev_end)
            delays.append(k[0] - ready)
            prev_end = k[1]
        res.append({
            "wall": wall / 1e3,
            "ar_n": len(nc),
            "ar_sum": sum(k[1] - k[0] for k in nc) / 1e3,
            "ar_each": [round((k[1] - k[0]) / 1e3, 3) for k in sorted(nc)],
            "ar_uncovered": _minus(un, uc) / 1e3,
            "compute_sum": sum(k[1] - k[0] for k in cp) / 1e3,
            "gemm_sum": sum(k[1] - k[0] for k in cp if _is_gemm(k[2])) / 1e3,
            "compute_busy": busy / 1e3,
            "compute_idle_in_block": (wall - busy) / 1e3,
            "fill_idle": (wall - busy - drain) / 1e3,
            "drain": drain / 1e3,
            "last_ar_end_minus_last_gemm_end": (res_tail / 1e3) if res_tail is not None else None,
            "nccl_delay_ms": [round(d / 1e3, 3) for d in delays],
            "nccl_grids": sorted({str(k[5]) for k in nc}),
            "nccl_names": sorted({k[2][:60] for k in nc}),
        })
    return res


def arms_phase(rank, block, compiled, call, inputs, pristine):
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    x = inputs[0]
    gname = block._tp_overlap_group_name
    base_chunks = block._tp_overlap_chunks
    res = {"group_name": str(gname), "base_chunks": base_chunks,
           "splits": {k: h3._row_chunks(M, 4, weights=w) for k, w in SPLITS.items()}}
    # Process groups. Each dedicated group is created (world-collective) in the same order on
    # both ranks; device memory before/after shows the cost of a second communicator.
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    pgs = {"shared": dist.distributed_c10d._resolve_process_group(gname)}
    mem = {}
    arm_set = os.environ.get("BLOCK_PAIR_ARMS", "0b")
    ARMS_SEL = ARMS_1 if arm_set == "1" else ARMS
    PROF_SEL = PROFILE_ARMS_1 if arm_set == "1" else PROFILE_ARMS
    check_only = os.environ.get("BLOCK_PAIR_CHECK_ONLY") == "1"
    if check_only:  # every arm of both sets
        seen = {a[0] for a in ARMS}
        ARMS_SEL = list(ARMS) + [a for a in ARMS_1 if a[0] not in seen]
        PROF_SEL = []
    res["arm_set"] = arm_set + ("+check_only" if check_only else "")
    needed = {a[1] for a in ARMS_SEL}
    for name, spec in PGS.items():
        if spec is None or name not in needed:
            continue
        f0 = torch.cuda.mem_get_info()[0]
        opts = dist.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = spec[1]
        if spec[0]:
            opts.config.min_ctas = spec[0]
            opts.config.max_ctas = spec[0]
        pgs[name] = dist.new_group([0, 1], backend="nccl", pg_options=opts)
        t = torch.ones(1 << 20, device="cuda", dtype=torch.bfloat16)
        dist.all_reduce(t, group=pgs[name])
        torch.cuda.synchronize()
        mem[name] = round((f0 - torch.cuda.mem_get_info()[0]) / 2**20, 1)
    # the shipped helper (default setting) must hand back an equivalent group
    helper_pg = h3.h3_dedicated_tp_group(gname, h3._TP_OVERLAP_PG_CTAS, h3._TP_OVERLAP_PG_HIGH_PRIORITY) \
        if hasattr(h3, "h3_dedicated_tp_group") else None
    res["helper_pg_built"] = helper_pg is not None
    res["pg_mem_mib"] = mem
    res["pg_mem_total_mib"] = round((free0 - torch.cuda.mem_get_info()[0]) / 2**20, 1)

    def set_arm(arm):
        _, pgn, sp = arm
        block._tp_overlap_pg = pgs[pgn]
        block._tp_overlap_weights = SPLITS[sp]
        block._tp_overlap_chunks = base_chunks

    # Reference: the serialized block as main compiles it (overlap off: full-M to_out -> AR ->
    # ... -> down -> AR through the shared group), the reference landing 8 was proven against.
    block._tp_overlap_chunks = 0
    with torch.inference_mode():
        eager = call(block).clone()
        ref = call(compiled).clone()
        ref2 = call(compiled)
    block._tp_overlap_chunks = base_chunks
    torch.cuda.synchronize()
    d = (eager.float() - ref.float())
    res["eager_serial_vs_compiled_serial"] = {"bitwise": bool(torch.equal(eager, ref)),
                                              "max_abs": d.abs().max().item()}
    res["compiled_serial_repeat_bitwise"] = bool(torch.equal(ref, ref2))
    peer = ref.clone()
    dist.broadcast(peer, src=0)
    res["reference_rank0_eq_rank1"] = bool(torch.equal(peer, ref))
    del eager, ref2, peer, d
    if rank == 0:
        log(f"reference (compiled serial) ready: {res['eager_serial_vs_compiled_serial']}, "
            f"ranks equal {res['reference_rank0_eq_rank1']}")

    # ---- timing: interleaved rounds, every arm in every round, rotated order -------------------
    ROUNDS, WU, TI = (0 if check_only else 5), 2, 8
    times = {a[0]: [] for a in ARMS_SEL}
    for arm in ARMS_SEL:  # compile/warm every arm once (no recompiles expected: pg/weights not guarded)
        set_arm(arm)
        for _ in range(WARMUP):
            call(compiled)
    torch.cuda.synchronize()
    for r in range(ROUNDS):
        order = ARMS_SEL[r % len(ARMS_SEL):] + ARMS_SEL[:r % len(ARMS_SEL)]
        if r % 2:
            order = order[::-1]
        for arm in order:
            set_arm(arm)
            for _ in range(WU):
                call(compiled)
            for _ in range(TI):
                dist.barrier()
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                call(compiled)
                e.record()
                e.synchronize()
                times[arm[0]].append(s.elapsed_time(e))
        if rank == 0:
            log(f"round {r} done: " + ", ".join(f"{k}={statistics.median(v):.2f}" for k, v in times.items()))

    def st(ts):
        ts = sorted(ts)
        n = len(ts)
        med = statistics.median(ts)
        p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1) + 0.5)]
        return {"median_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                "noise": round((p90 - p10) / med, 4), "n": n}

    res["timing"] = {k: st(v) for k, v in times.items() if v}

    # ---- untimed repeat check: same warmup/iteration counts, bitwise vs reference ---------------
    check = {}
    ok = True
    for arm in ARMS_SEL:
        set_arm(arm)
        counters.clear()
        bad = []
        stats_wu = None
        for it in range(WARMUP + ITERS):
            poison = torch.full_like(x, float("nan"))
            del poison
            y = call(compiled)
            torch.cuda.synchronize()
            if not torch.equal(y, ref):
                bad.append(it)
            peer = y.clone()
            dist.broadcast(peer, src=0)
            if not torch.equal(peer, y):
                bad.append(f"rank_mismatch@{it}")
            for t, p in zip((inputs[0], inputs[1], inputs[2], inputs[3][0], inputs[3][1]), pristine):
                if not torch.equal(t, p):
                    bad.append(f"input_changed@{it}")
            if it == WARMUP - 1:
                stats_wu = dict(counters["stats"])
            del y, peer
        rec = dict(counters["stats"]) != stats_wu
        check[arm[0]] = {"bitwise_all_iters": not bad, "bad": bad[:6], "recompiled_after_warmup": rec}
        ok = ok and not bad and not rec
    res["check"] = check
    res["check_passed"] = ok
    if rank == 0:
        log(f"check passed={ok}: {[(k, v['bitwise_all_iters']) for k, v in check.items()]}")
    res["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # ---- untimed torch.profiler pass, ~3 iterations per arm ------------------------------------
    from torch.profiler import ProfilerActivity, profile

    prof_res = {}
    arms_by_name = {a[0]: a for a in ARMS_SEL}
    plist = ([(n, arms_by_name[n]) for n in PROF_SEL] + [("serial", None)]) if PROF_SEL else []
    for name, arm in plist:
        if arm is None:
            block._tp_overlap_chunks = 0
        else:
            set_arm(arm)
        for _ in range(3):
            call(compiled)
        torch.cuda.synchronize()
        dist.barrier()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                dist.barrier()
                torch.cuda.synchronize()
                call(compiled)
                torch.cuda.synchronize()
                time.sleep(0.03)
        path = os.path.join(art, f"trace_{name}.r{rank}.json")
        prof.export_chrome_trace(path)
        try:
            it = _analyze_trace(path)
            prof_res[name] = it
        except Exception as ex:
            prof_res[name] = {"error": repr(ex)}
        if rank == 0:
            log(f"profiled {name}: {len(prof_res[name]) if isinstance(prof_res[name], list) else prof_res[name]}")
    block._tp_overlap_chunks = base_chunks
    # summarize: median over iterations
    summ = {}
    ser = prof_res.get("serial")
    ser_gemm = statistics.median([i["gemm_sum"] for i in ser]) if isinstance(ser, list) and ser else None
    ser_comp = statistics.median([i["compute_sum"] for i in ser]) if isinstance(ser, list) and ser else None
    ser_ar = statistics.median([e for i in ser for e in i["ar_each"]]) if isinstance(ser, list) and ser else None
    for name, its in prof_res.items():
        if not isinstance(its, list) or not its:
            continue
        md = lambda k: round(statistics.median([i[k] for i in its]), 3)
        d = {k: md(k) for k in ("wall", "ar_sum", "ar_uncovered", "compute_sum", "gemm_sum", "compute_busy",
                                "compute_idle_in_block", "fill_idle", "drain",
                                "last_ar_end_minus_last_gemm_end")}
        d["ar_n"] = its[0]["ar_n"]
        d["ar_each_med"] = [round(statistics.median([i["ar_each"][j] for i in its if len(i["ar_each"]) > j]), 3)
                            for j in range(its[0]["ar_n"])]
        dl = sorted(x for i in its for x in i["nccl_delay_ms"])
        d["nccl_delay_med_ms"] = round(statistics.median(dl), 3) if dl else None
        d["nccl_delay_max_ms"] = round(max(dl), 3) if dl else None
        d["nccl_grids"] = its[0]["nccl_grids"]
        d["nccl_names"] = its[0]["nccl_names"][:3]
        if ser_gemm:
            d["stretch_gemm"] = round(d["gemm_sum"] / ser_gemm, 4)
            d["stretch_compute_ms"] = round(d["compute_sum"] - ser_comp, 3)
            d["stretch_compute"] = round(d["compute_sum"] / ser_comp, 4)
        if ser_ar:
            d["serial_ar_full_M_ms"] = ser_ar
        summ[name] = d
    res["profile"] = summ
    return res


# ---------------------------------------------------------------------------------------------
# Idea 106 Stage 1: per-chunk final gated residual under the last down ARs, on main's ded_hp PG.
# Arms toggle MiniMaxH3TransformerBlock._tp_overlap_chunk_residual on the same compiled block
# (one dynamo variant each, compiled in warmup). --blocks N chains the block N times.
# ---------------------------------------------------------------------------------------------
STAGE1_ARMS = [("main_a", (False, 0)), ("stage1", (True, 0)), ("stage1_g188", (True, 188)),
               ("stage1_g64", (True, 64)), ("stage1_g24", (True, 24)), ("main_b", (False, 0))]
if os.environ.get("BLOCK_PAIR_STAGE1_ARMS"):
    _sel = os.environ["BLOCK_PAIR_STAGE1_ARMS"].split(",")
    STAGE1_ARMS = [a for a in STAGE1_ARMS if a[0] in _sel]


def _tail_metrics(path):
    """Per iteration: compute that ends after the last NCCL AR (exposed tail) and compute kernel
    time overlapping the last AR, from a kineto trace (ms)."""
    with open(path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", ""))
                  for e in evs if e.get("cat") == "kernel")
    iters, cur = [], []
    for k in kern:
        if cur and k[0] - max(c[1] for c in cur[-50:]) > 5000:
            iters.append(cur)
            cur = []
        cur.append(k)
    if cur:
        iters.append(cur)
    out = []
    for ks in iters:
        ks = [k for k in ks if not ("nccl" in k[2].lower() and "f32" in k[2].lower())]
        nc = sorted(k for k in ks if "nccl" in k[2].lower())
        cp = [k for k in ks if "nccl" not in k[2].lower()]
        if not nc or not cp:
            continue
        last = nc[-1]
        end = max(k[1] for k in cp)
        under = _measure(_union([(max(k[0], last[0]), min(k[1], last[1])) for k in cp
                                 if k[1] > last[0] and k[0] < last[1]]))
        out.append({"exposed_tail_after_last_ar": round(max(0.0, end - last[1]) / 1e3, 4),
                    "compute_under_last_ar": round(under / 1e3, 4),
                    "last_ar_ms": round((last[1] - last[0]) / 1e3, 4),
                    "launches_after_last_down_gemm": sum(
                        1 for k in cp if k[0] >= max((c[1] for c in cp if _is_gemm(c[2])), default=0))})
    return out


def stage1_phase(rank, block, call_chain, inputs, pristine, blocks_list, iters):
    from torch._dynamo.utils import counters

    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {"has_chunk_residual": hasattr(block, "_tp_overlap_chunk_residual")}
    spg = block._tp_overlap_pg
    try:
        be = spg._get_backend(torch.device("cuda"))
        res["pg"] = {"dedicated": True, "is_high_priority_stream": bool(be.options.is_high_priority_stream)}
    except Exception as ex:
        res["pg"] = {"dedicated": spg is not None, "error": repr(ex)}
    torch._dynamo.reset()
    compiled = torch.compile(block, mode="default", dynamic=None)

    def set_arm(flag):
        if isinstance(flag, tuple):
            flag, programs = flag
        else:
            programs = 0
        block._tp_overlap_chunk_residual = flag
        block._tp_overlap_residual_programs = programs

    # Reference: main's compiled chunked block (chunk_residual off), and the serialized block.
    nb_check = max(2, min(blocks_list))
    set_arm(False)
    ref = {nb: call_chain(compiled, nb).clone() for nb in sorted(set([1, nb_check]))}
    chunks = block._tp_overlap_chunks
    block._tp_overlap_chunks = 0
    serial1 = call_chain(compiled, 1).clone()
    block._tp_overlap_chunks = chunks
    torch.cuda.synchronize()
    res["main_chunked_eq_serial"] = bool(torch.equal(serial1, ref[1]))
    for flag in (True, False):  # compile both variants before any timing/counting
        set_arm(flag)
        for _ in range(3):
            call_chain(compiled, 1)
    torch.cuda.synchronize()

    # ---- untimed check: same warmup/iteration counts as timing, bitwise vs main, both ranks ----
    ok = res["main_chunked_eq_serial"]
    check = {}
    for name, flag in STAGE1_ARMS:
        set_arm(flag)
        counters.clear()
        bad, stats_wu = [], None
        for it in range(WARMUP + 20):
            for nb in sorted(ref):
                y = call_chain(compiled, nb)
                torch.cuda.synchronize()
                if not torch.equal(y, ref[nb]):
                    bad.append(f"nb{nb}@{it}")
                peer = y.clone()
                dist.broadcast(peer, src=0)
                if not torch.equal(peer, y):
                    bad.append(f"rank_mismatch_nb{nb}@{it}")
                del y, peer
            for t, p in zip((inputs[0], inputs[1], inputs[2], inputs[3][0], inputs[3][1]), pristine):
                if not torch.equal(t, p):
                    bad.append(f"input_changed@{it}")
            if it == WARMUP - 1:
                stats_wu = dict(counters["stats"])
        rec = dict(counters["stats"]) != stats_wu
        check[name] = {"bitwise_all_iters": not bad, "bad": bad[:6], "recompiled_after_warmup": rec}
        ok = ok and not bad and not rec
    res["check"] = check
    res["check_passed"] = ok
    if rank == 0:
        log(f"stage1 check passed={ok}: {check}")

    # ---- timing: interleaved arms, rotated order, per blocks setting ----
    def st(ts):
        ts = sorted(ts)
        n = len(ts)
        med = statistics.median(ts)
        p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1) + 0.5)]
        return {"median_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                "noise": round((p90 - p10) / med, 4), "n": n}

    res["timing"] = {}
    res["profile"] = {}
    if os.environ.get("BLOCK_PAIR_CHECK_ONLY") == "1":
        set_arm(True)
        res["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return res
    per_round = 4
    rounds = -(-iters // per_round)
    for nb in blocks_list:
        times = {a[0]: [] for a in STAGE1_ARMS}
        for r in range(rounds):
            k = r % len(STAGE1_ARMS)
            order = STAGE1_ARMS[k:] + STAGE1_ARMS[:k]
            if r % 2:
                order = order[::-1]
            for name, flag in order:
                set_arm(flag)
                call_chain(compiled, nb)
                for _ in range(per_round):
                    dist.barrier()
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    call_chain(compiled, nb)
                    e.record()
                    e.synchronize()
                    times[name].append(s.elapsed_time(e))
        t = {k: st(v) for k, v in times.items()}
        main_med = statistics.median(times["main_a"] + times["main_b"])
        t["aa_rel"] = round(t["main_b"]["median_ms"] / t["main_a"]["median_ms"] - 1, 5)
        for name, _ in STAGE1_ARMS:
            if name.startswith("stage1"):
                t[f"{name}_vs_main_rel"] = round(t[name]["median_ms"] / main_med - 1, 5)
                t[f"{name}_minus_main_ms_per_block"] = round((t[name]["median_ms"] - main_med) / nb, 4)
        res["timing"][f"blocks{nb}"] = t
        if rank == 0:
            log(f"blocks={nb}: {t}")

    # ---- untimed torch.profiler traces (1 block, 3 iterations per arm) ----
    from torch.profiler import ProfilerActivity, profile

    prof = {}
    for name, flag in STAGE1_ARMS:
        if name == "main_b":
            continue
        set_arm(flag)
        for _ in range(3):
            call_chain(compiled, 1)
        torch.cuda.synchronize()
        dist.barrier()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(3):
                dist.barrier()
                torch.cuda.synchronize()
                call_chain(compiled, 1)
                torch.cuda.synchronize()
                time.sleep(0.03)
        path = os.path.join(art, f"trace_stage1_{name}.r{rank}.json")
        p.export_chrome_trace(path)
        try:
            its = _analyze_trace(path)
            tails = _tail_metrics(path)
            md = lambda k: round(statistics.median([i[k] for i in its]), 3)
            prof[name] = {k: md(k) for k in ("wall", "ar_uncovered", "drain", "compute_busy",
                                               "last_ar_end_minus_last_gemm_end")}
            prof[name]["tail"] = tails
        except Exception as ex:
            prof[name] = {"error": repr(ex)}
    res["profile"] = prof
    if rank == 0:
        log(f"profile: {prof}")
    set_arm(True)
    res["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", default="time")
    ap.add_argument("--chunks", default="2,4,8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pair-check", action="store_true")
    ap.add_argument("--arms", default="0b", help="arm set for --mode arms: 0b or 1")
    ap.add_argument("--blocks", default="1", help="chained block forwards per timed call, e.g. 8 or 8,2")
    ap.add_argument("--timing-pair-only", action="store_true",
                    help="stage1: exit with wrong_pair unless the GPUs are the PIX pair 0-1 or 2-3")
    args = ap.parse_args()
    os.environ["BLOCK_PAIR_ARMS"] = args.arms
    os.environ["BLOCK_PAIR_BLOCKS"] = args.blocks
    global ITERS
    ITERS = args.iters
    os.environ["BLOCK_PAIR_ITERS"] = str(args.iters)  # spawned workers re-import this module
    chunks = [int(c) for c in args.chunks.split(",") if c]
    pair = None
    if args.pair_check:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ar_inplace_bench import pair_check

        pair = pair_check()
        # Timing only on the PIX x16 pairs 0-1, 2-3, 6-7 (pitfall 960, top-lead 07:45); else check-only.
        # Decided by physical ids + link widths (the node's PIX pairs are fixed facts, ruling 08:25);
        # a topo parse error must not refuse a PIX pair, and unknown ids fail loud.
        phys = pair.get("physical_ids")
        if not phys or len(phys) != 2:
            raise SystemExit(f"pair parser failed: physical_ids={phys!r}; fix the parser, do not resubmit")
        widths = pair.get("link_width_current") or []
        timing_pair = (set(phys) in ({0, 1}, {2, 3}, {6, 7})
                       and len(widths) == 2 and min(widths) >= 16)
        pair["timing_pair"] = timing_pair
        log(f"pair: phys {pair.get('physical_ids')} {pair.get('relation')} widths "
            f"{pair.get('link_width_current')} timing_pair={timing_pair}")
        if not timing_pair:
            node_x16 = (pair.get("relation") == "NODE" and not pair.get("wrong_pair")
                        and set(pair.get("physical_ids") or []) == {4, 6})
            if args.mode == "stage1" and args.timing_pair_only:
                print(json.dumps({"wrong_pair": True, "pair": pair}), flush=True)
                return
            if args.mode == "stage1":
                os.environ["BLOCK_PAIR_CHECK_ONLY"] = "1"
                log(f"pair {pair.get('physical_ids')}: stage1 untimed check only, wrong_pair for timing")
            elif args.mode == "check":
                log("non-timing pair: valid for the untimed bitwise check")
            elif args.mode == "check" and node_x16:
                log("NODE 4+6 x16: valid for the untimed bitwise check")
            elif args.mode == "arms" and node_x16:
                # NODE x16 (4+6) is valid for bitwise checks, not timing: run the untimed check only.
                os.environ["BLOCK_PAIR_CHECK_ONLY"] = "1"
                log("NODE 4+6: timing skipped, running the untimed bitwise check of every arm")
            else:
                print(json.dumps({"wrong_pair": True, "pair": pair}), flush=True)
                return
    if args.mode == "arms":
        art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV")
        os.environ.setdefault("NCCL_DEBUG_FILE", os.path.join(art, "nccl_debug.%h.%p.log"))
    # A fixed rendezvous port collides with sibling jobs (shared network namespace).
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(r, args.ckpt, args.mode, chunks, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = sorted([q.get(timeout=3000) for _ in range(2)], key=lambda r: r["rank"])
    for p in procs:
        p.join()
    log(f"done in {time.time() - t0:.0f}s; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    r0 = res[0]
    if args.mode == "time":
        t = r0["timing"]
        print(json.dumps({
            "batch": {"1": {"step_ms": t["median_ms"], "noise_floor": t["noise"]}},
            "output": {"class": "exact"},
            "detail": {"rank0": t, "rank1": res[1]["timing"], "overlap_path": r0["has_overlap"],
                       "chunks": r0["default_chunks"], "M": M, "blocks_per_call": args.blocks,
                       "pair": {k: (pair or {}).get(k) for k in (
                           "physical_ids", "relation", "link_width_current", "timing_pair")}},
        }))
    elif args.mode == "trace":
        print(json.dumps({"ms_per_block": r0["trace"], "rank1": res[1]["trace"]}))
    elif args.mode == "stage1":
        art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
        with open(os.path.join(art, "stage1_result.json"), "w") as f:
            json.dump({"pair": pair, "rank0": r0, "rank1": res[1]}, f, indent=1)
        print(json.dumps({
            "passed": bool(r0["check_passed"] and res[1]["check_passed"]),
            "compared": "compiled block chunk_residual on (stage1) and off (main_a/main_b) vs main's "
                        "compiled chunked block, 1 and 2 chained blocks, bitwise per iteration over 5 "
                        "warmup + 20, rank0 == rank1, inputs read-only, no recompiles after warmup",
            "timing_valid": bool((pair or {}).get("timing_pair")),
            "wrong_pair": not (pair or {}).get("timing_pair"),
            "pair": {k: (pair or {}).get(k) for k in ("physical_ids", "relation", "link_width_current")},
            "pg": [r0.get("pg"), res[1].get("pg")],
            "main_chunked_eq_serial": [r0.get("main_chunked_eq_serial"), res[1].get("main_chunked_eq_serial")],
            "check_rank0": r0["check"], "check_rank1": res[1]["check"],
            "timing_rank0": r0["timing"], "timing_rank1": res[1]["timing"],
            "profile_rank0": r0["profile"], "profile_rank1": res[1]["profile"],
            "peak_mem_gib": [r0["peak_mem_gib"], res[1]["peak_mem_gib"]],
        }))
    elif args.mode == "arms":
        import glob

        art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
        nccl_lines = []
        for fn in sorted(glob.glob(os.path.join(art, "nccl_debug.*.log"))):
            with open(fn, errors="replace") as f:
                for ln in f:
                    if any(k in ln for k in ("coll channels", "CTA", "NCHANNELS", "set by environment",
                                             "minCTAs", "maxCTAs", "Config")):
                        nccl_lines.append(ln.strip()[-220:])
        t0m = r0["timing"]
        base = t0m["main"]["median_ms"] if "main" in t0m else None
        rel = {k: round(v["median_ms"] / base - 1, 4) for k, v in t0m.items()} if base else {}
        full = {"pair": pair, "rank0": r0, "rank1": res[1], "nccl_debug_lines": nccl_lines[:80]}
        with open(os.path.join(art, "arms_result.json"), "w") as f:
            json.dump(full, f, indent=1)
        print(json.dumps({
            "passed": bool(r0["check_passed"] and res[1]["check_passed"]
                           and all(rr.get("reference_rank0_eq_rank1") for rr in res)),
            "timing_valid": bool(t0m) and (pair or {}).get("relation") == "PIX",
            "pair": {k: (pair or {}).get(k) for k in ("physical_ids", "relation", "link_width_current")},
            "rel_vs_main": rel, "aa": rel.get("main_aa"),
            "timing_rank0": t0m,
            "arm_set": r0.get("arm_set"),
            "reference": {k: [r0.get(k), res[1].get(k)] for k in (
                "eager_serial_vs_compiled_serial", "compiled_serial_repeat_bitwise", "reference_rank0_eq_rank1")},
            "check_rank0": {k: v["bitwise_all_iters"] for k, v in r0["check"].items()},
            "check_bad_rank0": {k: v["bad"] for k, v in r0["check"].items() if v["bad"]},
            "check_rank1": {k: v["bitwise_all_iters"] for k, v in res[1]["check"].items()},
            "pg_mem_mib": [r0["pg_mem_mib"], res[1]["pg_mem_mib"]],
            "peak_mem_gib": [r0["peak_mem_gib"], res[1]["peak_mem_gib"]],
            "helper_pg_built": [r0["helper_pg_built"], res[1]["helper_pg_built"]],
            "profile_rank0": r0["profile"], "profile_rank1": res[1]["profile"],
            "splits": r0["splits"],
            "nccl_debug_lines": nccl_lines[:30],
        }))
    else:
        print(json.dumps({
            "passed": bool(r0["passed"] and res[1]["passed"]),
            "compared": "compiled block (serial and chunked overlap) vs eager serialized block, bitwise per iteration over 5 warmup + 20 iterations, both ranks; inputs read-only; no dynamo recompiles after warmup",
            "rank0": r0["check"], "rank1": res[1]["check"],
            "eager_serial_vs_compiled_serial": r0.get("eager_serial_vs_compiled_serial"),
            "lazy_weights_setup_chunks": [r0.get("lazy_weights_setup_chunks"),
                                          res[1].get("lazy_weights_setup_chunks")],
            "shipped_pg": [r0.get("shipped_pg"), res[1].get("shipped_pg")],
            "pair": {k: (pair or {}).get(k) for k in ("physical_ids", "relation", "link_width_current")},
        }))


if __name__ == "__main__":
    main()
