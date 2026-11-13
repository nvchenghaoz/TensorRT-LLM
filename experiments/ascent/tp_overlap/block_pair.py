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
WARMUP = 5
ITERS = 20


def log(*a):
    print(*a, flush=True)


def build(ckpt, rank):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    vgm = VisualGenMapping(2, rank, tp_size=2)
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
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(block.named_parameters())):
            g.manual_seed(1000 * i + 17 + rank)
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

    def call(fn):
        with torch.inference_mode():
            return fn(x, temb, idx, rot, None, None, valid_seq_len=None)

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
        return {"median_ms": med, "p10": ts[2], "p90": ts[17], "noise": (ts[17] - ts[2]) / med}

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
    q.put(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", default="time")
    ap.add_argument("--chunks", default="2,4,8")
    args = ap.parse_args()
    chunks = [int(c) for c in args.chunks.split(",") if c]
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
                       "chunks": r0["default_chunks"], "M": M},
        }))
    elif args.mode == "trace":
        print(json.dumps({"ms_per_block": r0["trace"], "rank1": res[1]["trace"]}))
    else:
        print(json.dumps({
            "passed": bool(r0["passed"] and res[1]["passed"]),
            "compared": "compiled block (serial and chunked overlap) vs eager serialized block, bitwise per iteration over 5 warmup + 20 iterations, both ranks; inputs read-only; no dynamo recompiles after warmup",
            "rank0": r0["check"], "rank1": res[1]["check"],
            "eager_serial_vs_compiled_serial": r0.get("eager_serial_vs_compiled_serial"),
        }))


if __name__ == "__main__":
    main()
