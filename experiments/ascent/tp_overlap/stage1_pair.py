"""2-GPU check + interleaved timing of idea 136 / branch 49 Stage 1 on main 271ae68f (MiniMax-H3 TP2).

Builds NB real MiniMaxH3TransformerBlocks (checkpoint config, synthetic weights from a fixed seed per
block, per-rank TP shards), main's chunked TP overlap on the dedicated high-priority PG, landing 35's
AdaLN prefetch installed through the model's own setup_adaln_prefetch (drain hooks block i -> i+1),
and the engine.json attention config (FLASHINFER, FP8 q/k/v from forward K = attention_bf16_forwards,
BF16 fa2_sm120 before) with a DenoiseStepController in the attention metadata state, as the pipeline
does. Each block is compiled like the pipeline (torch.compile per block, mode=default, dynamic=None).
A call = set_step(forward) then nb chained blocks at the per-rank shape (M = 37966, BF16).

Arms (all in one process, switched per call on every block):
  main_a / main_b : _tp_overlap_chunk_residual False (main's code path verbatim; the A/A pair)
  stage1          : per-chunk residual, uncapped grid
  g188 / g64 / g24: per-chunk residual, chunks 0..n-2 capped at 188 / 64 / 24 programs
  noar            : TIMING ONLY, NOT EXACT: main's path with every chunk all-reduce skipped
                    (torch.distributed.all_reduce on the overlap PG returns a completed no-op work).
On main's tree (run_gpu_pair baseline) the chunk_residual attribute is ignored, so every non-noar
arm runs main's code: the in-job harness then shows its own null result.

  --mode check: untimed. Per phase (forward 9 BF16 attention, forward 20 FP8 attention) and per
      chain (1 and 2 blocks): reference = main_a's first output; every exact arm runs WARMUP+ITERS
      iterations, each bitwise vs the reference, rank0 == rank1 (broadcast), output buffers
      NaN-poisoned before every iteration (allocator pool filled with NaN blocks of the output size),
      inputs checked read-only before every reset point; dynamo stats after warmup vs end (no
      recompiles); adaln_prefetch hit/miss/prefetch/drop counts per arm equal to main_a's; event
      order of all-reduce issue / work.wait / drain hook per block equal to main's, with the hook
      after the last down all-reduce and before the first ff wait. Prints {"passed": ...}.
  --mode time : the check above with ITERS=25 first (same job), then per --blocks and per phase
      interleaved rotated rounds of every arm (1 untimed + 4 timed calls per arm per round, n>=40/arm),
      CUDA events after barrier+sync, median p10/p90 per arm and rank; A/A = main_b vs main_a;
      then an untimed torch.profiler pass per arm (2-block chain): per rank, block 0's AR d3, the
      compute busy under it (last down GEMM end -> AR d3 end), the tail from AR d3 end to block 1's
      QKV GEMM. Prints {"batch": {"1": {"step_ms", "noise_floor"}}, "output": {"class": "exact"}, ...}
      for --report-arm (main_a on the baseline tree, stage1 on the branch).
Usage: python stage1_pair.py <ckpt> --mode check|time [--blocks 8,2] [--iters 40] [--pair-check]
       [--timing-pair-only] [--report-arm stage1]
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M = 37966
WARMUP = 5
CHECK_ITERS = 25
ROOT = Path(__file__).resolve().parents[3]
PHASES = (("f9", 9), ("f20", 20))
ARMS = [
    ("main_a", (False, 0, False)),
    ("stage1", (True, None, False)),  # the shipped default grid cap (setup_tp_overlap: SM count)
    ("g0", (True, 0, False)),  # uncapped: one program per row
    ("g64", (True, 64, False)),
    ("main_b", (False, 0, False)),
    ("noar", (False, 0, True)),
]
EXACT_ARMS = [a for a in ARMS if a[0] != "noar"]


def log(*a):
    print(*a, flush=True)


def st(ts):
    ts = sorted(ts)
    n = len(ts)
    med = statistics.median(ts)
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1) + 0.5)]
    return {"median_ms": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
            "noise": round((p90 - p10) / med, 4), "n": n}


# ------------------------------------------------------------------------------------------------
# all_reduce wrapper (installed once, before any compile): NOAR switch + event-order log
# ------------------------------------------------------------------------------------------------
AR_STATE = {"noar": False, "log": None, "pgs": set()}
_ORIG_AR = dist.all_reduce


class _DoneWork:
    def wait(self, *a, **k):
        if AR_STATE["log"] is not None:
            AR_STATE["log"].append("W")
        return True

    def is_completed(self):
        return True


class _LogWork:
    def __init__(self, w):
        self._w = w

    def wait(self, *a, **k):
        AR_STATE["log"].append("W")
        return self._w.wait(*a, **k)

    def __getattr__(self, n):
        return getattr(self._w, n)


def _ar_wrapper(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    overlap = group is not None and id(group) in AR_STATE["pgs"]
    if overlap and AR_STATE["log"] is not None:
        AR_STATE["log"].append("AR")
    if overlap and AR_STATE["noar"]:
        return _DoneWork() if async_op else None
    w = _ORIG_AR(tensor, op=op, group=group, async_op=async_op)
    if overlap and AR_STATE["log"] is not None and w is not None:
        return _LogWork(w)
    return w


def build_block(h3, cfg, mc, rank, layer_idx):
    from tensorrt_llm._torch.modules.linear import Linear as _Lin

    block = h3.MiniMaxH3TransformerBlock(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        ffn_dim=cfg["ffn_dim"],
        time_embed_dim=cfg["time_embed_dim"],
        norm_eps=float(cfg.get("norm_eps", 1e-6)),
        qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-6)),
        model_config=mc,
        layer_idx=layer_idx,
        fuse_bf16_gate_up_swiglu=True,
    ).cuda()
    block.setup_tp_overlap(mc)
    sharded = set()
    for mname, mod in block.named_modules():
        if isinstance(mod, _Lin) and getattr(mod, "tp_size", 1) > 1 and getattr(mod, "tp_mode", None) is not None:
            row = "ROW" in str(mod.tp_mode)
            for pname, _ in mod.named_parameters(recurse=False):
                if pname == "weight" or not row:
                    sharded.add(f"{mname}.{pname}" if mname else pname)
    g = torch.Generator(device="cuda")
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(block.named_parameters())):
            g.manual_seed(1000 * i + 17 + 7919 * layer_idx + (rank if name in sharded else 0))
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
    return block


def _is_gemm(n):
    n = n.lower()
    return ("nvjet" in n or "gemm" in n or "cutlass" in n) and "flash" not in n and "fmha" not in n


def _union_len(iv):
    iv = sorted(iv)
    tot, cur = 0.0, None
    for s, e in iv:
        if e <= s:
            continue
        if cur is None or s > cur[1]:
            if cur is not None:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot


def worker(rank, ckpt, mode, port, iters, blocks_list, q):
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 64
    dist.all_reduce = _ar_wrapper  # before any import that may bind it, before any compile
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, create_attention_metadata_state
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (
        ATTENTION_BF16_FORWARDS_STATE_KEY, STEP_PRECISION_STATE_KEY, DenoiseStepController)
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    eng = json.loads((ROOT / "experiments" / "ascent-config" / "engine.json").read_text())
    vga = eng.get("visual_gen_args", {})
    vg = VisualGenArgs(**{k: vga[k] for k in ("attention_config", "step_precision") if k in vga})
    K = vg.step_precision.attention_bf16_forwards if vg.step_precision is not None else None
    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    ctrl = DenoiseStepController()
    state = create_attention_metadata_state()
    state[STEP_PRECISION_STATE_KEY] = ctrl
    state[ATTENTION_BF16_FORWARDS_STATE_KEY] = K
    vgm = VisualGenMapping(2, rank, tp_size=2)
    mc = DiffusionModelConfig(pretrained_config=None, mapping=vgm.to_llm_mapping(), visual_gen_mapping=vgm,
                              attention=vg.attention_config, attention_metadata_state=state,
                              step_precision=vg.step_precision)
    nbmax = max(max(blocks_list), 2)
    blocks = [build_block(h3, cfg, mc, rank, i) for i in range(nbmax)]
    holder = torch.nn.Module()
    holder.transformer_blocks = torch.nn.ModuleList(blocks)
    holder.step_precision_controller = ctrl
    h3.MiniMaxH3Transformer3DModel.setup_adaln_prefetch(holder)
    slot = holder._adaln_table_slot
    for b in blocks:
        if b._tp_overlap_pg is not None:
            AR_STATE["pgs"].add(id(b._tp_overlap_pg))
    has_stage1 = hasattr(h3.MiniMaxH3TransformerBlock, "_post_attn_overlap_residual")
    info = {"rank": rank, "has_stage1": has_stage1, "slot": slot is not None, "K": K,
            "attention_backend": str(vg.attention_config.backend),
            "quant_attention": str(vg.attention_config.quant_attention_config),
            "chunks": sorted({b._tp_overlap_chunks for b in blocks}),
            "dedicated_pg": all(b._tp_overlap_pg is not None for b in blocks), "nblocks": nbmax}
    try:
        be = blocks[0]._tp_overlap_pg._get_backend(torch.device("cuda"))
        info["pg_high_priority"] = bool(be.options.is_high_priority_stream)
    except Exception as ex:
        info["pg_error"] = repr(ex)
    if rank == 0:
        log(f"setup: {info}")
    rope = h3.MiniMaxH3RotaryPosEmbed(int(cfg.get("rope_freq_dim", 16)), float(cfg.get("rope_theta", 10000.0))).cuda()
    g = torch.Generator(device="cuda")
    g.manual_seed(42)
    D = cfg["hidden_size"]
    x = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16, generator=g)
    temb = torch.randn(2, cfg["time_embed_dim"], device="cuda", generator=g)
    idx = torch.randint(0, 6, (M,), device="cuda", generator=g)
    pos = torch.randint(0, 64, (M, 3), device="cuda", generator=g)
    rot = rope(pos)
    inputs = (x, temb, idx, rot[0], rot[1])
    pristine = [t.clone() for t in inputs]
    compiled = [torch.compile(b, mode="default", dynamic=None) for b in blocks]

    shipped_programs = getattr(blocks[0], "_tp_overlap_residual_programs", 0)
    info["shipped_residual_programs"] = shipped_programs
    if rank == 0:
        log(f"shipped residual programs: {shipped_programs}")

    def set_arm(flags):
        chunk_res, programs, noar = flags
        if programs is None:
            programs = shipped_programs
        for b in blocks:
            b._tp_overlap_chunk_residual = chunk_res
            b._tp_overlap_residual_programs = programs
        AR_STATE["noar"] = noar

    def call(nb, fwd):
        ctrl.set_step(fwd, 49)
        with torch.inference_mode():
            y = x
            for i in range(nb):
                y = compiled[i](y, temb, idx, rot, None, None, valid_seq_len=None)
            return y

    def poison():
        # Fill the allocator's free pool with NaN blocks of the block-output size so a fresh
        # torch.empty_like output that is not fully written shows up as NaN.
        ts = [torch.full((1, M, D), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(6)]
        del ts

    from torch._dynamo.utils import counters

    out = dict(info)
    # ---- warm/compile every arm, both phases, all chain lengths used ----
    chains = sorted(set([1, 2] + list(blocks_list)))
    for name, flags in ARMS:
        set_arm(flags)
        for _, fwd in PHASES:
            for _ in range(2):
                call(max(chains), fwd)
    torch.cuda.synchronize()
    set_arm(ARMS[0][1])
    stats_after_warmup = dict(counters["stats"])

    # ---- untimed check ----
    ok = True
    check = {}
    for ph, fwd in PHASES:
        for nb in (1, 2):
            set_arm(ARMS[0][1])
            poison()
            ref = call(nb, fwd).clone()
            torch.cuda.synchronize()
            peer = ref.clone()
            dist.broadcast(peer, src=0)
            ref_ranks_eq = bool(torch.equal(peer, ref))
            ref_finite = bool(torch.isfinite(ref).all())
            del peer
            for name, flags in EXACT_ARMS:
                set_arm(flags)
                torch.cuda.reset_peak_memory_stats()
                c0 = slot.counts() if slot is not None else {}
                bad = []
                for it in range(WARMUP + CHECK_ITERS):
                    poison()
                    y = call(nb, fwd)
                    torch.cuda.synchronize()
                    if not torch.equal(y, ref):
                        bad.append(f"neq@{it}")
                    peer = y.clone()
                    dist.broadcast(peer, src=0)
                    if not torch.equal(peer, y):
                        bad.append(f"rank_mismatch@{it}")
                    del y, peer
                    for t, p in zip(inputs, pristine):
                        if not torch.equal(t, p):
                            bad.append(f"input_changed@{it}")
                c1 = slot.counts() if slot is not None else {}
                cnt = {k: c1[k] - c0[k] for k in c1}
                check[f"{ph}/nb{nb}/{name}"] = {"bitwise_all_iters": not bad, "bad": bad[:6],
                                                "prefetch_counts": cnt,
                                                "peak_mem_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
                ok = ok and not bad
            ok = ok and ref_ranks_eq and ref_finite
            check[f"{ph}/nb{nb}/ref"] = {"rank0_eq_rank1": ref_ranks_eq, "finite": ref_finite}
            # prefetch counts equal to main_a's in every arm
            base_cnt = check[f"{ph}/nb{nb}/main_a"]["prefetch_counts"]
            for name, _ in EXACT_ARMS:
                if check[f"{ph}/nb{nb}/{name}"]["prefetch_counts"] != base_cnt:
                    ok = False
                    check[f"{ph}/nb{nb}/{name}"]["prefetch_counts_differ"] = True
            if rank == 0:
                log(f"check {ph} nb{nb}: " + json.dumps({k: v for k, v in check.items() if k.startswith(f'{ph}/nb{nb}/')}))
    stats_end = dict(counters["stats"])
    recompiled = stats_end != stats_after_warmup
    ok = ok and not recompiled
    out["dynamo"] = {"after_warmup": stats_after_warmup, "end": stats_end, "recompiled_after_warmup": recompiled,
                     "graph_breaks": len(counters["graph_break"])}

    # ---- event order: AR issue / wait / drain hook, 2-block chain, per exact arm and phase ----
    def hooked(h):
        def f(t, a):
            AR_STATE["log"].append("HOOK")
            return h(t, a)
        return f

    saved_hooks = [b._drain_hooks for b in blocks]
    for b in blocks:
        b._drain_hooks = [hooked(h) for h in b._drain_hooks]
    per_block = ["AR"] * 4 + ["W", "AR"] * 4 + ["HOOK"] + ["W"] * 4
    order = {}
    for ph, fwd in PHASES:
        for name, flags in (ARMS[0], ARMS[1], ARMS[3]):
            set_arm(flags)
            AR_STATE["log"] = []
            call(2, fwd)
            torch.cuda.synchronize()
            seq = AR_STATE["log"]
            AR_STATE["log"] = None
            order[f"{ph}/{name}"] = {"ok": seq == per_block * 2, "seq": " ".join(seq)}
    for b, h in zip(blocks, saved_hooks):
        b._drain_hooks = h
    order_ok = all(v["ok"] for v in order.values())
    ok = ok and order_ok
    out["event_order"] = order
    out["event_order_expected_per_block"] = " ".join(per_block)
    out["check"] = check
    out["check_passed"] = ok
    if rank == 0:
        log(f"event order ok={order_ok}: {order}")
        log(f"check passed={ok} recompiled_after_warmup={recompiled}")

    # ---- timing ----
    out["timing"] = {}
    out["profile"] = {}
    if mode == "time":
        per_round = 4
        rounds = -(-iters // per_round)
        for nb in blocks_list:
            for ph, fwd in PHASES:
                times = {a[0]: [] for a in ARMS}
                t0 = time.time()
                for r in range(rounds):
                    k = r % len(ARMS)
                    order_r = ARMS[k:] + ARMS[:k]
                    if r % 2:
                        order_r = order_r[::-1]
                    for name, flags in order_r:
                        set_arm(flags)
                        call(nb, fwd)
                        for _ in range(per_round):
                            dist.barrier()
                            torch.cuda.synchronize()
                            s = torch.cuda.Event(enable_timing=True)
                            e = torch.cuda.Event(enable_timing=True)
                            s.record()
                            call(nb, fwd)
                            e.record()
                            e.synchronize()
                            times[name].append(s.elapsed_time(e))
                    if rank == 0 and r % 3 == 2:
                        log(f"nb{nb} {ph} round {r + 1}/{rounds} ({time.time() - t0:.0f}s): " + ", ".join(
                            f"{a} {statistics.median(v):.2f}" for a, v in times.items()))
                t = {a: st(v) for a, v in times.items()}
                ma = t["main_a"]["median_ms"]
                main_med = statistics.median(times["main_a"] + times["main_b"])
                t["aa_rel"] = round(t["main_b"]["median_ms"] / ma - 1, 5)
                t["aa_ms_per_block"] = round((t["main_b"]["median_ms"] - ma) / nb, 4)
                for name, _ in ARMS:
                    if name.startswith("main"):
                        continue
                    t[f"{name}_vs_main_rel"] = round(t[name]["median_ms"] / main_med - 1, 5)
                    t[f"{name}_minus_main_ms_per_block"] = round((t[name]["median_ms"] - main_med) / nb, 4)
                t["main_a_minus_noar_ms_per_block"] = round((ma - t["noar"]["median_ms"]) / nb, 4)
                out["timing"][f"blocks{nb}/{ph}"] = t
                if rank == 0:
                    log(f"timing blocks{nb} {ph}: {json.dumps(t)}")

        # ---- untimed profiler pass (2-block chain), per exact arm and phase ----
        from torch.profiler import ProfilerActivity, profile

        for ph, fwd in PHASES:
            for name, flags in EXACT_ARMS:
                if name == "main_b":
                    continue
                set_arm(flags)
                for _ in range(2):
                    call(2, fwd)
                torch.cuda.synchronize()
                recs = []
                for _ in range(3):
                    dist.barrier()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        call(2, fwd)
                        torch.cuda.synchronize()
                    kev = sorted((e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                                 if str(e.device_type).endswith("CUDA"))
                    nccl = [k for k in kev if "nccl" in k[2].lower()]
                    if len(nccl) < 16:
                        recs.append({"error": f"{len(nccl)} nccl kernels"})
                        continue
                    d3 = nccl[7]
                    big = [k for k in kev if "nccl" not in k[2].lower() and _is_gemm(k[2]) and k[1] - k[0] > 500]
                    before = [k for k in big if k[0] < d3[0]]
                    after = [k for k in big if k[0] >= d3[1]]
                    if not before or not after:
                        recs.append({"error": "no anchor gemm"})
                        continue
                    anchor = before[-1][1]
                    comp = [(max(k[0], anchor), min(k[1], d3[1])) for k in kev
                            if "nccl" not in k[2].lower() and k[1] > anchor and k[0] < d3[1]]
                    busy = _union_len(comp)
                    recs.append({
                        "ar_d3_ms": (d3[1] - d3[0]) / 1000.0,
                        "busy_under_d3_ms": busy / 1000.0,
                        "window_gemm_end_to_d3_end_ms": (d3[1] - anchor) / 1000.0,
                        "tail_d3_end_to_next_qkv_ms": (after[0][0] - d3[1]) / 1000.0,
                        "launches_d3_end_to_next_qkv": sum(1 for k in kev if d3[1] <= k[0] < after[0][0]),
                    })
                good = [r for r in recs if "error" not in r]
                summ = {k: round(statistics.median(r[k] for r in good), 4) for k in good[0]} if good else {}
                summ["errors"] = [r["error"] for r in recs if "error" in r]
                out["profile"][f"{ph}/{name}"] = summ
                if rank == 0:
                    log(f"profile {ph} {name}: {summ}")
    set_arm(ARMS[0][1])
    out["max_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    q.put(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", default="check", choices=["check", "time"])
    ap.add_argument("--blocks", default="8,2")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pair-check", action="store_true")
    ap.add_argument("--timing-pair-only", action="store_true")
    ap.add_argument("--report-arm", default="stage1")
    args = ap.parse_args()
    blocks_list = [int(b) for b in args.blocks.split(",") if b]
    pair = None
    timing_pair = None
    if args.pair_check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ar_inplace_bench import pair_check

        pair = pair_check()
        phys = pair.get("physical_ids") or []
        gens, widths = [], []
        for line in (pair.get("pcie") or "").splitlines():
            p = [s.strip() for s in line.split(",")]
            try:
                gens.append(int(p[3]))
                widths.append(int(p[5]))
            except Exception:
                pass
        pair["link_gen_current"] = gens
        timing_pair = (len(phys) == 2 and None not in phys and 5 not in phys
                       and pair.get("relation") in ("PIX", "NODE")
                       and len(widths) == 2 and min(widths) >= 16 and len(gens) == 2 and min(gens) >= 5)
        pair["timing_pair"] = timing_pair
        log(f"PAIR phys {phys} bus {pair.get('cuda_bus_ids')} relation {pair.get('relation')} gen {gens} "
            f"width {widths} timing_pair={timing_pair}")
        if args.mode == "time" and args.timing_pair_only and not timing_pair:
            print(json.dumps({"wrong_pair": True, "pair": pair}), flush=True)
            return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(r, args.ckpt, args.mode, port, args.iters, blocks_list, q))
             for r in range(2)]
    for p in procs:
        p.start()
    res = sorted([q.get(timeout=5400) for _ in range(2)], key=lambda r: r["rank"])
    for p in procs:
        p.join()
    log(f"done in {time.time() - t0:.0f}s")
    r0, r1 = res
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, f"stage1_pair_{args.mode}.json"), "w") as f:
        json.dump({"pair": pair, "rank0": r0, "rank1": r1}, f, indent=1)
    passed = bool(r0["check_passed"] and r1["check_passed"])
    pair_lbl = {k: (pair or {}).get(k) for k in ("physical_ids", "cuda_bus_ids", "relation", "link_gen_current",
                                                 "link_width_current", "timing_pair")}
    common = {
        "passed": passed,
        "compared": ("exact arms (stage1 = shipped grid cap, g0 uncapped, g64, main_b) vs main_a = main's compiled chunked "
                     "path (chunk_residual off), forward 9 (BF16 attn) and 20 (FP8 attn), 1 and 2 chained "
                     f"blocks, bitwise per iteration over {WARMUP}+{CHECK_ITERS}, NaN-poisoned output pool, "
                     "rank0 == rank1, inputs read-only, prefetch counts == main_a, AR/wait/hook order == main, "
                     "no recompiles after warmup"),
        "has_stage1": r0["has_stage1"], "pair": pair_lbl,
        "setup": {k: r0.get(k) for k in ("K", "attention_backend", "quant_attention", "chunks", "dedicated_pg",
                                         "pg_high_priority", "nblocks", "slot", "shipped_residual_programs")},
        "dynamo": [r0["dynamo"]["recompiled_after_warmup"], r1["dynamo"]["recompiled_after_warmup"]],
        "event_order_ok": [all(v["ok"] for v in r["event_order"].values()) for r in res],
        "check_bad": {f"r{r['rank']}:{k}": v["bad"] for r in res for k, v in r["check"].items()
                      if v.get("bad")},
        "prefetch_counts_r0": {k: v.get("prefetch_counts") for k, v in r0["check"].items() if "prefetch_counts" in v
                               and k.endswith(("main_a", "stage1"))},
        "max_mem_alloc_gib": [r0["max_mem_alloc_gib"], r1["max_mem_alloc_gib"]],
    }
    if args.mode == "time":
        tkey = f"blocks{blocks_list[0]}/{PHASES[0][0]}"
        t = r0["timing"][tkey][args.report_arm]
        summary = {}
        for key, tt in r0["timing"].items():
            t1 = r1["timing"][key]
            summary[key] = {
                "main_a": tt["main_a"], "main_b": tt["main_b"], "stage1": tt["stage1"], "noar": tt["noar"],
                "aa_rel": tt["aa_rel"], "aa_ms_per_block": tt["aa_ms_per_block"],
                "stage1_minus_main_ms_per_block": [tt["stage1_minus_main_ms_per_block"],
                                                   t1["stage1_minus_main_ms_per_block"]],
                "stage1_vs_main_rel": [tt["stage1_vs_main_rel"], t1["stage1_vs_main_rel"]],
                "g0_uncapped": tt["g0"],
                "g0_minus_main_ms_per_block": [tt["g0_minus_main_ms_per_block"], t1["g0_minus_main_ms_per_block"]],
                "g64_minus_main_ms_per_block": [tt["g64_minus_main_ms_per_block"],
                                                t1["g64_minus_main_ms_per_block"]],
                "main_a_minus_noar_ms_per_block": [tt["main_a_minus_noar_ms_per_block"],
                                                   t1["main_a_minus_noar_ms_per_block"]],
            }
        common.update({
            "batch": {"1": {"step_ms": t["median_ms"], "noise_floor": t["noise"]}},
            "output": {"class": "exact"},
            "report_arm": args.report_arm, "report_key": tkey,
            "timing_valid": bool(timing_pair),
            "timing_summary_r0_r1": summary,
            "profile_r0": r0["profile"], "profile_r1": r1["profile"],
        })
    print(json.dumps(common), flush=True)


if __name__ == "__main__":
    main()
