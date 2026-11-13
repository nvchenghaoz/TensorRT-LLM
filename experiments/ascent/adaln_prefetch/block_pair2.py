"""2-GPU two-block benchmark/check of the AdaLN table prefetch (idea 105), MiniMax-H3 TP2.

Builds two consecutive real MiniMaxH3TransformerBlocks (checkpoint config, synthetic weights from a
fixed seed, per-rank shards; block 1's weights differ from block 0's), the chunked TP overlap on
main's dedicated high-priority PG, and installs the prefetch through the model's own
setup_adaln_prefetch on a holder module with a DenoiseStepController. Each block is compiled the way
the pipeline does (torch.compile per block, mode=default, dynamic=None). One "pair" = block 0 then
block 1 at the per-rank shape (M = 37966 rows, BF16, VANILLA attention, temb [2, 2688] FP32); the
controller's step is set before every pair, as the pipeline does per forward.

Arms (branch tree): main = block._adaln_slot_id None (main's adaln_proj.table path),
off = op path with the prefetch disabled (both blocks miss), on = prefetch on (block 1 hits).
On main's tree only the main arm exists.

  --mode time : per arm 5 warmup pairs, then ROUNDS rounds; each round times every arm twice
                (A/A copies, interleaved), CUDA events on rank 0 after barrier + sync.
                Then an untimed torch.profiler pass per arm: block 0's last all-reduce (down c3)
                kernel duration and the prefetch GEMV overlapping it.
                Prints {"batch": {"1": {"kernel_ms": <arm median>, "noise_floor": ...}}, ...}
                (on arm on the branch, main arm on main).
  --mode check: untimed, same warmup/iteration counts: every arm's pair output bitwise vs the
                main arm's first output, every iteration; inputs read-only; hits/misses per pair;
                dynamo recompiles after warmup; peek semantics on the live slot.
Usage: python block_pair2.py <checkpoint> --mode time|check
"""

import argparse
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M = 37966
WARMUP = 5
ROUNDS = int(os.environ.get("BP2_ROUNDS", "40"))


def log(*a):
    print(*a, flush=True)


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


def worker(rank, ckpt, mode, port, q):
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import DenoiseStepController

    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    vgm = VisualGenMapping(2, rank, tp_size=2)
    mc = DiffusionModelConfig(pretrained_config=None, mapping=vgm.to_llm_mapping(), visual_gen_mapping=vgm)
    try:
        mc.attention.backend = "VANILLA"
    except Exception:
        pass
    blocks = [build_block(h3, cfg, mc, rank, i) for i in range(2)]
    holder = torch.nn.Module()
    holder.transformer_blocks = torch.nn.ModuleList(blocks)
    ctrl = DenoiseStepController()
    holder.step_precision_controller = ctrl
    has_prefetch = hasattr(h3.MiniMaxH3Transformer3DModel, "setup_adaln_prefetch")
    slot = None
    if has_prefetch:
        h3.MiniMaxH3Transformer3DModel.setup_adaln_prefetch(holder)
        slot = holder._adaln_table_slot
    info = {
        "rank": rank,
        "has_prefetch": has_prefetch,
        "slot": slot is not None,
        "chunks": [b._tp_overlap_chunks for b in blocks],
        "dedicated_pg": [b._tp_overlap_pg is not None for b in blocks],
    }
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
    pristine = [x.clone(), temb.clone(), idx.clone(), rot[0].clone(), rot[1].clone()]
    compiled = [torch.compile(b, mode="default", dynamic=None) for b in blocks]
    slot_ids = [getattr(b, "_adaln_slot_id", None) for b in blocks]
    step = [0]

    def set_arm(arm):
        if not has_prefetch or slot is None:
            assert arm == "main"
            return
        for b, sid in zip(blocks, slot_ids):
            b._adaln_slot_id = None if arm == "main" else sid
        slot.enabled = arm == "on"

    def pair(arm):
        set_arm(arm)
        ctrl.set_step(step[0] % 49, 49)
        step[0] += 1
        with torch.inference_mode():
            y = compiled[0](x, temb, idx, rot, None, None, valid_seq_len=None)
            return compiled[1](y, temb, idx, rot, None, None, valid_seq_len=None)

    arms = ["main", "off", "on"] if slot is not None else ["main"]
    out = dict(info)
    from torch._dynamo.utils import counters

    for arm in arms:
        for _ in range(WARMUP):
            pair(arm)
    torch.cuda.synchronize()
    stats_after_warmup = dict(counters["stats"])

    if mode == "time":
        times = {f"{a}_{c}": [] for a in arms for c in "AB"}
        for r in range(ROUNDS):
            for c in "AB":
                for a in arms:
                    dist.barrier()
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    pair(a)
                    e.record()
                    e.synchronize()
                    times[f"{a}_{c}"].append(s.elapsed_time(e))
            if rank == 0 and r % 10 == 9:
                log(f"round {r + 1}/{ROUNDS}: " + ", ".join(
                    f"{k} {statistics.median(v):.3f}" for k, v in times.items()))
        summ = {}
        for k, v in times.items():
            v = sorted(v)
            med = statistics.median(v)
            lo, hi = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1) + 0.5)]
            summ[k] = {"median_ms": med, "p10": lo, "p90": hi, "noise": (hi - lo) / med, "n": len(v)}
        for a in arms:
            v = sorted(times[f"{a}_A"] + times[f"{a}_B"])
            med = statistics.median(v)
            lo, hi = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1) + 0.5)]
            summ[a] = {"median_ms": med, "p10": lo, "p90": hi, "noise": (hi - lo) / med, "n": len(v)}
        out["timing"] = summ
        if slot is not None:
            # paired per-round deltas (same round, same copy)
            for a, b in (("main", "on"), ("off", "on"), ("main", "off")):
                d = sorted(x1 - x2 for c in "AB" for x1, x2 in zip(times[f"{a}_{c}"], times[f"{b}_{c}"]))
                out.setdefault("paired_delta_ms", {})[f"{a}-{b}"] = {
                    "median": statistics.median(d), "p10": d[int(0.1 * (len(d) - 1))],
                    "p90": d[int(0.9 * (len(d) - 1) + 0.5)]}
            for a in arms:
                d = sorted(x1 - x2 for x1, x2 in zip(times[f"{a}_A"], times[f"{a}_B"]))
                out.setdefault("aa_delta_ms", {})[a] = {
                    "median": statistics.median(d), "p10": d[int(0.1 * (len(d) - 1))],
                    "p90": d[int(0.9 * (len(d) - 1) + 0.5)]}
        if rank == 0:
            log(f"timing: {json.dumps(summ)}")
            log(f"paired deltas: {out.get('paired_delta_ms')} A/A: {out.get('aa_delta_ms')}")
        # untimed profiler pass: block 0's last all-reduce kernel with and without the GEMV
        from torch.profiler import ProfilerActivity, profile

        prof_out = {}
        for a in arms:
            ar_durs, gemv_overlap, b1_first_gemv = [], [], []
            for _ in range(3):
                dist.barrier()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    pair(a)
                    torch.cuda.synchronize()
                kev = []
                for e in prof.events():
                    if str(e.device_type).endswith("CUDA"):
                        kev.append((e.time_range.start, e.time_range.end, e.name))
                kev.sort()
                nccl = [k for k in kev if "nccl" in k[2].lower()]
                if len(nccl) >= 16:
                    last_ar = nccl[7]  # block 0: 4 to_out + 4 down all-reduces; [7] = down c3
                    ar_durs.append((last_ar[1] - last_ar[0]) / 1000.0)
                    ov = [k for k in kev if "nccl" not in k[2].lower() and k[0] < last_ar[1] and k[1] > last_ar[0]]
                    gemv_overlap.append([(k[2][:60], round((k[1] - k[0]) / 1000.0, 4)) for k in ov][:6])
                    b1 = [k for k in kev if k[0] >= last_ar[1]]
                    b1_first_gemv.append([(k[2][:60], round((k[1] - k[0]) / 1000.0, 4)) for k in b1[:4]])
            prof_out[a] = {"block0_down_c3_ar_ms": ar_durs, "kernels_overlapping_it": gemv_overlap[:1],
                           "first_kernels_after_it": b1_first_gemv[:1]}
            if rank == 0:
                log(f"profile {a}: {prof_out[a]}")
        out["profile"] = prof_out
        main_arm = "on" if slot is not None else "main"
        t = summ[main_arm]
        out["batch"] = {"1": {"kernel_ms": t["median_ms"], "noise_floor": t["noise"]}}
        out["output"] = {"class": "exact"}
    elif mode == "check":
        ok = True
        ref = None
        res = {a: {"bad": [], "hits": 0, "misses": 0} for a in arms}
        for a in arms:
            for it in range(WARMUP + ROUNDS):
                c0 = slot.counts() if slot is not None else {}
                yb = pair(a)
                torch.cuda.synchronize()
                if ref is None:
                    ref = yb.clone()
                if not torch.equal(yb, ref):
                    res[a]["bad"].append(it)
                for t_, p_ in zip((x, temb, idx, rot[0], rot[1]), pristine):
                    if not torch.equal(t_, p_):
                        res[a]["bad"].append(f"input_changed@{it}")
                if slot is not None:
                    c1 = slot.counts()
                    res[a]["hits"] += c1["hits"] - c0["hits"]
                    res[a]["misses"] += c1["misses"] - c0["misses"]
            ok = ok and not res[a]["bad"]
        n = WARMUP + ROUNDS
        if slot is not None:
            ok = ok and res["on"]["hits"] == n and res["on"]["misses"] == n
            ok = ok and res["off"]["hits"] == 0 and res["off"]["misses"] == 2 * n
            ok = ok and res["main"]["hits"] == 0 and res["main"]["misses"] == 0
            # peek on the live slot: fill block 1's table via block 0, then peek before block 1
            set_arm("on")
            ctrl.set_step(0, 49)
            with torch.inference_mode():
                y0 = compiled[0](x, temb, idx, rot, None, None, valid_seq_len=None)
                cnt0, key0 = slot.counts(), slot._key
                p1, p1b, p0 = slot.peek(1), slot.peek(1), slot.peek(0)
                peek_ok = p1 is not None and p1 is p1b and p0 is None and slot.counts() == cnt0 and slot._key == key0
                y1 = compiled[1](y0, temb, idx, rot, None, None, valid_seq_len=None)
                torch.cuda.synchronize()
                peek_ok = peek_ok and slot.counts()["hits"] == cnt0["hits"] + 1 and slot.peek(1) is None
                peek_ok = peek_ok and torch.equal(y1, ref)
            res["peek_live_ok"] = peek_ok
            ok = ok and peek_ok
        stats_end = dict(counters["stats"])
        res["dynamo_stats_after_warmup"] = stats_after_warmup
        res["dynamo_stats_end"] = stats_end
        res["recompiled_after_warmup"] = stats_after_warmup != stats_end
        res["graph_breaks"] = {k: v for k, v in counters["graph_break"].items()}
        ok = ok and not res["recompiled_after_warmup"]
        out["check"] = res
        out["passed"] = ok
        if rank == 0:
            log(f"check: {json.dumps(res, default=str)}")
    out["max_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    q.put(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mode", default="time")
    a = ap.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, a.ckpt, a.mode, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    outs = [q.get() for _ in ps]
    for p in ps:
        p.join()
    outs.sort(key=lambda o: o["rank"])
    r0 = outs[0]
    final = {"ranks": outs}
    if a.mode == "time":
        final.update(batch=r0["batch"], output=r0["output"])
    else:
        final["passed"] = all(o.get("passed") for o in outs)
    print(json.dumps(final, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
