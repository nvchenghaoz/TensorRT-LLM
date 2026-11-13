"""2-GPU TP2 check + timing for branch 16 plan (a): norm_out FP32 tail + SM120 qk_norm_rope launch.

Runs identically in both arms of run_gpu_pair (main tree / branch tree):
  * Real MiniMaxH3Transformer3DModel from the checkpoint config with the LAST TWO blocks only
    (num_layers=2, layer_idx 0/1 standing in for 48/49), TP2 (VisualGenMapping tp_size=2),
    landing 8's TP overlap on (setup_tp_overlap in __init__, default chunks), VANILLA attention,
    per-block torch.compile exactly like pipeline.torch_compile, synthetic weights seeded by
    parameter name (identical in both arms and on both ranks).
  * Real packed layout at the benchmark shape (1344x768, 124 frames, T2VA, no keyframes), padded with
    text tokens to the per-rank M=37966; real adaln_indices = timestep_indices*3 + token_tags built by
    the model's own forward. Three timestep plans (n_t = 1 and 2) cycled per iteration.
  * NaN-poisoned allocator before every forward (an 8 GiB NaN block is allocated and freed, so the
    forward's outputs land in NaN-filled memory).
Sequence (each pass = 5 warmup + 20 iterations, iteration i uses plan i % 3):
  check pass -> timed round A -> timed round B (A/A) -> untimed repeat.
Every iteration of every pass compares video/audio outputs and the proj_out input (the norm_out tail
result) bitwise with the plan's reference and checks inputs read-only. Reference = first output of
the check pass; in the branch tree also main's code paths emulated in-process (norm_out.forward +
.to(fp32), qk launch (1, auto, 1)) must match bitwise. rank0 == rank1 hashes. Dynamo counters after
the check pass's warmup must not change afterwards.
Timing (rank 0, CUDA events): tail = end of last block -> end of audio_proj_out (norm_out, gather,
.to(fp32) x2, proj_out, audio_proj_out); qk = one trtllm::minimax_h3_qk_norm_rope at the rank shape
(28 heads x 128, M=37966); per forward = tail + 50 * qk. Round A and B each give median/p10/p90.
Last stdout line: {"batch": {"1": {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}, ...}.
Usage: python h3_tail_tp2.py <checkpoint>
"""

import hashlib
import json
import os
import socket
import statistics
import sys
import time
import zlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M_TARGET = 37966
WARMUP, ITERS = 5, 20
PLANS = [(1.0, 1.0), (0.6, 0.7), (0.2, 0.3)]
NUM_FORWARD_LAYERS = 50


def log(*a):
    print(*a, flush=True)


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = statistics.median(s)
    p10 = s[int(0.1 * (n - 1) + 0.5)]
    p90 = s[int(0.9 * (n - 1) + 0.5)]
    return {"median": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / med, "n": n}


def build(ckpt, rank):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    cfg["num_layers"] = 2
    if "num_refiner_layers" in cfg:
        cfg["num_refiner_layers"] = 1
    vgm = VisualGenMapping(2, rank, tp_size=2)
    mc = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**cfg),
        mapping=vgm.to_llm_mapping(),
        visual_gen_mapping=vgm,
    )
    model = MiniMaxH3Transformer3DModel(mc).cuda()
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator(device="cuda").manual_seed(zlib.crc32(name.encode()))
            r = torch.randn(p.shape, generator=g, device="cuda", dtype=torch.float32)
            if p.ndim == 1 and "norm" in name:
                p.copy_((1 + 0.1 * r).to(p.dtype))
            elif p.ndim == 1:
                p.copy_((0.02 * r).to(p.dtype))
            else:
                p.copy_((r / p.shape[-1] ** 0.5).to(p.dtype))
    for mod in model.modules():
        if hasattr(mod, "post_load_weights") and mod is not model:
            try:
                mod.post_load_weights()
            except Exception:
                pass
    model.eval()
    chunks = [getattr(b, "_tp_overlap_chunks", None) for b in model.transformer_blocks]
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.compile(b, mode="default", dynamic=None, fullgraph=False) for b in model.transformer_blocks]
    )
    return model, cfg, chunks


def make_inputs(model, cfg):
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
        MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_TEXT_TAG,
        audio_latent_num_frames,
        build_packed_sequence,
        build_row_timesteps,
        video_latent_num_frames,
    )

    patch = tuple(cfg["patch_size"])
    f_lat = video_latent_num_frames(124)
    a_lat = audio_latent_num_frames(124)
    chosen = None
    for ratio in (16, 32, 8):
        h, w = 768 // ratio, 1344 // ratio
        if h % patch[1] or w % patch[2]:
            continue
        rows = f_lat * (h // patch[1]) * (w // patch[2]) + a_lat * MINIMAX_H3_AUDIO_CHANNELS
        if 0 < M_TARGET - rows <= 4096:
            chosen = (ratio, h, w, M_TARGET - rows)
            break
    if chosen is None:
        raise RuntimeError("no VAE ratio reproduces M=37966")
    ratio, h, w, text_len = chosen
    layout = build_packed_sequence(
        torch.full((text_len,), MINIMAX_H3_TEXT_TAG, dtype=torch.long), f_lat, h, w, a_lat, patch
    )
    assert layout.sequence_length == M_TARGET, layout.sequence_length
    dev = "cuda"
    vid_dim = cfg["in_channels"] * patch[0] * patch[1] * patch[2]
    g = torch.Generator(device=dev).manual_seed(1)
    common = dict(
        hidden_states=torch.randn(1, layout.video_indices.numel(), vid_dim, generator=g, device=dev),
        audio_hidden_states=torch.randn(
            1, layout.audio_indices.numel(), cfg["audio_in_channels"], generator=g, device=dev
        ),
        token_tags=layout.token_tags.to(dev),
        video_indices=layout.video_indices.to(dev),
        audio_indices=layout.audio_indices.to(dev),
        text_indices=layout.text_indices.to(dev),
    )
    enc = torch.randn(1, text_len, cfg["text_dim"], generator=g, device=dev).to(torch.bfloat16)
    with torch.inference_mode():
        static = model.prepare_static_context(enc, layout.position_ids.to(dev))
    plans = []
    for vt, at in PLANS:
        uniq, tidx = build_row_timesteps(layout, vt, at, vt, 1.0)
        plans.append((uniq.to(dev), tidx.to(dev), torch.tensor([1.0 - vt], device=dev)))
    info = {"vae_ratio": ratio, "latent_hw": [h, w], "latent_frames": f_lat, "audio_latents": a_lat,
            "text_len": text_len, "M": layout.sequence_length,
            "n_t_per_plan": [int(p[0].numel()) for p in plans]}
    return common, static, plans, info


def worker(rank, ckpt, port, q):
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_rope as fr
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    t0 = time.time()
    model, cfg, chunks = build(ckpt, rank)
    common, static, plans, info = make_inputs(model, cfg)
    branch = hasattr(h3.MiniMaxH3AdaLayerNormOut, "forward_to")
    out = {"rank": rank, "branch_tree": branch, "overlap_chunks": chunks, "layout": info,
           "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid)}
    if rank == 0:
        log(f"built in {time.time() - t0:.0f}s branch_tree={branch} chunks={chunks} layout={info}")
    pristine = {k: v.clone() for k, v in common.items() if torch.is_tensor(v)}
    pristine_plans = [tuple(t.clone() for t in p) for p in plans]

    cap = {}
    model.proj_out.register_forward_pre_hook(lambda m, a: cap.__setitem__("tail", a[0]))
    ev = {}

    def h_start(m, a, o):
        if "s" in ev:
            ev["s"].record()

    def h_end(m, a, o):
        if "e" in ev:
            ev["e"].record()

    model.transformer_blocks[-1].register_forward_hook(h_start)
    model.audio_proj_out.register_forward_hook(h_end)

    def poison():
        p = torch.full((2 * 1024**3,), float("nan"), device="cuda", dtype=torch.float32)
        del p

    def fwd(i):
        uniq, tidx, att_t = plans[i % len(plans)]
        v, a = model(encoder_hidden_states=None, timestep=att_t, conditioning_timesteps=uniq,
                     timestep_indices=tidx, position_ids=None, return_dict=False,
                     static_context=static, **common)
        return v, a, cap["tail"]

    refs = {}
    main_refs = {}
    bad = []

    def compare(tag, it, outs):
        p = it % len(plans)
        if p not in refs:
            refs[p] = tuple(o.clone() for o in outs)
        else:
            for name, o, r in zip(("video", "audio", "tail"), outs, refs[p]):
                if not torch.equal(o, r):
                    bad.append(f"{tag}@{it}:{name}")
        if p in main_refs:
            for name, o, r in zip(("video", "audio", "tail"), outs, main_refs[p]):
                if not torch.equal(o, r):
                    bad.append(f"{tag}@{it}:{name}_vs_main_emul")
        for name, o in zip(("video", "audio", "tail"), outs):
            if not bool(torch.isfinite(o).all()):
                bad.append(f"{tag}@{it}:{name}_nonfinite")
        for k, v in pristine.items():
            if not torch.equal(common[k], v):
                bad.append(f"{tag}@{it}:input_changed:{k}")
                common[k].copy_(v)
        for pp, ppr in zip(plans, pristine_plans):
            for t_, tr in zip(pp, ppr):
                if not torch.equal(t_, tr):
                    bad.append(f"{tag}@{it}:plan_changed")
                    t_.copy_(tr)

    with torch.inference_mode():
        if branch:
            # Main's code paths, in-process: eager norm_out + .to(dtype), qk launch (1, auto, 1).
            orig_ft = h3.MiniMaxH3AdaLayerNormOut.forward_to
            h3.MiniMaxH3AdaLayerNormOut.forward_to = (
                lambda self, hs, temb, ti, dt: self.forward(hs, temb, ti).to(dt)
            )
            fr._launch_by_device[torch.cuda.current_device()] = (1, None, 1)
            for i in range(len(plans)):
                poison()
                main_refs[i] = tuple(o.clone() for o in fwd(i))
            torch.cuda.synchronize()
            h3.MiniMaxH3AdaLayerNormOut.forward_to = orig_ft
            fr._launch_by_device.clear()
            out["launch_after_restore"] = list(fr._default_launch(torch.device("cuda", rank)))
            if rank == 0:
                log(f"main-emulated refs done; branch launch {out['launch_after_restore']}")

        # check pass
        stats_after_warmup = None
        for it in range(WARMUP + ITERS):
            poison()
            o = fwd(it)
            torch.cuda.synchronize()
            compare("check", it, o)
            if it == WARMUP - 1:
                stats_after_warmup = {k: dict(v) for k, v in counters.items()
                                      if k in ("stats", "graph_break", "recompiles")}
        out["hashes"] = {str(p): {n: sha(t) for n, t in zip(("video", "audio", "tail"), r)}
                         for p, r in refs.items()}
        if rank == 0:
            log(f"check pass done, bad={bad[:5]} hashes={out['hashes']}")

        def timed_round(tag):
            ts = []
            for it in range(WARMUP + ITERS):
                poison()
                dist.barrier()
                torch.cuda.synchronize()
                ev["s"] = torch.cuda.Event(enable_timing=True)
                ev["e"] = torch.cuda.Event(enable_timing=True)
                o = fwd(it)
                ev["e"].synchronize()
                if it >= WARMUP:
                    ts.append(ev["s"].elapsed_time(ev["e"]))
                ev.clear()
                torch.cuda.synchronize()
                compare(tag, it, o)
            return ts

        tail_a = timed_round("timedA")
        tail_b = timed_round("timedB")
        if rank == 0:
            log(f"tail A {stats(tail_a)}\ntail B {stats(tail_b)}")
        for it in range(WARMUP + ITERS):
            poison()
            o = fwd(it)
            torch.cuda.synchronize()
            compare("repeat", it, o)
        stats_end = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "recompiles")}
        out["recompiled_after_warmup"] = stats_end != stats_after_warmup
        out["dynamo_end"] = stats_end

        # qk_norm_rope at the rank shape, the tree's default launch through the custom op.
        H = cfg["num_attention_heads"] // 2
        hd = cfg["attention_head_dim"]
        g = torch.Generator(device="cuda").manual_seed(5)
        qkv = torch.randn(1, M_TARGET, 3 * H * hd, generator=g, device="cuda").to(torch.bfloat16)
        wq = (1 + 0.1 * torch.randn(hd, generator=g, device="cuda")).to(torch.bfloat16)
        wk = (1 + 0.1 * torch.randn(hd, generator=g, device="cuda")).to(torch.bfloat16)
        rot = static.rotary_emb
        cos, sin = rot[0], rot[1]
        qk_in = (qkv.clone(), wq.clone(), wk.clone(), cos.clone(), sin.clone())
        eps = float(cfg.get("qk_norm_eps", 1e-6))

        def qk():
            return torch.ops.trtllm.minimax_h3_qk_norm_rope(qkv, wq, wk, cos, sin, eps, H, hd)

        qref = None
        qk_bad = []

        def qk_round(tag, timed):
            nonlocal qref
            ts = []
            for it in range(WARMUP + ITERS):
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                r = qk()
                e.record()
                e.synchronize()
                if timed and it >= WARMUP:
                    ts.append(s.elapsed_time(e))
                if qref is None:
                    qref = tuple(x.clone() for x in r)
                elif not all(torch.equal(x, y) for x, y in zip(r, qref)):
                    qk_bad.append(f"{tag}@{it}")
                for x, y in zip((qkv, wq, wk, cos, sin), qk_in):
                    if not torch.equal(x, y):
                        qk_bad.append(f"{tag}@{it}:input_changed")
            return ts

        qk_round("check", False)
        qk_a = qk_round("timedA", True)
        qk_b = qk_round("timedB", True)
        qk_round("repeat", False)
        out["qk_hash"] = [sha(x) for x in qref]
        if branch:
            fr._launch_by_device[torch.cuda.current_device()] = (1, None, 1)
            r = qk()
            out["qk_vs_main_launch_bitwise"] = all(torch.equal(x, y) for x, y in zip(r, qref))
            fr._launch_by_device.clear()
            real = torch.cuda.get_device_capability
            try:
                torch.cuda.get_device_capability = lambda *a, **k: (10, 0)
                out["sm100_launch"] = list(fr._default_launch(torch.device("cuda", rank)))
            finally:
                torch.cuda.get_device_capability = real
                fr._launch_by_device.clear()
        out["qk_bad"] = qk_bad[:10]
        out["launch"] = list(fr._default_launch(torch.device("cuda", rank))) if branch else [1, "auto", 1]

    out["bad"] = bad[:20]
    out["n_bad"] = len(bad)
    per_fwd_a = [t + NUM_FORWARD_LAYERS * k for t, k in zip(tail_a, qk_a)]
    per_fwd_b = [t + NUM_FORWARD_LAYERS * k for t, k in zip(tail_b, qk_b)]
    out["timing"] = {
        "tail_A": stats(tail_a), "tail_B": stats(tail_b),
        "qk_A": stats(qk_a), "qk_B": stats(qk_b),
        "per_forward_A": stats(per_fwd_a), "per_forward_B": stats(per_fwd_b),
    }
    q.put(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ckpt = sys.argv[1]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(r, ckpt, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = sorted([q.get(timeout=3000) for _ in range(2)], key=lambda r: r["rank"])
    for p in procs:
        p.join()
    r0, r1 = res
    log(f"done in {time.time() - t0:.0f}s CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        import subprocess

        log(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout)
    except Exception as e:  # noqa: BLE001
        log(f"topo unavailable: {e}")
    ranks_equal = r0["hashes"] == r1["hashes"] and r0["qk_hash"] == r1["qk_hash"]
    passed = (
        ranks_equal
        and r0["n_bad"] == 0 and r1["n_bad"] == 0
        and not r0["qk_bad"] and not r1["qk_bad"]
        and not r0["recompiled_after_warmup"] and not r1["recompiled_after_warmup"]
        and all(c and c > 1 for c in r0["overlap_chunks"])
    )
    if r0["branch_tree"]:
        passed = passed and r0.get("qk_vs_main_launch_bitwise") and r1.get("qk_vs_main_launch_bitwise")
        passed = passed and r0.get("sm100_launch") == [1, None, 1]
    t = r0["timing"]
    pa, pb = t["per_forward_A"], t["per_forward_B"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": pa["median"], "noise_floor": max(pa["noise"], pb["noise"])}},
        "output": {"class": "exact"},
        "passed": bool(passed),
        "aa": {"per_forward_A_median": pa["median"], "per_forward_B_median": pb["median"],
               "aa_diff_ms": abs(pa["median"] - pb["median"])},
        "timing_rank0": t,
        "timing_rank1": r1["timing"],
        "rank0_equals_rank1": ranks_equal,
        "hashes": r0["hashes"], "qk_hash": r0["qk_hash"],
        "branch_tree": r0["branch_tree"], "launch": r0["launch"], "sm100_launch": r0.get("sm100_launch"),
        "qk_vs_main_launch_bitwise": [r0.get("qk_vs_main_launch_bitwise"), r1.get("qk_vs_main_launch_bitwise")],
        "bad": [r0["bad"], r1["bad"]], "qk_bad": [r0["qk_bad"], r1["qk_bad"]],
        "recompiled_after_warmup": [r0["recompiled_after_warmup"], r1["recompiled_after_warmup"]],
        "overlap_chunks": r0["overlap_chunks"], "layout": r0["layout"],
        "gpus": [r0["gpu_uuid"], r1["gpu_uuid"]], "cuda_visible": r0["cuda_visible"],
        "compared": "per iteration (check, timed A, timed B, untimed repeat; 5 warmup + 20 each): video/audio "
                    "outputs and the norm_out tail (proj_out input) bitwise vs the plan reference and, in the "
                    "branch tree, vs main's code paths emulated in-process; rank0==rank1 hashes; inputs read-only; "
                    "NaN-poisoned allocator; no dynamo recompiles after warmup. Cross-arm: compare 'hashes' and "
                    "'qk_hash' with the baseline arm.",
    }))


if __name__ == "__main__":
    main()
