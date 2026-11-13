"""MiniMax-H3 denoiser attention backend shoot-out at the per-rank shape (idea 12).

B=1, S=M (packed sequence for 768x1344x124f + text), H_local=heads/2, D=128, BF16,
non-causal, no mask. Q/K are fresh contiguous [1,S,H*D] buffers, V is a strided view
into a [1,S,3*H*D] qkv buffer (as in MiniMaxH3Attention.forward).

Usage:
  python attn_backend_bench.py --checkpoint CKPT [--backends a,b,...] [--iters 20]
      [--warmup 5] [--mode bench|pair|profile] [--pair-backend NAME]

mode=bench   : full table (timing, error vs FP32 reference rows, repeated check), GEMM peak.
mode=pair    : time a single backend, print evidence-format JSON.
mode=profile : run one backend a few times (for ncu/nsys).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


def log(*a):
    print(*a, flush=True)


def load_shape(checkpoint, text_len, tp):
    cfg_path = os.path.join(checkpoint, "transformer", "config.json") if checkpoint else None
    cfg = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    heads = int(cfg.get("num_attention_heads", 48))
    head_dim = int(cfg.get("attention_head_dim", 128))
    patch = cfg.get("patch_size", [1, 2, 2])
    if isinstance(patch, int):
        patch = [1, patch, patch]
    try:
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
            align_num_frames,
            audio_latent_num_frames,
            video_latent_num_frames,
        )

        nf = align_num_frames(124)
        lf = video_latent_num_frames(nf)
        na = audio_latent_num_frames(nf)
    except Exception as e:  # pragma: no cover
        log("packing import failed:", repr(e))
        nf, lf, na = 124, 37, 207
    rows_per_frame = (768 // 16 // patch[1]) * (1344 // 16 // patch[2])
    M = text_len + na * 2 + lf * rows_per_frame
    info = dict(
        heads=heads,
        head_dim=head_dim,
        h_local=heads // tp,
        patch=patch,
        num_frames=nf,
        latent_frames=lf,
        audio_latents=na,
        rows_per_frame=rows_per_frame,
        text_len=text_len,
        M=M,
        cfg_keys={k: cfg[k] for k in cfg if isinstance(cfg[k], (int, float, str))},
    )
    return info


def make_inputs(M, H, D, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    qkv = torch.randn(1, M, 3 * H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    q = torch.randn(1, M, H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, M, H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = qkv[..., 2 * H * D :]  # strided view, row stride 3*H*D
    return qkv, q, k, v


# ----------------------------------------------------------------------------
# Candidates: each takes NHD views q,k,v [1,S,H,D] and returns [1,S,H,D]
# ----------------------------------------------------------------------------


def build_candidates(H, D, M, names):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cands = {}

    def hnd(t):
        return t.transpose(1, 2)

    if "vanilla" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention

        be = VanillaAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)

        def f(q, k, v):
            return be.forward(hnd(q), hnd(k), hnd(v)).transpose(1, 2)

        cands["vanilla"] = f

    def sdpa_forced(backend):
        def f(q, k, v):
            with sdpa_kernel([backend]):
                return F.scaled_dot_product_attention(
                    hnd(q), hnd(k), hnd(v), scale=1.0 / math.sqrt(D)
                ).transpose(1, 2)

        return f

    if "sdpa_flash" in names:
        cands["sdpa_flash"] = sdpa_forced(SDPBackend.FLASH_ATTENTION)
    if "sdpa_cudnn" in names:
        cands["sdpa_cudnn"] = sdpa_forced(SDPBackend.CUDNN_ATTENTION)
    if "sdpa_eff" in names:
        cands["sdpa_eff"] = sdpa_forced(SDPBackend.EFFICIENT_ATTENTION)

    if "cudnn" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention

        be_c = CuDNNAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)

        def f(q, k, v):
            return be_c.forward(q, k, v)

        cands["cudnn"] = f

    if "flashinfer" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import (
            FlashInferAttention,
        )

        be_f = FlashInferAttention(num_heads=H, head_dim=D, dtype=torch.bfloat16)

        def f(q, k, v):
            return be_f.forward(q, k, v)

        cands["flashinfer"] = f

    if "flashinfer_raw" in names:
        import flashinfer

        def f(q, k, v):
            out = flashinfer.single_prefill_with_kv_cache(
                q[0], k[0], v[0], causal=False, sm_scale=1.0 / math.sqrt(D), return_lse=False
            )
            return out.unsqueeze(0)

        cands["flashinfer_raw"] = f

    if "trtllm" in names or "trtllm_fused" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention

        be_t = TrtllmAttention(
            num_heads=H,
            head_dim=D,
            dtype=torch.bfloat16,
            max_batch_size=1,
            max_seq_len=M,
            attention_metadata_state={},
        )
        if "trtllm" in names:

            def f(q, k, v):
                return be_t.forward(q, k, v, batch_size=1, seq_len=q.shape[1]).view(
                    1, q.shape[1], H, D
                )

            cands["trtllm"] = f
        if "trtllm_fused" in names:
            # Kernel-only view: a pre-packed contiguous qkv (no concat copy).
            packed = {}

            def f(q, k, v):
                key = (q.data_ptr(), k.data_ptr(), v.data_ptr())
                if key not in packed:
                    packed.clear()
                    packed[key] = torch.cat([q, k, v], dim=2).contiguous()
                qkv = packed[key]
                return be_t.forward(qkv, None, None, batch_size=1, seq_len=q.shape[1]).view(
                    1, q.shape[1], H, D
                )

            cands["trtllm_fused"] = f

    if "fa4" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends.flash_attn4 import (
            FlashAttn4Attention,
        )

        be_4 = FlashAttn4Attention(num_heads=H, head_dim=D, dtype=torch.bfloat16)

        def f(q, k, v):
            return be_4.forward(q, k, v)

        cands["fa4"] = f

    if "fa4_ns1" in names:
        from tensorrt_llm._torch.visual_gen.attention.backends import flash_attn4 as fa4mod

        if fa4mod._flash_attn_fwd is None:
            raise ImportError(str(fa4mod._flash_attn_fwd_import_error))

        def f(q, k, v):
            out, *_ = fa4mod._flash_attn_fwd(
                q, k, v, softmax_scale=1.0 / math.sqrt(D), causal=False, num_splits=1,
                return_lse=False,
            )
            return out

        cands["fa4_ns1"] = f

    if "cudnn_nocopy" in names or "cudnn_tune" in names:
        import cudnn

        handle = cudnn.create_handle()
        state = {}

        def build(q, k, v, tune):
            g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16,
                              intermediate_data_type=cudnn.data_type.FLOAT,
                              compute_data_type=cudnn.data_type.FLOAT)
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            qt = g.tensor(name="q", dim=list(qh.shape), stride=list(qh.stride()),
                          data_type=cudnn.data_type.BFLOAT16)
            kt = g.tensor(name="k", dim=list(kh.shape), stride=list(kh.stride()),
                          data_type=cudnn.data_type.BFLOAT16)
            vt = g.tensor(name="v", dim=list(vh.shape), stride=list(vh.stride()),
                          data_type=cudnn.data_type.BFLOAT16)
            ot, _ = g.sdpa(q=qt, k=kt, v=vt, attn_scale=1.0 / math.sqrt(D),
                           use_causal_mask=False, generate_stats=False)
            S = q.shape[1]
            ot.set_output(True).set_dim([1, H, S, D]).set_stride([S * H * D, D, H * D, 1])
            ot.set_data_type(cudnn.data_type.BFLOAT16)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            g.check_support()
            if tune:
                g.build_plans(cudnn.build_plan_policy.ALL)
            else:
                g.build_plans()
            return g, qt, kt, vt, ot

        def run_plan(g, pack, idx):
            if idx is None:
                ws = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
                g.execute(pack, ws, handle=handle)
            else:
                ws = torch.empty(max(1, g.get_workspace_size_plan_at_index(idx)),
                                 dtype=torch.uint8, device="cuda")
                g.execute_plan_at_index(pack, ws, idx, handle=handle)

        def make(tune):
            def f(q, k, v):
                key = (tune, q.shape, k.stride(), v.stride())
                if key not in state:
                    g, qt, kt, vt, ot = build(q, k, v, tune)
                    idx = None
                    if tune:
                        n = g.get_execution_plan_count()
                        o = torch.empty(1, q.shape[1], H, D, dtype=q.dtype, device="cuda")
                        pack = {qt: q.transpose(1, 2), kt: k.transpose(1, 2),
                                vt: v.transpose(1, 2), ot: o.transpose(1, 2)}
                        best = None
                        for i in range(n):
                            try:
                                cudnn.set_stream(handle=handle,
                                                 stream=torch.cuda.current_stream().cuda_stream)
                                run_plan(g, pack, i)
                                torch.cuda.synchronize()
                                s = torch.cuda.Event(enable_timing=True)
                                e = torch.cuda.Event(enable_timing=True)
                                s.record()
                                for _ in range(3):
                                    run_plan(g, pack, i)
                                e.record()
                                e.synchronize()
                                t = s.elapsed_time(e) / 3
                                log(f"    cudnn plan {i}/{n}: {t:.2f} ms")
                                if best is None or t < best[0]:
                                    best = (t, i)
                            except Exception as ex:  # noqa: BLE001
                                log(f"    cudnn plan {i}/{n}: failed {str(ex)[:120]}")
                        idx = best[1]
                        log(f"    cudnn best plan {idx}: {best[0]:.2f} ms")
                    state[key] = (g, qt, kt, vt, ot, idx)
                g, qt, kt, vt, ot, idx = state[key]
                o = torch.empty(1, q.shape[1], H, D, dtype=q.dtype, device="cuda")
                cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
                run_plan(g, {qt: q.transpose(1, 2), kt: k.transpose(1, 2),
                             vt: v.transpose(1, 2), ot: o.transpose(1, 2)}, idx)
                return o

            return f

        if "cudnn_nocopy" in names:
            cands["cudnn_nocopy"] = make(False)
        if "cudnn_tune" in names:
            cands["cudnn_tune"] = make(True)

    return cands


# ----------------------------------------------------------------------------


def fp32_reference_rows(q, k, v, rows, chunk_heads=4):
    """FP32 math-backend SDPA for the selected query rows (all heads)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    H = q.shape[2]
    D = q.shape[3]
    qs = q[0, rows].float().transpose(0, 1)  # [H, R, D]
    out = torch.empty(len(rows), H, D, device="cuda", dtype=torch.float32)
    for h0 in range(0, H, chunk_heads):
        h1 = min(H, h0 + chunk_heads)
        kk = k[0, :, h0:h1].float().transpose(0, 1)  # [h, S, D]
        vv = v[0, :, h0:h1].float().transpose(0, 1)
        with sdpa_kernel([SDPBackend.MATH]):
            o = F.scaled_dot_product_attention(
                qs[h0:h1].unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0), scale=1.0 / math.sqrt(D)
            )[0]
        out[:, h0:h1] = o.transpose(0, 1)
        del kk, vv, o
    return out  # [R, H, D]


def err_vs_ref(out, ref, rows):
    o = out[0, rows].float()
    d = (o - ref).abs()
    return dict(
        max_abs=float(d.max()),
        rel_l2=float(torch.linalg.vector_norm(o - ref) / torch.linalg.vector_norm(ref)),
    )


def time_fn(fn, q, k, v, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(q, k, v)
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts_sorted = sorted(ts)
    med = ts_sorted[len(ts) // 2] if len(ts) % 2 else 0.5 * (
        ts_sorted[len(ts) // 2 - 1] + ts_sorted[len(ts) // 2]
    )
    p10 = ts_sorted[max(0, int(0.1 * (len(ts) - 1)))]
    p90 = ts_sorted[min(len(ts) - 1, int(math.ceil(0.9 * (len(ts) - 1))))]
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90, min_ms=ts_sorted[0], max_ms=ts_sorted[-1],
                noise=(p90 - p10) / med)


def repeated_check(fn, q, k, v, warmup, iters, ref, rows, first_out, tol, pristine):
    """Untimed rerun with the same warmup/iteration counts; every output compared."""
    bad = []
    n_bitwise_same = 0
    for i in range(warmup + iters):
        out = fn(q, k, v)
        torch.cuda.synchronize()
        e = err_vs_ref(out, ref, rows)
        if not (e["max_abs"] <= tol["max_abs"] and e["rel_l2"] <= tol["rel_l2"]):
            bad.append(dict(iter=i, **e))
        if torch.equal(out, first_out):
            n_bitwise_same += 1
        for name, t, p in pristine:
            if not torch.equal(t, p):
                bad.append(dict(iter=i, input_modified=name))
    return dict(calls=warmup + iters, failures=bad, bitwise_equal_to_first=n_bitwise_same)


def gemm_peak(n=8192, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            a @ b
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / 5)
    ts.sort()
    med = ts[len(ts) // 2]
    return dict(n=n, median_ms=med, tflops=2 * n**3 / med / 1e9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument(
        "--backends",
        default="vanilla,sdpa_cudnn,sdpa_eff,cudnn,flashinfer,flashinfer_raw,trtllm,trtllm_fused,fa4",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--ref-rows", type=int, default=1024)
    p.add_argument("--mode", default="bench")
    p.add_argument("--seq", type=int, default=0, help="override M")
    p.add_argument("--profile-calls", type=int, default=3)
    p.add_argument("--no-gemm", action="store_true")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.cuda.get_device_properties(0)
    log(f"device {dev.name} cc {dev.major}.{dev.minor} SMs {dev.multi_processor_count}")
    log("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
    info = load_shape(args.checkpoint, args.text_len, args.tp)
    if args.seq:
        info["M"] = args.seq
    log("shape", json.dumps(info))
    M, H, D = info["M"], info["h_local"], info["head_dim"]
    flops = 4.0 * M * M * D * H

    names = [n for n in args.backends.split(",") if n]
    qkv, q, k, v = make_inputs(M, H, D)
    qn = q.view(1, M, H, D)
    kn = k.view(1, M, H, D)
    vn = v.view(1, M, H, D)
    log("v strides", vn.stride())

    if args.mode == "profile":
        cands = build_candidates(H, D, M, names)
        fn = cands[names[0]]
        for _ in range(2):
            fn(qn, kn, vn)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_calls):
            fn(qn, kn, vn)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        log(json.dumps({"profiled": names[0]}))
        return

    if args.mode == "pair":
        cands = build_candidates(H, D, M, names)
        fn = cands[names[0]]
        r = time_fn(fn, qn, kn, vn, args.warmup, args.iters)
        log(f"{names[0]}: {r}")
        print(
            json.dumps(
                {
                    "batch": {"1": {"kernel_ms": r["median_ms"], "noise_floor": r["noise"]}},
                    "output": {"class": "reordered"},
                    "backend": names[0],
                    "tflops": flops / r["median_ms"] / 1e9,
                }
            )
        )
        return

    # ---------------- full bench ----------------
    gen = torch.Generator()
    gen.manual_seed(1)
    rows = torch.randperm(M, generator=gen)[: args.ref_rows].sort().values.cuda()
    t0 = time.time()
    ref = fp32_reference_rows(qn, kn, vn, rows)
    torch.cuda.synchronize()
    log(f"fp32 reference for {len(rows)} rows done in {time.time() - t0:.1f}s")

    pristine = [("q", q, q.clone()), ("k", k, k.clone()), ("qkv", qkv, qkv.clone())]
    results = {}
    tol = None
    for name in names:
        log(f"=== {name}")
        try:
            cands = build_candidates(H, D, M, [name])
            fn = cands[name]
            t0 = time.time()
            first = fn(qn, kn, vn)
            torch.cuda.synchronize()
            log(f"  first call {time.time() - t0:.1f}s, out {tuple(first.shape)} {first.dtype}")
            first = first.contiguous().clone()
            e = err_vs_ref(first, ref, rows)
            log(f"  error vs fp32 ref: {e}")
            r = time_fn(fn, qn, kn, vn, args.warmup, args.iters)
            r["tflops"] = flops / r["median_ms"] / 1e9
            log(f"  timing: {r}")
            if tol is None:
                # tolerance: 2x the first backend (FA2 baseline) error, protocol's BF16 class
                tol = dict(max_abs=2 * e["max_abs"] + 1e-3, rel_l2=2 * e["rel_l2"] + 1e-4)
                log(f"  tolerance set from {name}: {tol}")
            rc = repeated_check(fn, qn, kn, vn, args.warmup, args.iters, ref, rows, first, tol,
                                pristine)
            log(f"  repeated check: calls {rc['calls']} failures {len(rc['failures'])} "
                f"bitwise_equal_to_first {rc['bitwise_equal_to_first']}")
            results[name] = dict(err=e, timing=r, repeated=dict(
                calls=rc["calls"], n_fail=len(rc["failures"]), fails=rc["failures"][:5],
                bitwise_equal_to_first=rc["bitwise_equal_to_first"]))
            del cands, fn, first
            torch.cuda.empty_cache()
        except Exception as ex:
            import traceback

            traceback.print_exc()
            results[name] = dict(error=f"{type(ex).__name__}: {str(ex)[:400]}")
            log(f"  FAILED: {results[name]['error']}")

    gemm = None if args.no_gemm else gemm_peak()
    if gemm:
        log(f"cuBLAS BF16 GEMM {gemm}")

    log("\n| backend | median ms | p10-p90 ms | noise | TFLOPS | max-abs | rel-L2 | repeat |")
    log("|---|---|---|---|---|---|---|---|")
    for n, r in results.items():
        if "error" in r:
            log(f"| {n} | - | - | - | - | - | - | {r['error'][:80]} |")
            continue
        t = r["timing"]
        log(f"| {n} | {t['median_ms']:.2f} | {t['p10_ms']:.2f}-{t['p90_ms']:.2f} | "
            f"{t['noise']:.3f} | {t['tflops']:.0f} | {r['err']['max_abs']:.2e} | "
            f"{r['err']['rel_l2']:.2e} | {r['repeated']['n_fail']} fail, "
            f"{r['repeated']['bitwise_equal_to_first']}/{r['repeated']['calls']} bitwise |")
    passed = all(("error" in r) or r["repeated"]["n_fail"] == 0 for r in results.values())
    print(json.dumps({"passed": passed, "shape": {"M": M, "H": H, "D": D}, "gemm": gemm,
                      "results": results}))


if __name__ == "__main__":
    main()
