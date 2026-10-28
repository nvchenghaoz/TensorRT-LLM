"""Branch 18: SM120 NVFP4 QKV on CUTLASS FP4GemmRunner + QK-RMSNorm/RoPE pass.

  table : one run, interleaved per iteration, CUDA events at real clocks, median of ITERS per M:
          cute_fused (main's op: flux3_sm120_nvfp4.run('qkv') incl. its F.pad), gemm alone
          (the 3 pinned-tactic CUTLASS GEMMs, q on current + k/v on side streams), pass alone
          (qknorm_rope from separate q/k/v into packed out), composed (the branch's custom op).
  check : untimed sequence with the same warmup/iteration count as the timed one; every iteration
          compared bitwise (all of q, k, v planes) with the protected reference (unmodified CuTe
          fused kernel) and with iteration 0; inputs compared with pristine copies; both the
          per-plane-alpha (checkpoint) and single-alpha routes; several text lengths; packed
          [M, 9216] row stride; CUDA graph capture + 3 replays; no AutoTuner GEMM entries; the
          pass compiles once per layout (no JIT per M).
"""

import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope as qr
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

WARMUP = 5
ITERS = 60
H, K = 3072, 3072


def log(*a):
    print(*a, flush=True)


def mk(M, seed, per_plane=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    ws = [torch.randn(H, K, device="cuda", dtype=torch.bfloat16, generator=g) * s
          for s in (0.02, 0.035, 0.012)]

    def q(t, s=None):
        s = float(t.abs().amax().float() / 2688.0) if s is None else s
        d, sf = fp.nvfp4_quantize(t, torch.tensor([1.0 / s], device="cuda"))
        return d, sf, s

    a, asf, ags = q(x)
    if per_plane:
        qw = [q(w) for w in ws]
    else:
        s = max(float(w.abs().amax().float() / 2688.0) for w in ws)
        qw = [q(w, s) for w in ws]
    w = torch.cat([t[0] for t in qw]).contiguous()
    wsf = torch.cat([t[1] for t in qw]).contiguous()
    qs = torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32) if per_plane else None
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    return (a, w, asf, wsf, ags, qw[0][2], qn, kn, cs, None, qs)


op = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope


def ref(args):
    a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
    pos = torch.arange(a.shape[0], device="cuda", dtype=torch.int64)
    return flux3_sm120_nvfp4.run("qkv", a, w, asf, wsf, ags, wgs, q_norm=qn, k_norm=kn,
                                 cos_sin=cs, positions=pos, qkv_scale=qs)


def tensors(args):
    return [t for t in args if isinstance(t, torch.Tensor)]


def table():
    runner = fp._sm120_fp4_gemm_runner()
    side = fp._sm120_qkv_side_streams(torch.cuda.current_device())
    out = {}
    fns_by_m = {}
    for M in (4096, 4176, 4256):
        args = mk(M, M)
        a, w, asf, wsf, ags, wgs, qn, kn, cs, _, qs = args
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        seg = wsf.numel() // 3

        def gemm(i, a=a, w=w, asf=asf, wsf=wsf, qs=qs, seg=seg):
            return runner.run_gemm(a, w[i * H:(i + 1) * H], asf, wsf[i * seg:(i + 1) * seg],
                                   qs[i:i + 1], 0, 0, None, None)

        def gemms(gemm=gemm):
            cur = torch.cuda.current_stream()
            for s in side:
                s.wait_stream(cur)
            q = gemm(0)
            p = []
            for i, s in enumerate(side):
                with torch.cuda.stream(s):
                    p.append(gemm(i + 1))
            for s in side:
                cur.wait_stream(s)
            for t in p:
                t.record_stream(cur)
            return q, p[0], p[1]

        q, k, v = gemms()
        dst = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
        fns_by_m[M] = {
            "cute_fused": lambda args=args: ref(args),
            "gemm_alone": gemms,
            "pass_alone": lambda q=q, k=k, v=v, dst=dst, qn=qn, kn=kn, cs=cs, pos=pos:
                qr.qknorm_rope(q, k, v, dst, qn, kn, cs, pos),
            "composed": lambda args=args: op(*args),
        }
    for M, fns in fns_by_m.items():
        for f in fns.values():
            for _ in range(WARMUP):
                f()
    torch.cuda.synchronize()
    ev = {(M, n): [] for M, fns in fns_by_m.items() for n in fns}
    for _ in range(ITERS):
        for M, fns in fns_by_m.items():
            for n, f in fns.items():
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                f()
                e.record()
                ev[(M, n)].append((s, e))
    torch.cuda.synchronize()
    for (M, n), lst in ev.items():
        t = sorted(s.elapsed_time(e) * 1000.0 for s, e in lst)
        med = t[len(t) // 2]
        out.setdefault(str(M), {})[n] = [round(med, 1), round((t[(len(t) * 9) // 10] - t[len(t) // 10]) / med, 4)]
    for M, row in out.items():
        row["composed_vs_cute"] = round(row["composed"][0] / row["cute_fused"][0], 4)
        log(M, json.dumps(row))
    print(json.dumps({"passed": True, "table_us_median_noise": out, "iters": ITERS}))


def check():
    from tensorrt_llm._torch.autotuner import AutoTuner

    AutoTuner.get().clear_cache()
    results, passed = {}, True
    log("backend", fp._sm120_qkv_backend())
    compiled_after_first = None
    for per_plane in (True, False):
        for M in (4176, 4096, 4256, 4336, 4416, 4496):
            args = mk(M, M + (0 if per_plane else 1), per_plane)
            pristine = [t.clone() for t in tensors(args)]
            reference = ref(args).clone()
            for _ in range(WARMUP):
                op(*args)
            first = None
            n_ref = n_first = 0
            plane_mm = [0, 0, 0]
            max_abs = 0.0
            for i in range(ITERS):
                o = op(*args)
                if first is None:
                    first = o.clone()
                n_ref += bool(torch.equal(o, reference))
                n_first += bool(torch.equal(o, first))
                for p in range(3):
                    plane_mm[p] += int((o[:, p * H:(p + 1) * H] != reference[:, p * H:(p + 1) * H]).sum())
                max_abs = max(max_abs, float((o.float() - reference.float()).abs().max()))
            inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, tensors(args)))
            layout_ok = tuple(first.shape) == (M, 3 * H) and first.stride() == (3 * H, 1)
            ok = n_ref == ITERS and n_first == ITERS and inputs_ok and layout_ok
            passed &= ok
            key = f"{'per_plane_alpha' if per_plane else 'single_alpha'}_M{M}"
            results[key] = {"bitwise_vs_cute_ref": f"{n_ref}/{ITERS}", "bitwise_vs_iter0": f"{n_first}/{ITERS}",
                            "mismatch_q_k_v_total": plane_mm, "max_abs": max_abs,
                            "inputs_unchanged": inputs_ok, "packed_stride_9216": layout_ok, "ok": ok}
            log(key, results[key])
            if compiled_after_first is None:
                compiled_after_first = len(qr._COMPILED)
        if per_plane:
            per_plane_compiles = len(qr._COMPILED)
    no_jit_per_m = per_plane_compiles == compiled_after_first and len(qr._COMPILED) <= 2
    results["pass_compiles"] = {"after_first_M": compiled_after_first, "after_all_per_plane_M": per_plane_compiles,
                                "total_layouts": len(qr._COMPILED), "no_jit_per_M": no_jit_per_m}
    passed &= no_jit_per_m
    # CUDA graph capture + 3 replays (per-plane route, side streams inside capture)
    args = mk(4176, 7)
    r = ref(args).clone()
    op(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            g_out = op(*args)
    gok = True
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        gok &= bool(torch.equal(g_out, r))
    results["cuda_graph_replay_x3_bitwise"] = gok
    passed &= gok
    entries = [str(k) for k in AutoTuner.get().profiling_cache.cache.keys() if "gemm" in str(k).lower()]
    results["autotuner_gemm_cache_entries"] = entries
    passed &= not entries
    log("graph", gok, "autotuner entries", entries)
    print(json.dumps({"passed": bool(passed), "backend": fp._sm120_qkv_backend(),
                      "reference": "unmodified CuTe fused kernel flux3_sm120_nvfp4.run('qkv') (main's op), bitwise",
                      "results": results}))


if __name__ == "__main__":
    torch.manual_seed(0)
    (table if sys.argv[1:] == ["table"] else check)()
