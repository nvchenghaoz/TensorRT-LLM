"""Branch 34 (idea 78): SM120 CUTLASS QKV path writes Q/K/V as three contiguous planes.

  bench : runs in BOTH trees (base and candidate). The real chain of one NVFP4 attention's QKV:
          op = trtllm::flux3_primsts_nvfp4_qkv_rope (per-plane alpha, checkpoint route), then
          flux3_primsts.qkv_planes (as primsts.qkv_forward reads it), then q/k/v.contiguous()
          (as Flux3Attention._attend does). Base: packed [M,9216] -> 3 strided copies; candidate:
          planar -> no-op .contiguous(). CUDA events, interleaved M, median of ITERS samples of
          REPS chains each. Also times the op alone. Last line = evidence JSON.
  check : candidate only. Untimed, bitwise:
          (1) op planar vs packed path (switch off), per-plane and one-alpha, M=4176/4096/80:
              planes == packed planes .contiguous(), 50 repeated calls vs reference and iter 0,
              inputs restored from pristine copies before every call and verified unchanged.
          (2) pass-level in-place (R1a: q/k are planes 0/1 of out, src==dst) and copy_v=False
              (R1b: v already plane 2), restored before every call, 50 calls each.
          (3) qkv_planes layout decision (inferred and explicit), CUDA graph capture + 3 replays,
              one pass compile per layout (no JIT per M), CuTe fused kernel reference bitwise.
"""

import json
import statistics
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope as qr
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

WARMUP = 5
ITERS = 50
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
    qs = (torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32)
          if per_plane else None)
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    return (a, w, asf, wsf, ags, qw[0][2], qn, kn, cs, None, qs)


op = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope


def tensors(args):
    return [t for t in args if isinstance(t, torch.Tensor)]


def chain(args):
    a, w = args[0], args[1]
    o = op(*args)
    q, k, v = fp.qkv_planes(o, a, w, 128, qkv_scale=args[-1] is not None)
    return q.contiguous(), k.contiguous(), v.contiguous()


# ----------------------------------------------------------------------------- bench
def bench():
    reps = 5
    iters = 60
    fns = {}
    for M in (4176, 4096):
        args = mk(M, M)
        fns[(M, "chain")] = lambda args=args: chain(args)
        fns[(M, "op")] = lambda args=args: op(*args)
    args1 = mk(4176, 1)
    o = op(*args1)
    layout = ("planar" if fp.qkv_planes(o, args1[0], args1[1], 128, qkv_scale=True)[1].is_contiguous()
              else "packed")
    log("layout", layout)
    for f in fns.values():
        for _ in range(WARMUP):
            f()
    torch.cuda.synchronize()
    ev = {k: [] for k in fns}
    for it in range(iters):
        for key, f in fns.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                f()
            e.record()
            ev[key].append((s, e))
        if it % 20 == 0:
            log("iter", it)
    torch.cuda.synchronize()
    res = {}
    for (M, n), lst in ev.items():
        t = sorted(s.elapsed_time(e) / reps for s, e in lst)
        med = statistics.median(t)
        p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
        res[f"{n}_M{M}"] = {"ms": round(med, 5), "noise": round((p90 - p10) / med, 5)}
        log(f"{n} M={M}: {med * 1000:.1f} us noise {(p90 - p10) / med:.4f}")
    main = res["chain_M4176"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": main["ms"], "noise_floor": main["noise"]}},
        "output": {"class": "exact"},
        "layout": layout,
        "what": "per-attention QKV chain: NVFP4 QKV op (per-plane alpha) + qkv_planes + q/k/v "
                ".contiguous(), ms per call, M=4176 headline",
        "detail": res,
    }))


# ----------------------------------------------------------------------------- check
def set_planar(on):
    fp.SM120_QKV_PLANAR = bool(on)


def cute_ref(args):
    a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
    pos = torch.arange(a.shape[0], device="cuda", dtype=torch.int64)
    return flux3_sm120_nvfp4.run("qkv", a, w, asf, wsf, ags, wgs, q_norm=qn, k_norm=kn,
                                 cos_sin=cs, positions=pos, qkv_scale=qs)


def restore(dst_list, src_list):
    for d, s in zip(dst_list, src_list):
        d.copy_(s)


def check_op(M, per_plane, results):
    args = mk(M, M + (0 if per_plane else 1), per_plane)
    a, w = args[0], args[1]
    ins = tensors(args)
    pristine = [t.clone() for t in ins]
    set_planar(False)
    packed = op(*args).clone()
    ref_planes = [p.contiguous() for p in packed.split(H, dim=-1)]
    cute_ok = bool(torch.equal(packed, cute_ref(args)))
    set_planar(True)
    for _ in range(WARMUP):
        restore(ins, pristine)
        op(*args)
    first = None
    n_ref = n_first = n_inputs = 0
    layout_ok = True
    for _ in range(ITERS):
        n_inputs += all(torch.equal(t, p) for t, p in zip(ins, pristine))
        restore(ins, pristine)
        o = op(*args)
        planes = fp.qkv_planes(o, a, w, 128, qkv_scale=per_plane)
        layout_ok &= tuple(o.shape) == (M, 3 * H) and o.is_contiguous() and all(
            p.is_contiguous() and p.data_ptr() == o.data_ptr() + i * M * H * 2
            for i, p in enumerate(planes))
        if first is None:
            first = o.clone()
        n_ref += all(torch.equal(p, r) for p, r in zip(planes, ref_planes))
        n_first += bool(torch.equal(o, first))
    ok = (n_ref == ITERS and n_first == ITERS and n_inputs == ITERS and layout_ok and cute_ok)
    key = f"op_{'per_plane' if per_plane else 'one_alpha'}_M{M}"
    results[key] = {"planes_bitwise_vs_packed_contiguous": f"{n_ref}/{ITERS}",
                    "bitwise_vs_iter0": f"{n_first}/{ITERS}",
                    "inputs_unchanged_before_reset": f"{n_inputs}/{ITERS}",
                    "planar_layout_contiguous_planes": layout_ok,
                    "packed_path_bitwise_vs_cute_fused": cute_ok, "ok": ok}
    log(key, results[key])
    return ok


def check_pass_modes(M, results):
    """R1a (src == dst in place on planes 0/1) and R1b (copy_v False), pass level."""
    args = mk(M, 7 * M + 3, True)
    a, w, asf, wsf, ags, wgs, qn, kn, cs, _, qs = args
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    runner = fp._sm120_fp4_gemm_runner()
    seg = wsf.numel() // 3
    gq, gk, gv = [runner.run_gemm(a, w[i * H:(i + 1) * H], asf, wsf[i * seg:(i + 1) * seg],
                                  qs[i:i + 1], 0, 0, None, None).clone() for i in range(3)]
    packed = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
    qr.qknorm_rope(gq, gk, gv, packed, qn, kn, cs, pos)
    ref_planes = [p.contiguous() for p in packed.split(H, dim=-1)]
    raw = torch.stack([gq, gk, gv]).contiguous()  # pristine planar GEMM outputs
    all_ok = True
    for mode in ("separate", "inplace", "copy_v_false"):
        buf = torch.empty(3, M, H, device="cuda", dtype=torch.bfloat16)
        srcs = [gq.clone(), gk.clone(), gv.clone()]
        pristine_srcs = [t.clone() for t in srcs]
        first = None
        n_ref = n_first = n_inputs = 0
        for it in range(WARMUP + ITERS):
            if mode == "separate":
                n_inputs += all(torch.equal(t, p) for t, p in zip(srcs, pristine_srcs))
                restore(srcs, pristine_srcs)
                qr.qknorm_rope(srcs[0], srcs[1], srcs[2], buf, qn, kn, cs, pos, planar=True)
            elif mode == "inplace":
                # in place overwrites its own input: restore every call
                buf.copy_(raw)
                n_inputs += 1
                qr.qknorm_rope(buf[0], buf[1], buf[2], buf, qn, kn, cs, pos, planar=True)
            else:
                n_inputs += all(torch.equal(t, p) for t, p in zip(srcs, pristine_srcs))
                restore(srcs, pristine_srcs)
                buf.zero_()
                buf[2].copy_(raw[2])
                qr.qknorm_rope(srcs[0], srcs[1], buf[2], buf, qn, kn, cs, pos, planar=True)
            if it < WARMUP:
                continue
            if first is None:
                first = buf.clone()
            n_ref += all(torch.equal(buf[i], ref_planes[i]) for i in range(3))
            n_first += bool(torch.equal(buf, first))
        ok = n_ref == ITERS and n_first == ITERS and n_inputs == ITERS + WARMUP
        all_ok &= ok
        key = f"pass_{mode}_M{M}"
        results[key] = {"bitwise_vs_packed_planes": f"{n_ref}/{ITERS}",
                        "bitwise_vs_iter0": f"{n_first}/{ITERS}", "ok": ok}
        log(key, results[key])
    return all_ok


def check():
    results, passed = {}, True
    log("backend", fp._sm120_qkv_backend(), "planar default", fp.SM120_QKV_PLANAR)
    # qkv_planes decision: inferred from NVFP4 weight, explicit flag, FP8 weight stays packed
    w_fp4 = torch.empty(3 * H, K // 2, device="cuda", dtype=torch.uint8)
    w_fp8 = torch.empty(3 * H, K, device="cuda", dtype=torch.float8_e4m3fn)
    o = torch.empty(80, 3 * H, device="cuda", dtype=torch.bfloat16)
    act = torch.empty(80, K // 2, device="cuda", dtype=torch.uint8)
    dec = {
        "nvfp4_inferred_planar": fp.qkv_planes(o, act, w_fp4, 128)[1].is_contiguous(),
        "fp8_inferred_packed": not fp.qkv_planes(o, act, w_fp8, 128)[1].is_contiguous(),
        "explicit_false_packed": not fp.qkv_planes(o, act, w_fp4, 128, planar=False)[1].is_contiguous(),
        "helper_true": fp.sm120_qkv_planar(o.device, 3 * H, 128, 128),
        "helper_uncovered_false": not fp.sm120_qkv_planar(o.device, 3 * 384, 128, 128),
    }
    results["layout_decision"] = dec
    passed &= all(dec.values())
    log("layout_decision", dec)
    for per_plane in (True, False):
        for M in (4176, 4096, 80):
            passed &= check_op(M, per_plane, results)
    n_compiled_ops = len(qr._COMPILED)
    for M in (4176, 4096, 80):
        passed &= check_pass_modes(M, results)
    results["pass_compiles"] = {"layouts_after_op_checks": n_compiled_ops,
                                "layouts_total": len(qr._COMPILED),
                                "keys": [str(k) for k in qr._COMPILED]}
    # one compile per layout (packed per-plane, packed one-alpha in place, planar per-plane,
    # planar one-alpha, planar in place copy_v False): no JIT per M
    passed &= len(qr._COMPILED) <= 6
    log("pass_compiles", results["pass_compiles"])
    # CUDA graph capture + 3 replays, planar per-plane route
    set_planar(True)
    args = mk(4176, 11)
    set_planar(False)
    r = [p.contiguous() for p in op(*args).split(H, dim=-1)]
    set_planar(True)
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
        gok &= all(torch.equal(a, b) for a, b in zip(g_out.view(3, -1, H), r))
    results["cuda_graph_replay_x3_bitwise"] = gok
    passed &= gok
    log("graph", gok)
    print(json.dumps({"passed": bool(passed),
                      "reference": "packed path (SM120_QKV_PLANAR=False, = main's layout) split "
                                   "planes .contiguous(), bitwise; packed path also bitwise vs the "
                                   "unmodified CuTe fused QKV kernel",
                      "results": results}))


if __name__ == "__main__":
    torch.manual_seed(0)
    (bench if sys.argv[1:] == ["bench"] else check)()
