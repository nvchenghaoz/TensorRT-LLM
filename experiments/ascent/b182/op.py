"""Branch 182 (idea 268): static RMSNorm-bound Q/K E4M3 scale in the VEC qknorm_rope_fp8 kernel.

Modes (python op.py <mode> [base_rev]):
  check : untimed, candidate tree. MAIN = main's flux3_sm120_qknorm_rope_fp8.py extracted with
          `git show <base_rev>:<path>` and imported as its own module in this process (same fresh
          caches). REF_BF16 = flux3_sm120_qknorm_rope.qknorm_rope (main's BF16 norm+RoPE planes,
          which main's E4M3 bytes are the per-tensor quantization of).
          lever 0 (STATIC_QK=0): qkv8 + scales[8] bitwise MAIN.
          lever 1 (STATIC_QK=1): v bytes + scales[2], scales[6], scales[7] bitwise MAIN; q/k bytes
          bitwise a torch static-factor reference (B' = mul_rn(mul_rn(sqrt128, max|w|), 1+2^-6),
          scale = bf16(max(B'/448, min)), e4m3(rn(rot_bf16 * factor)) with satfinite) on REF_BF16;
          scales[0,1,4,5] and scales[3] = (q_d * k_d) * sm_scale bitwise the torch scalars;
          0 saturated (|rot*factor| > 448) in the torch reference.
          M in 4176/4096/4100/4160/4256 and 1/7/33/63, plus a near-bound input (one-hot heads at
          the max|w| column, cos=1) and large-weight norms. 100 repeats == call 0 with inputs
          checked against pristine copies before each reset; CUDA graph x3 == eager.
  bench : production path of this tree (base: main; candidate: static default). One sample = one
          CUDA-graph replay of 28 x M4176 + 4 x M4096 routed calls (planar NVFP4 QKV GEMMs ->
          VEC mode 1 -> mode 2), distinct NVFP4 weights per call. kernel_ms = per call.
  ab    : candidate tree, diagnostic. Arms A (STATIC_QK=0), S (=1), A2 (=0) graphs of the same
          chain + the routed FMHA, interleaved, order reversed every block; per-QKV paired saving
          (A - S)/32 with p10/p90 and A/A (A - A2)/32.
"""

import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x
from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import _MIN_SCALING_FACTOR
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope as qr
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

H, K = 3072, 3072
HEADS, D = 24, 128
SM_SCALE = 1.0 / (128 ** 0.5)
QR8_PATH = "tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/sm120/flux3_sm120_qknorm_rope_fp8.py"
HAS_STATIC = hasattr(qr8, "STATIC_QK")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def u8(t):
    return t.contiguous().view(torch.uint8)


def same(a, b):
    return bool(torch.equal(u8(a), u8(b)))


def f32(x):
    return torch.tensor(x, dtype=torch.float32)


def scale_of(amax):
    s = torch.maximum(f32(amax) / f32(448.0), f32(_MIN_SCALING_FACTOR))
    s = s.to(torch.bfloat16).to(torch.float32)
    return s, f32(1.0) / s


def static_ref(rot, w):
    """Torch static-factor reference: E4M3 bytes, descale, factor, saturated count."""
    maxw = float(w.float().abs().max())
    b = (f32(math.sqrt(128.0)) * f32(maxw)) * f32(1.0 + 2.0 ** -6)
    s, f = scale_of(b.item())
    v = rot.float() * f.item()
    sat = int((v.abs() > 448.0).sum())
    q = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, s, f, sat, float(rot.float().abs().max() / b)


def load_main(rev):
    src = subprocess.run(["git", "show", f"{rev}:{QR8_PATH}"], capture_output=True, text=True,
                         check=True).stdout
    path = "/tmp/b182_main_qr8.py"
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("b182_main_qr8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log("MAIN module from", rev, "VEC", mod.VEC, "FOLD", mod.FOLD,
        "has STATIC_QK", hasattr(mod, "STATIC_QK"))
    return mod


# --------------------------------------------------------------------------- check
def mk_planes(M, seed, wscale=1.0, near_bound=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    planes = torch.empty(3, M, H, device="cuda", dtype=torch.bfloat16)
    for i, s in enumerate((3.0, 2.0, 1.5)):
        planes[i] = (torch.randn(M, H, device="cuda", generator=g) * s).bfloat16()
    qn = (wscale * (1 + 0.3 * torch.randn(128, device="cuda", generator=g))).bfloat16()
    kn = (wscale * (1 + 0.3 * torch.randn(128, device="cuda", generator=g))).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    if near_bound:
        # one-hot heads at the max|w| column with cos = 1, sin = 0 on half of the rows
        th[: (M + 1) // 2] = 0.0
        for plane, w in ((0, qn), (1, kn)):
            col = int(w.float().abs().argmax())
            x = planes[plane].view(M, HEADS, D)
            x[: (M + 1) // 2] = 0
            x[: (M + 1) // 2, :, col] = 7.0
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return planes, (planes[0], planes[1], planes[2], qn, kn, cs, pos)


def set_static(flag):
    qr8.STATIC_QK = flag
    qr8.VEC, qr8.FOLD = True, True


def run_cand(flag, ins):
    set_static(flag)
    return qr8.qknorm_rope_fp8(*ins, SM_SCALE)


def check_case(main, M, seed, results, repeats=100, **kw):
    planes, ins = mk_planes(M, seed, **kw)
    tensors = [planes] + list(ins[3:])
    pristine = [t.clone() for t in tensors]
    m8, ms = main.qknorm_rope_fp8(*ins, SM_SCALE)
    # BF16 reference planes (main's in-place qknorm_rope arithmetic)
    ref = torch.empty(3, M, H, device="cuda", dtype=torch.bfloat16)
    qr.qknorm_rope(ins[0], ins[1], ins[2], ref, *ins[3:], planar=True)
    # main's per-tensor bytes reproduce from REF_BF16 (sanity of the reference)
    _, fq = scale_of(float(ref[0].float().abs().max()))
    main_q_repro = same((ref[0].float() * fq.item()).clamp(-448, 448).to(torch.float8_e4m3fn),
                        m8[0])
    l0_8, l0_s = run_cand(False, ins)
    rec = {"M": M, "kw": kw, "lever0_qkv8": same(l0_8, m8), "lever0_scales": same(l0_s, ms),
           "main_q_repro_from_ref_bf16": main_q_repro}
    c8, cs_ = run_cand(True, ins)
    qd, sq, fq_, satq, rq = static_ref(ref[0], ins[3])
    kd, sk, fk_, satk, rk = static_ref(ref[1], ins[4])
    sc = cs_.cpu()
    msc = ms.cpu()
    rec.update({
        "lever1_v_bytes": same(c8[2], m8[2]),
        "lever1_v_scale": bool(sc[2] == msc[2] and sc[6] == msc[6] and sc[7] == msc[7]),
        "lever1_q_bytes": same(c8[0], qd), "lever1_k_bytes": same(c8[1], kd),
        "lever1_q_mism": int((u8(c8[0]) != u8(qd)).sum()),
        "lever1_k_mism": int((u8(c8[1]) != u8(kd)).sum()),
        "lever1_qk_scales": bool(sc[0] == sq and sc[1] == sk and sc[4] == fq_ and sc[5] == fk_),
        "lever1_s3": bool(sc[3] == (sq * sk) * f32(SM_SCALE)),
        "sat_ref": satq + satk, "max_rot_over_B": max(rq, rk),
        "q_bytes_at_448": int((u8(c8[0]) & 0x7F == 0x7E).sum()),
        "scales_static": [float(x) for x in sc], "scales_main": [float(x) for x in msc],
    })
    # repeats == call 0, inputs unchanged before each reset
    n_first = n_in = 0
    first = (c8.clone(), cs_.clone())
    for _ in range(repeats):
        n_in += all(torch.equal(t, p) for t, p in zip(tensors, pristine))
        for t, p in zip(tensors, pristine):
            t.copy_(p)
        o8, os_ = run_cand(True, ins)
        n_first += same(o8, first[0]) and same(os_, first[1])
    rec["repeats_eq_call0"] = f"{n_first}/{repeats}"
    rec["inputs_unchanged"] = f"{n_in}/{repeats}"
    ok = (rec["lever0_qkv8"] and rec["lever0_scales"] and rec["lever1_v_bytes"]
          and rec["lever1_v_scale"] and rec["lever1_q_bytes"] and rec["lever1_k_bytes"]
          and rec["lever1_qk_scales"] and rec["lever1_s3"] and rec["sat_ref"] == 0
          and n_first == repeats and n_in == repeats and main_q_repro)
    rec["ok"] = ok
    log("case", json.dumps({k: v for k, v in rec.items() if not k.startswith("scales")}))
    results.append(rec)
    return ok


def check_graph(main, M, results):
    _, ins = mk_planes(M, 1000 + M)
    set_static(True)
    eager = qr8.qknorm_rope_fp8(*ins, SM_SCALE)
    eager = (eager[0].clone(), eager[1].clone())
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        qr8.qknorm_rope_fp8(*ins, SM_SCALE)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = qr8.qknorm_rope_fp8(*ins, SM_SCALE)
    ok = True
    for _ in range(3):
        for t in g_out:
            t.view(torch.uint8).fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        ok &= same(g_out[0], eager[0]) and same(g_out[1], eager[1])
    # attention output of the static bytes (info) vs main's
    m8, ms = main.qknorm_rope_fp8(*ins, SM_SCALE)
    a_m = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(m8, ms, 1, HEADS)
    a_s = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(eager[0], eager[1], 1, HEADS)
    rel = float((a_s.float() - a_m.float()).norm() / a_m.float().norm())
    results.append({"graph_M": M, "replay_x3_eq_eager": ok, "fmha_rel_l2_static_vs_main": rel})
    log("graph", M, ok, "fmha rel static vs main", rel)
    return ok


def check(rev):
    main = load_main(rev)
    results, passed = [], True
    log("STATIC_QK default", qr8.STATIC_QK, "VEC", qr8.VEC, "FOLD", qr8.FOLD)
    for M in (4176, 4096, 4100, 4160, 4256, 1, 7, 33, 63):
        passed &= check_case(main, M, 7 * M + 1, results, repeats=100 if M >= 4096 else 20)
    for M in (4176, 63):
        passed &= check_case(main, M, 11 * M, results, repeats=10, near_bound=True)
        passed &= check_case(main, M, 13 * M, results, repeats=10, wscale=6.0, near_bound=True)
    for M in (4176, 4096, 33):
        passed &= check_graph(main, M, results)
    keys = [str(k) for k in qr8._COMPILED]
    print(json.dumps({
        "passed": bool(passed),
        "reference": f"MAIN = git show {rev}:{QR8_PATH} imported in-process (lever 0 bitwise; "
                     "lever 1 V bytes/scales bitwise); lever 1 Q/K = torch static-factor "
                     "reference on main's BF16 qknorm_rope planes (bitwise)",
        "compile_keys": keys, "results": results}))


# --------------------------------------------------------------------------- timing
def mk_call(M, seed, x):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ws = [torch.randn(H, K, device="cuda", dtype=torch.bfloat16, generator=g) * s
          for s in (0.02, 0.035, 0.012)]

    def q(t):
        s = float(t.abs().amax().float() / 2688.0)
        d, sf = fp.nvfp4_quantize(t, torch.tensor([1.0 / s], device="cuda"))
        return d, sf, s

    a, asf, ags = x
    qw = [q(w) for w in ws]
    w = torch.cat([t[0] for t in qw]).contiguous()
    wsf = torch.cat([t[1] for t in qw]).contiguous()
    qs = torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32)
    qn = (1 + 0.3 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.3 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return (a, w, asf, wsf, qn, kn, cs, pos, qs)


def mk_chain():
    xs = {}
    for M in (4176, 4096):
        g = torch.Generator(device="cuda").manual_seed(M)
        xx = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
        s = float(xx.abs().amax().float() / 2688.0)
        d, sf = fp.nvfp4_quantize(xx, torch.tensor([1.0 / s], device="cuda"))
        xs[M] = (d, sf, s)
    seq = [4176] * 28 + [4096] * 4
    return [mk_call(M, 100 + i, xs[M]) for i, M in enumerate(seq)]


def chain_graph(calls, fmha=False):
    def body():
        for c in calls:
            qkv8, scales, _ = fp.nvfp4_qkv_rope_fp8_impl(*c, SM_SCALE)
            if fmha:
                fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(qkv8, scales, 1, HEADS)

    body()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    return graph


def soak(graphs, seconds=10.0):
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        for gr in graphs:
            gr.replay()
        torch.cuda.synchronize()
        n += 1
    return n


def stats(t):
    t = sorted(t)
    med = statistics.median(t)
    p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
    return med, p10, p90


def bench():
    route = ("static_qk" if HAS_STATIC and qr8.STATIC_QK else "main per-tensor") + \
        f" vec={qr8.VEC}"
    log("route", route)
    calls = mk_chain()
    gr = chain_graph(calls)
    log("soak blocks", soak([gr]))
    t = []
    for it in range(200):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        gr.replay()
        e.record()
        e.synchronize()
        t.append(s.elapsed_time(e) / 32)
        if it % 50 == 0:
            log("sample", it, round(t[-1] * 1000, 2), "us/call")
    med, p10, p90 = stats(t)
    log(f"per call {med * 1000:.2f} us [p10 {p10 * 1000:.2f}, p90 {p90 * 1000:.2f}]")
    out_class = "precision" if HAS_STATIC and qr8.STATIC_QK else "exact"
    print(json.dumps({
        "batch": {"1": {"kernel_ms": round(med, 6), "noise_floor": round((p90 - p10) / med, 5)}},
        "output": {"class": out_class,
                   "error": "Q/K E4M3 at the static RMSNorm-bound scale; V bitwise; see "
                            "experiments/ascent/b182/capture.py STEP 0 (fp32-SDPA rel_l2 d/b)"},
        "route": route,
        "what": "per routed call: one CUDA graph of 28 x M4176 + 4 x M4096 planar NVFP4 QKV "
                "GEMMs -> VEC qknorm_rope_fp8 mode 1 -> mode 2, distinct weights per call, "
                "200 samples after a 10 s soak",
        "p10_ms": p10, "p90_ms": p90,
    }))


def ab():
    calls = mk_chain()
    arms = {}
    for name, flag in (("A", False), ("S", True), ("A2", False)):
        set_static(flag)
        arms[name] = chain_graph(calls, fmha=True)
        log("captured", name)
    set_static(True)
    log("soak blocks", soak(list(arms.values())))
    names = list(arms)
    per = {k: [] for k in names}
    for blk in range(200):
        order = names if blk % 2 == 0 else names[::-1]
        ev = {}
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            arms[n].replay()
            e.record()
            ev[n] = (s, e)
        torch.cuda.synchronize()
        for n in names:
            per[n].append(ev[n][0].elapsed_time(ev[n][1]) * 1000)
        if blk % 50 == 0:
            log("block", blk, {n: round(per[n][-1], 1) for n in names})
    res = {}
    for n in names:
        med, p10, p90 = stats(per[n])
        res[n] = {"us_per_graph": round(med, 1), "noise": round((p90 - p10) / med, 4)}
    d = [(a - s) / 32 for a, s in zip(per["A"], per["S"])]
    aa = [(a - b) / 32 for a, b in zip(per["A"], per["A2"])]
    dm, d10, d90 = stats(d)
    am, a10, a90 = stats(aa)
    res["saving_us_per_qkv"] = {"median": round(dm, 2), "p10": round(d10, 2), "p90": round(d90, 2),
                                "frac_faster": sum(x > 0 for x in d) / len(d)}
    res["aa_us_per_qkv"] = {"median": round(am, 2), "p10": round(a10, 2), "p90": round(a90, 2)}
    log(json.dumps(res))
    print(json.dumps({"passed": True, "diagnostic": "in-process interleaved graph-replay chain "
                      "28xM4176+4xM4096 (QKV GEMMs + VEC mode1/2 + routed FMHA), untraced",
                      **res}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else "0761b30b")
    elif mode == "bench":
        bench()
    else:
        ab()
