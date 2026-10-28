"""Branch 100 (idea 159): FLUX.3 SM120 NVFP4 QKV straight to E4M3 (no BF16 q/k, no K1/K2).

main      : trtllm::flux3_primsts_nvfp4_qkv_rope (per-plane alphas, planar in-place: 3 strided GEMMs,
            in-place BF16 norm+RoPE) + fp8_sm12x.quantize_qkv_e4m3 (K1 amax partials, K2 quantize).
candidate : flux3_primsts.nvfp4_qkv_rope_fp8_impl (same 3 GEMMs, then qknorm_rope_fp8: amax pass,
            Triton scale reduce, norm+RoPE->E4M3 pass).

Modes:
  bench : both trees (run_gpu_pair). One sample = CUDA-event-timed replay of ONE CUDA graph holding
          the per-request-step served mix of routed calls, 24 x M=4176 (single blocks) + 4 x M=4096
          (image mode blocks), QKV + quantize only (fmha_v2 is unchanged and gets identical inputs).
          Reported per step (28 calls). 10 s warm soak, 200 samples, median, noise (p90-p10)/med.
          Each tree runs its production path (base: main chain; candidate: new op).
  ab    : candidate tree, diagnostic only: same graphs for main chain and new op in one process,
          interleaved.
  check : candidate tree, untimed. Bitwise vs main at M=4096/4176/4256: qkv8 (q/k/v E4M3 bytes),
          scales[0:4], v plane, attention output (prequant op vs visual_gen_fp8_attention_sm120);
          warmup 5 + 50 calls each compared with reference and call 0, inputs checked against
          pristine copies before each reset; pass-level random magnitudes and inf/NaN rows; CUDA graph
          capture + 3 replays bitwise; compile keys independent of M.
"""

import json
import statistics
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_qknorm_rope as qr
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )
except ImportError:  # base tree
    qr8 = None

H, K = 3072, 3072
HEADS, D = 24, 128
SM_SCALE = 1.0 / (128 ** 0.5)
MS = (4096, 4176, 4256)
WARMUP, ITERS = 5, 50
SOAK_S = 10.0
SAMPLES = 200

op_main = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope


def log(*a):
    print(*a, flush=True)


def mk(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    ws = [torch.randn(H, K, device="cuda", dtype=torch.bfloat16, generator=g) * s
          for s in (0.02, 0.035, 0.012)]

    def q(t):
        s = float(t.abs().amax().float() / 2688.0)
        d, sf = fp.nvfp4_quantize(t, torch.tensor([1.0 / s], device="cuda"))
        return d, sf, s

    a, asf, ags = q(x)
    qw = [q(w) for w in ws]
    w = torch.cat([t[0] for t in qw]).contiguous()
    wsf = torch.cat([t[1] for t in qw]).contiguous()
    qs = torch.tensor([ags * t[2] for t in qw], device="cuda", dtype=torch.float32)
    qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
    cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int64)
    return (a, w, asf, wsf, ags, qw[0][2], qn, kn, cs, pos, qs)


def main_chain(args):
    out = op_main(*args)
    M = out.shape[0]
    planes = out.view(3, M, H)
    q8, k8, v8, scales = fp8_sm12x.quantize_qkv_e4m3(
        *(p.view(1, M, HEADS, D) for p in planes), SM_SCALE)
    return out, q8, k8, v8, scales


def cand_chain(args):
    a, w, asf, wsf, ags, wgs, qn, kn, cs, pos, qs = args
    return fp.nvfp4_qkv_rope_fp8_impl(a, w, asf, wsf, qn, kn, cs, pos, qs, SM_SCALE)


def production_chain(args):
    return cand_chain(args) if qr8 is not None else main_chain(args)


# --------------------------------------------------------------------------- timing
def step_graph(fn, sets):
    """One graph: 24 x M=4176 + 4 x M=4096 calls (one denoise step's routed calls)."""
    seq = [sets[4176]] * 24 + [sets[4096]] * 4

    def body():
        for args in seq:
            fn(args)

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


def time_graphs(graphs):
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        for gr in graphs.values():
            gr.replay()
        torch.cuda.synchronize()
    ev = {k: [] for k in graphs}
    for it in range(SAMPLES):
        for name, gr in graphs.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
            ev[name].append((s, e))
        if it % 50 == 0:
            torch.cuda.synchronize()
            log("sample", it)
    torch.cuda.synchronize()
    res = {}
    for name, lst in ev.items():
        t = sorted(s.elapsed_time(e) for s, e in lst)
        med = statistics.median(t)
        p10, p90 = t[len(t) // 10], t[(len(t) * 9) // 10]
        res[name] = {"ms": round(med, 5), "noise": round((p90 - p10) / med, 5),
                     "p10": round(p10, 5), "p90": round(p90, 5)}
        log(f"{name}: {med * 1000:.1f} us/step noise {(p90 - p10) / med:.4f}")
    return res


def bench():
    route = "fp8_direct" if qr8 is not None else "main_qknorm_rope+K1K2"
    log("route", route)
    sets = {M: mk(M, M) for M in (4176, 4096)}
    graphs = {"step": step_graph(production_chain, sets)}
    res = time_graphs(graphs)["step"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": res["ms"], "noise_floor": res["noise"]}},
        "output": {"class": "exact"},
        "route": route,
        "what": "per denoise step: 24 x M=4176 + 4 x M=4096 routed NVFP4 QKV -> E4M3 q/k/v + "
                "scales (main: nvfp4_qkv_rope + quantize_qkv_e4m3; candidate: "
                "nvfp4_qkv_rope_fp8_impl), one CUDA graph per sample, 200 samples",
        "detail": res,
    }))


def ab():
    sets = {M: mk(M, M) for M in (4176, 4096)}
    graphs = {"main": step_graph(main_chain, sets), "cand": step_graph(cand_chain, sets),
              "main2": step_graph(main_chain, sets)}
    res = time_graphs(graphs)
    d = (res["cand"]["ms"] - res["main"]["ms"]) * 1000
    aa = (res["main2"]["ms"] - res["main"]["ms"]) * 1000
    log(f"cand - main = {d:.1f} us/step (A/A {aa:.1f}); per call {d / 28:.2f} us")
    print(json.dumps({"passed": True, "diagnostic": "in-process interleaved, not pair evidence",
                      "cand_minus_main_us_per_step": round(d, 2),
                      "aa_us_per_step": round(aa, 2), "detail": res}))


# --------------------------------------------------------------------------- check
def u8(t):
    return t.contiguous().view(torch.uint8)


def same(a, b):
    return bool(torch.equal(u8(a), u8(b)))


def op_outputs_equal(ref, cand, M):
    out, q8, k8, v8, scales = ref
    qkv8, sc, gemm = cand
    return {
        "q8": same(qkv8[0], q8.view(M, H)),
        "k8": same(qkv8[1], k8.view(M, H)),
        "v8": same(qkv8[2], v8.view(M, H)),
        "scales": same(sc[:4], scales),
        "v_plane": same(gemm.view(3, M, H)[2], out.view(3, M, H)[2]),
    }


def attention_equal(ref, cand, M):
    out, *_ = ref
    planes = out.view(3, M, H)
    a_main = fp8_sm12x.visual_gen_fp8_attention_sm120(
        *(p.view(1, M, HEADS, D) for p in planes), SM_SCALE, False)
    a_cand = fp8_sm12x.visual_gen_fp8_attention_sm120_prequant(cand[0], cand[1], 1, HEADS)
    return same(a_main, a_cand)


def check_op(M, results):
    args = mk(M, 3 * M + 1)
    ins = [t for t in args if isinstance(t, torch.Tensor)]
    pristine = [t.clone() for t in ins]
    ref = tuple(t.clone() for t in main_chain(args))
    first = None
    n_ref = n_first = n_inputs = 0
    per_field_fail = {}
    for it in range(WARMUP + ITERS):
        n_inputs += all(torch.equal(t, p) for t, p in zip(ins, pristine))
        for t, p in zip(ins, pristine):
            t.copy_(p)
        junk = torch.empty(3 * M * H * 2, device="cuda", dtype=torch.uint8).fill_(0x7F)
        del junk
        o = cand_chain(args)
        if it < WARMUP:
            continue
        eq = op_outputs_equal(ref, o, M)
        for k, v in eq.items():
            if not v:
                per_field_fail[k] = per_field_fail.get(k, 0) + 1
        n_ref += all(eq.values())
        if first is None:
            first = tuple(t.clone() for t in o)
        n_first += all(same(a, b) for a, b in zip(o, first))
    n_inputs_final = all(torch.equal(t, p) for t, p in zip(ins, pristine))
    attn_ok = attention_equal(ref, first, M)
    ok = (n_ref == ITERS and n_first == ITERS and n_inputs == WARMUP + ITERS and n_inputs_final
          and attn_ok)
    results[f"op_M{M}"] = {
        "bitwise_vs_main(q8,k8,v8,scales,v_plane)": f"{n_ref}/{ITERS}",
        "bitwise_vs_call0": f"{n_first}/{ITERS}",
        "inputs_unchanged_before_reset": f"{n_inputs}/{WARMUP + ITERS}",
        "attention_output_bitwise": attn_ok,
        "field_failures": per_field_fail,
        "scales": [float(x) for x in first[1][:4].cpu()],
        "ok": ok,
    }
    log(f"op_M{M}", results[f"op_M{M}"])
    return ok


def pass_ref(q, k, v, qn, kn, cs, pos):
    M = q.shape[0]
    out = torch.empty(M, 3 * H, device="cuda", dtype=torch.bfloat16)
    qr.qknorm_rope(q, k, v, out, qn, kn, cs, pos, planar=True)
    planes = out.view(3, M, H)
    q8, k8, v8, sc = fp8_sm12x.quantize_qkv_e4m3(
        *(p.view(1, M, HEADS, D) for p in planes), SM_SCALE)
    return torch.stack([q8.view(M, H), k8.view(M, H), v8.view(M, H)]), sc


def check_pass(results):
    """Pass-level: random magnitudes per plane, packed and planar sources, inf/NaN rows."""
    ok_all = True
    cases = []
    g = torch.Generator(device="cuda").manual_seed(7)
    for case in range(12):
        M = MS[case % 3]
        mags = [10.0 ** float(torch.randint(-3, 4, (1,), generator=g, device="cuda"))
                for _ in range(3)]
        packed = case % 2 == 1
        src = torch.randn(M, 3 * H, device="cuda", generator=g, dtype=torch.float32)
        for i in range(3):
            src[:, i * H:(i + 1) * H] *= mags[i]
        src = src.bfloat16()
        special = None
        if case >= 8:
            r = int(torch.randint(0, M, (1,), generator=g, device="cuda"))
            special = ["inf_q", "nan_k", "inf_v", "nan_all"][case - 8]
            if special == "inf_q":
                src[r, 5] = float("inf")
            elif special == "nan_k":
                src[r, H + 7] = float("nan")
            elif special == "inf_v":
                src[r, 2 * H + 9] = float("-inf")
            else:
                src[r, :] = float("nan")
        if packed:
            q, k, v = src[:, :H], src[:, H:2 * H], src[:, 2 * H:]
        else:
            planar = src.view(M, 3, H).permute(1, 0, 2).contiguous()
            q, k, v = planar[0], planar[1], planar[2]
        qn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        kn = (1 + 0.1 * torch.randn(128, device="cuda", generator=g)).bfloat16()
        th = torch.rand(M, 64, device="cuda", generator=g) * 6.28
        cs = torch.cat([th.cos(), th.sin()], -1).float().contiguous()
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        src_copy = src.clone()
        ref8, ref_sc = pass_ref(q.clone(), k.clone(), v.clone(), qn, kn, cs, pos)
        c8, c_sc = qr8.qknorm_rope_fp8(q, k, v, qn, kn, cs, pos, SM_SCALE)
        eq8 = same(c8, ref8)
        eqs = same(c_sc[:4], ref_sc)
        unchanged = same(src, src_copy)
        ok = eq8 and eqs and unchanged
        ok_all &= ok
        nbad = int((u8(c8) != u8(ref8)).sum())
        cases.append({"M": M, "packed": packed, "mags": mags, "special": special,
                      "qkv8_bitwise": eq8, "scales_bitwise": eqs, "mismatched_bytes": nbad,
                      "inputs_unchanged": unchanged,
                      "scales": [float(x) for x in c_sc[:4].cpu()]})
        log("pass case", cases[-1])
    results["pass_cases"] = cases
    return ok_all


def check_graph(results):
    ok = True
    for M in (4176, 4096):
        args = mk(M, 11 + M)
        ref = tuple(t.clone() for t in main_chain(args))
        eager = tuple(t.clone() for t in cand_chain(args))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out = cand_chain(args)
        torch.cuda.synchronize()
        gok = True
        for _ in range(3):
            for t in g_out:
                t.view(torch.uint8).fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            gok &= all(same(a, b) for a, b in zip(g_out, eager))
            gok &= all(op_outputs_equal(ref, g_out, M).values())
        results[f"cuda_graph_M{M}"] = {"replay_x3_bitwise_vs_eager_and_main": gok}
        ok &= gok
        log(f"cuda_graph_M{M}", gok)
    return ok


def check():
    results, passed = {}, True
    for M in MS:
        passed &= check_op(M, results)
    passed &= check_pass(results)
    passed &= check_graph(results)
    results["compile_keys"] = [str(k) for k in qr8._COMPILED]
    passed &= len({k[3] for k in qr8._COMPILED}) == 2 and len(
        {(k[1], k[2], k[3]) for k in qr8._COMPILED if k[2] == H}) == 2
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main: trtllm::flux3_primsts_nvfp4_qkv_rope (per-plane planar in-place) + "
                     "fp8_sm12x.quantize_qkv_e4m3 (and qknorm_rope + quantize_qkv_e4m3 at pass "
                     "level), visual_gen_fp8_attention_sm120 for the attention output; bitwise",
        "results": results,
    }))


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"bench": bench, "ab": ab, "check": check}[mode]()
