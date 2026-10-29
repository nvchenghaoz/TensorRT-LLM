"""Idea 17 run_gpu_pair unit: one denoise step's routed GEMMs at the served 2048^2 shapes.

Same script in both trees (base = main: Pingpong 128x128x128 QKV3, 128x128x128 LPT pair, FP8R
arm 1; branch = large-M routes). Calls go through the served entry points only:
flux3_sm120_strided_gemm.gemm_qkv3, flux3_sm120_lpt_pair.run, flux3_fp8r_gemm.gemm (default arm).
SwiGLU gate_up is not routed (identical in both trees) and not in the unit.

Cases (one CUDA graph each = one denoise step's routed GEMM calls, NSETS=2 input sets per M
alternating, so A + W + out per call exceed the 128 MB L2):
  Served single-block M (T=80, knowledge 384): t2i 16464, i2i 32848 (BENCH_SINGLE_MS overrides,
  e.g. "16896,33280" for the old T=512 shapes).
  t2i-2048: 28 single blocks @ M 16464 + 4 image mode blocks @ M 16384 (QKV3 + tail pair each)
            + image.0 FP8R (QKV, gate_up, down, attn_out) @ M 16384.
  i2i-2048: 28 single blocks @ M 32848 + 4 image mode blocks (1-4) @ M 16384 + image.0 FP8R
            @ M 16384 (knowledge 316: image_cond mode blocks are never primsts-prepared, so they
            take the generic linear route, not these GEMMs; the M=512 text blocks stay < 8192).
Inputs: NVFP4 quantized from BF16 normals (flashinfer nvfp4_quantize, 128x4 scales, per-plane
global scales), FP8R from quantize_fp8r. Fixed seeds; pristine copies checked after timing.
Timing: 5 warm replays, then ROUNDS replays per case, cases interleaved; kernel_ms = median per
replay (one step), noise_floor = (p90 - p10) / median.
Verification (untimed, after timing): references for every output buffer from main's modules
called directly before capture; then the timed sequence again (5 warm + ROUNDS alternating
rounds) with every output poisoned before each replay and compared bitwise against the
reference and the graph's first replay, inputs checked against pristine copies before each
replay. output.class comes from this check.
"""

import json
import os
import sys
import time

import torch

ROUNDS = int(os.environ.get("BENCH_ROUNDS", "30"))
NSETS = 2
H, FF = 3072, 9216


def log(*a):
    print(*a, flush=True)


def _quant(x):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    gs = float(x.abs().amax().float() / (448.0 * 6.0))
    inv = torch.tensor([1.0 / gs], device="cuda", dtype=torch.float32)
    q, sf = fp.nvfp4_quantize(x, inv)
    return q, sf, gs


def nvfp4_inputs(m, g):
    """QKV3 (3 planes, own global scales) + down + attn_out operands at M rows."""
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16, generator=g)
    a, a_sf, a_gs = _quant(x)
    planes = [_quant(torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * s)
              for s in (0.02, 0.035, 0.012)]
    wqkv = torch.cat([p[0] for p in planes], 0).contiguous()
    sqkv = torch.cat([p[1] for p in planes], 0).contiguous()
    aqkv = torch.tensor([a_gs * p[2] for p in planes], device="cuda", dtype=torch.float32)
    xd = torch.randn(m, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    d, d_sf, d_gs = _quant(xd)
    wd, swd, wd_gs = _quant(torch.randn(H, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    wo, swo, wo_gs = _quant(torch.randn(H, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    return {
        "m": m, "a": a, "a_sf": a_sf, "wqkv": wqkv, "sqkv": sqkv, "aqkv": aqkv,
        "qkv_out": torch.empty(3 * m * H, device="cuda", dtype=torch.bfloat16),
        "d": d, "d_sf": d_sf, "wd": wd, "swd": swd,
        "alpha_d": torch.tensor([d_gs * wd_gs], device="cuda", dtype=torch.float32),
        "wo": wo, "swo": swo,
        "alpha_o": torch.tensor([a_gs * wo_gs], device="cuda", dtype=torch.float32),
        "down_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
        "attn_out": torch.empty(m, H, device="cuda", dtype=torch.bfloat16),
    }


def fp8r_inputs(m, g):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    res = {}
    for name, (n, k) in {"qkv": (3 * H, H), "gate_up": (2 * FF, H), "down": (H, FF),
                         "attn_out": (H, H)}.items():
        x = (torch.randn((m, k), device="cuda", generator=g) * 1.5).to(torch.bfloat16)
        x[:, :8] *= 20
        w = (torch.randn((n, k), device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        data, scales = quantize_fp8r(x)
        wd, ws = quantize_fp8r_weight(w)
        res[name] = (data.contiguous(), wd, scales.float().reshape(-1).contiguous(), ws,
                     torch.empty((m, n), device="cuda", dtype=torch.bfloat16))
    return res


def qkv3(c):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    strided.gemm_qkv3(c["a"], c["wqkv"], c["a_sf"], c["sqkv"], c["aqkv"], c["qkv_out"])


def pair(c):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    lpt.run(c["d"], c["wd"], c["d_sf"], c["swd"], c["alpha_d"], c["down_out"],
            c["a"], c["wo"], c["a_sf"], c["swo"], c["alpha_o"], c["attn_out"])


def fp8r(f):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg

    for name in ("qkv", "gate_up", "down", "attn_out"):
        a, w, sa, sw, out = f[name]
        fg.gemm(a, w, sa, sw, out=out)


def preload():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda", torch.cuda.current_device())
    fp.sm120_linear_preload(dev)
    strided.preload(dev)
    lpt.preload(dev)
    fg.preload(dev)


def case_calls(sets, f8, singles_m, n_single, mode_m, n_mode):
    calls = []
    for i in range(n_single):
        c = sets[singles_m][i % NSETS]
        calls += [lambda c=c: qkv3(c), lambda c=c: pair(c)]
    for i in range(n_mode):
        c = sets[mode_m][i % NSETS]
        calls += [lambda c=c: qkv3(c), lambda c=c: pair(c)]
    calls.append(lambda: fp8r(f8))
    return calls


def capture(calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    torch.cuda.synchronize()
    return g


def tensors_of(sets, f8):
    out = []
    for lst in sets.values():
        for c in lst:
            out += [v for k, v in c.items() if isinstance(v, torch.Tensor) and not k.endswith("_out")]
    for v in f8.values():
        out += list(v[:4])
    return out


def _bits(t):
    return t.view(torch.int16)


def references(sets, f8):
    """Main's modules called directly (as in check.py), each on its own workspace.

    QKV3: strided.qkv_pingpong_module().gemm_qkv3_pingpong with main's raster/swizzle; pair:
    lpt.module().lpt_pair with lpt.sched_mode() on a separate zeroed workspace; FP8R: arm 1.
    Returns {id(out buffer): reference tensor}.
    """
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda", torch.cuda.current_device())
    qws = torch.zeros(strided._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    pws = torch.zeros(lpt._WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    main_qkv = strided.qkv_pingpong_module()
    main_pair = lpt.module()
    refs = {}
    for lst in sets.values():
        for c in lst:
            rq = torch.empty_like(c["qkv_out"])
            main_qkv.gemm_qkv3_pingpong(c["a"], c["wqkv"], c["a_sf"], c["sqkv"], c["aqkv"], rq, qws,
                                        int(strided.QKV3_RASTER), int(strided.QKV3_SWIZZLE))
            rd = torch.empty_like(c["down_out"])
            ra = torch.empty_like(c["attn_out"])
            main_pair.lpt_pair(c["d"], c["wd"], c["d_sf"], c["swd"], c["alpha_d"], rd,
                               c["a"], c["wo"], c["a_sf"], c["swo"], c["alpha_o"], ra, pws,
                               lpt.sched_mode(), 0)
            refs[id(c["qkv_out"])] = rq
            refs[id(c["down_out"])] = rd
            refs[id(c["attn_out"])] = ra
    for name in ("qkv", "gate_up", "down", "attn_out"):
        a, w, sa, sw, out = f8[name]
        refs[id(out)] = fg.gemm(a, w, sa, sw, arm=1, slot=1)
    torch.cuda.synchronize()
    return refs


def case_outputs(sets, f8, singles_m, mode_m):
    outs = []
    for m in (singles_m, mode_m):
        for c in sets[m]:
            outs += [c["qkv_out"], c["down_out"], c["attn_out"]]
    outs += [v[4] for v in f8.values()]
    return outs


def verify(graphs, outs, refs, sets, f8, pristine, rounds):
    """Untimed copy of the timed sequence: 5 warm replays of both graphs, then `rounds` rounds
    alternating (t2i,i2i on even rounds, i2i,t2i on odd). Before each replay: inputs vs pristine,
    poison every output buffer that graph writes. After: bitwise vs main reference and vs that
    graph's first replay."""
    names = list(graphs)
    seq = [names] * 5 + [names if r % 2 == 0 else names[::-1] for r in range(rounds)]
    stat = {k: {"matched": 0, "replays": 0, "match_ref": 0, "match_first": 0} for k in names}
    first = {}
    inputs_ok = True
    order_log = []
    for i, step in enumerate(seq):
        if i % 10 == 0:
            log(f"verify step {i}/{len(seq)}: " + " ".join(
                f"{k}={stat[k]['matched']}/{stat[k]['replays']}" for k in names))
        order_log.append(",".join(step))
        for k in step:
            if not all(torch.equal(a, b) for a, b in zip(pristine, tensors_of(sets, f8))):
                inputs_ok = False
            for o in outs[k]:
                _bits(o).fill_(-1)
            graphs[k].replay()
            torch.cuda.synchronize()
            eq_ref = all(torch.equal(_bits(o), _bits(refs[id(o)])) for o in outs[k])
            if k not in first:
                first[k] = [o.clone() for o in outs[k]]
            eq_first = all(torch.equal(_bits(o), _bits(f)) for o, f in zip(outs[k], first[k]))
            s = stat[k]
            s["replays"] += 1
            s["match_ref"] += eq_ref
            s["match_first"] += eq_first
            s["matched"] += eq_ref and eq_first
    if not all(torch.equal(a, b) for a, b in zip(pristine, tensors_of(sets, f8))):
        inputs_ok = False
    alt = {"warm": "5 x (" + ",".join(names) + ")",
           "rounds": f"{rounds} rounds: even {','.join(names)} / odd {','.join(names[::-1])}",
           "first_rounds": order_log[5:9]}
    return stat, inputs_ok, alt


def main():
    import tensorrt_llm  # noqa: F401

    t0 = time.time()
    preload()
    try:
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_large_m as lm

        log(f"tree: branch (large-M routes {sorted(lm._enabled_routes())}, min M {lm.LARGE_M_MIN})")
    except ImportError:
        log("tree: base (no flux3_sm120_large_m)")
    log(f"preload {time.time() - t0:.0f}s")
    g = torch.Generator(device="cuda")
    g.manual_seed(2048)
    m_t2i, m_i2i = (int(v) for v in os.environ.get("BENCH_SINGLE_MS", "16464,32848").split(","))
    log(f"single-block M: t2i {m_t2i}, i2i {m_i2i}; mode blocks + image.0 FP8R M 16384")
    sets = {m: [nvfp4_inputs(m, g) for _ in range(NSETS)] for m in (16384, m_t2i, m_i2i)}
    f8 = fp8r_inputs(16384, g)
    pristine = [t.clone() for t in tensors_of(sets, f8)]
    refs = references(sets, f8)
    log(f"references from main's modules: {len(refs)} buffers ({time.time() - t0:.0f}s)")
    outs = {"t2i-2048": case_outputs(sets, f8, m_t2i, 16384),
            "i2i-2048": case_outputs(sets, f8, m_i2i, 16384)}
    cases = {
        "t2i-2048": case_calls(sets, f8, m_t2i, 28, 16384, 4),
        "i2i-2048": case_calls(sets, f8, m_i2i, 28, 16384, 4),
    }
    graphs = {k: capture(v) for k, v in cases.items()}
    log(f"captured {list(graphs)} ({time.time() - t0:.0f}s)")
    for _ in range(5):
        for gr in graphs.values():
            gr.replay()
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    names = list(graphs)
    for r in range(ROUNDS):
        order = names if r % 2 == 0 else names[::-1]
        for k in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[k].replay()
            e1.record()
            e1.synchronize()
            times[k].append(e0.elapsed_time(e1))
        if r % 10 == 0:
            log(f"round {r}: " + " ".join(f"{k}={times[k][-1]:.3f}ms" for k in names))
    ok = all(torch.equal(a, b) for a, b in zip(pristine, tensors_of(sets, f8)))
    log("verification pass (untimed, same sequence as timing)")
    stat, vin_ok, alt = verify(graphs, outs, refs, sets, f8, pristine, ROUNDS)
    for k, s in stat.items():
        log(f"verify {k}: matched {s['matched']}/{s['replays']} (ref {s['match_ref']}, "
            f"first {s['match_first']})")
    exact = vin_ok and ok and all(s["matched"] == s["replays"] for s in stat.values())
    verification = {k: {"matched": f"{s['matched']}/{s['replays']}",
                        "matched_vs_main_ref": f"{s['match_ref']}/{s['replays']}",
                        "matched_vs_first_replay": f"{s['match_first']}/{s['replays']}",
                        "buffers_per_replay": len(outs[k])} for k, s in stat.items()}
    batch = {}
    for k, v in times.items():
        v = sorted(v)
        med = v[len(v) // 2]
        p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
        batch[k] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med,
                    "verify_matched": verification[k]["matched"]}
        log(f"{k}: median {med:.3f} ms/step p10 {p10:.3f} p90 {p90:.3f}")
    print(json.dumps({"batch": batch,
                      "output": {"class": "exact" if exact else "mismatch"},
                      "verification": verification, "inputs_unchanged": ok and vin_ok,
                      "inputs_unchanged_after_timing": ok, "inputs_unchanged_in_verify": vin_ok,
                      "alternation": alt,
                      "reference": "main modules direct: gemm_qkv3_pingpong (QKV3_RASTER/"
                                   "SWIZZLE, own workspace), lpt.module().lpt_pair (sched_mode, "
                                   "own workspace), fg.gemm(arm=1, slot 1)",
                      "shapes": {"t2i_single_M": m_t2i, "i2i_single_M": m_i2i, "mode_M": 16384},
                      "unit": "one CUDA graph replay = one denoise step's QKV3 + tail-pair calls "
                              "(28 single + 4 image mode blocks) + image.0's 4 FP8R GEMMs"}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
