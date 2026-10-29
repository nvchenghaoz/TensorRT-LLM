"""Idea 184 STEP 3: graph-replay microbenchmark of the served attention -> tail-pair chain of the
32 NVFP4 blocks (FLUX.3 klein, SM120), synthetic tensors at the served shapes, no model load.

One chain ("block back") = the ops the NVFP4 tail-pair block runs from FMHA to the residual add:
  main: trtllm::flux3_sm120_fp8_prequant_attention_side_gate_up (mode 3: gate_up forked onto its
        stream, routed FMHA -> BF16 O, join) -> trtllm::flux3_sm120_nvfp4_out_pair (attn_out
        quantize, LPT args, one LPT GEMM) -> residual add
  nfo : trtllm::flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up (same fork; FMHA writes
        attn_out's FP4 + SF) -> trtllm::flux3_sm120_nvfp4_out_pair_prequant (LPT args, same LPT
        GEMM) -> residual add
qkv8 / fp8 scales come from main's trtllm::flux3_primsts_nvfp4_qkv_rope_fp8 at setup (pristine
inputs, never written). Two weight/activation sets per S, cycled, so the working set exceeds L2.

  pair  : evidence unit. One graph = 28 x S4176 + 4 x S4096 chains (one denoise step's NVFP4
          blocks). Arm = nfo when the tree has it (branch), else main (base tree). Soak, then
          SAMPLES CUDA-event samples of REPLAYS replays each; evidence JSON.
  inter : branch only. Graphs mainA, nfoB, mainA2, nfoB2 of the same unit, ROUNDS interleaved
          rounds; medians, B-A deltas, A/A and B/B spreads, per-sample paired ratios.
  check : untimed. Same warmup / iteration / reset points as timing: per S, the nfo chain vs main's
          chain (block output, attn_out, down, packed gate_up, gate_up scales) bitwise on every
          iteration and vs iteration 0; graph replay x5 vs main; inputs unchanged; the nfo FP4/SF
          vs nvfp4_quantize_prepared(main's BF16 attention output).
"""

import json
import math
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

H, HEADS, D, FF = 3072, 24, 128, 9216
SAMPLES = int(os.environ.get("IDEA184_SAMPLES", "200"))
REPLAYS = 3
SOAK_S = float(os.environ.get("IDEA184_SOAK", "15"))
ROUNDS = int(os.environ.get("IDEA184_ROUNDS", "60"))
SPIN = 6_000_000
HAS_NFO = hasattr(fp, "nvfp4_out_pair_prequant")


def log(*a):
    print(*a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def make_case(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x = torch.randn(S, H, device=dev, dtype=torch.bfloat16, generator=g)
    w_qkv = torch.randn(3 * H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_up = torch.randn(2 * FF, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(H, FF, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    s_x, s_qkv, s_up, s_down, s_out = map(_gs, (x, w_qkv, w_up, w_down, w_out))
    # attn_out keeps a unit input global scale as served (_flux3_nvfp4_missing_input_scale).
    s_attn, s_down_in = 1.0, 0.05
    a_x, sf_x = fp.nvfp4_quantize(x, _inv(s_x))
    q_qkv, sf_qkv = fp.nvfp4_quantize(w_qkv, _inv(s_qkv))
    q_up, sf_up = fp.nvfp4_quantize(w_up, _inv(s_up))
    q_down, sf_down = fp.nvfp4_quantize(w_down, _inv(s_down))
    q_out, sf_out = fp.nvfp4_quantize(w_out, _inv(s_out))
    qkv_ad = PreparedNVFP4(
        q_qkv, sf_qkv, s_x, s_qkv,
        qkv_scale=torch.tensor([s_x * s_qkv] * 3, device=dev, dtype=torch.float32),
    )
    up_ad = PreparedNVFP4(q_up, sf_up, s_x, s_up, swiglu=True)
    down_ad = PreparedNVFP4(q_down, sf_down, s_down_in, s_down)
    out_ad = PreparedNVFP4(q_out, sf_out, s_attn, s_out)
    theta = torch.rand(S, 64, device=dev, generator=g) * 6.28
    cos_sin = torch.cat((theta.cos(), theta.sin()), dim=-1).float().contiguous()
    q_norm = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    k_norm = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    residual = torch.randn(S, H, device=dev, dtype=torch.bfloat16, generator=g)
    qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
        a_x, qkv_ad.weight, sf_x, qkv_ad.scale, q_norm, k_norm, cos_sin, qkv_ad.qkv_scale, x,
        qkv_ad.weight, 1.0 / math.sqrt(D),
    )
    torch.cuda.synchronize()
    return {"S": S, "x": x, "a_x": a_x, "sf_x": sf_x, "qkv": qkv_ad, "up": up_ad,
            "down": down_ad, "out": out_ad, "residual": residual, "qkv8": qkv8.clone(),
            "scales": scales.clone()}


def inputs(c):
    ts = [c[k] for k in ("x", "a_x", "sf_x", "residual", "qkv8", "scales")]
    for k in ("up", "down", "out"):
        ts += [c[k].weight, c[k].scale]
    return ts


def chain(c, arm):
    S = c["S"]
    up, down, out = c["up"], c["down"], c["out"]
    if arm == "main":
        attn, packed, sf = torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_side_gate_up(
            c["qkv8"], c["scales"], 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale,
            down.inverse_scale, c["x"], c["qkv"].weight, 3,
        )
        attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair(
            packed, down.weight, sf, down.scale, attn.reshape(S, H), out.weight, out.scale,
        )
        extra = (attn,)
    else:
        fp4, osf, packed, sf = (
            torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up(
                c["qkv8"], c["scales"], 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale,
                down.inverse_scale, c["x"], c["qkv"].weight, 3, out.weight,
            )
        )
        attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
            packed, down.weight, sf, down.scale, fp4.reshape(S, H // 2), osf, out.weight,
            out.scale,
        )
        extra = (fp4, osf)
    return (c["residual"] + (attn_y + mlp_y), attn_y, mlp_y, packed, sf), extra


_CAPTURE_STREAM = []


def capture(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(torch.cuda.Stream())
    with torch.cuda.graph(g, stream=_CAPTURE_STREAM[0], capture_error_mode="thread_local"):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def stat(v):
    v = sorted(v)
    med = v[len(v) // 2]
    return med, (v[(len(v) * 9) // 10] - v[len(v) // 10]) / med, v[len(v) // 10], v[(len(v) * 9) // 10]


def make_unit():
    sets = {4176: [make_case(4176, 1), make_case(4176, 2)], 4096: [make_case(4096, 3),
                                                                     make_case(4096, 4)]}
    plan = [(sets[4176][i % 2]) for i in range(28)] + [(sets[4096][i % 2]) for i in range(4)]
    return sets, plan


def unit_fn(plan, arm):
    def step():
        last = None
        for c in plan:
            last = chain(c, arm)[0][0]
        return last
    return step


def sample(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(SPIN)
    a.record()
    for _ in range(REPLAYS):
        g.replay()
    b.record()
    return a, b


def pair():
    arm = os.environ.get("IDEA184_ARM") or ("nfo" if HAS_NFO and fp.attn_nvfp4_out_enabled()
                                            else "main")
    sets, plan = make_unit()
    pristine = {id(c): [t.clone() for t in inputs(c)] for cs in sets.values() for c in cs}
    g, _ = capture(unit_fn(plan, arm))
    log(f"pair arm={arm} has_nfo={HAS_NFO}: captured 32 chains; soak {SOAK_S}s")
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        g.replay()
        torch.cuda.synchronize()
    times = []
    for i in range(SAMPLES):
        torch.cuda.synchronize()
        a, b = sample(g)
        b.synchronize()
        times.append(a.elapsed_time(b) / REPLAYS)
        if i % 40 == 0:
            log(f"  sample {i}: {times[-1]:.4f} ms")
    ok = all(torch.equal(p, t) for cs in sets.values() for c in cs
             for p, t in zip(pristine[id(c)], inputs(c)))
    med, noise, p10, p90 = stat(times)
    log(f"arm {arm}: {med:.4f} ms/unit noise {noise:.4f} (p10 {p10:.4f} p90 {p90:.4f}) "
        f"inputs_unchanged={ok}")
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": noise}},
        "unit": "one graph replay = 28 x S4176 + 4 x S4096 NVFP4-block backs (side gate_up fork "
                "+ FMHA [+ attn_out quantize] + LPT pair + residual add)",
        "arm": arm, "samples": SAMPLES, "replays_per_sample": REPLAYS, "soak_s": SOAK_S,
        "p10_ms": p10, "p90_ms": p90, "inputs_unchanged": ok, "output": {"class": "exact"},
    }))


def inter():
    assert HAS_NFO, "inter needs the branch tree"
    sets, plan = make_unit()
    graphs = {}
    for name, arm in (("mainA", "main"), ("nfoB", "nfo"), ("mainA2", "main"), ("nfoB2", "nfo")):
        graphs[name], _ = capture(unit_fn(plan, arm))
    log(f"inter: captured {list(graphs)}; soak {SOAK_S}s")
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        for gg in graphs.values():
            gg.replay()
        torch.cuda.synchronize()
    keys = list(graphs)
    times = {k: [] for k in keys}
    for r in range(ROUNDS):
        order = keys if r % 2 == 0 else keys[::-1]
        evs = []
        for k in order:
            torch.cuda.synchronize()
            evs.append((k, *sample(graphs[k])))
        torch.cuda.synchronize()
        for k, a, b in evs:
            times[k].append(a.elapsed_time(b) / REPLAYS)
        if r % 10 == 0:
            log(f"  round {r}: " + " ".join(f"{k}={times[k][-1]:.4f}" for k in keys))
    row = {k: dict(zip(("median_ms", "noise", "p10", "p90"), stat(times[k]))) for k in keys}
    ratios = sorted(b / a for a, b in zip(times["mainA"], times["nfoB"]))
    ratios2 = sorted(b / a for a, b in zip(times["mainA2"], times["nfoB2"]))
    aa = sorted(b / a for a, b in zip(times["mainA"], times["mainA2"]))
    n = len(ratios)
    res = {
        "table": row,
        "delta_us_per_unit_B_minus_A": (row["nfoB"]["median_ms"] - row["mainA"]["median_ms"]) * 1e3,
        "delta_us_per_unit_B2_minus_A2": (row["nfoB2"]["median_ms"] - row["mainA2"]["median_ms"])
        * 1e3,
        "aa_spread_us": abs(row["mainA"]["median_ms"] - row["mainA2"]["median_ms"]) * 1e3,
        "bb_spread_us": abs(row["nfoB"]["median_ms"] - row["nfoB2"]["median_ms"]) * 1e3,
        "paired_ratio_B_over_A": {"median": ratios[n // 2], "p10": ratios[n // 10],
                                  "p90": ratios[(9 * n) // 10],
                                  "frac_faster": sum(r < 1 for r in ratios) / n},
        "paired_ratio_B2_over_A2_median": ratios2[n // 2],
        "paired_ratio_A2_over_A": {"median": aa[n // 2], "p10": aa[n // 10],
                                   "p90": aa[(9 * n) // 10]},
        "per_call_saving_us": (row["mainA"]["median_ms"] + row["mainA2"]["median_ms"]
                               - row["nfoB"]["median_ms"] - row["nfoB2"]["median_ms"]) / 2 * 1e3
        / 32,
        "rounds": ROUNDS, "replays_per_sample": REPLAYS,
    }
    print(json.dumps(res))


def check():
    assert HAS_NFO
    results, passed = {}, True
    for S, seed in ((4176, 11), (4096, 12)):
        c = make_case(S, seed)
        pristine = [t.clone() for t in inputs(c)]
        ref, ref_extra = chain(c, "main")
        ref = [t.clone() for t in ref]
        ref_d, ref_s = fp.nvfp4_quantize_prepared(ref_extra[0].reshape(S, H), c["out"].weight)
        torch.cuda.synchronize()
        for _ in range(2):  # warmup as in capture()
            chain(c, "nfo")
        n_ref = n_first = n_q = 0
        first = None
        iters = 20
        for _ in range(iters):
            out, extra = chain(c, "nfo")
            if first is None:
                first = [t.clone() for t in out]
            n_ref += all(torch.equal(a, b) for a, b in zip(out, ref))
            n_first += all(torch.equal(a, b) for a, b in zip(out, first))
            n_q += bool(torch.equal(extra[0].reshape(S, H // 2), ref_d)
                        and torch.equal(extra[1], ref_s))
        inputs_ok = all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
        g, gout = capture(lambda: chain(c, "nfo")[0])
        g_ok = 0
        for _ in range(5):
            for t in gout:
                t.zero_()
            g.replay()
            torch.cuda.synchronize()
            g_ok += all(torch.equal(a, b) for a, b in zip(gout, ref))
        del g
        inputs_ok &= all(torch.equal(p, t) for p, t in zip(pristine, inputs(c)))
        ok = n_ref == iters and n_first == iters and n_q == iters and g_ok == 5 and inputs_ok
        passed &= ok
        results[f"S{S}"] = {"eager_bitwise_vs_main": f"{n_ref}/{iters}",
                            "eager_vs_call0": f"{n_first}/{iters}",
                            "fp4_sf_vs_quantize_prepared_of_main_O": f"{n_q}/{iters}",
                            "graph_replay_bitwise_vs_main": f"{g_ok}/5",
                            "inputs_unchanged": inputs_ok, "ok": ok}
        log(f"S={S}: {json.dumps(results[f'S{S}'])}")
    print(json.dumps({
        "passed": bool(passed),
        "reference": "main's chain on pristine inputs: flux3_sm120_fp8_prequant_attention_side_gate_up"
                     " (mode 3) + flux3_sm120_nvfp4_out_pair (LPT); bitwise on block output, "
                     "attn_out, down, packed gate_up and its scales",
        "results": results, "counters": dict(fp.SIDE_GATE_UP_COUNTERS),
    }))


if __name__ == "__main__":
    {"pair": pair, "inter": inter, "check": check}[sys.argv[1]]()
