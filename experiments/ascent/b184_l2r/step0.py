"""Branch 184 (idea 244): STEP 0 untraced kill gates and STEP 2 arms, synthetic tensors at the
served shapes (FLUX.3 klein NVFP4, SM120), no model load. CUDA events around graph replays,
arms interleaved round by round in ONE process (same slot).

  d  : STEP 0a (stage D). One graph = one denoise step's 32 NVFP4 blocks (28 x S4176 + 4 x S4096,
       two weight/activation sets per S, cycled): trtllm::flux3_primsts_nvfp4_qkv_rope_fp8 (QKV
       GEMMs + VEC mode 1/2) -> trtllm::flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up
       (nfo FMHA || side gate_up) -> flux3_sm120_nvfp4_out_pair_prequant -> residual add.
       Arms: base (L2DISCARD off = main), U (discard q/k/v, keep_v False = served PAG-off),
       Uqk (discard q/k only), E (evict_first hint, no discard), base2 (A/A).
  r  : STEP 0b (stage R). One graph = image.0's 16 FP8R calls at S4096 (2 sets, cycled):
       A = main: flux3_sm120_fp8r_qkv_qknorm_rope (packed) -> visual_gen_fp8_attention_sm120
       B = trtllm::flux3_sm120_fp8r_qkv_rope_fp8 (L2DISCARD off) -> ..._prequant
       BD = B with L2DISCARD on (keep_v False), A2 = A/A.
  both : STEP 2. One graph = one step: image.0's FP8R call x1 + the 32 NVFP4 blocks (as d), so
       a request is 16 replays. Arms: base (R off, D off), D-only, R-only, both, base2 (A/A).
Last stdout line: JSON.
"""

import json
import math
import os
import sys
import time

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "idea184"))
import bench as b184  # noqa: E402

from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402,F401
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as f8r  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402

H, HEADS, D = 3072, 24, 128
SM = 1.0 / math.sqrt(D)
ROUNDS = int(os.environ.get("B184_ROUNDS", "60"))
REPLAYS = int(os.environ.get("B184_REPLAYS", "2"))
SOAK_S = float(os.environ.get("B184_SOAK", "10"))
log = b184.log


def stat(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2]
    return {"median_ms": med, "p10": v[n // 10], "p90": v[(9 * n) // 10],
            "noise": (v[(9 * n) // 10] - v[n // 10]) / med}


def set_levers(discard, hint=False, r=None):
    qr8.L2DISCARD = bool(discard)
    qr8.L2HINT = bool(hint)
    if r is not None:
        f8r.FP8R_QKV_FP8_DIRECT = bool(r)


# ----------------------------------------------------------------------------- NVFP4 block (D)
def nvfp4_block(c, keep_v):
    S = c["S"]
    up, down, out, qkv = c["up"], c["down"], c["out"], c["qkv"]
    qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
        c["a_x"], qkv.weight, c["sf_x"], qkv.scale, c["q_norm"], c["k_norm"], c["cos_sin"],
        qkv.qkv_scale, c["x"], qkv.weight, SM, keep_v,
    )
    fp4, osf, packed, sf = torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up(
        qkv8, scales, 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale, down.inverse_scale,
        c["x"], qkv.weight, 3, out.weight,
    )
    attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
        packed, down.weight, sf, down.scale, fp4.reshape(S, H // 2), osf, out.weight, out.scale,
    )
    return c["residual"] + (attn_y + mlp_y), qkv8, scales


def make_nvfp4_case(S, seed):
    c = b184.make_case(S, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    theta = torch.rand(S, 64, device="cuda", generator=g) * 6.28
    c["cos_sin"] = torch.cat((theta.cos(), theta.sin()), dim=-1).float().contiguous()
    c["q_norm"] = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    c["k_norm"] = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    return c


def nvfp4_plan():
    sets = {4176: [make_nvfp4_case(4176, 1), make_nvfp4_case(4176, 2)],
            4096: [make_nvfp4_case(4096, 3), make_nvfp4_case(4096, 4)]}
    return [sets[4176][i % 2] for i in range(28)] + [sets[4096][i % 2] for i in range(4)]


# ----------------------------------------------------------------------------- FP8R image.0 (R)
def make_fp8r_case(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, H, device="cuda", generator=g) * 2
    w = torch.randn(3 * H, H, device="cuda", generator=g) * 0.02
    xs = x.abs().amax(-1).clamp_min(1e-6) / 448.0
    ws = w.abs().amax(-1).clamp_min(1e-8) / 448.0
    act = (x / xs[:, None]).to(torch.float8_e4m3fn)
    wq = (w / ws[:, None]).to(torch.float8_e4m3fn)
    theta = torch.rand(S, 64, device="cuda", generator=g) * 6.28
    return {
        "S": S, "act": act, "w": wq, "a_s": xs.float().contiguous(), "w_s": ws.float().contiguous(),
        "qn": (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16),
        "kn": (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16),
        "cs": torch.cat((theta.cos(), theta.sin()), dim=-1).float().contiguous(),
    }


def fp8r_main(c):
    S = c["S"]
    out = torch.ops.trtllm.flux3_sm120_fp8r_qkv_qknorm_rope(
        c["act"], c["w"], c["a_s"], c["w_s"], c["qn"], c["kn"], c["cs"], False)
    q, k, v = (t.reshape(1, S, H).view(1, S, HEADS, D) for t in out.split(H, -1))
    o = torch.ops.trtllm.visual_gen_fp8_attention_sm120(q, k, v, SM, False)
    return o.flatten(2), out[:, 2 * H:]


def fp8r_direct(c, keep_v=False):
    qkv8, sc, gemm = torch.ops.trtllm.flux3_sm120_fp8r_qkv_rope_fp8(
        c["act"], c["w"], c["a_s"], c["w_s"], c["qn"], c["kn"], c["cs"], SM, keep_v)
    o = torch.ops.trtllm.visual_gen_fp8_attention_sm120_prequant(qkv8, sc, 1, HEADS)
    return o.flatten(2), gemm[:, 2 * H:], qkv8, sc


def fp8r_arm(c, r):
    return fp8r_direct(c)[0] if r else fp8r_main(c)[0]


# ----------------------------------------------------------------------------- timing harness
_STREAM = []


def capture(fn, levers):
    set_levers(*levers)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    if not _STREAM:
        _STREAM.append(torch.cuda.Stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_STREAM[0], capture_error_mode="thread_local"):
        fn()
    torch.cuda.synchronize()
    set_levers(False, False, False)
    return g


def sample(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(3_000_000)
    a.record()
    for _ in range(REPLAYS):
        g.replay()
    b.record()
    return a, b


def interleave(graphs, base="base", aa="base2"):
    keys = list(graphs)
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        for gg in graphs.values():
            gg.replay()
        torch.cuda.synchronize()
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
    table = {k: stat(times[k]) for k in keys}
    n = ROUNDS
    res = {"table": table}
    aa_r = sorted(b / a for a, b in zip(times[base], times[aa]))
    res["aa_ratio"] = {"median": aa_r[n // 2], "p10": aa_r[n // 10], "p90": aa_r[(9 * n) // 10]}
    base_med = (table[base]["median_ms"] + table[aa]["median_ms"]) / 2
    res["aa_band_us"] = abs(table[base]["median_ms"] - table[aa]["median_ms"]) * 1e3
    for k in keys:
        if k in (base, aa):
            continue
        rr = sorted(b / a for a, b in zip(times[base], times[k]))
        res[f"{k}_minus_base_us"] = (table[k]["median_ms"] - base_med) * 1e3
        res[f"{k}_paired_ratio"] = {"median": rr[n // 2], "p10": rr[n // 10],
                                    "p90": rr[(9 * n) // 10],
                                    "frac_faster": sum(x < 1 for x in rr) / n}
    return res


def step0_d():
    plan = nvfp4_plan()

    def unit(keep_v):
        def f():
            last = None
            for c in plan:
                last = nvfp4_block(c, keep_v)[0]
            return last
        return f

    graphs = {
        "base": capture(unit(False), (False, False)),
        "U": capture(unit(False), (True, False)),
        "Uqk": capture(unit(True), (True, False)),
        "E": capture(unit(False), (False, True)),
        "base2": capture(unit(False), (False, False)),
    }
    log("d: captured", list(graphs), "compiled keys", [str(k) for k in qr8._COMPILED])
    res = interleave(graphs)
    res["unit"] = "one graph = 32 NVFP4 blocks (28xS4176 + 4xS4096): QKV op + nfo fork + out pair"
    res["per_block_us"] = {k: v / 32 for k, v in res.items() if k.endswith("_minus_base_us")}
    return res


def step0_r():
    sets = [make_fp8r_case(4096, 7), make_fp8r_case(4096, 8)]

    def unit(r):
        def f():
            last = None
            for i in range(16):
                last = fp8r_arm(sets[i % 2], r)
            return last
        return f

    graphs = {
        "base": capture(unit(False), (False, False, False)),
        "B": capture(unit(True), (False, False, True)),
        "BD": capture(unit(True), (True, False, True)),
        "base2": capture(unit(False), (False, False, False)),
    }
    res = interleave(graphs)
    res["unit"] = "one graph = 16 image.0 FP8R calls at S4096 (QKV GEMM .. FMHA)"
    res["per_call_us"] = {k: v / 16 for k, v in res.items() if k.endswith("_minus_base_us")}
    # bitwise: B's FMHA output vs A's, and qkv8/scales vs quantize_qkv_e4m3 on main's planes
    ok = True
    for c in sets:
        ref, _ = fp8r_main(c)
        for disc in (False, True):
            set_levers(disc, False, True)
            got = fp8r_direct(c)[0]
            ok &= bool(torch.equal(ref, got))
        set_levers(False, False, False)
    res["B_fmha_out_bitwise_vs_A"] = ok
    return res


def step2_both():
    plan = nvfp4_plan()
    r_sets = [make_fp8r_case(4096, 7), make_fp8r_case(4096, 8)]

    def unit(r):
        def f():
            fp8r_arm(r_sets[0], r)
            last = None
            for c in plan:
                last = nvfp4_block(c, False)[0]
            return last
        return f

    graphs = {
        "base": capture(unit(False), (False, False, False)),
        "D": capture(unit(False), (True, False, False)),
        "R": capture(unit(True), (False, False, True)),
        "both": capture(unit(True), (True, False, True)),
        "base2": capture(unit(False), (False, False, False)),
    }
    res = interleave(graphs)
    res["unit"] = "one graph = one denoise step: image.0 FP8R call + 32 NVFP4 blocks"
    res["per_request_ms"] = {k: v * 16 / 1e3 for k, v in res.items()
                             if k.endswith("_minus_base_us")}
    res["aa_band_per_request_ms"] = res["aa_band_us"] * 16 / 1e3
    return res


HAS_BRANCH = hasattr(f8r, "fp8r_qkv_rope_fp8")


def pair():
    """Evidence unit for run_gpu_pair, each tree at its served default: one denoise step =
    image.0's FP8R call (base tree: main's chain; branch: R) + the 32 NVFP4 blocks (branch: QKV op
    with keep_v=False as served with PAG off, L2DISCARD default on)."""
    plan = nvfp4_plan()
    r_sets = [make_fp8r_case(4096, 7), make_fp8r_case(4096, 8)]
    samples = int(os.environ.get("B184_SAMPLES", "150"))
    # B184_UNIT=qkvfork: per block only the ops the change acts on (QKV op incl. VEC mode 1/2,
    # then the nfo FMHA || side gate_up fork, where the dropped writeback frees DRAM); the
    # out-pair GEMM + residual add are left out.
    qkvfork = os.environ.get("B184_UNIT", "step") == "qkvfork"

    def block(c):
        S = c["S"]
        up, down, out, qkv = c["up"], c["down"], c["out"], c["qkv"]
        extra = (False,) if HAS_BRANCH else ()
        qkv8, scales, _ = torch.ops.trtllm.flux3_primsts_nvfp4_qkv_rope_fp8(
            c["a_x"], qkv.weight, c["sf_x"], qkv.scale, c["q_norm"], c["k_norm"], c["cos_sin"],
            qkv.qkv_scale, c["x"], qkv.weight, SM, *extra,
        )
        fp4, osf, packed, sf = (
            torch.ops.trtllm.flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up(
                qkv8, scales, 1, HEADS, c["a_x"], up.weight, c["sf_x"], up.scale,
                down.inverse_scale, c["x"], qkv.weight, 3, out.weight,
            )
        )
        if qkvfork:
            return fp4
        attn_y, mlp_y = torch.ops.trtllm.flux3_sm120_nvfp4_out_pair_prequant(
            packed, down.weight, sf, down.scale, fp4.reshape(S, H // 2), osf, out.weight,
            out.scale,
        )
        return c["residual"] + (attn_y + mlp_y)

    r_served = bool(getattr(f8r, "FP8R_QKV_FP8_DIRECT", False))

    def step():
        if r_served:
            fp8r_direct(r_sets[0])
        else:
            fp8r_main(r_sets[0])
        last = None
        for c in plan:
            last = block(c)
        return last

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    _STREAM.append(torch.cuda.Stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=_STREAM[0], capture_error_mode="thread_local"):
        step()
    torch.cuda.synchronize()
    log(f"pair: branch={HAS_BRANCH} L2DISCARD={getattr(qr8, 'L2DISCARD', None)} "
        f"R={getattr(f8r, 'FP8R_QKV_FP8_DIRECT', None)}; soak {SOAK_S}s")
    t0 = time.time()
    while time.time() - t0 < SOAK_S:
        g.replay()
        torch.cuda.synchronize()
    times = []
    for i in range(samples):
        torch.cuda.synchronize()
        a, b = sample(g)
        b.synchronize()
        times.append(a.elapsed_time(b) / REPLAYS)
        if i % 30 == 0:
            log(f"  sample {i}: {times[-1]:.4f} ms")
    st = stat(times)
    unit = ("one graph replay = one denoise step's image.0 FP8R call (QKV GEMM .. FMHA) + 32 "
            "NVFP4 blocks' " + ("QKV op + nfo FMHA||gate_up fork" if qkvfork else
                                "QKV op .. nfo fork .. out pair .. residual"))
    return {"batch": {"1": {"step_ms": st["median_ms"], "noise_floor": st["noise"]}},
            "unit": unit,
            "branch_tree": HAS_BRANCH, "p10_ms": st["p10"], "p90_ms": st["p90"],
            "samples": samples, "output": {"class": "exact"}}


if __name__ == "__main__":
    torch.manual_seed(0)
    mode = sys.argv[1]
    t0 = time.time()
    out = {"d": step0_d, "r": step0_r, "both": step2_both, "pair": pair}[mode]()
    out["mode"] = mode
    out["rounds"], out["replays"] = ROUNDS, REPLAYS
    out["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(out))
