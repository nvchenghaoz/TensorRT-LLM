"""Idea 42: last single block (single_blocks[27]) computes only the rows the final layer reads.

Unit = block 27's tail as served on main 8f606f93 (B=1, NVFP4 tail-pair block, side gate_up mode 3,
FMHA NVFP4-out, 4-warp nfo kernel), synthetic tensors at the served shapes, no model load:
  attention+gate_up : trtllm::flux3_sm120_fp8_prequant_attention_nvfp4_out_side_gate_up
                      (FMHA over qkv8 [3, S, 3072] E4M3 || packed-SwiGLU gate_up on a side stream)
  pair              : trtllm::flux3_sm120_nvfp4_out_pair_prequant (down K=9216 || attn_out K=3072)
  residual          : residual + gate * (attn_y + mlp_y) (torch.compile, as Inductor fuses it)
Base tree (main): all S rows. Branch (STEP 1, as primsts.qkv_fp8_attention_forward(keep_rows=n)):
FMHA still on all S rows, the side gate_up on the first n = 16384 rows (the final layer's image
rows), the attention output's first n rows into the pair at M=n, residual at n.
S = 16464 (t2i-2048) and 32848 (i2i-2048), n = 16384 in both.

  bench : run_gpu_pair unit; per case one CUDA graph of the unit, WARM replays, ITERS timed
          replays (CUDA events), median ms and (p90 - p10) / median.
  check : untimed. Per S (sequence t2i, i2i, t2i, i2i): the cut unit's every output bitwise vs
          rows [0, n) of the full unit's (fp4, sf row blocks, packed, packed sf, pair outputs,
          residual), WARM+ITERS eager calls vs the reference and call 0 with outputs poisoned,
          CUDA-graph replay x3 vs the reference, inputs unchanged.
"""

import json
import math
import os
import statistics
import sys
import time

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

HID, FF, HEADS, HD = 3072, 9216, 24, 128
N_KEEP = 16384
CASES = {"t2i-2048": 16464, "i2i-2048": 32848}
WARM = int(os.environ.get("WARM", "5"))
ITERS = int(os.environ.get("ITERS", "30"))
MODE = 3


def log(*a):
    print(f"[idea42 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _gs(t):
    return float(t.abs().amax().float() / (448.0 * 6.0))


def _inv(s):
    return torch.tensor([1.0 / s], device="cuda", dtype=torch.float32)


def has_rows():
    """True in the branch tree (qkv_fp8_attention_forward takes keep_rows)."""
    import inspect

    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

    return "keep_rows" in inspect.signature(primsts.qkv_fp8_attention_forward).parameters


def make_weights(seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w_up = torch.randn(2 * FF, HID, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_down = torch.randn(HID, FF, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_out = torch.randn(HID, HID, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    s_x, s_h, s_a = 4.0 / (448 * 6), 2.0 / (448 * 6), 3.0 / (448 * 6)
    ups = []
    for w in (w_up, w_down, w_out):
        sw = _gs(w)
        q, sf = fp.nvfp4_quantize(w, _inv(sw))
        ups.append((q, sf, sw))
    up = PreparedNVFP4(ups[0][0], ups[0][1], s_x, ups[0][2], swiglu=True)
    down = PreparedNVFP4(ups[1][0], ups[1][1], s_h, ups[1][2])
    out = PreparedNVFP4(ups[2][0], ups[2][1], s_a, ups[2][2])
    return up, down, out


def make_case(S, w, seed):
    up, down, out = w
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, HID, device="cuda", dtype=torch.bfloat16, generator=g)
    up_data, up_sf = fp.nvfp4_quantize(x, _inv(up.input_global_scale))
    qkv = torch.randn((3, 1, S, HEADS, HD), device="cuda", dtype=torch.float32, generator=g) * 2.0
    q, k, v = qkv.to(torch.bfloat16).unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(HD))
    qkv8 = torch.stack([t.reshape(S, HID) for t in (q8, k8, v8)])
    residual = torch.randn(1, S, HID, device="cuda", dtype=torch.bfloat16, generator=g)
    gate = torch.randn(1, S, HID, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    del qkv, q, k, v, q8, k8, v8
    return dict(S=S, x=x, up_data=up_data, up_sf=up_sf, qkv8=qkv8, scales=scales,
                residual=residual, gate=gate)


def inputs(c):
    return [c[k] for k in ("x", "up_data", "up_sf", "qkv8", "scales", "residual", "gate")]


@torch.compile(dynamic=False, options={"emulate_precision_casts": True})
def _residual(residual, gate, attn_y, mlp_y):
    n = attn_y.shape[0]
    output = attn_y.reshape(1, n, -1) + mlp_y.reshape(1, n, -1)
    return residual + gate * output


def unit(c, w, cut):
    """Block 27's tail; cut=True: the first N_KEEP rows only (branch op)."""
    up, down, out = w
    if cut:
        # As primsts.qkv_fp8_attention_forward(keep_rows=n): FMHA on all S rows, gate_up on the
        # first n rows, the attention output's first n rows into the pair.
        n = N_KEEP
        sf_cols = (HID // 16 + 3) // 4 * 4
        fp4, sf, packed, psf = fp.fp8_prequant_attention_nvfp4_out_side_gate_up(
            c["qkv8"], c["scales"], 1, HEADS, c["up_data"][:n], up.weight,
            c["up_sf"][: n * sf_cols], up.scale, down.inverse_scale, c["x"][:n], up.weight,
            MODE, out.weight,
        )
        fp4, sf = fp4[:, :n], sf[: n * sf_cols]
        residual, gate = c["residual"][:, :n], c["gate"][:, :n]
    else:
        fp4, sf, packed, psf = fp.fp8_prequant_attention_nvfp4_out_side_gate_up(
            c["qkv8"], c["scales"], 1, HEADS, c["up_data"], up.weight, c["up_sf"], up.scale,
            down.inverse_scale, c["x"], up.weight, MODE, out.weight,
        )
        residual, gate = c["residual"], c["gate"]
    attn_y, mlp_y = fp.nvfp4_out_pair_prequant(
        packed, down.weight, psf, down.scale, fp4.reshape(-1, fp4.shape[-1]), sf, out.weight,
        out.scale,
    )
    res = _residual(residual, gate, attn_y, mlp_y)
    return dict(fp4=fp4, sf=sf, packed=packed, psf=psf, attn_y=attn_y, mlp_y=mlp_y, res=res)


def capture(fn, stream):
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = fn()
    torch.cuda.synchronize()
    return graph, outs


def bench():
    cut = has_rows() and os.environ.get("TRTLLM_FLUX3_LAST_BLOCK_ROWS", "1") != "0"
    log(f"tree: {'branch' if has_rows() else 'base'}; cut={cut}; warm={WARM} iters={ITERS}")
    fp.sm120_linear_preload(torch.device("cuda", torch.cuda.current_device()))
    w = make_weights()
    stream = torch.cuda.Stream()
    res = {}
    for label, S in CASES.items():
        c = make_case(S, w, 1000 + S)
        graph, _ = capture(lambda: unit(c, w, cut), stream)
        for _ in range(WARM):
            graph.replay()
        torch.cuda.synchronize()
        ts = []
        for i in range(ITERS):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(stream)
            with torch.cuda.stream(stream):
                graph.replay()
            b.record(stream)
            b.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        med = statistics.median(ts)
        p10, p90 = ts[len(ts) // 10], ts[(len(ts) * 9) // 10]
        res[label] = {"kernel_ms": round(med, 4), "noise_floor": round((p90 - p10) / med, 4),
                      "p10_ms": round(p10, 4), "p90_ms": round(p90, 4), "S": S,
                      "rows": N_KEEP if cut else S}
        log(f"{label} S={S}: median {med:.3f} ms  p10 {p10:.3f}  p90 {p90:.3f}")
        del graph, c
        torch.cuda.empty_cache()
    print(json.dumps({"batch": res, "output": {"class": "exact"}, "cut": cut,
                      "unit": "block-27 tail: FMHA nvfp4_out || side gate_up, LPT pair, residual"}),
          flush=True)


def _eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8) if
                                                                     a.dtype != torch.uint8 else a,
                                                                     b.view(torch.uint8) if
                                                                     b.dtype != torch.uint8 else b)


def prefix(name, t, n):
    """Rows [0, n) of a full-unit output, as the cut unit lays it out."""
    if name == "fp4":
        return t[:, :n]
    if name == "sf":
        return t[: n * ((HID // 16 + 3) // 4 * 4)]
    if name == "psf":
        return t[: n * ((FF // 16 + 3) // 4 * 4)]
    if name == "res":
        return t[:, :n]
    return t[:n]


def check():
    log(f"tree: {'branch' if has_rows() else 'base'}")
    fp.sm120_linear_preload(torch.device("cuda", torch.cuda.current_device()))
    w = make_weights()
    stream = torch.cuda.Stream()
    n = N_KEEP
    report, ok = [], True
    cases = {S: make_case(S, w, 1000 + S) for S in CASES.values()}
    pristine = {S: [t.clone() for t in inputs(c)] for S, c in cases.items()}
    refs = {}
    # Full-unit references (main's op) and the cut outputs, per S.
    for S, c in cases.items():
        full = {k: v.clone() for k, v in unit(c, w, False).items()}
        torch.cuda.synchronize()
        refs[S] = {k: prefix(k, v, n).clone() for k, v in full.items()}
        del full
    sequence = [16464, 32848, 16464, 32848]
    for step, S in enumerate(sequence):
        c = cases[S]
        rec = {"step": step, "S": S}
        call0 = None
        bad_ref, bad_c0 = [], []
        for it in range(WARM + ITERS):
            outs = unit(c, w, True)
            torch.cuda.synchronize()
            for k, v in outs.items():
                if not _eq(v, refs[S][k]):
                    bad_ref.append((it, k))
            if call0 is None:
                call0 = {k: v.clone() for k, v in outs.items()}
            else:
                for k, v in outs.items():
                    if not _eq(v, call0[k]):
                        bad_c0.append((it, k))
            # Poison the outputs before the next call (they are fresh allocations, but the
            # caching allocator may hand the same blocks back).
            for v in outs.values():
                v.view(torch.uint8).fill_(0xA5) if v.dtype == torch.uint8 else v.fill_(float("nan"))
            del outs
        unchanged = all(torch.equal(a, b) for a, b in zip(inputs(c), pristine[S]))
        rec.update(eager_vs_full_prefix_bad=bad_ref[:6], eager_vs_call0_bad=bad_c0[:6],
                   inputs_unchanged=unchanged)
        # CUDA graph replay x3 vs eager.
        graph, gouts = capture(lambda: unit(c, w, True), stream)
        gbad = []
        for r in range(3):
            for v in gouts.values():
                v.view(torch.uint8).fill_(0x5A) if v.dtype == torch.uint8 else v.fill_(-1.0)
            with torch.cuda.stream(stream):
                graph.replay()
            torch.cuda.synchronize()
            for k, v in gouts.items():
                if not _eq(v, refs[S][k]):
                    gbad.append((r, k))
        del graph, gouts
        rec["graph_bad"] = gbad
        step_ok = not bad_ref and not bad_c0 and unchanged and not gbad
        rec["passed"] = step_ok
        ok &= step_ok
        log(json.dumps(rec))
        report.append(rec)
    print(json.dumps({"passed": bool(ok), "compared": (
        "cut unit (FMHA nvfp4_out on all S rows || gate_up at M=n, attention output rows [0,n) "
        "into the pair at M=n, residual at n) vs rows [0,n) of main's full unit (protected "
        "reference: main's ops at M=S), bitwise per output, t2i->i2i->t2i->i2i, WARM+ITERS eager "
        "repeats vs reference and call 0, CUDA-graph replays x3, inputs unchanged"),
        "n": n, "warm": WARM, "iters": ITERS, "steps": report}), flush=True)


if __name__ == "__main__":
    {"bench": bench, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "bench"]()
