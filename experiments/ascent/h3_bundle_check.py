"""Correctness check for the H3 elementwise bundle (candidate tree only; untimed).

References:
  main_boundary = Inductor-compiled ``x + mod_prev[idx, gate_mlp] * ff`` (the block's poi kernel) followed
                  by main's _minimax_h3_adaln_kernel (vendored verbatim below) as norm_mod.
  main_norm2    = main's kernel in gate_res mode (the in-block norm2), vs the edited kernel.
  main_tail     = compiled residual + eager norm_out (flashinfer RMSNorm, 1+scale, mul, add) + .to(fp32).
  fp32_ref      = independent fp32 torch math (tolerance information only).
New outputs must be bitwise equal to the main references. A repeated-execution sequence (same warmup
and iteration count as the timing script) compares every iteration bitwise and checks that read-only
inputs are unchanged. Also runs a small odd token count (M=1001) and batch 2.
Last stdout line: {"passed": ...}.
"""

import json
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "experiments/ascent")
import h3_bundle_bench as bb  # noqa: E402

from tensorrt_llm._torch.modules.rms_norm import RMSNorm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln as fa  # noqa: E402

_CH = 1024


# ---- main (e64bf1e4) kernel, vendored verbatim ----
@triton.jit
def _ld(ptr, off, D, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    return tl.load(ptr + c, mask=c < D, other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, off, D, val, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    tl.store(ptr + c, val.to(tl.bfloat16), mask=c < D)


@triton.jit
def _main_adaln_kernel(
    x_ptr, a_ptr, x1_ptr, m_ptr, w_ptr, mod_ptr, idx_ptr, D, x_stride, a_stride, x1_stride,
    m_stride, mod_stride, gate_off, scale_off, shift_off, eps, SEQ,
    HAS_RES: tl.constexpr, CH: tl.constexpr,
):
    row = tl.program_id(0)
    r64 = row.to(tl.int64)
    mrow = tl.load(idx_ptr + row % SEQ).to(tl.int64)
    mb = mod_ptr + mrow * mod_stride
    xr = x_ptr + r64 * x_stride
    x0 = _ld(xr, 0 * CH, D, CH)
    x1 = _ld(xr, 1 * CH, D, CH)
    x2 = _ld(xr, 2 * CH, D, CH)
    x3 = _ld(xr, 3 * CH, D, CH)
    x4 = _ld(xr, 4 * CH, D, CH)
    x5 = _ld(xr, 5 * CH, D, CH)
    if HAS_RES:
        ar = a_ptr + r64 * a_stride
        gb = mb + gate_off
        x0 = (x0 + _ld(gb, 0 * CH, D, CH) * _ld(ar, 0 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x1 = (x1 + _ld(gb, 1 * CH, D, CH) * _ld(ar, 1 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 + _ld(gb, 2 * CH, D, CH) * _ld(ar, 2 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x3 = (x3 + _ld(gb, 3 * CH, D, CH) * _ld(ar, 3 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x4 = (x4 + _ld(gb, 4 * CH, D, CH) * _ld(ar, 4 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x5 = (x5 + _ld(gb, 5 * CH, D, CH) * _ld(ar, 5 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        o1 = x1_ptr + r64 * x1_stride
        _st(o1, 0 * CH, D, x0, CH)
        _st(o1, 1 * CH, D, x1, CH)
        _st(o1, 2 * CH, D, x2, CH)
        _st(o1, 3 * CH, D, x3, CH)
        _st(o1, 4 * CH, D, x4, CH)
        _st(o1, 5 * CH, D, x5, CH)
    ss = (
        tl.sum(x0 * x0, 0) + tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)
        + tl.sum(x3 * x3, 0) + tl.sum(x4 * x4, 0) + tl.sum(x5 * x5, 0)
    )
    r = tl.math.rsqrt(ss / D + eps)
    om = m_ptr + r64 * m_stride
    sb = mb + scale_off
    hb = mb + shift_off
    for i in tl.static_range(6):
        if i == 0:
            xi = x0
        elif i == 1:
            xi = x1
        elif i == 2:
            xi = x2
        elif i == 3:
            xi = x3
        elif i == 4:
            xi = x4
        else:
            xi = x5
        n = (xi * r * _ld(w_ptr, i * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        _st(om, i * CH, D, n * (1.0 + _ld(sb, i * CH, D, CH)) + _ld(hb, i * CH, D, CH), CH)


def main_launch(x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
    D = x.shape[-1]
    seq_len = x.shape[-2]
    x2 = x.reshape(-1, D)
    rows = x2.shape[0]
    m = torch.empty((rows, D), dtype=x.dtype, device=x.device)
    has_res = a is not None
    a2 = a.reshape(-1, D) if has_res else x2
    x1 = torch.empty((rows, D), dtype=x.dtype, device=x.device) if has_res else m
    _main_adaln_kernel[(rows,)](
        x2, a2, x1, m, w, mod, idx, D, x2.stride(0), a2.stride(0), x1.stride(0), m.stride(0),
        mod.stride(0), gate_col * D, scale_col * D, shift_col * D, eps, seq_len,
        HAS_RES=has_res, CH=_CH, num_warps=4,
    )
    return x1.view(x.shape), m.view(x.shape)


def neq(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum()) if a.dtype == torch.bfloat16 else int(
        (a.view(torch.int32) != b.view(torch.int32)).sum()
    )


def run_case(c, m, batch, report, tag, repeat):
    D = c["hidden_size"]
    eps = c["norm_eps"]
    t = bb.make_inputs(c, m=m, seed=7 + m)
    if batch > 1:
        t["x"] = t["x"].repeat(batch, 1, 1) + torch.randn_like(t["x"].repeat(batch, 1, 1)).to(torch.bfloat16)
        t["ff"] = t["ff"].repeat(batch, 1, 1)
    pristine = {k: v.clone() for k, v in t.items()}
    res = bb.compiled_residual()
    norm = RMSNorm(hidden_size=D, eps=c["final_norm_eps"], dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        norm.weight.copy_(t["wout"])

    # --- main references ---
    x1_ref = res(t["x"], t["mod_prev"], t["idx"], t["ff"])
    _, m_ref = main_launch(x1_ref, None, t["w1"], t["mod"], t["idx"], eps, 0, 1, 0)
    x1b_ref, mb_ref = main_launch(t["x"], t["ff"], t["w1"], t["mod"], t["idx"], eps, 2, 4, 3)
    shift, scale = t["out_table"].chunk(2, dim=-1)
    n_ref = norm(x1_ref.reshape(-1, D)).view(x1_ref.shape)
    tail_ref = (n_ref * (1.0 + scale.index_select(0, t["tidx"])) + shift.index_select(0, t["tidx"])).to(torch.float32)

    # --- independent fp32 reference (information) ---
    gate = t["mod_prev"][:, 5 * D : 6 * D].float()[t["idx"]]
    x1_32 = t["x"].float() + gate * t["ff"].float()
    x1_bf = x1_32.to(torch.bfloat16).float()
    rr = torch.rsqrt(x1_bf.pow(2).mean(-1, keepdim=True) + eps)
    n32 = (x1_bf * rr * t["w1"].float()).to(torch.bfloat16).float()
    m32 = n32 * (1 + t["mod"][:, D : 2 * D].float()[t["idx"]]) + t["mod"][:, 0:D].float()[t["idx"]]

    def new_all():
        x1n, mn = fa.h3_prev_gate_res_norm_mod(t["x"], t["ff"], t["w1"], t["mod_prev"], t["mod"], t["idx"], eps, 5, 1, 0)
        x1b, mb = fa.h3_gate_res_norm_mod(t["x"], t["ff"], t["w1"], t["mod"], t["idx"], eps, 2, 4, 3)
        xt = fa.h3_gate_res(t["x"], t["ff"], t["mod_prev"], t["idx"], 5)
        nt = norm(xt.reshape(-1, D)).view(xt.shape)
        tail = fa.h3_mod_out(nt, t["out_table"], t["tidx"], 1, 0, torch.float32)
        return {"x1": x1n, "m": mn, "norm2_x1": x1b, "norm2_m": mb, "tail_x": xt, "tail": tail}

    refs = {"x1": x1_ref, "m": m_ref, "norm2_x1": x1b_ref, "norm2_m": mb_ref, "tail_x": x1_ref, "tail": tail_ref}
    ok = True
    first = new_all()
    torch.cuda.synchronize()
    mism = {k: neq(first[k], refs[k]) for k in refs}
    ok &= all(v == 0 for v in mism.values())
    info = {
        "mismatch_vs_main": mism,
        "maxabs_x1_vs_fp32": float((first["x1"].float() - x1_32).abs().max()),
        "maxabs_main_x1_vs_fp32": float((x1_ref.float() - x1_32).abs().max()),
        "maxabs_m_vs_fp32": float((first["m"].float() - m32).abs().max()),
        "maxabs_main_m_vs_fp32": float((m_ref.float() - m32).abs().max()),
    }
    # repeated execution: same warmup + iteration count as the timing script, inputs checked each iter
    bad_iters = []
    if repeat:
        for it in range(bb.WARM + bb.ITERS):
            out = new_all()
            for k in refs:
                if neq(out[k], refs[k]):
                    bad_iters.append((it, k))
            for k, v in pristine.items():
                if not torch.equal(t[k], v):
                    bad_iters.append((it, "input_changed:" + k))
                    t[k].copy_(v)
    ok &= not bad_iters
    info["repeat_bad"] = bad_iters[:10]
    info["repeat_iters"] = (bb.WARM + bb.ITERS) if repeat else 0
    report[tag] = info
    print(tag, json.dumps(info), flush=True)
    return ok


def qk_case(c, m, report, tag, repeat):
    """Default launch (SM120: 4 tokens x 1 head x 2 warps) vs the former launch (1 token x auto heads
    x 1 warp, which the new code keeps on every other GPU), bitwise; tail-masked when m % 4 != 0."""
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_rope as fr

    H = c["num_attention_heads"] // 2
    hd = c["attention_head_dim"]
    eps = c["qk_norm_eps"]
    qk, tens = bb.make_qk(c, m=m)
    pristine = {k: v.clone() for k, v in tens.items()}
    q_ref = torch.empty(1, m, H * hd, device="cuda", dtype=torch.bfloat16)
    k_ref = torch.empty_like(q_ref)
    fr.launch_minimax_h3_qk_norm_rope(
        tens["qkv"], q_ref, k_ref, tens["wq"], tens["wk"], tens["cos"], tens["sin"], eps, H, hd,
        tokens_per_program=1, heads_per_program=None, num_warps=1,
    )
    q, k = qk()
    torch.cuda.synchronize()
    info = {"m": m, "tail_mask": m % 4 != 0, "mismatch": neq(q, q_ref) + neq(k, k_ref),
            "launch": list(fr._default_launch(q.device))}
    bad = []
    if repeat:
        for it in range(bb.WARM + bb.ITERS):
            q, k = qk()
            if neq(q, q_ref) or neq(k, k_ref):
                bad.append(it)
            for kk, v in pristine.items():
                if not torch.equal(tens[kk], v):
                    bad.append((it, "input_changed:" + kk))
                    tens[kk].copy_(v)
    info["repeat_bad"] = bad[:10]
    # SM100 keeps its launch: the selector returns the former default for any capability but (12, 0).
    real = torch.cuda.get_device_capability
    saved = dict(fr._launch_by_device)
    try:
        fr._launch_by_device.clear()
        torch.cuda.get_device_capability = lambda *a, **k: (10, 0)
        info["sm100_launch"] = list(fr._default_launch(q.device))
    finally:
        torch.cuda.get_device_capability = real
        fr._launch_by_device.clear()
        fr._launch_by_device.update(saved)
    ok = info["mismatch"] == 0 and not bad and info["sm100_launch"] == [1, None, 1]
    if torch.cuda.get_device_capability() == (12, 0):
        ok &= info["launch"] == [4, 1, 2]
    report[tag] = info
    print(tag, json.dumps(info), flush=True)
    return ok


def main():
    c = bb.cfg()
    report = {}
    ok = True
    with torch.inference_mode():
        ok &= run_case(c, bb.M, 1, report, f"M{bb.M}_b1", True)
        ok &= run_case(c, 1001, 1, report, "M1001_b1", True)
        ok &= run_case(c, 517, 2, report, "M517_b2", False)
        ok &= qk_case(c, bb.M, report, f"qk_M{bb.M}", True)
        ok &= qk_case(c, 1001, report, "qk_M1001", True)
    torch.cuda.empty_cache()
    # Model level (separate process): real MiniMaxH3Transformer3DModel with per-block torch.compile,
    # deferred path vs main's block path bitwise, no recompiles after the first forward, kernel counts.
    if bb.CKPT:
        import subprocess

        proc = subprocess.run(
            [sys.executable, "experiments/ascent/h3_model_trace.py", bb.CKPT, "check"],
            capture_output=True, text=True,
        )
        print(proc.stdout[-6000:], proc.stderr[-3000:] if proc.returncode else "", flush=True)
        try:
            model = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            model = {"passed": False, "returncode": proc.returncode}
        report["model_level"] = model
        ok &= bool(model.get("passed"))
    print(json.dumps({"passed": bool(ok), "compared": "new ops vs main kernel (vendored) + Inductor residual + eager tail, bitwise", "cases": report}))


if __name__ == "__main__":
    main()
