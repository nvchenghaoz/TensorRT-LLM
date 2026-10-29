"""STEP 0 probe (idea 276): torch._scaled_mm e4m3 at m=16384 n=512 k=16384 on SM120.

mat1 = P e4m3 [T,T] row-major, mat2 = V^T column-major ([c,T] row-major .t()).
Modes: (i) per-tensor scales bf16 out, (ii) per-tensor fp32 out, (iii) rowwise [M,1]x[1,N] bf16 out.
Reference timing: main's bf16 bmm2 o = bmm(v[1,c,T], p[1,T,T]^T).
"""
import json
import subprocess
import sys

import torch

T, C = 16384, 512
FP8_FLOOR = 0.340
MAIN_BMM2 = 0.717


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:  # noqa
        return str(e)


def ancestry():
    print(sh("git log --oneline -3"), flush=True)
    for x in ("da43d58f", "59981b4e", "b1758d79"):
        rc = subprocess.run(f"git merge-base --is-ancestor {x} HEAD", shell=True).returncode
        print(f"ancestor {x} rc={rc}", flush=True)


def time_fn(fn, warm=10, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    n = len(ts)
    return ts[n // 2], ts[n // 10], ts[(9 * n) // 10]


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            names.append(ev.name)
    return sorted(set(names))


def main():
    ancestry()
    torch.manual_seed(0)
    dev = "cuda"
    # realistic-ish P: softmax rows of random logits, unnormalized E in [0,1]
    s = torch.randn(T, T, device=dev, dtype=torch.float32) * 3.0
    e = torch.exp(s - s.amax(-1, keepdim=True))
    l = e.sum(-1, keepdim=True)
    p_bf16 = (e / l).to(torch.bfloat16)
    qe = (e * 448.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    del s, e
    v = torch.randn(C, T, device=dev, dtype=torch.bfloat16)  # [c,T] row-major
    amax = v.float().abs().amax(-1)  # [c]
    sb = (amax / 448.0).clamp(min=1e-12)
    vq = (v.float() / sb[:, None]).to(torch.float8_e4m3fn)  # [c,T]
    mat2 = vq.t()  # [T,c] column-major
    one = torch.ones((), device=dev, dtype=torch.float32)
    sa_row = (1.0 / (448.0 * l)).float().contiguous()  # [T,1]
    sb_row = sb[None, :].float().contiguous()  # [1,c]
    pristine = [t.clone() for t in (qe, vq, sa_row, sb_row)]

    # reference fp32 O^T = P V^T
    ref = (p_bf16.float() @ v.float().t())
    rows = {}

    def base():
        return torch.bmm(v[None], p_bf16[None].transpose(1, 2))

    modes = {
        "bf16_bmm2_main": base,
        "i_pertensor_bf16": lambda: torch._scaled_mm(qe, mat2, one, one, out_dtype=torch.bfloat16, use_fast_accum=False),
        "ii_pertensor_fp32": lambda: torch._scaled_mm(qe, mat2, one, one, out_dtype=torch.float32, use_fast_accum=False),
        "iii_rowwise_bf16": lambda: torch._scaled_mm(qe, mat2, sa_row, sb_row, out_dtype=torch.bfloat16, use_fast_accum=False),
        "iii_rowwise_fp32": lambda: torch._scaled_mm(qe, mat2, sa_row, sb_row, out_dtype=torch.float32, use_fast_accum=False),
        "iii_rowwise_bf16_fastacc": lambda: torch._scaled_mm(qe, mat2, sa_row, sb_row, out_dtype=torch.bfloat16, use_fast_accum=True),
    }
    for name, fn in modes.items():
        print(f"--- {name}", flush=True)
        row = {}
        try:
            out = fn()
            torch.cuda.synchronize()
            row["works"] = True
            row["kernels"] = kernel_names(fn)
            med, p10, p90 = time_fn(fn)
            row.update(median_ms=round(med, 4), p10_ms=round(p10, 4), p90_ms=round(p90, 4),
                       pct_of_fp8_floor=round(100 * FP8_FLOOR / med, 1), ratio_to_main_bmm2=round(med / MAIN_BMM2, 3))
            # check inputs pristine, then 5x bitwise
            ok_in = all(torch.equal(a, b) for a, b in zip((qe, vq, sa_row, sb_row), pristine))
            o0 = fn().clone()
            bit = all(torch.equal(fn(), o0) for _ in range(5))
            row["inputs_unchanged"] = ok_in
            row["bitwise_5x"] = bit
            o = o0.float()
            if name == "bf16_bmm2_main":
                o = o[0].t()
            elif name.startswith("i_") or name.startswith("ii_"):
                o = o * sa_row * sb_row
            err = (o - ref).abs()
            row["max_abs_vs_fp32"] = float(err.max())
            row["rel_l2_vs_fp32"] = float(err.norm() / ref.norm())
        except Exception as ex:  # noqa
            row["works"] = False
            row["error"] = repr(ex)[:500]
        print(json.dumps(row), flush=True)
        rows[name] = row

    # compile check for (iii) under decoder options
    try:
        import torch._inductor.config as icfg
        icfg.layout_optimization = False
        icfg.emulate_precision_casts = True
        from torch._inductor.utils import run_and_get_code

        def f(a, b, sa, sb):
            return torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)

        cf = torch.compile(f, dynamic=False)
        out, codes = run_and_get_code(cf, qe, vq, sa_row, sb_row)
        code = "\n".join(codes)
        rows["compile_iii"] = {
            "n_scaled_mm_extern": code.count("_scaled_mm"),
            "bitwise_vs_eager": bool(torch.equal(out, modes["iii_rowwise_bf16"]())),
            "snippet": [ln.strip() for ln in code.splitlines() if "scaled_mm" in ln][:6],
        }
    except Exception as ex:  # noqa
        rows["compile_iii"] = {"error": repr(ex)[:500]}
    print(json.dumps(rows["compile_iii"]), flush=True)

    b = rows["bf16_bmm2_main"]
    best = min((r["median_ms"] for k, r in rows.items() if k != "bf16_bmm2_main" and r.get("works") and "median_ms" in r), default=None)
    print(json.dumps({"passed": True, "rows": rows, "best_fp8_ms": best, "main_bmm2_ms": b.get("median_ms")}))


if __name__ == "__main__":
    main()
