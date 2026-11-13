"""1-GPU unit check of the Stage 1 per-chunk residual kernels (idea 106), no model load.

Shapes: M = 37966 packed rows, D from <ckpt>/transformer/config.json, main's 4 row chunks
(_row_chunks(M, 4) = 9600 x 3 + 9166), mod table [6, 6D] bf16, adaln_indices in the packed
layout (256 text + 414 audio + 37296 video rows) and a random variant.

Compares, bitwise, per iteration over 5 warmup + 20 iterations (x_out NaN-poisoned before each):
  (a) h3_gate_res_norm_mod_into(x_out[:, s:e], ...) -> x1 rows and m  vs  main's h3_gate_res_norm_mod;
  (b) h3_gate_res_into(x_out[:, s:e], x_out[:, s:e], ff_i, ...) in place  vs  main's compiled tail
      torch.cat(residual_chunks) + gate_mlp.index_select(0, idx) * torch.cat(ff_chunks)
      (torch.compile, the code main's block runs after the graph break);
inputs read-only. Also times main's compiled tail vs the 4 per-chunk launches (CUDA events).
Last stdout line: {"passed": ...}.
"""

import json
import os
import statistics
import sys

import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import (
    h3_gate_res_norm_mod,
    h3_norm_mod,
)

try:  # the branch; main has no row-view launchers (then only the digests run)
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import (
        h3_gate_res_into,
        h3_gate_res_norm_mod_into,
    )
except ImportError:
    h3_gate_res_into = h3_gate_res_norm_mod_into = None

M = 37966


def _digest(t):
    import hashlib

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def log(*a):
    print(*a, flush=True)


def main():
    ckpt = sys.argv[1]
    with open(os.path.join(ckpt, "transformer", "config.json")) as f:
        cfg = json.load(f)
    D = cfg["hidden_size"]
    eps = float(cfg.get("norm_eps", 1e-6))
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    bounds = h3._row_chunks(M, 4)
    log(f"D={D} M={M} bounds={bounds}")
    hs = torch.randn(1, M, D, device=dev, dtype=torch.bfloat16, generator=g)
    attn = [torch.randn(e - s, D, device=dev, dtype=torch.bfloat16, generator=g) for s, e in bounds]
    ff = [torch.randn(e - s, D, device=dev, dtype=torch.bfloat16, generator=g) for s, e in bounds]
    mod = (0.3 * torch.randn(6, 6 * D, device=dev, generator=g)).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    idx_layout = torch.cat([
        torch.full((256,), 0), torch.full((414,), 4), torch.full((M - 670,), 2)
    ]).to(dev)
    idx_rand = torch.randint(0, 6, (M,), device=dev, generator=g)
    inputs = [hs, mod, w, idx_layout, idx_rand] + attn + ff
    pristine = [t.clone() for t in inputs]

    def main_tail(r0, r1, r2, r3, f0, f1, f2, f3, mod, idx):
        gate_mlp = mod[:, 5 * D: 6 * D]
        return torch.cat([r0, r1, r2, r3], dim=1) + gate_mlp.index_select(0, idx) * torch.cat(
            [f0, f1, f2, f3], dim=1)

    tail = torch.compile(main_tail, mode="default", dynamic=None)

    def run_main(idx):
        res, ms = [], []
        for i, (s, e) in enumerate(bounds):
            r, m = h3_gate_res_norm_mod(hs[:, s:e], attn[i].view(1, e - s, D), w, mod, idx[s:e], eps, 2, 4, 3)
            res.append(r)
            ms.append(m)
        out = tail(*res, *[f.view(1, -1, D) for f in ff], mod, idx)
        return res, ms, out

    def run_stage1(idx, x_out, programs=0):
        ms = []
        for i, (s, e) in enumerate(bounds):
            ms.append(h3_gate_res_norm_mod_into(x_out[:, s:e], hs[:, s:e], attn[i], w, mod, idx[s:e], eps, 2, 4, 3))
        x1 = x_out.clone()
        for i, (s, e) in enumerate(bounds):
            rows = x_out[:, s:e]
            h3_gate_res_into(rows, rows, ff[i], mod, idx[s:e], 5, max_programs=programs if i < 3 else 0)
        return x1, ms, x_out

    result = {"bounds": bounds, "D": D}
    # Digests of main's public ops (and main's compiled tail) on these inputs: equal between the
    # baseline (main) and candidate arms of a pair proves the kernel refactor is bitwise vs main.
    dig = {}
    for name, idx in (("layout_idx", idx_layout), ("random_idx", idx_rand)):
        res_ref, m_ref, out_ref = run_main(idx)
        dig[name] = {
            "x1": _digest(torch.cat(res_ref, dim=1)),
            "m": _digest(torch.cat([m.view(-1, D) for m in m_ref], 0)),
            "tail": _digest(out_ref),
            "norm_mod": _digest(h3_norm_mod(hs, w, mod, idx, eps, 1, 0)),
        }
    result["main_op_digests"] = dig
    log(f"digests: {dig}")
    if h3_gate_res_into is None:
        result["passed"] = True
        result["compared"] = "main tree: digests only"
        print(json.dumps(result), flush=True)
        return
    ok = True
    for name, idx, programs in (("layout_idx", idx_layout, 0), ("random_idx", idx_rand, 0),
                                ("layout_idx_g64", idx_layout, 64), ("random_idx_g24", idx_rand, 24),
                                ("layout_idx_g188", idx_layout, 188)):
        res_ref, m_ref, out_ref = run_main(idx)
        x1_ref = torch.cat(res_ref, dim=1)
        # fp32 eager formula for information: x + g*a with one rounding
        gate = mod[:, 5 * D: 6 * D].index_select(0, idx).float()
        one_round = (x1_ref.float() + gate * torch.cat(ff, 0).view(1, M, D).float()).to(torch.bfloat16)
        info = {"inductor_tail_eq_fp32_one_rounding": bool(torch.equal(one_round, out_ref))}
        bad = []
        for it in range(5 + 20):
            x_out = torch.full((1, M, D), float("nan"), device=dev, dtype=torch.bfloat16)
            x1, ms, out = run_stage1(idx, x_out, programs)
            torch.cuda.synchronize()
            if not torch.equal(x1, x1_ref):
                bad.append(f"x1@{it}")
            if not all(torch.equal(a.view(-1, D), b.view(-1, D)) for a, b in zip(ms, m_ref)):
                bad.append(f"m@{it}")
            if not torch.equal(out, out_ref):
                d = (out.float() - out_ref.float()).abs()
                bad.append(f"out@{it}:maxabs={d.nan_to_num(1e9).max().item():.3g},n={(d != 0).sum().item()}")
            for t, p in zip(inputs, pristine):
                if not torch.equal(t, p):
                    bad.append(f"input_changed@{it}")
                    break
            # main's own path must also be repeatable
            _, _, out_m = run_main(idx)
            if not torch.equal(out_m, out_ref):
                bad.append(f"main_repeat@{it}")
        info["bitwise_all_iters"] = not bad
        info["bad"] = bad[:8]
        result[name] = info
        ok = ok and not bad
        log(f"{name}: {info}")

    # In-place + strided view guard: a non-row view must raise, not copy.
    try:
        h3_gate_res_into(torch.empty(D, 8, device=dev, dtype=torch.bfloat16).t(), hs[:, :8], ff[0][:8], mod,
                         idx_layout[:8], 5)
        result["non_row_view_raises"] = False
        ok = False
    except ValueError:
        result["non_row_view_raises"] = True

    # Timing (informational): main's compiled tail (cat + gated residual) vs 4 per-chunk launches.
    res_ref, _, _ = run_main(idx_layout)
    x_out = torch.cat(res_ref, dim=1).contiguous()
    ffv = [f.view(1, -1, D) for f in ff]

    def t_main():
        tail(*res_ref, *ffv, mod, idx_layout)

    def t_stage1():
        for i, (s, e) in enumerate(bounds):
            rows = x_out[:, s:e]
            h3_gate_res_into(rows, rows, ff[i], mod, idx_layout[s:e], 5)

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med = statistics.median(ts)
        return {"median_ms": round(med, 4), "noise": round((ts[26] - ts[3]) / med, 4)}

    result["time_main_tail"] = bench(t_main)
    result["time_stage1_4_chunks"] = bench(t_stage1)
    for p in (188, 64, 24):
        def t_chunk0(p=p):
            rows = x_out[:, 0:9600]
            h3_gate_res_into(rows, rows, ff[0], mod, idx_layout[0:9600], 5, max_programs=p)
        result[f"time_one_9600_chunk_g{p}"] = bench(t_chunk0)

    def t_chunk0_full():
        rows = x_out[:, 0:9600]
        h3_gate_res_into(rows, rows, ff[0], mod, idx_layout[0:9600], 5)
    result["time_one_9600_chunk_full"] = bench(t_chunk0_full)
    log(f"timing: main tail {result['time_main_tail']}, stage1 {result['time_stage1_4_chunks']}")
    result["passed"] = ok
    result["compared"] = ("Stage 1 eager row-view kernels vs main's h3_gate_res_norm_mod and main's "
                          "torch.compile'd cat + gate*cat tail, bitwise, 25 iterations, 2 index layouts")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
