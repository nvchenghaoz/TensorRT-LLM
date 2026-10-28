"""Idea 91 microbench: image-stream _packed_stream + Flux3LastLayer at real shapes.

Runs the TREE's Flux3Model._packed_stream and Flux3LastLayer.forward (bound to
stand-in modules with the real shapes: S=4096 image tokens, D=3072, flow-map
ref row, uniform timesteps, one segment). Base expands vec to [1,S,D] and runs
SiLU + Linear 3072->6144 at M=4096; the candidate keeps vec [1,1,D].

Also decides the output class: is Linear(silu(v[1 row])) bitwise equal to row 0
of Linear(silu(v expanded to 4096 rows))?  And the full final-layer output of
the tree vs an explicit expanded-vec reference (main's math).

Last stdout line: evidence JSON.
"""

import functools
import json
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as T


def log(*a):
    print(*a, flush=True)


dev = "cuda"
bf = torch.bfloat16
D, S, CIN, COUT = 3072, 4096, 3072, 128
torch.manual_seed(0)


def lin(i, o, std=0.02):
    m = nn.Linear(i, o, bias=False, device=dev, dtype=bf)
    with torch.no_grad():
        m.weight.normal_(0, std)
    return m


class TimeIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer, self.out_layer = lin(256, D), lin(D, D)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = lin(D, 3 * D)

    def forward(self, vec):
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        return out.chunk(3, dim=-1)


class LastLayer(nn.Module):
    forward = T.Flux3LastLayer.forward

    def __init__(self):
        super().__init__()
        self.norm_final = LayerNorm(
            hidden_size=D, eps=1e-6, has_weights=False, has_bias=False, dtype=bf, device=dev
        )
        self.linear = lin(D, COUT)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), lin(D, 2 * D))


fake = types.SimpleNamespace()
fake.time_in = TimeIn()
fake.emb_in = {"image": nn.Identity()}
fake.txt_in = nn.Identity()
fake.early_stream_modulations = {"image": Mod()}
fake.single_stream_modulations = {"image": Mod()}
fake.sharder = types.SimpleNamespace(is_active=False)
fake._shard_packed = functools.partial(T.Flux3Model._shard_packed, fake)
final = LastLayer()

inputs = {
    "x_image": torch.randn(S, CIN, device=dev, dtype=bf),
    "x_image_timesteps": torch.full((S,), 0.6875, device=dev, dtype=torch.float32),
    "_flux3_uniform_timesteps": True,
}
ref = (torch.randn(1, D, device=dev, dtype=bf) * 0.5)
x_final = torch.randn(1, S, D, device=dev, dtype=bf)
pristine = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
pristine_ref, pristine_x = ref.clone(), x_final.clone()
lengths = (S,)


_final_vec = getattr(T.Flux3Model, "_final_layer_vec", None)  # absent on the base tree


def region():
    st = T.Flux3Model._packed_stream(fake, "image", inputs, lengths, ref)
    vec = st["vec"] if _final_vec is None else _final_vec(fake, st["vec"], inputs, lengths)
    return final(x_final, vec)[0], vec


@torch.no_grad()
def main():
    # ---- 1. output class: GEMM M=1 vs row 0 of M=4096 ----
    W = final.adaLN_modulation[1].weight
    v1 = torch.randn(1, 1, D, device=dev, dtype=bf)
    m1 = F.linear(F.silu(v1), W)
    vexp = torch.cat([v1.expand(-1, S, -1)], dim=1)
    mS = F.linear(F.silu(vexp), W)
    gemm_bitwise = bool(torch.equal(m1[0, 0], mS[0, 0]))
    rows_equal = bool(torch.equal(mS[0], mS[0, :1].expand(S, -1)))
    d = (m1[0, 0].float() - mS[0, 0].float()).abs()
    gemm_max_abs = float(d.max())
    gemm_max_rel = float((d / mS[0, 0].float().abs().clamp_min(1e-6)).max())
    log(f"gemm M=1 vs row0 of M=4096 bitwise={gemm_bitwise} max_abs={gemm_max_abs:.3e} "
        f"max_rel={gemm_max_rel:.3e} M4096 rows identical={rows_equal}")

    # ---- 2. tree output vs explicit expanded reference (main's math) ----
    out, vec = region()
    log("tree vec shape", tuple(vec.shape))
    vec_row = vec[:, :1]
    ref_out = final(x_final, torch.cat([vec_row.expand(-1, S, -1)], dim=1))[0]
    full_bitwise = bool(torch.equal(out, ref_out))
    fd = (out.float() - ref_out.float()).abs()
    full_max_abs = float(fd.max())
    full_frac = float((fd > 0).float().mean())
    # fp32 reference of the same math, for scale
    mod32 = F.linear(F.silu(vec_row.float()), final.adaLN_modulation[1].weight.float())
    sh32, sc32 = mod32.chunk(2, dim=-1)
    xn = F.layer_norm(x_final.float(), (D,), eps=1e-6)
    out32 = F.linear((1 + sc32) * xn + sh32, final.linear.weight.float())[0]
    err_tree_fp32 = float((out.float() - out32).abs().max())
    err_ref_fp32 = float((ref_out.float() - out32).abs().max())
    log(f"final out vs expanded ref: bitwise={full_bitwise} max_abs={full_max_abs:.3e} "
        f"frac_diff={full_frac:.3e}; vs fp32: tree={err_tree_fp32:.3e} expanded={err_ref_fp32:.3e}")

    # ---- 3. timing: CUDA graph of N region calls, median of reps ----
    N, REPS = 10, 80
    for _ in range(3):
        region()
    torch.cuda.synchronize()
    mode = "graph"
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                region()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(N):
                region()

        def run():
            g.replay()
    except Exception as e:  # noqa: BLE001
        log("graph capture failed, eager timing:", repr(e)[:200])
        mode = "eager"

        def run():
            for _ in range(N):
                region()

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    times = []
    for i in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / N)
        if i % 20 == 0:
            log(f"rep {i} {times[-1]*1e3:.1f} us/call")
    times.sort()
    med = times[len(times) // 2]
    p10, p90 = times[int(0.1 * REPS)], times[int(0.9 * REPS)]
    log(f"mode={mode} median {med*1e3:.1f} us/call p10 {p10*1e3:.1f} p90 {p90*1e3:.1f}")

    # ---- 4. untimed repeated-execution check: same N per graph, reset per rep ----
    rep_ok = True
    out0 = None
    for i in range(5):
        for k, v in pristine.items():
            if torch.is_tensor(v):
                assert torch.equal(inputs[k], v), f"input {k} mutated"
        assert torch.equal(ref, pristine_ref) and torch.equal(x_final, pristine_x)
        for _ in range(N):
            o, _ = region()
        if out0 is None:
            out0 = o.clone()
        rep_ok &= bool(torch.equal(o, out0)) and bool(torch.equal(o, out))
    log("repeated execution bitwise:", rep_ok)

    cls = "exact" if (gemm_bitwise and full_bitwise) else "reordered"
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
        "output": {"class": cls, "gemm_m1_vs_row0_bitwise": gemm_bitwise,
                   "gemm_max_abs": gemm_max_abs, "gemm_max_rel": gemm_max_rel,
                   "final_vs_expanded_bitwise": full_bitwise, "final_max_abs": full_max_abs,
                   "final_frac_diff": full_frac, "err_tree_vs_fp32": err_tree_fp32,
                   "err_expanded_vs_fp32": err_ref_fp32},
        "vec_shape": list(vec.shape), "timing_mode": mode,
        "repeated_execution_passed": rep_ok, "passed": rep_ok,
    }))


if __name__ == "__main__":
    main()
    sys.exit(0)
