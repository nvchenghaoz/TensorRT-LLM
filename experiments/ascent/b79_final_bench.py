"""Branch 79 (idea 123) microbench: the per-step final-layer SECTION at real shapes.

Runs the TREE's code: the tree's Flux3Pipeline.torch_compile on a holder transformer whose
final_layer ModuleDict holds a real Flux3LastLayer (BF16, unquantized Linear, D=3072 -> 128),
so the base tree leaves it eager and the branch compiles it. The token selection mirrors the
tree's final-layer loop: base = combined[:, T:T+S].contiguous() + torch.cat([part]); branch
(detected from the tree's Flux3Model.forward source) = part[0].unsqueeze(0), no copy.
mod is the step-batched [16,1,1,2D] row tensor indexed by step, as _image_step_stream gives.

Timing: CUDA graph of N=20 section calls, median of 60 replays (per-call us). Then T=160
calls to check that the compiled final layer keeps one Dynamo cache entry. Untimed repeat:
same warmup/N, each replay's output compared bitwise with iteration 0, inputs checked
unchanged before each reset. Reference: eager Flux3LastLayer on the cat copy (main's math)
and an fp32 torch reference of the same math.

Last stdout line: evidence JSON.
"""

import inspect
import json
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as P
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as T


def log(*a):
    print(*a, flush=True)


dev = "cuda"
bf = torch.bfloat16
D, S, COUT, STEPS = 3072, 4096, 128, 16
N, REPS = 20, 60
torch.manual_seed(0)

CAND = "parts[0][0].unsqueeze(0)" in inspect.getsource(T.Flux3Model.forward)
log("tree single-part view path:", CAND)

cfg = types.SimpleNamespace(
    torch_dtype=bf, quant_config=None, skip_create_weights_in_init=False,
    force_dynamic_quantization=False,
)
with torch.device(dev):
    layer = T.Flux3LastLayer(D, COUT, config=cfg, eps=1e-6)
layer = layer.to(dev)
with torch.no_grad():
    for p in layer.parameters():
        p.normal_(0, 0.02)
layer.eval()
eager = layer


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = nn.ModuleDict({"image": layer})


holder = Holder()
fake_pipe = types.SimpleNamespace(
    pipeline_config=types.SimpleNamespace(
        torch_compile=types.SimpleNamespace(enable_fullgraph=False)
    ),
    transformer=holder,
)
P.Flux3Pipeline.torch_compile(fake_pipe)
compiled = holder.final_layer["image"] is not layer
log("tree compiles final layer:", compiled, type(holder.final_layer["image"]).__name__)
final = holder.final_layer["image"]

combined = {t: torch.randn(1, t + S, D, device=dev, dtype=bf) for t in (80, 160)}
mods = torch.randn(STEPS, 1, 1, 2 * D, device=dev, dtype=bf) * 0.5
pristine = {t: v.clone() for t, v in combined.items()}
pristine_mods = mods.clone()


def section(t=80, step=3):
    part = combined[t][:, t : t + S].contiguous()
    if CAND and part.shape[0] == 1:
        tokens = part[0].unsqueeze(0)
    else:
        tokens = torch.cat([part], dim=1)
    return final(tokens, None, mod=mods[step])[0]


def cache_entries():
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        return len(_debug_get_cache_entry_list(T.Flux3LastLayer.forward.__code__))
    except Exception as e:  # noqa: BLE001
        return f"n/a {e!r}"[:80]


@torch.no_grad()
def main():
    # ---- reference: eager layer on the cat copy (main's math), and fp32 ----
    ref = eager(torch.cat([combined[80][:, 80 : 80 + S]], dim=1), None, mod=mods[3])[0]
    out = section()
    torch.cuda.synchronize()
    d = (out.float() - ref.float()).abs()
    max_abs = float(d.max())
    max_rel = float((d / ref.float().abs().clamp_min(1e-3)).max())
    frac = float((d > 0).float().mean())
    sh, sc = mods[3].float().chunk(2, dim=-1)
    xn = F.layer_norm(combined[80][:, 80 : 80 + S].float(), (D,), eps=1e-6)
    out32 = F.linear((1 + sc) * xn + sh, layer.linear.weight.float())[0]
    err_tree = float((out.float() - out32).abs().max())
    err_ref = float((ref.float() - out32).abs().max())
    mse = float(((out.float() - ref.float()) ** 2).mean())
    peak = float(ref.float().abs().max())
    psnr = float(10 * torch.log10(torch.tensor(peak * peak / max(mse, 1e-30))))
    bitwise = bool(torch.equal(out, ref))
    log(f"tree section vs eager-cat reference: bitwise={bitwise} max_abs={max_abs:.3e} "
        f"max_rel={max_rel:.3e} frac_diff={frac:.3e} psnr={psnr:.1f} dB; vs fp32: "
        f"tree={err_tree:.3e} eager={err_ref:.3e}")

    # ---- T=160 then T=80: compiled layer must not recompile ----
    ce0 = cache_entries()
    section(160)
    section(80)
    torch.cuda.synchronize()
    ce1 = cache_entries()
    log(f"Flux3LastLayer.forward cache entries after T=80: {ce0}, after T=160/T=80: {ce1}")

    # ---- timing ----
    for _ in range(3):
        section()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            section()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        outs = [section() for _ in range(N)]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for i in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / N)
        if i % 20 == 0:
            log(f"rep {i} {times[-1] * 1e3:.1f} us/call")
    st = sorted(times)
    med = st[len(st) // 2]
    p10, p90 = st[int(0.1 * REPS)], st[int(0.9 * REPS)]
    log(f"graph N={N} median {med * 1e3:.2f} us/call p10 {p10 * 1e3:.2f} p90 {p90 * 1e3:.2f}")

    # ---- untimed repeat: same warmup (5) + REPS replays, reset check before each ----
    rep_ok = True
    first = None
    for _ in range(5):
        g.replay()
    for i in range(REPS):
        for t, v in pristine.items():
            assert torch.equal(combined[t], v), "combined mutated"
        assert torch.equal(mods, pristine_mods), "mods mutated"
        g.replay()
        torch.cuda.synchronize()
        cur = [o.clone() for o in outs]
        if first is None:
            first = cur
        rep_ok &= all(torch.equal(c, f) for c, f in zip(cur, first))
        rep_ok &= all(torch.equal(c, out) for c in cur)
    log("untimed repeat (graph replays bitwise vs iteration 0 and vs eager-mode call):", rep_ok)

    cls = "exact" if bitwise else "reordered"
    print(json.dumps({
        "batch": {"1": {"kernel_ms": med, "noise_floor": (p90 - p10) / med}},
        "output": {"class": cls, "vs_eager_cat_bitwise": bitwise, "max_abs": max_abs,
                   "max_rel": max_rel, "frac_diff": frac, "psnr_db": psnr,
                   "err_tree_vs_fp32": err_tree, "err_eager_vs_fp32": err_ref},
        "tree_view_path": CAND, "tree_compiles_final_layer": compiled,
        "cache_entries": [ce0, ce1], "graph_calls": N, "reps": REPS,
        "repeated_execution_passed": rep_ok, "passed": rep_ok,
    }))


if __name__ == "__main__":
    main()
    sys.exit(0)
