"""Branch 71 (idea 121): fused FP8R head (LN + adaLN-modulate + per-token E4M3) vs main.

Main's head = per-block-compiled (dynamic=None, emulate_precision_casts) LN+modulate followed by
quantize_fp8r (warpPerTokenFp8QuantKernel). Shapes of one denoise step's FP8R heads:
  4 x single (M=4176, [1,2,D] shift/scale tables + [1,S,1] select)
  1 x mode   (M=4096, [1,1,D] rows that are views of a [16,1,1,3D] step table)
Modes:
  bench   : CUDA graph of 20 step-sequences, median of 60 replays; prints the evidence JSON
            (kernel_ms per step-sequence of 5 heads). Candidate (tree has the op): the served
            config, store_x_mod=False (the FP8R consumers read only the quantized data).
            Baseline (main): Inductor LN+modulate + quantize_fp8r.
  check   : untimed correctness, for store_x_mod True and False. Fused vs main's compiled head
            (static and dynamic-shape Inductor kernels) at M=4176/4256 single and M=4096 mode
            (several step rows, offset 0 included): bitwise E4M3 data + fp32 scales, plus x_mod
            when stored; repeat sequence (same warmup/iters as bench) bitwise vs main and vs
            iteration 0; inputs unchanged.
"""

import json
import sys

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_fp8r as fused_mod
except ImportError:
    fused_mod = None

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
D = 3072
dev = "cuda"
WARMUP, ITERS = 3, 20
torch.manual_seed(1234)
pre_norm = LayerNorm(
    hidden_size=D, eps=1e-6, has_weights=False, has_bias=False, dtype=torch.bfloat16
).to(dev)


class Single(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = pre_norm

    def forward(self, x, shift_t, scale_t, select):
        shift, scale = (torch.where(select, v[:, 1:2], v[:, 0:1]) for v in (shift_t, scale_t))
        return (1 + scale) * self.pre_norm(x) + shift


class Mode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = pre_norm

    def forward(self, x, shift, scale):
        return (1 + scale) * self.pre_norm(x) + shift


def compiled(module):
    return torch.compile(module, dynamic=None, options={"emulate_precision_casts": True})


single_c, mode_c = compiled(Single()), compiled(Mode())


def make_single(M):
    x = torch.randn(1, M, D, device=dev, dtype=torch.bfloat16) * 2.0
    x[..., 7] *= 40  # an outlier channel like the residual stream
    shift_t = torch.randn(1, 2, D, device=dev, dtype=torch.bfloat16) * 0.3
    scale_t = torch.randn(1, 2, D, device=dev, dtype=torch.bfloat16) * 0.3
    select = torch.zeros(1, M, 1, device=dev, dtype=torch.bool)
    select[:, 4096:] = True  # text rows pick row 1
    return x, shift_t, scale_t, select


table = torch.randn(16, 1, 1, 3 * D, device=dev, dtype=torch.bfloat16) * 0.3


def make_mode(step):
    x = torch.randn(1, 4096, D, device=dev, dtype=torch.bfloat16) * 2.0
    x[..., 11] *= 40
    shift, scale, _ = table[step].chunk(3, dim=-1)
    return x, shift, scale


def main_single(x, shift_t, scale_t, select, fn=None):
    x_mod = (single_c if fn is None else fn)(x, shift_t, scale_t, select)
    data, scales = quantize_fp8r(x_mod.reshape(-1, D))
    return x_mod, data, scales


def main_mode(x, shift, scale):
    x_mod = mode_c(x, shift, scale)
    data, scales = quantize_fp8r(x_mod.reshape(-1, D))
    return x_mod, data, scales


STORE = [False]  # served config; check runs both


def fused_single(x, shift_t, scale_t, select):
    return torch.ops.trtllm.flux3_fp8r_adaln_quant(
        x, shift_t, scale_t, select, pre_norm.weight, pre_norm.bias, 1e-6, STORE[0])


def fused_mode(x, shift, scale):
    return torch.ops.trtllm.flux3_fp8r_adaln_quant(
        x, shift, scale, None, pre_norm.weight, pre_norm.bias, 1e-6, STORE[0])


def inductor_configs():
    from torch._inductor.codecache import PyCodeCache

    mods = PyCodeCache.modules
    mods = list(mods.values()) if isinstance(mods, dict) else list(mods)
    out = []
    for mod in mods:
        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, "launchers") and "layer_norm" in name:
                out.append((name, [str(launcher.config) for launcher in obj.launchers]))
    return out


def u8(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def bitwise(a, b):
    if not STORE[0]:  # x_mod not written: compare E4M3 data + fp32 scales only
        a, b = a[1:], b[1:]
    return all(torch.equal(u8(p), u8(q)) for p, q in zip(a, b))


def mismatch(a, b):
    return {
        "x_mod_bitwise": torch.equal(a[0], b[0]) if STORE[0] else "not stored",
        "data_diff_elems": int((u8(a[1]) != u8(b[1])).sum()),
        "scales_bitwise": torch.equal(a[2], b[2]),
    }


def step_seq(singles, mode_in, use_fused):
    outs = []
    for s in singles:
        outs.append(fused_single(*s) if use_fused else main_single(*s))
    outs.append(fused_mode(*mode_in) if use_fused else main_mode(*mode_in))
    return outs


def bench(singles, mode_in):
    """Two input sets, graphs replayed interleaved:
    cold: 5 distinct x (128 MB working set, x mostly from DRAM);
    hot : the 5 heads read one x (L2-resident after the first head), like the model, where x
          was just written by the previous block's tail. kernel_ms reports the hot set."""
    use_fused = fused_mod is not None
    x_hot = singles[0][0]
    hot_singles = [(x_hot,) + tuple(s[1:]) for s in singles]
    hot_mode = (x_hot[:, :4096],) + tuple(mode_in[1:])
    sets = {"cold": (singles, mode_in), "hot": (hot_singles, hot_mode)}
    graphs = {}
    for name, (ss, mm) in sets.items():
        for _ in range(WARMUP):
            step_seq(ss, mm, use_fused)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(ITERS):
                step_seq(ss, mm, use_fused)
        for _ in range(5):
            g.replay()
        graphs[name] = g
    torch.cuda.synchronize()
    times = {name: [] for name in graphs}
    for i in range(60):
        for name, g in graphs.items():
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            g.replay()
            en.record()
            en.synchronize()
            times[name].append(st.elapsed_time(en) / ITERS)
        if i % 20 == 0:
            print("replay", i, {n: t[-1] for n, t in times.items()}, flush=True)
    stats = {}
    for name, t in times.items():
        t.sort()
        med = t[len(t) // 2]
        stats[name] = {
            "kernel_ms": med,
            "noise_floor": (t[(len(t) * 9) // 10] - t[len(t) // 10]) / med,
            "per_head_us": med * 1000 / 5,
        }
    arm = "fused(store_x_mod=False)" if use_fused else "main"
    print("arm", arm, "configs", inductor_configs(), flush=True)
    print(json.dumps({
        "batch": {"1": {"kernel_ms": stats["hot"]["kernel_ms"],
                        "noise_floor": stats["hot"]["noise_floor"]}},
        "output": {"class": "exact"},
        "arm": arm,
        "unit": "ms per denoise step of FP8R heads (4 single M=4176 + 1 mode M=4096); "
        "kernel_ms = L2-hot x (in-model-like); cold = 5 distinct x",
        "hot": stats["hot"],
        "cold": stats["cold"],
    }))


def run_checks(tag, singles, mode_in, fails, report):
    cases = {"single4176": make_single(4176), "single4256": make_single(4256)}
    # Static Inductor kernels (first shapes seen by single_c).
    for name, inp in cases.items():
        ref, got = main_single(*inp), fused_single(*inp)
        ok = bitwise(got, ref)
        report[tag + name + "_static"] = "bitwise" if ok else mismatch(got, ref)
        if not ok:
            fails.append(tag + name + "_static")
    # Dynamic-shape Inductor kernel: a fresh compile that has seen 4176 then 4256.
    single_dyn = compiled(Single())
    single_dyn(*cases["single4176"])
    single_dyn(*cases["single4256"])  # automatic dynamic recompile
    for name, inp in cases.items():
        ref, got = main_single(*inp, fn=single_dyn), fused_single(*inp)
        ok = bitwise(got, ref)
        report[tag + name + "_dynamic"] = "bitwise" if ok else mismatch(got, ref)
        if not ok:
            fails.append(tag + name + "_dynamic")
    for step in (0, 3, 15):
        inp = make_mode(step)
        ref, got = main_mode(*inp), fused_mode(*inp)
        key = f"{tag}mode4096_step{step}_off{inp[1].storage_offset()}"
        ok = bitwise(got, ref)
        report[key] = "bitwise" if ok else mismatch(got, ref)
        if not ok:
            fails.append(key)
    # Repeat sequence matching the bench (same warmup/iters), vs main's reference and iter 0.
    pristine = [tuple(t.clone() for t in s) for s in singles + [mode_in]]
    refs = step_seq(singles, mode_in, False)
    first = None
    for it in range(WARMUP + ITERS):
        outs = step_seq(singles, mode_in, True)
        if first is None:
            first = [tuple(t.clone() for t in o) for o in outs]
        for k, (o, r, f) in enumerate(zip(outs, refs, first)):
            if not bitwise(o, r):
                fails.append(f"{tag}repeat it{it} head{k} vs main")
            if not bitwise(o, f):
                fails.append(f"{tag}repeat it{it} head{k} vs iter0")
    for k, (cur, orig) in enumerate(zip(singles + [mode_in], pristine)):
        if not all(torch.equal(a, b) for a, b in zip(cur, orig)):
            fails.append(f"{tag}input {k} mutated")
    report[tag + "repeat_sequence"] = f"{WARMUP + ITERS} iterations x 5 heads compared"


with torch.inference_mode():
    singles = [make_single(4176) for _ in range(4)]
    mode_in = make_mode(3)
    if MODE == "bench":
        bench(singles, mode_in)
        sys.exit(0)
    assert fused_mod is not None, "check needs the fused op"
    fails, report = [], {}
    for store in (True, False):
        STORE[0] = store
        run_checks("store_" if store else "nostore_", singles, mode_in, fails, report)
    report["inductor_configs"] = inductor_configs()
    print(json.dumps(report, default=str), flush=True)
    print(json.dumps({
        "passed": not fails,
        "failures": fails[:20],
        "compared_against": "main's per-block-compiled LN+adaLN-modulate (static and dynamic "
        "Inductor kernels, emulate_precision_casts) + quantize_fp8r. store_x_mod=True: x_mod, "
        "E4M3 data, fp32 scales bitwise; store_x_mod=False (served): data + scales bitwise. "
        "Repeat sequence (3 warmup + 20 iters) vs main and vs iteration 0; inputs unchanged",
        "report": report,
    }, default=str))
