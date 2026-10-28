"""Branch 58 (idea 113): FP8R op with planar qknorm_rope dst (no q/k/v .contiguous() copies).

Same op-level setup as experiments/ascent/b52_fp8r/bench.py (pairs 842/843), at M=4176 / 4096:
primsts.qkv_forward -> flux3_fp8r_qkv._sm120_fp8r_qkv_forward under torch.compile
(emulate_precision_casts), returning q/k/v exactly as attention receives them.
- base tree (main 1e9ffb0a): op runs the in-place packed pass, then 3 .contiguous() copies.
- candidate tree: op runs the planar pass into a fresh [3, M, W] buffer; q/k/v are its planes.
Timing: CUDA graph of 20 calls, median of 60 replays, inputs restored from pristine copies.

Modes: bench (timed; last line = evidence JSON), check (untimed correctness, candidate tree):
- reference = main's op, reproduced in this process two ways: (1) an eager copy of main's op
  body (_scaled_mm + in-place qknorm_rope + split + .contiguous()), (2) the candidate module with
  _FP8R_QKV_PLANAR=False (main's code path). q/k/v must be BITWISE equal to both.
- same sequence as timing (graph of 20 calls, 3 warm + 60 replays, reset before): every
  iteration bitwise equal to iteration 0 and to the reference; inputs unchanged after each replay;
  graph == eager bitwise; outputs contiguous.
"""

import json
import sys
import types

import torch

from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as mod
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedFP8Sm120, qkv_forward

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
SHAPES = (4176, 4096)
H, D = 24, 128
W = H * D
K = 3072
EPS = 1e-6
CALLS = 20
REPLAYS = 60
WARM = 3
DEV = "cuda"
HAS_PLANAR = hasattr(mod, "_FP8R_QKV_PLANAR")


def log(*a):
    print(*a, flush=True)


def make_inputs(m):
    g = torch.Generator(device=DEV).manual_seed(1234 + m)
    w = (torch.randn(3 * W, K, device=DEV, generator=g) * 0.02).bfloat16()
    adapter = PreparedFP8Sm120(w)
    x = torch.randn(1, m, K, device=DEV, generator=g).bfloat16()
    qn = (1 + 0.2 * torch.randn(D, device=DEV, generator=g)).bfloat16()
    kn = (1 + 0.2 * torch.randn(D, device=DEV, generator=g)).bfloat16()
    idx = torch.arange(m, device=DEV)
    axes = torch.stack([idx * 0, idx // 4096, (idx % 4096) // 64, idx % 64], -1).float()
    pieces = []
    for a in range(4):
        freqs = 1.0 / (2000.0 ** (torch.arange(0, 32, 2, device=DEV).float() / 32))
        pieces.append(axes[:, a:a + 1] * freqs[None])
    ang = torch.cat(pieces, -1)
    cos = ang.cos().repeat_interleave(2, -1).view(1, m, 1, D).contiguous()
    sin = ang.sin().repeat_interleave(2, -1).view(1, m, 1, D).contiguous()
    cos_sin = torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), -1)
    cos_sin = cos_sin.float().contiguous()
    data, scales = quantize_fp8r(x.reshape(-1, K))
    attn = types.SimpleNamespace(
        _flux3_primsts_nvfp4_qkv=None,
        _primsts_fp8_qkv_rope=False,
        _flux3_sm120_fp8_qkv=adapter,
        _flux3_sm120_fp8r_qknorm_rope=True,
        norm_q=types.SimpleNamespace(weight=qn, variance_epsilon=EPS),
        norm_k=types.SimpleNamespace(weight=kn, variance_epsilon=EPS),
    )
    return dict(m=m, adapter=adapter, x=x, data=data, scales=scales, qn=qn, kn=kn,
                cos=cos, sin=sin, cos_sin=cos_sin, attn=attn)


def make_fn(inp):
    attn = inp["attn"]

    def fn(x, data, scales, cos, sin, cos_sin, qn, kn):
        xq = Fp8RowwiseQuantizedTensor(x, data, scales)
        q, k, v = qkv_forward(attn, xq, (cos, sin, cos_sin))
        return q, k, v

    return torch.compile(fn, dynamic=False, options={"emulate_precision_casts": True})


def main_op_eager(inp):
    """Eager copy of main 1e9ffb0a's op body + _sm120_fp8r_qkv_forward's split/contiguous."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120.flux3_sm120_qknorm_rope import (
        qknorm_rope,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts import _token_positions

    a = inp["adapter"]
    out = torch._scaled_mm(inp["data"], a.weight.t(), scale_a=inp["scales"].reshape(-1, 1),
                           scale_b=a.scale.reshape(1, -1), out_dtype=torch.bfloat16)
    width = out.shape[1] // 3
    positions = _token_positions(out.shape[0], out.device)
    qknorm_rope(out[:, :width], out[:, width:2 * width], out[:, 2 * width:], out,
                inp["qn"], inp["kn"], inp["cos_sin"], positions)
    q, k, v = out.split(width, -1)
    return tuple(t.contiguous().reshape(1, out.shape[0], width) for t in (q, k, v))


def args_of(inp):
    return (inp["x"], inp["data"], inp["scales"], inp["cos"], inp["sin"], inp["cos_sin"],
            inp["qn"], inp["kn"])


def pristine(inp):
    return [t.clone() for t in args_of(inp)]


def restore(inp, copy):
    for t, c in zip(args_of(inp), copy):
        t.copy_(c)


def unchanged(inp, copy):
    return all(torch.equal(a, b) for a, b in zip(args_of(inp), copy))


def graph_of(fn, args):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS - 1):
            fn(*args)
        outs = fn(*args)
    return g, outs


def time_graph(g):
    ts = []
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    for _ in range(REPLAYS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / CALLS)
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med


def bench():
    arm = "candidate_planar" if HAS_PLANAR and mod._FP8R_QKV_PLANAR else "base_contiguous"
    log(f"arm={arm} torch={torch.__version__}")
    result = {}
    for m in SHAPES:
        inp = make_inputs(m)
        copy = pristine(inp)
        fn = make_fn(inp)
        g, outs = graph_of(fn, args_of(inp))
        restore(inp, copy)
        med, nf = time_graph(g)
        log(f"M={m} arm={arm} median {med * 1000:.2f} us/call noise {nf:.4f} "
            f"contiguous={[t.is_contiguous() for t in outs]}")
        result[m] = (med, nf)
        del g, outs
        torch.cuda.synchronize()
    out = {
        "batch": {"1": {"kernel_ms": result[4176][0], "noise_floor": result[4176][1]}},
        "shapes": {str(m): {"kernel_ms": v[0], "noise_floor": v[1]} for m, v in result.items()},
        "arm": arm,
        "note": "batch 1 = M=4176 (single_blocks.24-27); M=4096 (content_mode_blocks.image.0) "
                "in shapes; op time incl. q/k/v as handed to attention",
        "output": {"class": "exact"},
    }
    print(json.dumps(out), flush=True)


def eq3(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def check():
    assert HAS_PLANAR, "check runs in the candidate tree"
    report = {"shapes": {}}
    passed = True
    for m in SHAPES:
        inp = make_inputs(m)
        copy = pristine(inp)
        rep = {}
        with torch.no_grad():
            ref = [t.clone() for t in main_op_eager(inp)]
            restore(inp, copy)
            mod._FP8R_QKV_PLANAR = False
            off = [t.clone() for t in make_fn(inp)(*args_of(inp))]
            torch._dynamo.reset()
            restore(inp, copy)
            mod._FP8R_QKV_PLANAR = True
            fn = make_fn(inp)
            eager = fn(*args_of(inp))
            rep["outputs_contiguous"] = all(t.is_contiguous() for t in eager)
            rep["outputs_share_planar_buffer"] = (
                eager[1].data_ptr() - eager[0].data_ptr() == m * W * 2
                and eager[2].data_ptr() - eager[1].data_ptr() == m * W * 2)
            eager = [t.clone() for t in eager]
        rep["inputs_unchanged_eager"] = unchanged(inp, copy)
        rep["main_eager_vs_main_flag_off_bitwise"] = eq3(ref, off)
        rep["planar_vs_main_bitwise"] = {n: torch.equal(a, b)
                                         for n, a, b in zip("qkv", eager, ref)}
        rep["max_abs_vs_main"] = max((a.float() - b.float()).abs().max().item()
                                     for a, b in zip(eager, ref))
        ok = (rep["outputs_contiguous"] and rep["inputs_unchanged_eager"]
              and rep["main_eager_vs_main_flag_off_bitwise"]
              and all(rep["planar_vs_main_bitwise"].values()))
        # Same sequence as timing: graph of CALLS calls, WARM + REPLAYS replays, reset before.
        restore(inp, copy)
        g, outs = graph_of(fn, args_of(inp))
        restore(inp, copy)
        iter0, rep_ok, ref_ok, inputs_ok = None, True, True, True
        for i in range(WARM + REPLAYS):
            g.replay()
            torch.cuda.synchronize()
            cur = [t.clone() for t in outs]
            if iter0 is None:
                iter0 = cur
                rep["graph_eq_eager"] = eq3(cur, eager)
            else:
                rep_ok &= eq3(cur, iter0)
            ref_ok &= eq3(cur, ref)
            inputs_ok &= unchanged(inp, copy)
        rep.update(every_iteration_bitwise_iter0=rep_ok, every_iteration_bitwise_main=ref_ok,
                   inputs_unchanged_every_replay=inputs_ok, iterations=WARM + REPLAYS)
        ok = ok and rep_ok and ref_ok and inputs_ok and rep["graph_eq_eager"]
        rep["passed"] = bool(ok)
        passed &= ok
        report["shapes"][str(m)] = rep
        log(f"M={m}: {json.dumps(rep)}")
        del g, outs
        torch._dynamo.reset()
    report["passed"] = bool(passed)
    report["compared"] = ("candidate planar FP8R op q/k/v (compiled qkv_forward, eager and "
                          "CUDA-graph replays) vs main 1e9ffb0a's op (eager copy of its body + "
                          "contiguous copies, and the module with _FP8R_QKV_PLANAR=False): bitwise")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        check() if MODE == "check" else bench()
