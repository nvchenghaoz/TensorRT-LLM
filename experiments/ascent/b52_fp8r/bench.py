"""Branch 52 (idea 104): FP8R blocks' QKV + QK-RMSNorm + RoPE, op level, at M=4096 / 4176.

Arms (one script for both trees of run_gpu_pair):
- base (main, no flux3_fp8r_qkv module): PreparedFP8Sm120.gemm (torch._scaled_mm) -> split ->
  Flux3Attention.apply_qk_norm math (FP32) -> apply_rotary_emb -> BF16, i.e. attn_pre_out's
  fallback for the 5 FP8R blocks.
- candidate: primsts.qkv_forward -> flux3_fp8r_qkv._sm120_fp8r_qkv_forward (one custom op:
  _scaled_mm + in-place CuTe qknorm_rope).
Both are compiled like the block (torch.compile, emulate_precision_casts) and end with the
q/k/v .contiguous() that Flux3Attention._attend does for the FLASHINFER backend, so the time
covers everything that changes between the arms. Timing: a CUDA graph of 20 calls, median of
60 replays (knowledge 477), inputs restored from pristine copies before timing.

Modes: bench (timed; last line = evidence JSON), check (untimed correctness, candidate tree).
ITEM 2 v2 TOLERANCE (frozen before the run, ruling knowledge 663 / event 1747), per shape and
separately for q and k, candidate vs main (the compiled base path, same inputs):
 (a) >= 99.9% of elements within 1 bf16 ulp of main;
 (b) candidate-vs-fp64 max abs error <= main-vs-fp64 max abs error at bf16 resolution
     (relative 1e-3);
 (c) V bitwise equal to main.
Plus every iteration bitwise equal to iteration 0, inputs unchanged, graph == eager bitwise.
"""

import json
import sys
import types

import torch

from tensorrt_llm._torch.visual_gen.attention.attention import apply_rotary_emb
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
from tensorrt_llm._torch.modules.linear import Fp8RowwiseQuantizedTensor
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedFP8Sm120, qkv_forward

try:
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as cand_mod
except ImportError:
    cand_mod = None

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


def log(*a):
    print(*a, flush=True)


def make_inputs(m):
    g = torch.Generator(device=DEV).manual_seed(1234 + m)
    w = (torch.randn(3 * W, K, device=DEV, generator=g) * 0.02).bfloat16()
    adapter = PreparedFP8Sm120(w)
    x = torch.randn(1, m, K, device=DEV, generator=g).bfloat16()
    qn = (1 + 0.2 * torch.randn(D, device=DEV, generator=g)).bfloat16()
    kn = (1 + 0.2 * torch.randn(D, device=DEV, generator=g)).bfloat16()
    # FLUX-like multi-axis RoPE: 4 axes of 32 dims, positions (0, t, row, col).
    idx = torch.arange(m, device=DEV)
    axes = torch.stack([idx * 0, idx // 4096, (idx % 4096) // 64, idx % 64], -1).float()
    pieces = []
    for a in range(4):
        freqs = 1.0 / (2000.0 ** (torch.arange(0, 32, 2, device=DEV).float() / 32))
        pieces.append(axes[:, a:a + 1] * freqs[None])
    ang = torch.cat(pieces, -1)  # [m, 64]
    cos = ang.cos().repeat_interleave(2, -1).view(1, m, 1, D).contiguous()
    sin = ang.sin().repeat_interleave(2, -1).view(1, m, 1, D).contiguous()
    cos_sin = torch.cat((cos.reshape(-1, 128)[:, ::2], sin.reshape(-1, 128)[:, ::2]), -1)
    cos_sin = cos_sin.float().contiguous()
    data, scales = quantize_fp8r(x.reshape(-1, K))
    attn = types.SimpleNamespace(
        _flux3_primsts_nvfp4_qkv=None,
        _primsts_fp8_qkv_rope=False,
        _flux3_sm120_fp8_qkv=adapter,
        _flux3_sm120_fp8r_qknorm_rope=cand_mod is not None,
        norm_q=types.SimpleNamespace(weight=qn, variance_epsilon=EPS),
        norm_k=types.SimpleNamespace(weight=kn, variance_epsilon=EPS),
    )
    return dict(m=m, adapter=adapter, x=x, data=data, scales=scales, qn=qn, kn=kn,
                cos=cos, sin=sin, cos_sin=cos_sin, attn=attn)


def base_tail(out, cos, sin, qn, kn):
    s = out.shape[0]
    q, k, v = out.split(W, dim=-1)
    q = q.reshape(1, s, W).view(1, s, H, D)
    k = k.reshape(1, s, W).view(1, s, H, D)
    v = v.reshape(1, s, W).view(1, s, H, D)
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS) * qn.float()
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + EPS) * kn.float()
    q = apply_rotary_emb(q, cos, sin).to(v.dtype)
    k = apply_rotary_emb(k, cos, sin).to(v.dtype)
    return q.flatten(2), k.flatten(2), v.flatten(2)


def _strided_attention_on_main():
    # Landing 29 (fp8_sm12x attention, accepts_strided_qkv on the served fp8/fp8 config):
    # _attend no longer copies q/k/v to contiguous, so neither arm does here.
    try:
        import tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x  # noqa: F401
        return True
    except Exception:
        return False


STRIDED = _strided_attention_on_main()


def contig(t):
    # Strided q/k/v are passed to attention as-is; return a non-aliasing view for the graph.
    return t if STRIDED else t.contiguous()


def make_base(inp):
    adapter = inp["adapter"]

    def fn(x, data, scales, cos, sin, cos_sin, qn, kn):
        out = adapter.gemm(data, scales)
        q, k, v = base_tail(out, cos, sin, qn, kn)
        return contig(q), contig(k), contig(v)

    return torch.compile(fn, dynamic=False, options={"emulate_precision_casts": True})


def make_cand(inp):
    attn = inp["attn"]

    def fn(x, data, scales, cos, sin, cos_sin, qn, kn):
        xq = Fp8RowwiseQuantizedTensor(x, data, scales)
        q, k, v = qkv_forward(attn, xq, (cos, sin, cos_sin))
        return contig(q), contig(k), contig(v)

    return torch.compile(fn, dynamic=False, options={"emulate_precision_casts": True})


def args_of(inp):
    return (inp["x"], inp["data"], inp["scales"], inp["cos"], inp["sin"], inp["cos_sin"],
            inp["qn"], inp["kn"])


def pristine(inp):
    return [t.clone() for t in args_of(inp)]


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


def time_graph(g, replays=REPLAYS):
    ts = []
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / CALLS)
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med


def ordered(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def fp64_ref(inp):
    out = inp["adapter"].gemm(inp["data"], inp["scales"]).double()
    s = out.shape[0]
    res = []
    for plane, nw in ((0, inp["qn"]), (1, inp["kn"])):
        t = out[:, plane * W:(plane + 1) * W].reshape(s, H, D)
        t = t * torch.rsqrt(t.square().mean(-1, keepdim=True) + EPS) * nw.double()
        c = inp["cos_sin"].double()[:, None, :64]
        sn = inp["cos_sin"].double()[:, None, 64:]
        e, o = t[..., 0::2], t[..., 1::2]
        r = torch.stack([e * c - o * sn, o * c + e * sn], -1).flatten(-2)
        res.append(r.reshape(s, W))
    return res


def bench():
    arm = "candidate" if cand_mod is not None else "base"
    log(f"arm={arm} torch={torch.__version__}")
    result = {}
    for m in SHAPES:
        inp = make_inputs(m)
        copy = pristine(inp)
        fn = make_cand(inp) if cand_mod is not None else make_base(inp)
        g, _ = graph_of(fn, args_of(inp))
        for t, c in zip(args_of(inp), copy):
            t.copy_(c)
        med, nf = time_graph(g)
        log(f"M={m} arm={arm} median {med * 1000:.2f} us/call noise {nf:.4f}")
        result[m] = (med, nf)
        del g
        torch.cuda.synchronize()
    out = {
        "batch": {"1": {"kernel_ms": result[4176][0], "noise_floor": result[4176][1]}},
        "shapes": {str(m): {"kernel_ms": v[0], "noise_floor": v[1]} for m, v in result.items()},
        "arm": arm,
        "note": "batch 1 = M=4176 (single_blocks.24-27); M=4096 (content_mode_blocks.image.0) in shapes",
        "output": {"class": "reordered" if arm == "candidate" else "exact"},
    }
    print(json.dumps(out), flush=True)


def check():
    assert cand_mod is not None, "check runs in the candidate tree"
    report = {"shapes": {}}
    passed = True
    for m in SHAPES:
        inp = make_inputs(m)
        copy = pristine(inp)
        base = make_base(inp)
        cand = make_cand(inp)
        with torch.no_grad():
            bq, bk, bv = [t.clone() for t in base(*args_of(inp))]
            eq, ek, ev = [t.clone() for t in cand(*args_of(inp))]  # eager-call (no graph)
        ok_inputs = unchanged(inp, copy)
        ref_q, ref_k = fp64_ref(inp)
        shape_rep = {}
        for name, c, b, r in (("q", eq, bq, ref_q), ("k", ek, bk, ref_k)):
            ulp = (ordered(c) - ordered(b)).abs()
            within1 = (ulp <= 1).double().mean().item()
            c_err = (c.double().reshape(m, W) - r).abs().max().item()
            b_err = (b.double().reshape(m, W) - r).abs().max().item()
            ok = within1 >= 0.999 and c_err <= b_err * (1 + 1e-3)
            shape_rep[name] = dict(frac_within_1ulp=within1, max_ulp=int(ulp.max().item()),
                                   n_gt1ulp=int((ulp > 1).sum().item()),
                                   cand_fp64_maxabs=c_err, base_fp64_maxabs=b_err,
                                   max_abs_vs_base=(c.float() - b.float()).abs().max().item(),
                                   passed=ok)
            passed &= ok
        v_bitwise = torch.equal(ev, bv)
        passed &= v_bitwise and ok_inputs
        # Same sequence as timing: graph of CALLS calls, WARM + REPLAYS replays, reset before.
        for t, c in zip(args_of(inp), copy):
            t.copy_(c)
        g, outs = graph_of(cand, args_of(inp))
        for t, c in zip(args_of(inp), copy):
            t.copy_(c)
        iter0 = None
        rep_ok = True
        graph_eq_eager = None
        for i in range(WARM + REPLAYS):
            g.replay()
            torch.cuda.synchronize()
            cur = [t.clone() for t in outs]
            if iter0 is None:
                iter0 = cur
                graph_eq_eager = all(torch.equal(a, b) for a, b in zip(cur, (eq, ek, ev)))
            else:
                rep_ok &= all(torch.equal(a, b) for a, b in zip(cur, iter0))
            if not unchanged(inp, copy):
                rep_ok = False
        # Base repeatability too (reference stable).
        bq2, bk2, bv2 = base(*args_of(inp))
        base_stable = torch.equal(bq2, bq) and torch.equal(bk2, bk) and torch.equal(bv2, bv)
        passed &= rep_ok and graph_eq_eager and base_stable
        shape_rep.update(v_bitwise=v_bitwise, inputs_unchanged=ok_inputs,
                         every_iteration_bitwise_iter0=rep_ok, graph_eq_eager=graph_eq_eager,
                         base_repeat_bitwise=base_stable, iterations=WARM + REPLAYS)
        report["shapes"][str(m)] = shape_rep
        log(f"M={m}: {json.dumps(shape_rep)}")
        del g
    report["passed"] = bool(passed)
    report["compared"] = ("candidate qkv_forward (FP8R op) vs main's compiled FP8R fallback path "
                          "(same inputs) and an fp64 reference; item 2 v2 tolerance")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        check() if MODE == "check" else bench()
