"""Untimed correctness check for embed_scale_rmsnorm + gathered final norm.

For every case: main's chain (reference a) vs the fused path, bitwise, for h,
layer-0 normed and final-norm rows; the repeated-call sequence (5 warmup +
30 calls, as the timing run) with every call compared to the first call and
to main; FP32 torch reference (b) errors for main and the candidate;
inputs unchanged; out-of-range ids (-1, V, V+7) give zero rows like main;
A,B,A shape order. Also imports modeling_gemma4 (the integration)."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import CASES, build_modules, fused_chain, main_chain, make_inputs, read_dims  # noqa: E402

WARMUP, ITERS = 5, 30


def bits_eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def fp32_ref(emb, norm0, normf, ids, x_last, seq_lens):
    V = emb.weight.shape[0]
    ok = (ids >= 0) & (ids < V)
    rows = emb.weight[torch.where(ok, ids, 0).long()].float() * torch.where(ok, 1.0, 0.0)[:, None]
    h32 = rows * float(emb.embed_scale.float())
    hb = h32.bfloat16().float()  # the norm reads the stored bf16 h

    def rms(x, w, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    n32 = rms(hb, norm0.weight, norm0.variance_epsilon)
    last = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1
    f32 = rms(x_last.float()[last], normf.weight, normf.variance_epsilon)
    return h32, n32, f32


def err(o, r):
    d = (o.float() - r).abs()
    return {"max_abs": float(d.max()), "rel_fro": float(d.norm() / r.norm().clamp_min(1e-30))}


def main():
    print(json.dumps(run(read_dims(sys.argv[1]))))


def run(dims, modules=None):
    emb, norm0, normf = modules if modules is not None else build_modules(dims)
    import tensorrt_llm._torch.models.modeling_gemma4 as m4  # noqa: F401
    from tensorrt_llm._torch.kernels.fused_ops.embed_scale_rmsnorm import rmsnorm_replay_supported
    replay = rmsnorm_replay_supported(dims["hidden"], torch.bfloat16, torch.device("cuda"))
    print("replay supported", replay, flush=True)
    w_before = emb.weight.clone()
    passed = replay
    report = {}
    names = ("h", "norm0", "final")
    for label in CASES:
        ids, x_last, seq_lens = make_inputs(label, dims, seed=1)
        ids_before, x_before = ids.clone(), x_last.clone()
        ref = main_chain(emb, norm0, normf, ids, x_last, seq_lens)
        first = None
        rep_ok = True
        for i in range(WARMUP + ITERS):
            outs = fused_chain(emb, norm0, normf, ids, x_last, seq_lens)
            if first is None:
                first = [o.clone() for o in outs]
            for k in range(3):
                rep_ok &= bits_eq(outs[k], first[k]) and bits_eq(outs[k], ref[k])
        torch.cuda.synchronize()
        eq = {n: bits_eq(first[k], ref[k]) for k, n in enumerate(names)}
        r32 = fp32_ref(emb, norm0, normf, ids, x_last, seq_lens)
        e_main = {n: err(ref[k], r32[k]) for k, n in enumerate(names)}
        e_cand = {n: err(first[k], r32[k]) for k, n in enumerate(names)}
        maxdiff = {n: float((first[k].float() - ref[k].float()).abs().max()) for k, n in enumerate(names)}
        finite = all(bool(torch.isfinite(o).all()) for o in first)
        inputs_ok = torch.equal(ids, ids_before) and bits_eq(x_last, x_before)
        ok = all(eq.values()) and rep_ok and finite and inputs_ok
        passed &= ok
        report[label] = {"bitwise": eq, "repeat_ok": rep_ok, "finite": finite, "inputs_unchanged": inputs_ok,
                         "cand_vs_main_maxabs": maxdiff, "err_main_vs_fp32": e_main,
                         "err_cand_vs_fp32": e_cand}
        print(label, json.dumps(report[label]), flush=True)

    # Out-of-range ids (rejection-sampler -1 padding, ids >= V): zero rows like main.
    V = dims["vocab"]
    ids, x_last, seq_lens = make_inputs("1000", dims, seed=2)
    ids[3], ids[10], ids[500], ids[999] = -1, V, V + 7, -1
    ref = main_chain(emb, norm0, normf, ids, x_last, seq_lens)
    got = fused_chain(emb, norm0, normf, ids, x_last, seq_lens)
    oor_ok = all(bits_eq(got[k], ref[k]) for k in range(3)) and bool((got[0][[3, 10, 500, 999]] == 0).all())
    report["out_of_range_ids"] = oor_ok
    print("out_of_range_ids", oor_ok, flush=True)
    passed &= oor_ok

    # A, B, A order: no state carried across shapes.
    seqs = []
    for label in ("8000", "300", "8000"):
        ids, x_last, seq_lens = make_inputs(label, dims, seed=3)
        seqs.append(fused_chain(emb, norm0, normf, ids, x_last, seq_lens))
    aba_ok = all(bits_eq(seqs[0][k], seqs[2][k]) for k in range(3))
    report["aba"] = aba_ok
    passed &= aba_ok
    table_ok = bits_eq(emb.weight, w_before)
    report["table_unchanged"] = table_ok
    passed &= table_ok
    return {"passed": bool(passed), "compared": "fused embed_scale_rmsnorm + gathered final norm vs main "
            "chain (masked embedding * embed_scale, flashinfer rmsnorm, final norm on all rows then gather), "
            "bitwise; FP32 torch reference errors reported", "summary": {
                k: (v if not isinstance(v, dict) else {"bitwise": v["bitwise"], "repeat_ok": v["repeat_ok"]})
                for k, v in report.items()}}


if __name__ == "__main__":
    main()
