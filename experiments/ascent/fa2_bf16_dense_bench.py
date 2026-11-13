"""FlashInferAttention._run_bf16_dense: timing (mode=bench) and bitwise check (mode=check).

Calls the function exactly as _run_fp8_gated does: NHD [1,S,28,128] BF16 buffers (q, k
contiguous, v a strided view of a packed QKV buffer), scale 1/sqrt(128), non-causal.
q-sliced cases pass q[:, a:b] against full k/v (parallel_comm's chunked BF16 attention).
In the baseline tree _run_bf16_dense is torch SDPA (FA2); in the candidate tree it takes
the fa2_sm120 op through fa2_sm120.try_attention.
"""

import json
import os
import subprocess
import sys
import types

import torch

H, D = 28, 128
SCALE = 1.0 / (D**0.5)
WARMUP, ITERS = 5, 20


def log(*a):
    print(*a, flush=True)


def make_inputs(s, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((1, s, 3, H, D), generator=g, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :, 0].contiguous()
    k = qkv[:, :, 1].contiguous()
    v = qkv[:, :, 2]  # strided view, row stride 3*H*D
    return q, k, v


def get_fn():
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention

    ns = types.SimpleNamespace(scale=SCALE, num_heads=H, num_kv_heads=H)
    return lambda q, k, v: FlashInferAttention._run_bf16_dense(ns, q, k, v, False)


def sdpa_flash_ref(q, k, v):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=SCALE
        )
    return o.transpose(1, 2)


# (label, S_kv, q slice) ; q slice None = full
CASES_BENCH = [("37966", 37966, None), ("q12032", 37966, (0, 12032))]
CASES_CHECK = [
    ("37966", 37966, None),
    ("37769", 37769, None),
    ("q12032", 37966, (0, 12032)),
    ("q12032b", 37966, (12032, 24064)),
    ("q1870", 37966, (36096, 37966)),
    ("2", 2, None),
    ("1000", 1000, None),
    ("4097", 4097, None),
]


def bench():
    fn = get_fn()
    out = {}
    for label, s, sl in CASES_BENCH:
        q, k, v = make_inputs(s)
        if sl is not None:
            q = q[:, sl[0] : sl[1]]
        for _ in range(WARMUP):
            fn(q, k, v)
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn(q, k, v)
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        med = ts[len(ts) // 2]
        p10, p90 = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1))]
        log(f"[bench] {label}: median {med:.3f} ms p10 {p10:.3f} p90 {p90:.3f}")
        out[label] = {"kernel_ms": med, "p10": p10, "p90": p90, "noise_floor": (p90 - p10) / med}
        del q, k, v
        torch.cuda.empty_cache()
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


def check():
    from tensorrt_llm._torch.visual_gen.attention.backends import fa2_sm120

    fn = get_fn()
    results, passed = {}, True
    log(f"[check] fa2_sm120 enabled={fa2_sm120.enabled()} min_q_row_blocks={fa2_sm120._min_q_row_blocks()}")
    passed &= fa2_sm120.enabled()
    for label, s, sl in CASES_CHECK:
        q, k, v = make_inputs(s)
        if sl is not None:
            q = q[:, sl[0] : sl[1]]
        q0, k0, v0 = q.clone(), k.clone(), v.clone()
        takes_op = fa2_sm120.supports(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        n0 = fa2_sm120._calls
        ref = sdpa_flash_ref(q, k, v)
        torch.cuda.synchronize()
        bad = []
        for i in range(WARMUP + ITERS):
            o = fn(q, k, v)
            if not (torch.equal(o, ref) and o.stride() == ref.stride() and o.shape == ref.shape):
                bad.append(i)
            if not (torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0)):
                bad.append(f"input_changed@{i}")
        op_calls = fa2_sm120._calls - n0
        # direct op vs SDPA (diagnostic, also where the gate falls back)
        direct = fa2_sm120.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), SCALE).transpose(1, 2)
        direct_eq = bool(torch.equal(direct, ref))
        # FP32 math reference on a 256-row q sample (rows are independent)
        rows = min(256, q.shape[1])
        qs = q[:, :rows].transpose(1, 2).float()
        att = torch.softmax((qs @ k.transpose(1, 2).float().transpose(-1, -2)) * SCALE, dim=-1)
        r32 = (att @ v.transpose(1, 2).float()).transpose(1, 2)
        err_o = ((o[:, :rows].float() - r32).norm() / r32.norm()).item()
        err_ref = ((ref[:, :rows].float() - r32).norm() / r32.norm()).item()
        ok = not bad and (op_calls == (WARMUP + ITERS if takes_op else 0))
        expect_op = label not in ("2", "1000")
        ok &= takes_op == expect_op
        passed &= ok
        results[label] = {
            "S_q": q.shape[1], "S_kv": s, "takes_op": takes_op, "op_calls": op_calls,
            "bitwise_all_iters": not bad, "bad": bad[:5], "strides": list(o.stride()),
            "ref_strides": list(ref.stride()), "direct_op_equals_sdpa": direct_eq,
            "relL2_vs_fp32_sample": err_o, "sdpa_relL2_vs_fp32_sample": err_ref, "ok": ok,
        }
        log(f"[check] {label}: {results[label]}")
        del q, k, v, q0, k0, v0, ref, o, direct
        torch.cuda.empty_cache()
    # TLLM_FA2_SM120=0 disables the gate
    env = dict(os.environ, TLLM_FA2_SM120="0")
    code = (
        "from tensorrt_llm._torch.visual_gen.attention.backends import fa2_sm120 as f;"
        "print('ENABLED', f.enabled())"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    disabled_ok = "ENABLED False" in r.stdout
    log(f"[check] TLLM_FA2_SM120=0 -> {r.stdout.strip()[-80:]} ok={disabled_ok}")
    passed &= disabled_ok
    print(json.dumps({"passed": bool(passed), "compared": "candidate _run_bf16_dense vs torch SDPA FLASH backend, bitwise + strides, 25 repeated calls per case, inputs unchanged", "cases": results, "env_disable_ok": disabled_ok}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    bench() if mode == "bench" else check()
