"""Untimed correctness for branch 66 (idea 104).

1. Repeated sequence (same WARMUP + 3*ITERS iterations x 40 layers as b104_bench.py), M = 1/4/16/64/512/2048:
   every call NaN-poisons o_lora, runs mla.fp8_block_scaling_bmm_out (candidate) and compares it bitwise
   with the base copy path (new_empty + bmm + copy_) on the same inputs; plus a dequantized FP32
   reference for layer 0 (tolerance sanity).
2. Guard unit checks: misaligned base pointer and non-16B row stride -> helper False, fallback runs
   (copy kernel seen in torch.profiler) and is bitwise equal; aligned o_lora view -> no copy kernel.
3. Decode-shape CUDA graph capture/replay at M = 1/2/4/8 with changing inputs, bitwise vs eager copy path.
4. STEP 0: project_sparse_attn_output (deepseek_v4/module.py) driven with a stand-in module at M=2048
   reaches mla.fp8_block_scaling_bmm_out and the direct branch.
"""
import json
import types

import torch
from b104_common import (
    D,
    G,
    ITERS,
    LAYERS,
    WARMUP,
    R,
    copy_path_into,
    fp8_utils,
    make_inputs,
    make_layer_weights,
    mla,
)

from tensorrt_llm._utils import get_sm_version, is_sm_120f


def bits(t):
    return t.contiguous().view(torch.int16)


def dequant_ref(mat1, sf_w, w):
    # FP32 reference of the math: per-token 1x128 quantized A (as the op does) is not
    # reproduced; we compare BF16 A @ dequant(W)^T in fp32 with BF16 tolerance.
    wf = w.float() * sf_w.repeat_interleave(128, 1).repeat_interleave(128, 2)
    a = mat1.float().transpose(0, 1)  # [G, M, D]
    return torch.bmm(a, wf.transpose(1, 2)).transpose(0, 1)  # [M, G, R]


def kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def main():
    out = {"passed": False}
    sm = get_sm_version()
    out["sm"] = sm
    has_helper = hasattr(mla, "_sm120_bmm_out_direct_ok")
    out["candidate_helper_present"] = has_helper
    ok = is_sm_120f(sm)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    weights = make_layer_weights(gen)

    # ---- 1. repeated sequence ----
    seq = {}
    for m in (1, 4, 16, 64, 512, 2048):
        gen_m = torch.Generator(device="cuda").manual_seed(4321 + m)  # re-create inputs from seed
        inputs = make_inputs(gen_m, m)
        n_calls = 0
        n_bad = 0
        max_abs = 0.0
        all_finite = True
        for it in range(WARMUP + 3 * ITERS):
            for i in range(LAYERS):
                w, s, _ = weights[i]
                o_lora = torch.full([m, G, R], float("nan"), device="cuda", dtype=torch.bfloat16)
                mla.fp8_block_scaling_bmm_out(inputs[i], w, s, o_lora.transpose(0, 1), None, False)
                ref = torch.full([m, G, R], float("nan"), device="cuda", dtype=torch.bfloat16)
                copy_path_into(inputs[i], w, s, ref.transpose(0, 1))
                eq = torch.equal(bits(o_lora), bits(ref))
                fin = bool(torch.isfinite(o_lora).all())
                all_finite &= fin
                if not eq:
                    n_bad += 1
                    max_abs = max(max_abs, float((o_lora.float() - ref.float()).abs().nan_to_num(1e30).max()))
                n_calls += 1
        # fp32 sanity for layer 0
        o_lora = chain_out = torch.empty([m, G, R], device="cuda", dtype=torch.bfloat16)
        mla.fp8_block_scaling_bmm_out(inputs[0], weights[0][0], weights[0][1], chain_out.transpose(0, 1), None, False)
        r32 = dequant_ref(inputs[0], weights[0][2], weights[0][0])
        rel = float((o_lora.float() - r32).norm() / r32.norm())
        seq[str(m)] = {
            "calls": n_calls,
            "bitwise_mismatch_calls": n_bad,
            "max_abs_err_vs_copy_path": max_abs,
            "max_rel_err_vs_copy_path": 0.0 if n_bad == 0 else None,
            "all_finite_after_nan_poison": all_finite,
            "rel_fro_err_vs_fp32_dequant_ref": rel,
        }
        print(f"M={m} {seq[str(m)]}", flush=True)
        ok &= n_bad == 0 and all_finite and rel < 6e-2
        del inputs
        torch.cuda.empty_cache()
    out["repeated_sequence"] = seq

    # ---- 2. guard unit checks ----
    guard = {}
    m = 64
    x = make_inputs(torch.Generator(device="cuda").manual_seed(7), m, 1)[0]
    w, s, _ = weights[0]
    ref = torch.empty([m, G, R], device="cuda", dtype=torch.bfloat16)
    copy_path_into(x, w, s, ref.transpose(0, 1))
    # misaligned base pointer (2-byte offset)
    flat = torch.full([m * G * R + 8], float("nan"), device="cuda", dtype=torch.bfloat16)
    mis = flat[1 : 1 + m * G * R].view(m, G, R)
    mis_view = mis.transpose(0, 1)
    # non-16B row stride: [m, G, R+1] sliced to R
    padded = torch.full([m, G, R + 1], float("nan"), device="cuda", dtype=torch.bfloat16)
    pad_view = padded[:, :, :R].transpose(0, 1)
    aligned = torch.full([m, G, R], float("nan"), device="cuda", dtype=torch.bfloat16)
    al_view = aligned.transpose(0, 1)
    if has_helper:
        guard["helper_misaligned_ptr"] = mla._sm120_bmm_out_direct_ok(mis_view)
        guard["helper_bad_stride"] = mla._sm120_bmm_out_direct_ok(pad_view)
        guard["helper_aligned_o_lora_view"] = mla._sm120_bmm_out_direct_ok(al_view)
        ok &= (not guard["helper_misaligned_ptr"]) and (not guard["helper_bad_stride"]) and guard["helper_aligned_o_lora_view"]
    for name, view, holder in (("misaligned_ptr", mis_view, mis), ("bad_stride", pad_view, padded[:, :, :R]),
                               ("aligned", al_view, aligned)):
        names = kernel_names(lambda: mla.fp8_block_scaling_bmm_out(x, w, s, view, None, False))
        n_copy = sum(1 for k in names if "copy" in k.lower() or "elementwise" in k.lower())
        eq = torch.equal(bits(holder), bits(ref))
        guard[name] = {"bitwise_equal": eq, "copy_kernels": n_copy, "kernels": names}
        ok &= eq
    if has_helper:
        ok &= guard["misaligned_ptr"]["copy_kernels"] >= 1 and guard["bad_stride"]["copy_kernels"] >= 1
        ok &= guard["misaligned_ptr"]["copy_kernels"] > guard["aligned"]["copy_kernels"]
        ok &= len(guard["aligned"]["kernels"]) < len(guard["misaligned_ptr"]["kernels"])
    # padded column R must remain NaN (no write past the view)
    guard["pad_column_untouched"] = bool(torch.isnan(padded[:, :, R]).all())
    ok &= guard["pad_column_untouched"]
    print(f"guard {guard}", flush=True)
    out["guard"] = guard

    # ---- 3. decode-shape CUDA graph capture / replay ----
    graphs = {}
    for m in (1, 2, 4, 8):
        static_x = torch.zeros([m, G, D], device="cuda", dtype=torch.bfloat16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                o = torch.empty([m, G, R], device="cuda", dtype=torch.bfloat16)
                mla.fp8_block_scaling_bmm_out(static_x, w, s, o.transpose(0, 1), None, False)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_o = torch.empty([m, G, R], device="cuda", dtype=torch.bfloat16)
            mla.fp8_block_scaling_bmm_out(static_x, w, s, static_o.transpose(0, 1), None, False)
        bad = 0
        gx = torch.Generator(device="cuda").manual_seed(99 + m)
        for rep in range(10):
            static_x.copy_(torch.randn([m, G, D], generator=gx, device="cuda").to(torch.bfloat16))
            static_o.fill_(float("nan"))
            g.replay()
            r = torch.empty([m, G, R], device="cuda", dtype=torch.bfloat16)
            copy_path_into(static_x, w, s, r.transpose(0, 1))
            torch.cuda.synchronize()
            if not torch.equal(bits(static_o), bits(r)):
                bad += 1
        graphs[str(m)] = {"replays": 10, "bitwise_mismatch": bad}
        ok &= bad == 0
    print(f"graphs {graphs}", flush=True)
    out["decode_graph_capture_replay"] = graphs

    # ---- 4. STEP 0: project_sparse_attn_output at M=2048 reaches the direct branch ----
    step0 = {}
    try:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import module as dsv4_module

        calls = []
        orig_fn = mla.fp8_block_scaling_bmm_out
        orig_op = torch.ops.trtllm.fp8_block_scaling_bmm_out

        def spy(mat1, mat2_fp8, mat2_scale, o, *a, **k):
            calls.append(
                {
                    "M": int(mat1.shape[0]),
                    "out_strides": list(o.stride()),
                    "direct_ok": bool(mla._sm120_bmm_out_direct_ok(o)) if has_helper else None,
                }
            )
            return orig_fn(mat1, mat2_fp8, mat2_scale, o, *a, **k)

        mla.fp8_block_scaling_bmm_out = spy
        m = 2048
        heads, nope, rope = 16, 448, 64
        fake = types.SimpleNamespace(
            num_heads_tp=heads,
            n_local_groups=G,
            o_lora_rank=R,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            o_a_proj=w,
            o_a_proj_scale=s,
            o_a_proj_dequant=None,
            use_cute_dsl_blockscaling_bmm=False,
            use_cute_dsl_bf16_bmm=False,
            inverse_rotary_emb=types.SimpleNamespace(
                rotary_cos_sin=torch.randn([65536, rope], device="cuda", dtype=torch.float32),
                is_neox=False,
            ),
            o_b_proj=lambda t: t,
        )
        attn = torch.randn([m, heads * (nope + rope)], device="cuda").to(torch.bfloat16)
        pos = torch.arange(m, device="cuda", dtype=torch.int32) + 32768
        res = dsv4_module.project_sparse_attn_output(fake, [attn], pos)
        torch.cuda.synchronize()
        step0 = {"calls": calls, "output_shape": list(res.shape), "finite": bool(torch.isfinite(res).all()),
                 "sm120_branch": bool(is_sm_120f(sm)),
                 "b12x_wo_accepts_at_M2048": "False by contract (csa2/b12x_wo.py accepts needs 1<=M<=MAX_DECODE_ROWS=8)"}
        mla.fp8_block_scaling_bmm_out = orig_fn
        del orig_op
    except Exception as exc:  # report only; STEP 0 is a trace, not a pass criterion
        step0 = {"error": repr(exc)}
    print(f"step0 {step0}", flush=True)
    out["step0"] = step0

    out["passed"] = bool(ok)
    out["compared_against"] = "base copy path (new_empty + fp8_block_scaling_bmm_out + copy_), bitwise; fp32 dequant ref"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
