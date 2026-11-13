"""Idea 63 Stages 1-2: the Sage2 middle tier wired into FlashInferAttention's step gate.

One TP2 rank's shape: B=1, S=37966, H=28, D=128, non-causal, Q/K contiguous NHD, V a strided
view of a packed QKV buffer (row stride 3*H*D), inputs from bakeoff.make_inputs (set i).

Modes (argv[1]):
  pair   The same script runs in both trees of run_gpu_pair. It times one gated attention call
         on denoising forward TIER_STEP (default 6) through FlashInferAttention.forward with
         attention_metadata_state = {controller, attention_bf16_forwards=14,
         attention_sage_onset=6}. Main ignores the onset key, so its call runs the BF16 phase
         (fa2_sm120); the branch runs the Sage2 tier. Diagnostic arms in the same job, same
         palindrome: forward 0 (BF16 phase) and forward 14 (main's quantized recipe, FP8).
         Error of each arm vs FP32 SDPA math (TF32 off).
  check  Untimed: parse validation (Ks < K), controller tier table, routing bitwise per tier
         (phase 0 == _run_bf16_dense, tier == raw cake, >= K == _run_fp8_sm120, refiner == BF16,
         field absent == main routing), and the repeated-execution sequence (WARM + ITERS calls of
         the wired tier, every output bitwise vs the raw 'cake' call, inputs unchanged).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("BO_S", "37966")
os.environ.setdefault("BO_H", "28")
import bakeoff as bo  # noqa: E402

S, H, D = bo.S, bo.H, bo.D
K_ONSET = int(os.environ.get("TIER_K", 14))
KS = int(os.environ.get("TIER_KS", 6))
TIER_STEP = int(os.environ.get("TIER_STEP", KS))
NUM_STEPS = 49
WARM = int(os.environ.get("TIER_WARM", 5))
ITERS = int(os.environ.get("TIER_ITERS", 24))
dev = bo.dev


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_backend(with_sage: bool, layer_idx: int = 1):
    from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (
        DenoiseStepController,
    )
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    controller = DenoiseStepController()
    state = {"step_precision_controller": controller, "attention_bf16_forwards": K_ONSET}
    if with_sage:
        state["attention_sage_onset"] = KS
    backend = FlashInferAttention(
        layer_idx=layer_idx,
        num_heads=H,
        head_dim=D,
        dtype=torch.bfloat16,
        quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
        attention_metadata_state=state,
    )
    return backend, controller


def gated_call(backend, controller, step, q4, k4, v4):
    if step is None:
        controller.reset()
    else:
        controller.set_step(step, NUM_STEPS)
    return backend.forward(q4, k4, v4)


def raw_cake(q4, k4, v4):
    """Independent raw call: contiguous BHSD copies, own block list (bakeoff A_cake)."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
    from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import quantize_sage_qkv_sm120

    sq, sk = q4.shape[1], k4.shape[1]
    nq, nkv = (sq + 63) // 64, (sk + 63) // 64
    idx = torch.arange(nkv, dtype=torch.int32, device=dev).view(1, 1, 1, nkv).expand(1, H, nq, nkv).contiguous()
    bsz = torch.full((nkv,), 64, dtype=torch.int32, device=dev)
    bsz[-1] = sk - 64 * (nkv - 1)
    out = torch.empty(1, H, sq, D, device=dev, dtype=torch.bfloat16)
    ops = quantize_sage_qkv_sm120(
        q4.transpose(1, 2).contiguous(), k4.transpose(1, 2).contiguous(), v4.transpose(1, 2).contiguous()
    )
    bsa_attn_sm120_blk64_sage_fwd(*ops, idx, nkv, block_sizes=bsz, softmax_scale=1.0 / math.sqrt(D),
                                  out=out, uniform_block_count=True, contiguous_block_indices=True,
                                  backend="cake")
    return out.transpose(1, 2)


def inputs():
    q, k, v, qkv = bo.make_inputs(1234, False, False)
    return q, k, v, qkv, q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)


def stats(ts):
    t = sorted(ts)
    med = t[len(t) // 2]
    p10 = t[int(0.1 * (len(t) - 1))]
    p90 = t[int(0.9 * (len(t) - 1))]
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "noise": (p90 - p10) / med, "n": len(t)}


# ----------------------------------------------------------------------------------- pair
def run_pair():
    log("pair mode, S", S, "H", H, "tier step", TIER_STEP, "K", K_ONSET, "Ks", KS)
    q, k, v, qkv, q4, k4, v4 = inputs()
    pristine = (q.clone(), k.clone(), qkv.clone())
    backend, controller = build_backend(with_sage=True)
    arms = {"tier_call": TIER_STEP, "bf16_phase_f0": 0, "quant_recipe_f14": K_ONSET}
    res = {}
    with torch.no_grad():
        ref = bo.reference(q, k, v)
        log("reference done")
        for name, step in arms.items():
            o = gated_call(backend, controller, step, q4, k4, v4)
            torch.cuda.synchronize()
            res[name] = {"step": step, "err": bo.errors(o.reshape(S, H * D), ref)}
            log(name, res[name])
            del o
        del ref
        torch.cuda.empty_cache()
        names = list(arms)
        times = {n: [] for n in names}
        for n in names:
            for _ in range(WARM):
                gated_call(backend, controller, arms[n], q4, k4, v4)
        torch.cuda.synchronize()
        log("warmup done")
        for r in range(ITERS // 2):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                for _ in range(2):
                    controller.set_step(arms[n], NUM_STEPS)
                    s_ev = torch.cuda.Event(enable_timing=True)
                    e_ev = torch.cuda.Event(enable_timing=True)
                    s_ev.record()
                    backend.forward(q4, k4, v4)
                    e_ev.record()
                    e_ev.synchronize()
                    times[n].append(s_ev.elapsed_time(e_ev))
            if r % 4 == 0:
                log("round", r, {n: round(times[n][-1], 2) for n in names})
    for n in names:
        res[n].update(stats(times[n]))
        log(n, res[n])
    unchanged = (torch.equal(q, pristine[0]) and torch.equal(k, pristine[1])
                 and torch.equal(qkv, pristine[2]))
    sage_wired = hasattr(backend, "_run_sage_sm12x")
    tier = res["tier_call"]
    print(json.dumps({
        "batch": {"1": {"kernel_ms": tier["median_ms"], "noise_floor": tier["noise"]}},
        "output": {
            "class": "precision" if sage_wired else "exact",
            "tier_call_path": "sage2_cake" if sage_wired else "bf16_fa2_sm120",
            "rel_l2_vs_fp32": tier["err"]["rel_l2"],
            "max_abs_vs_fp32": tier["err"]["max_abs"],
            "bf16_rel_l2_vs_fp32": res["bf16_phase_f0"]["err"]["rel_l2"],
            "fp8_recipe_rel_l2_vs_fp32": res["quant_recipe_f14"]["err"]["rel_l2"],
        },
        "arms": res,
        "inputs_unchanged": unchanged,
        "shape": {"B": 1, "S": S, "H": H, "D": D, "v_row_stride": v.stride(0)},
        "K": K_ONSET, "Ks": KS,
    }))


# ----------------------------------------------------------------------------------- check
def parse_checks():
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    attn = {"backend": "FLASHINFER", "quant_attention_config": {"qk_dtype": "fp8", "v_dtype": "fp8"}}
    cases = {
        "K14_Ks6_ok": ({"attention_bf16_forwards": 14, "attention_sage_onset": 6}, True),
        "K14_Ks4_ok": ({"attention_bf16_forwards": 14, "attention_sage_onset": 4}, True),
        "K14_Ks0_ok": ({"attention_bf16_forwards": 14, "attention_sage_onset": 0}, True),
        "K14_absent_ok": ({"attention_bf16_forwards": 14}, True),
        "K14_Ks14_raises": ({"attention_bf16_forwards": 14, "attention_sage_onset": 14}, False),
        "K14_Ks20_raises": ({"attention_bf16_forwards": 14, "attention_sage_onset": 20}, False),
        "Knone_Ks6_raises": ({"attention_sage_onset": 6}, False),
        "Ks_neg_raises": ({"attention_bf16_forwards": 14, "attention_sage_onset": -1}, False),
    }
    out = {}
    for name, (sp, should_pass) in cases.items():
        try:
            args = VisualGenArgs(attention_config=attn, step_precision=sp)
            ok = True
            msg = repr(args.step_precision)
        except Exception as e:  # noqa: BLE001
            ok = False
            msg = repr(e)[:300]
        out[name] = {"parsed": ok, "expected": should_pass, "pass": ok == should_pass, "msg": msg}
        log("parse", name, out[name])
    return out


def controller_table():
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.step_precision import (
        DenoiseStepController,
    )

    c = DenoiseStepController()
    tiers = []
    for s in range(NUM_STEPS):
        c.set_step(s, NUM_STEPS)
        bf16 = c.runs_bf16(K_ONSET)
        sage = c.runs_sage(KS, K_ONSET)
        tiers.append("sage" if sage else "bf16" if bf16 else "quant")
    expected = ["bf16"] * KS + ["sage"] * (K_ONSET - KS) + ["quant"] * (NUM_STEPS - K_ONSET)
    c.reset()
    refiner = (c.runs_bf16(K_ONSET), c.runs_sage(KS, K_ONSET))
    c.set_step(6, NUM_STEPS, warmup=True)
    warm = (c.runs_bf16(K_ONSET), c.runs_sage(KS, K_ONSET))
    c.set_step(6, NUM_STEPS)
    absent = c.runs_sage(None, K_ONSET)
    ok = tiers == expected and refiner == (True, False) and warm == (False, False) and absent is False
    return {"pass": ok, "tiers": "".join(t[0] for t in tiers), "expected": "".join(t[0] for t in expected),
            "refiner_bf16_sage": refiner, "warmup_bf16_sage": warm, "absent_sage": absent}


def run_check():
    log("check mode, S", S, "H", H, "K", K_ONSET, "Ks", KS)
    result = {"parse": parse_checks(), "controller": controller_table()}
    log("controller", result["controller"])
    q, k, v, qkv, q4, k4, v4 = inputs()
    pristine = (q.clone(), k.clone(), qkv.clone())

    def unchanged():
        return (torch.equal(q, pristine[0]) and torch.equal(k, pristine[1])
                and torch.equal(qkv, pristine[2]))

    sage_b, sage_c = build_backend(with_sage=True)
    main_b, main_c = build_backend(with_sage=False)
    routing = {}
    with torch.no_grad():
        bf16_ref = main_b._run_bf16_dense(q4, k4, v4, False).clone()
        fp8_ref, _ = main_b._run_fp8_sm120(q4, k4, v4, is_causal=False, with_lse=False)
        fp8_ref = fp8_ref.clone()
        cake_ref = raw_cake(q4, k4, v4).clone()
        torch.cuda.synchronize()
        log("references done")
        refs = {"bf16": bf16_ref, "sage": cake_ref, "fp8": fp8_ref}
        plan = [
            ("sage_tree", sage_b, sage_c, None, "bf16"),
            ("sage_tree", sage_b, sage_c, 0, "bf16"),
            ("sage_tree", sage_b, sage_c, KS - 1, "bf16"),
            ("sage_tree", sage_b, sage_c, KS, "sage"),
            ("sage_tree", sage_b, sage_c, K_ONSET - 1, "sage"),
            ("sage_tree", sage_b, sage_c, K_ONSET, "fp8"),
            ("sage_tree", sage_b, sage_c, NUM_STEPS - 1, "fp8"),
            ("field_absent", main_b, main_c, None, "bf16"),
            ("field_absent", main_b, main_c, 0, "bf16"),
            ("field_absent", main_b, main_c, KS, "bf16"),
            ("field_absent", main_b, main_c, K_ONSET - 1, "bf16"),
            ("field_absent", main_b, main_c, K_ONSET, "fp8"),
        ]
        for tree, b, c, step, want in plan:
            o = gated_call(b, c, step, q4, k4, v4)
            torch.cuda.synchronize()
            matches = [name for name, r in refs.items() if torch.equal(o, r)]
            routing[f"{tree}_f{step}"] = {"expected": want, "bitwise_equal_to": matches,
                                         "pass": matches == [want]}
            log("routing", tree, step, routing[f"{tree}_f{step}"])
            del o
        # wired tier vs bakeoff's 'A_cake' (strided quantize input) and phase 0 vs _run_bf16_dense
        result["routing"] = routing

        # repeated-execution sequence on the wired tier: same warmup/iteration counts as timing
        diffs, inputs_ok = [], True
        for i in range(WARM + ITERS):
            o = gated_call(sage_b, sage_c, KS, q4, k4, v4)
            if not torch.equal(o, cake_ref):
                diffs.append(i)
            if not unchanged():
                inputs_ok = False
            del o
        result["repeat"] = {"calls": WARM + ITERS, "all_bitwise_raw_cake": not diffs,
                            "diff_iters": diffs[:10], "inputs_unchanged": inputs_ok}
        log("repeat", result["repeat"])
        # phase-0 repeated sequence: bitwise main's _run_bf16_dense
        diffs0 = []
        for i in range(WARM + ITERS):
            o = gated_call(sage_b, sage_c, 0, q4, k4, v4)
            if not torch.equal(o, bf16_ref):
                diffs0.append(i)
            del o
        result["repeat_phase0"] = {"calls": WARM + ITERS, "all_bitwise_bf16_dense": not diffs0,
                                   "diff_iters": diffs0[:10]}
        log("repeat phase0", result["repeat_phase0"])
        # odd-length shape (pitfall 622): another packed S on the same backend
        s2 = int(os.environ.get("TIER_S2", 37769))
        q2, k2, v2 = q4[:, :s2].contiguous(), k4[:, :s2].contiguous(), v4[:, :s2]
        o2 = gated_call(sage_b, sage_c, KS, q2, k2, v2)
        r2 = raw_cake(q2, k2, v2)
        result["second_shape"] = {"S": s2, "bitwise_raw_cake": bool(torch.equal(o2, r2))}
        o3 = gated_call(sage_b, sage_c, KS, q4, k4, v4)
        result["second_shape"]["back_to_S_bitwise"] = bool(torch.equal(o3, cake_ref))
        log("second shape", result["second_shape"])

    passed = (
        all(v["pass"] for v in result["parse"].values())
        and result["controller"]["pass"]
        and all(v["pass"] for v in routing.values())
        and result["repeat"]["all_bitwise_raw_cake"]
        and result["repeat"]["inputs_unchanged"]
        and result["repeat_phase0"]["all_bitwise_bf16_dense"]
        and result["second_shape"]["bitwise_raw_cake"]
        and result["second_shape"]["back_to_S_bitwise"]
    )
    print(json.dumps({"passed": passed, "compared": "wired Sage tier vs raw FlashInfer cake call; "
                      "BF16 phase vs main's _run_bf16_dense; >=K vs _run_fp8_sm120",
                      "S": S, "H": H, "K": K_ONSET, "Ks": KS, **result}, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pair"
    run_pair() if mode == "pair" else run_check()
