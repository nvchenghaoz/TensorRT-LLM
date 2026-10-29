"""Branch 182 (idea 268) STEP 0: accuracy pre-check of a static RMSNorm-bound Q/K E4M3 scale.

In-process pipeline as served (engine.json overlay), graph runner disabled so every routed
FP8-direct QKV call goes through flux3_primsts.nvfp4_qkv_rope_fp8_impl in Python. Request 1
(prompt A) is a warm-up (the first eager request routes extra calls); on request 2 (and on a
second prompt, supplementary) every call is logged (layer = call % 32, max|w_q|, max|w_k|, the
kernel's q/k descales) and 18 calls are captured: steps 0/8/15 x double first/mid/last
(idx 0, 2, 3; M4096) and single first/mid/last (idx 4, 18, 31; M4176).

Per capture (all fp32 SDPA, TF32 off, rel_l2 against (a)):
  (a)   BF16 norm+RoPE q/k (torch replica of the kernel math) and BF16 v;
  (b)   main: the kernel's own E4M3 qkv8 dequantized with its own descales;
  (bt)  torch per-tensor rule on the torch replica (reports byte match vs the kernel);
  (d)   static Q/K: B' = mul_rn(mul_rn(sqrt(128), max|w|), 1 + 2^-6), scale = bf16(max(B'/448,
        min)), V per-tensor (= the kernel's V bytes);
  (b_r) controls: per-tensor with amax * r, r in {1.19, 1.41, 1.68}.
Saturation in (d) = count of |rot * factor| > 448 before the clamp.
Gate (plan 1873): mean d/b <= 1.05, max d/b <= 1.20, 0 saturated.
Usage: python capture.py <checkpoint>
"""

import json
import math
import os
import sys
import time

import torch

CKPT = sys.argv[1]
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b182")
os.makedirs(ART, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import (  # noqa: E402
    _MIN_SCALING_FACTOR,
)
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402

HAS_STATIC = hasattr(qr8, "STATIC_QK")
if HAS_STATIC:
    qr8.STATIC_QK = False  # (b) = main's per-tensor route; the static kernel runs as (dk)
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

STEPS = (0, 8, 15)
IDX = {0: "double_first", 2: "double_mid", 3: "double_last", 4: "single_first",
       18: "single_mid", 31: "single_last"}
SQRT128 = math.sqrt(128.0)
MARGIN = 1.0 + 2.0 ** -6
RS = (1.19, 1.41, 1.68)

state = {"on": False, "n": 0, "calls": [], "caps": []}
_impl = fp.nvfp4_qkv_rope_fp8_impl


def f32(x):
    return torch.tensor(x, dtype=torch.float32)


def bound(maxw):
    """B' exactly as the kernel computes it (two mul.rn.f32)."""
    return (f32(SQRT128) * f32(maxw)) * f32(MARGIN)


def scale_of(amax):
    """_scale_factor: scale = bf16(max(div_rn(amax, 448), min)), factor = div_rn(1, scale)."""
    s = torch.maximum(f32(amax) / f32(448.0), f32(_MIN_SCALING_FACTOR))
    s = s.to(torch.bfloat16).to(torch.float32)
    return s, f32(1.0) / s


def _hook(activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin,
          positions, qkv_scale, sm_scale):
    qkv8, scales, out = _impl(activation, weight, activation_scale, weight_scale, q_norm,
                              k_norm, cos_sin, positions, qkv_scale, sm_scale)
    if state["on"]:
        i = state["n"]
        state["n"] += 1
        step, idx = divmod(i, 32)
        sc = scales.float().cpu()
        wq = float(q_norm.float().abs().max())
        wk = float(k_norm.float().abs().max())
        state["calls"].append({"call": i, "step": step, "layer": idx, "M": int(out.shape[0]),
                               "maxw_q": wq, "maxw_k": wk, "q_descale": float(sc[0]),
                               "k_descale": float(sc[1])})
        if step in STEPS and idx in IDX:
            state["caps"].append({
                "call": i, "step": step, "idx": idx, "name": IDX[idx], "sm_scale": sm_scale,
                "out": out.clone(), "qn": q_norm.clone(), "kn": k_norm.clone(),
                "cs": cos_sin.clone(), "pos": positions.clone(), "qkv8": qkv8.clone(),
                "scales": scales.clone()})
    return qkv8, scales, out


fp.nvfp4_qkv_rope_fp8_impl = _hook


def rot_ref(x, w, cs, pos):
    """Torch replica of the kernel's norm + RoPE, rounded to BF16. x [M, W] BF16."""
    M, W = x.shape
    H = W // 128
    xf = x.float().reshape(M, H, 128)
    ss = (xf * xf).sum(-1, keepdim=True)
    inv = torch.rsqrt(ss / 128.0 + 1e-6)
    y = xf * inv
    wf = w.float().view(1, 1, 128)
    e = y[..., 0::2] * wf[..., 0::2]
    o = y[..., 1::2] * wf[..., 1::2]
    t = cs[pos.long()].float()
    c = t[:, None, :64]
    s = t[:, None, 64:]
    re = e * c - o * s
    ro = o * c + e * s
    return torch.stack([re, ro], -1).reshape(M, W).to(torch.bfloat16)


def quant(x, factor):
    v = x.float() * factor
    sat = int((v.abs() > 448.0).sum())
    q = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, sat


def sdpa(q, k, v, sm_scale):
    """fp32 attention, [M, W] each, heads of 128, TF32 off."""
    M, W = q.shape
    H = W // 128
    qh = q.float().reshape(M, H, 128).transpose(0, 1)
    kh = k.float().reshape(M, H, 128).transpose(0, 1)
    vh = v.float().reshape(M, H, 128).transpose(0, 1)
    outs = []
    for h0 in range(0, H, 4):
        s = torch.matmul(qh[h0:h0 + 4], kh[h0:h0 + 4].transpose(1, 2)) * sm_scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.matmul(p, vh[h0:h0 + 4]))
        del s, p
    return torch.cat(outs, 0)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def analyze(cap):
    out = cap["out"]
    M = out.shape[0]
    W = out.shape[1] // 3
    planes = out.view(3, M, W)  # planar GEMM output: three contiguous [M, W] planes
    q, k, v = planes[0], planes[1], planes[2]
    qr = rot_ref(q, cap["qn"], cap["cs"], cap["pos"])
    kr = rot_ref(k, cap["kn"], cap["cs"], cap["pos"])
    sm = cap["sm_scale"]
    ref = sdpa(qr, kr, v, sm)
    sc = cap["scales"].float()
    q8, k8, v8 = cap["qkv8"][0], cap["qkv8"][1], cap["qkv8"][2]
    s0, s1, s2 = sc[0].item(), sc[1].item(), sc[2].item()
    ob = sdpa(q8.float() * s0, k8.float() * s1, v8.float() * s2, sm)
    rb = rel(ob, ref)
    amq = float(qr.float().abs().max())
    amk = float(kr.float().abs().max())
    amv = float(v.float().abs().max())
    res = {k_: cap[k_] for k_ in ("call", "step", "idx", "name")}
    res.update({"M": M, "amax_q": amq, "amax_k": amk, "amax_v": amv, "rel_b": rb})
    # (bt) torch per-tensor replica
    sq, fq = scale_of(amq)
    sk, fk = scale_of(amk)
    sv, fv = scale_of(amv)
    qb, _ = quant(qr, fq.item())
    kb, _ = quant(kr, fk.item())
    vb, _ = quant(v, fv.item())
    u8 = torch.uint8
    res["bt_bytes_match_q"] = float((qb.view(u8) == q8.view(u8)).float().mean())
    res["bt_bytes_match_k"] = float((kb.view(u8) == k8.view(u8)).float().mean())
    res["bt_bytes_match_v"] = float((vb.view(u8) == v8.view(u8)).float().mean())
    res["bt_scales_match"] = bool(sq.item() == s0 and sk.item() == s1 and sv.item() == s2)
    obt = sdpa(qb.float() * sq.item(), kb.float() * sk.item(), vb.float() * sv.item(), sm)
    res["rel_bt"] = rel(obt, ref)
    # (d) static Q/K
    wq = float(cap["qn"].float().abs().max())
    wk = float(cap["kn"].float().abs().max())
    Bq = bound(wq).item()
    Bk = bound(wk).item()
    sdq, fdq = scale_of(Bq)
    sdk, fdk = scale_of(Bk)
    qd, satq = quant(qr, fdq.item())
    kd, satk = quant(kr, fdk.item())
    od = sdpa(qd.float() * sdq.item(), kd.float() * sdk.item(), v8.float() * s2, sm)
    res.update({"maxw_q": wq, "maxw_k": wk, "B_q": Bq, "B_k": Bk,
                "binades_q": math.log2(Bq / amq), "binades_k": math.log2(Bk / amk),
                "max_rot_over_B": max(amq / Bq, amk / Bk),
                "sat_d": satq + satk, "rel_d": rel(od, ref)})
    res["d_over_b"] = res["rel_d"] / rb
    if HAS_STATIC:
        # (dk) this branch's static kernel on the same planes: bytes vs torch (d), its rel_l2
        qr8.STATIC_QK = True
        k8s, ks = qr8.qknorm_rope_fp8(q, k, v, cap["qn"], cap["kn"], cap["cs"], cap["pos"], sm)
        qr8.STATIC_QK = False
        ksc = ks.float()
        u8_ = torch.uint8
        res["dk_q_bytes_eq_d"] = bool(torch.equal(k8s[0].view(u8_), qd.view(u8_)))
        res["dk_k_bytes_eq_d"] = bool(torch.equal(k8s[1].view(u8_), kd.view(u8_)))
        res["dk_q_match"] = float((k8s[0].view(u8_) == qd.view(u8_)).float().mean())
        res["dk_v_eq_b"] = bool(torch.equal(k8s[2].view(u8_), v8.view(u8_)))
        res["dk_scales_eq_d"] = bool(ksc[0].item() == sdq.item() and ksc[1].item() == sdk.item()
                                     and ksc[2].item() == s2)
        odk = sdpa(k8s[0].float() * ksc[0].item(), k8s[1].float() * ksc[1].item(),
                   k8s[2].float() * ksc[2].item(), sm)
        res["rel_dk"] = rel(odk, ref)
        res["dk_over_b"] = res["rel_dk"] / rb
    tiny = 2.0 ** -6
    res["subn_q_b"] = float(((qr.float() * fq).abs() < tiny).float().mean())
    res["subn_q_d"] = float(((qr.float() * fdq).abs() < tiny).float().mean())
    res["subn_k_b"] = float(((kr.float() * fk).abs() < tiny).float().mean())
    res["subn_k_d"] = float(((kr.float() * fdk).abs() < tiny).float().mean())
    res["qin_rel_b"] = rel(q8.float() * s0, qr.float())
    res["qin_rel_d"] = rel(qd.float() * sdq.item(), qr.float())
    res["kin_rel_b"] = rel(k8.float() * s1, kr.float())
    res["kin_rel_d"] = rel(kd.float() * sdk.item(), kr.float())
    for r in RS:
        a1, g1 = scale_of(amq * r)
        a2, g2 = scale_of(amk * r)
        q1, _ = quant(qr, g1.item())
        k1, _ = quant(kr, g2.item())
        o1 = sdpa(q1.float() * a1.item(), k1.float() * a2.item(), v8.float() * s2, sm)
        res[f"rel_b{r}"] = rel(o1, ref)
        res[f"b{r}_over_b"] = res[f"rel_b{r}"] / rb
    return res


here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **engine,
)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None}")
if runner is not None:
    runner.enabled = False


def generate(prompt, seed):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
        with autotune_context(pipe.device, pipe.rank, tune=False):
            pipe._prepare_t2i(ctx=ctx, height=1024, width=1024, num_inference_steps=16,
                              seed=seed)
    torch.cuda.synchronize()
    return ctx[0].shape[1]


PROMPTS = [("A", "A red fox standing in fresh snow at sunrise", 11),
           ("B", "A vintage travel poster that says VISIT MARS in bold letters", 108)]
report = {"requests": []}
generate(PROMPTS[0][1], PROMPTS[0][2])
log("warm-up eager request done")
for tag, prompt, seed in PROMPTS:
    state.update(on=True, n=0, calls=[], caps=[])
    T = generate(prompt, seed)
    state["on"] = False
    log(f"{tag}: T={T}, routed calls {state['n']}, captures {len(state['caps'])}")
    caps = []
    with torch.inference_mode():
        for cap in state["caps"]:
            r = analyze(cap)
            caps.append(r)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in r.items()}))
    state["caps"] = []
    torch.cuda.empty_cache()
    layers = []
    for layer in range(32):
        cl = [c for c in state["calls"] if c["layer"] == layer]
        if not cl:
            continue
        wq, wk = cl[0]["maxw_q"], cl[0]["maxw_k"]
        Bq, Bk = bound(wq).item(), bound(wk).item()
        amq = max(c["q_descale"] for c in cl) * 448.0
        amk = max(c["k_descale"] for c in cl) * 448.0
        amq_min = min(c["q_descale"] for c in cl) * 448.0
        amk_min = min(c["k_descale"] for c in cl) * 448.0
        layers.append({"layer": layer, "M": cl[0]["M"], "maxw_q": wq, "maxw_k": wk,
                       "B_q": Bq, "B_k": Bk, "amax_q_max": amq, "amax_k_max": amk,
                       "binades_q_min": math.log2(Bq / amq),
                       "binades_k_min": math.log2(Bk / amk),
                       "binades_q_max": math.log2(Bq / amq_min),
                       "binades_k_max": math.log2(Bk / amk_min),
                       "weights_const": all(c["maxw_q"] == wq and c["maxw_k"] == wk
                                            for c in cl)})
    for l in layers:
        log("LAYER", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in l.items()}))
    db = [c["d_over_b"] for c in caps]
    summ = {"tag": tag, "T": T, "routed": state["n"], "n_caps": len(caps),
            "mean_d_over_b": sum(db) / len(db), "max_d_over_b": max(db),
            "min_d_over_b": min(db), "sat_d_total": sum(c["sat_d"] for c in caps),
            "n_d_worse": sum(1 for x in db if x > 1.0),
            "mean_rel_b": sum(c["rel_b"] for c in caps) / len(caps),
            "bt_bytes_match_q_min": min(c["bt_bytes_match_q"] for c in caps),
            "bt_scales_match_all": all(c["bt_scales_match"] for c in caps)}
    if HAS_STATIC:
        dk = [c["dk_over_b"] for c in caps]
        summ.update({"mean_dk_over_b": sum(dk) / len(dk), "max_dk_over_b": max(dk),
                     "dk_bytes_eq_d_all": all(c["dk_q_bytes_eq_d"] and c["dk_k_bytes_eq_d"]
                                              for c in caps),
                     "dk_v_eq_b_all": all(c["dk_v_eq_b"] for c in caps),
                     "dk_scales_eq_d_all": all(c["dk_scales_eq_d"] for c in caps)})
    for r in RS:
        x = [c[f"b{r}_over_b"] for c in caps]
        summ[f"mean_b{r}_over_b"] = sum(x) / len(x)
        summ[f"max_b{r}_over_b"] = max(x)
    summ["gate_pass"] = (summ["mean_d_over_b"] <= 1.05 and summ["max_d_over_b"] <= 1.20
                         and summ["sat_d_total"] == 0 and len(caps) == 18)
    log("SUMMARY", json.dumps(summ))
    report["requests"].append({"summary": summ, "captures": caps, "layers": layers})

with open(os.path.join(ART, "step0.json"), "w") as f:
    json.dump(report, f, indent=1)
first = report["requests"][0]["summary"]
print(json.dumps({"passed": bool(first["gate_pass"]),
                  "gate": "plan 1873: mean d/b <= 1.05, max d/b <= 1.20, 0 saturated (prompt A)",
                  "summaries": [r["summary"] for r in report["requests"]]}))
