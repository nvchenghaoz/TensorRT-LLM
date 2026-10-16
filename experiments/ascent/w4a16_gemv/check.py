# Branch 63 correctness check (untimed): trtllm::nvfp4_w4a16_gemv through the real
# FP8 Linear (FP8QDQLinearMethod.apply), decode rows 1..4, against
#  - the FP32 reference of the original op (FP8 weight * per-tensor scale, FP64 matmul),
#  - Marlin on the same NVFP4 decode copy (what the base runs for these rows),
#  - the NVFP4-dequant FP64 reference (same values, exact math).
# Pass criteria per shape and M (several input draws):
#  GEMV max/mean abs error vs FP32 ref <= Marlin's + 1 BF16 ulp (ulp at max |ref|);
#  10 warmup + 50 NaN-poisoned CUDA-graph replays bitwise equal to replay 1 (== eager);
#  kill switch (TRTLLM_NVFP4_DECODE_GEMV=0) builds no GEMV copy and is bitwise Marlin;
#  rows 5..16 on the GEMV module are bitwise the kill-switch module (Marlin unchanged).
# Usage: check.py [N,K ...]   (default 5120,6144)
import json
import os
import subprocess
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
SHAPES = [tuple(int(v) for v in a.split(",")) for a in sys.argv[1:]] or [(5120, 6144)]
WARMUP, ITERS = 10, 50


def tree_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        return h
    except Exception:  # noqa: BLE001
        return "unknown"


def make_lin(n, k, seed, gemv_on):
    os.environ[L._NVFP4_DECODE_GEMV_ENV] = "1" if gemv_on else "0"
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=gen, device=dev) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0  # a few outlier input channels
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(k, n, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                       "input_scale": torch.tensor(0.04, device=dev)}])
    lin.post_load_weights()
    os.environ[L._NVFP4_DECODE_GEMV_ENV] = "1"
    return lin, w8, s


def dequant_nvfp4(gemv, n, k):
    wq, sf, g = gemv
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                       device=dev, dtype=torch.float64)
    v = torch.stack([lut[(wq & 0xF).long()], lut[(wq >> 4).long()]], dim=-1).reshape(n, k)
    return v * sf.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1) * g


def marlin_direct(lin, x):
    return torch.ops.trtllm.marlin_nvfp4_gemm(
        x, lin._nvfp4_decode_weight, scale_a=None, scale_b=lin._nvfp4_decode_scale, alpha=None,
        weight_global_scale=lin._nvfp4_decode_global_scale, bias=None, out_dtype=torch.bfloat16,
        size_n=lin.out_features, size_k=lin.in_features, workspace_mode=0)


def bits(t):
    return t.contiguous().view(torch.int16)


def bf16_ulp(v):
    e = torch.floor(torch.log2(torch.tensor(abs(v), dtype=torch.float64).clamp(min=1e-30)))
    return float(2.0 ** (e - 7))


def check_shape(n, k):
    res = {}
    ok_all = True
    lin_on, w8, s = make_lin(n, k, 1234, True)
    lin_off, _, _ = make_lin(n, k, 1234, False)
    gemv = lin_on.__dict__.get("_nvfp4_decode_gemv")
    res["gemv_copy_built"] = gemv is not None
    res["killswitch_no_copy"] = lin_off.__dict__.get("_nvfp4_decode_gemv") is None
    ok_all &= res["gemv_copy_built"] and res["killswitch_no_copy"]
    # Both modules hold the identical Marlin copy.
    same_marlin = (torch.equal(lin_on._nvfp4_decode_weight, lin_off._nvfp4_decode_weight)
                   and torch.equal(lin_on._nvfp4_decode_scale, lin_off._nvfp4_decode_scale))
    res["same_marlin_copy"] = bool(same_marlin)
    ok_all &= same_marlin
    w_fp32ref = w8.double() * s.double()
    w_nvfp4 = dequant_nvfp4(gemv, n, k)
    for m in (1, 2, 3, 4):
        rec = dict(gemv_max=0.0, gemv_mean=0.0, marlin_max=0.0, marlin_mean=0.0, gemv_vs_nvfp4ref_max=0.0,
                   marlin_vs_nvfp4ref_max=0.0, ulp=0.0, eq_marlin_frac=0.0)
        draws = 4
        for d in range(draws):
            gen = torch.Generator(device=dev).manual_seed(100 * m + d)
            x = (torch.randn(m, k, generator=gen, device=dev) * (0.3 + 0.4 * d)).to(torch.bfloat16)
            x[:, 7::97] *= 20.0
            ref = x.double() @ w_fp32ref.t()
            ref4 = x.double() @ w_nvfp4.t()
            og = lin_on(x)
            om = lin_off(x)
            od = marlin_direct(lin_on, x)
            torch.cuda.synchronize()
            # Kill switch == base path (direct Marlin call) bitwise.
            ks = bool(torch.equal(bits(om), bits(od)))
            rec.setdefault("killswitch_bitwise_marlin", True)
            rec["killswitch_bitwise_marlin"] &= ks
            eg = (og.double() - ref).abs()
            em = (om.double() - ref).abs()
            rec["gemv_max"] = max(rec["gemv_max"], eg.max().item())
            rec["marlin_max"] = max(rec["marlin_max"], em.max().item())
            rec["gemv_mean"] += eg.mean().item() / draws
            rec["marlin_mean"] += em.mean().item() / draws
            rec["gemv_vs_nvfp4ref_max"] = max(rec["gemv_vs_nvfp4ref_max"], (og.double() - ref4).abs().max().item())
            rec["marlin_vs_nvfp4ref_max"] = max(rec["marlin_vs_nvfp4ref_max"],
                                                (om.double() - ref4).abs().max().item())
            rec["ulp"] = max(rec["ulp"], bf16_ulp(ref.abs().max().item()))
            rec["eq_marlin_frac"] += float((bits(og) == bits(om)).float().mean().item()) / draws
            rec.setdefault("finite", True)
            rec["finite"] &= bool(torch.isfinite(og.float()).all().item())
            # 3D input path.
            o3 = lin_on(x.unsqueeze(0))
            rec.setdefault("input3d_bitwise", True)
            rec["input3d_bitwise"] &= bool(o3.shape == (1, m, n) and torch.equal(bits(o3[0]), bits(og)))
        rec["err_ok"] = (rec["gemv_max"] <= rec["marlin_max"] + rec["ulp"]
                         and rec["gemv_mean"] <= rec["marlin_mean"] + rec["ulp"])
        # NaN-poisoned graph replays.
        gen = torch.Generator(device=dev).manual_seed(999 + m)
        xs = (torch.randn(m, k, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        eager = lin_on(xs)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                lin_on(xs)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gout = lin_on(xs)
        torch.cuda.synchronize()
        first = None
        stable = True
        max_dev = 0.0
        for it in range(WARMUP + ITERS):
            gout.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            if first is None:
                first = gout.clone()
                rec["replay1_eq_eager"] = bool(torch.equal(bits(first), bits(eager)))
            else:
                same = bool(torch.equal(bits(gout), bits(first)))
                stable &= same
                if not same:
                    max_dev = max(max_dev, (gout.float() - first.float()).abs().nan_to_num(1e30).max().item())
        rec["replays"] = WARMUP + ITERS
        rec["replays_bitwise_stable"] = stable
        rec["replay_max_dev"] = max_dev
        rec["ok"] = bool(rec["err_ok"] and rec["killswitch_bitwise_marlin"] and rec["finite"]
                         and rec["input3d_bitwise"] and rec["replay1_eq_eager"] and stable)
        ok_all &= rec["ok"]
        res[f"M{m}"] = rec
        log(f"[check] N={n} K={k} M={m}: " + json.dumps(rec))
    # Rows above the GEMV gate stay on Marlin, bitwise.
    above = True
    for m in (5, 8, 16):
        gen = torch.Generator(device=dev).manual_seed(500 + m)
        x = (torch.randn(m, k, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        above &= bool(torch.equal(bits(lin_on(x)), bits(lin_off(x))))
    res["rows5_16_bitwise_marlin"] = above
    ok_all &= above
    log(f"[check] N={n} K={k} rows 5..16 bitwise Marlin: {above}")
    res["ok"] = bool(ok_all)
    del lin_on, lin_off, w8, w_fp32ref, w_nvfp4
    torch.cuda.empty_cache()
    return res


def main():
    log(f"TREE_COMMIT={tree_commit()}")
    log(f"[check] shapes {SHAPES}, has op: {hasattr(torch.ops.trtllm, 'nvfp4_w4a16_gemv')}")
    out = {}
    passed = True
    for n, k in SHAPES:
        r = check_shape(n, k)
        out[f"{n}x{k}"] = r
        passed &= r["ok"]
    print(json.dumps({"passed": bool(passed),
                      "compared": "nvfp4_w4a16_gemv via FP8QDQLinearMethod.apply vs FP32 ref (FP8 W * scale), "
                                  "Marlin (base path), kill switch, NaN-poisoned graph replays",
                      "shapes": out}), flush=True)


if __name__ == "__main__":
    main()
