# Correctness check (untimed): trtllm::nvfp4_w4a16_gemv through the real FP8 Linear
# (FP8QDQLinearMethod.apply), decode rows 1..cap, where cap is the module's per-shape GEMV row cap
# (linear._NVFP4_DECODE_GEMV_ROW_CAPS: (5120,6144) 3, (16384,5120) 3, (14336,5120) 2), against
#  - the FP32 reference of the original op (FP8 weight * per-tensor scale, FP64 matmul),
#  - Marlin on the same NVFP4 decode copy (what the base runs for these rows),
#  - the NVFP4-dequant FP64 reference (same values, exact math).
# Pass criteria per shape and M (several input draws):
#  GEMV max/mean abs error vs FP32 ref <= Marlin's + 1 BF16 ulp (ulp at max |ref|);
#  10 warmup + 50 NaN-poisoned CUDA-graph replays bitwise equal to replay 1 (== eager);
#  TRTLLM_NVFP4_DECODE_GEMV=0 builds no GEMV copy and is bitwise Marlin at rows 1..16;
#  TRTLLM_NVFP4_DECODE_GEMV_INPROJ=0 builds no GEMV copy for the in_proj shapes (bitwise Marlin at
#  rows 1..16) and changes nothing for (5120,6144) (bitwise the default module at rows 1..16);
#  rows above the cap (cap+1..16) on the GEMV module are bitwise the kill-switch module;
#  a misaligned input (data_ptr % 16 != 0) at rows <= cap routes to Marlin, not the GEMV (op spies),
#  and equals Marlin on the same values bitwise.
# Usage: check.py [N,K ...]   (default 5120,6144 16384,5120 14336,5120)
import json
import os
import subprocess
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
os.environ.pop("TRTLLM_NVFP4_DECODE_GEMV", None)
os.environ.pop("TRTLLM_NVFP4_DECODE_GEMV_INPROJ", None)
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
SHAPES = ([tuple(int(v) for v in a.split(",")) for a in sys.argv[1:] if "," in a]
          or [(5120, 6144), (16384, 5120), (14336, 5120)])
EXPECTED_CAP = {(5120, 6144): 3, (16384, 5120): 3, (14336, 5120): 2}
INPROJ = ((16384, 5120), (14336, 5120))
WARMUP, ITERS = 10, 50
GEMV_ENV = "TRTLLM_NVFP4_DECODE_GEMV"
INPROJ_ENV = "TRTLLM_NVFP4_DECODE_GEMV_INPROJ"


def tree_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        return h
    except Exception:  # noqa: BLE001
        return "unknown"


def make_lin(n, k, seed, env):
    for key in (GEMV_ENV, INPROJ_ENV):
        os.environ.pop(key, None)
    os.environ.update(env)
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
    for key in (GEMV_ENV, INPROJ_ENV):
        os.environ.pop(key, None)
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


def misaligned_check(lin_on, lin_off, k, cap):
    """Rows <= cap with data_ptr % 16 != 0: apply must call Marlin, never the GEMV."""
    ns = torch.ops.trtllm
    real_gemv = ns.nvfp4_w4a16_gemv
    real_marlin = ns.marlin_nvfp4_gemm
    calls = {"gemv": 0, "marlin": 0, "marlin_misaligned": 0}

    def spy_gemv(*a, **kw):
        calls["gemv"] += 1
        return real_gemv(*a, **kw)

    def spy_marlin(x, *a, **kw):
        calls["marlin"] += 1
        if x.data_ptr() % 16 != 0:
            calls["marlin_misaligned"] += 1
            x = x.clone()  # Marlin's cp.async also needs 16-B aligned A; run it on the same values
        return real_marlin(x, *a, **kw)

    ok = True
    recs = {}
    setattr(ns, "nvfp4_w4a16_gemv", spy_gemv)
    setattr(ns, "marlin_nvfp4_gemm", spy_marlin)
    try:
        for m in range(1, cap + 1):
            gen = torch.Generator(device=dev).manual_seed(700 + m)
            buf = (torch.randn(m * k + 8, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            x = buf[3:3 + m * k].view(m, k)  # contiguous, storage offset 3 elements = 6 bytes
            before = dict(calls)
            out = lin_on(x)
            torch.cuda.synchronize()
            d_gemv = calls["gemv"] - before["gemv"]
            d_mis = calls["marlin_misaligned"] - before["marlin_misaligned"]
            ref = real_marlin(x.clone(), lin_on._nvfp4_decode_weight, scale_a=None,
                              scale_b=lin_on._nvfp4_decode_scale, alpha=None,
                              weight_global_scale=lin_on._nvfp4_decode_global_scale, bias=None,
                              out_dtype=torch.bfloat16, size_n=lin_on.out_features,
                              size_k=lin_on.in_features, workspace_mode=0)
            off = lin_off(x.clone())
            torch.cuda.synchronize()
            r = dict(ptr_mod16=int(x.data_ptr() % 16), contiguous=bool(x.is_contiguous()),
                     gemv_calls=d_gemv, marlin_misaligned_calls=d_mis,
                     bitwise_marlin=bool(torch.equal(bits(out), bits(ref))),
                     bitwise_killswitch=bool(torch.equal(bits(out), bits(off))))
            r["ok"] = bool(r["ptr_mod16"] != 0 and r["contiguous"] and d_gemv == 0 and d_mis == 1
                           and r["bitwise_marlin"] and r["bitwise_killswitch"])
            # The aligned control on the same module does take the GEMV.
            before = dict(calls)
            lin_on(x.clone())
            torch.cuda.synchronize()
            r["aligned_control_gemv_calls"] = calls["gemv"] - before["gemv"]
            r["ok"] &= r["aligned_control_gemv_calls"] == 1
            ok &= r["ok"]
            recs[f"M{m}"] = r
    finally:
        setattr(ns, "nvfp4_w4a16_gemv", real_gemv)
        setattr(ns, "marlin_nvfp4_gemm", real_marlin)
    return ok, recs


def check_shape(n, k):
    res = {}
    ok_all = True
    lin_on, w8, s = make_lin(n, k, 1234, {})
    lin_off, _, _ = make_lin(n, k, 1234, {GEMV_ENV: "0"})
    lin_ip, _, _ = make_lin(n, k, 1234, {INPROJ_ENV: "0"})
    gemv = lin_on.__dict__.get("_nvfp4_decode_gemv")
    cap = int(lin_on.__dict__.get("_nvfp4_decode_gemv_max_rows", 0))
    res["gemv_copy_built"] = gemv is not None
    res["gemv_max_rows"] = cap
    res["cap_expected"] = EXPECTED_CAP.get((n, k))
    res["killswitch_no_copy"] = (lin_off.__dict__.get("_nvfp4_decode_gemv") is None
                                 and lin_off.__dict__.get("_nvfp4_decode_gemv_max_rows", 0) == 0)
    if (n, k) in INPROJ:
        res["inproj_switch_no_copy"] = lin_ip.__dict__.get("_nvfp4_decode_gemv") is None
    else:
        res["inproj_switch_no_copy"] = None
        res["inproj_switch_keeps_copy"] = (
            lin_ip.__dict__.get("_nvfp4_decode_gemv") is not None
            and lin_ip.__dict__.get("_nvfp4_decode_gemv_max_rows") == cap)
    ok_all &= (res["gemv_copy_built"] and res["killswitch_no_copy"] and cap == res["cap_expected"]
               and (res["inproj_switch_no_copy"] is not False)
               and res.get("inproj_switch_keeps_copy", True))
    # All modules hold the identical Marlin copy.
    same_marlin = all(torch.equal(lin_on._nvfp4_decode_weight, o._nvfp4_decode_weight)
                      and torch.equal(lin_on._nvfp4_decode_scale, o._nvfp4_decode_scale)
                      for o in (lin_off, lin_ip))
    res["same_marlin_copy"] = bool(same_marlin)
    ok_all &= same_marlin
    w_fp32ref = w8.double() * s.double()
    w_nvfp4 = dequant_nvfp4(gemv, n, k)
    for m in range(1, cap + 1):
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
        eq_eager_all = True
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
                eq_eager_all &= bool(torch.equal(bits(gout), bits(eager)))
                if not same:
                    max_dev = max(max_dev, (gout.float() - first.float()).abs().nan_to_num(1e30).max().item())
        rec["replays"] = WARMUP + ITERS
        rec["replays_bitwise_stable"] = stable
        rec["replays_bitwise_eager"] = eq_eager_all
        rec["replay_max_dev"] = max_dev
        rec["ok"] = bool(rec["err_ok"] and rec["killswitch_bitwise_marlin"] and rec["finite"]
                         and rec["input3d_bitwise"] and rec["replay1_eq_eager"] and stable
                         and eq_eager_all)
        ok_all &= rec["ok"]
        res[f"M{m}"] = rec
        log(f"[check] N={n} K={k} M={m}: " + json.dumps(rec))
    # Rows above the GEMV cap stay on Marlin, bitwise; both kill switches vs main's path at 1..16.
    above = True
    ks_all = True
    ip_all = True
    for m in range(1, 17):
        gen = torch.Generator(device=dev).manual_seed(500 + m)
        x = (torch.randn(m, k, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        o_on, o_off, o_ip = lin_on(x), lin_off(x), lin_ip(x)
        od = marlin_direct(lin_on, x)
        torch.cuda.synchronize()
        if m > cap:
            above &= bool(torch.equal(bits(o_on), bits(o_off)))
        ks_all &= bool(torch.equal(bits(o_off), bits(od)))
        if (n, k) in INPROJ:
            ip_all &= bool(torch.equal(bits(o_ip), bits(o_off)))
        else:
            ip_all &= bool(torch.equal(bits(o_ip), bits(o_on)))
    res["rows_above_cap_bitwise_marlin"] = above
    res["killswitch_gemv0_bitwise_marlin_rows_1_16"] = ks_all
    res["inproj0_bitwise_main_rows_1_16"] = ip_all
    ok_all &= above and ks_all and ip_all
    log(f"[check] N={n} K={k} rows {cap + 1}..16 bitwise Marlin: {above}; GEMV=0 bitwise Marlin 1..16: "
        f"{ks_all}; INPROJ=0 bitwise main 1..16: {ip_all}")
    mis_ok, mis = misaligned_check(lin_on, lin_off, k, cap)
    res["misaligned"] = mis
    res["misaligned_ok"] = mis_ok
    ok_all &= mis_ok
    log(f"[check] N={n} K={k} misaligned input -> Marlin: {mis_ok} " + json.dumps(mis))
    res["ok"] = bool(ok_all)
    del lin_on, lin_off, lin_ip, w8, w_fp32ref, w_nvfp4
    torch.cuda.empty_cache()
    return res


def main():
    for word in sys.argv[1:]:
        if "," not in word:
            log(word)
    log(f"TREE_COMMIT={tree_commit()}")
    log(f"[check] shapes {SHAPES}, has op: {hasattr(torch.ops.trtllm, 'nvfp4_w4a16_gemv')}, caps "
        f"{getattr(L, '_NVFP4_DECODE_GEMV_ROW_CAPS', None)}")
    out = {}
    passed = True
    for n, k in SHAPES:
        r = check_shape(n, k)
        out[f"{n}x{k}"] = r
        passed &= r["ok"]
    print(json.dumps({"passed": bool(passed),
                      "compared": "nvfp4_w4a16_gemv via FP8QDQLinearMethod.apply vs FP32 ref (FP8 W * scale), "
                                  "Marlin (base path), both kill switches, rows above cap, misaligned input, "
                                  "NaN-poisoned graph replays",
                      "shapes": out}), flush=True)


if __name__ == "__main__":
    main()
