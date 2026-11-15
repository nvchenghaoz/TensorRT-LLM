"""Idea 60 stage 0: M-chunked gate_up -> gelu_tanh_mul_fp4 chain (Gemma4 MLP).

usage: stage0.py <checkpoint> [check|time|all]

check: chunk gate_up rows (int16) == main's full-M rows and the chunked act's
       FP4 payload + full SF buffer byte-equal to main's full-M act, for
       synthetic and real (checkpoint layers 0/mid/last) weights, 2 seeds +
       heavy-tailed rows, M in {2000,4800,8000,8100}, C in {1024,1280,2560},
       gate_up forced to main's full-M autotuned pick.
time:  cold-weight chain timing (flush before every chain) per arm:
       base (full M) and chunked C x act cfg {main (8,512,8), (1,1024,4)} x
       order {fwd, rev} x L2 window {none, persist}. Reports chain, gemm sum,
       act sum, act t_in (in chain), act t_cold (after flush) and H.
"""
import inspect
import json
import os
import sys
import threading
import time

import torch
import triton
import triton.language as tl


def _hb():
    t0 = time.time()
    while True:
        time.sleep(30)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


threading.Thread(target=_hb, daemon=True).start()

CK = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
cfg = json.load(open(os.path.join(CK, "config.json")))
tc = cfg.get("text_config", cfg)
H = int(tc["hidden_size"])
I = int(tc["intermediate_size"])
NL = int(tc["num_hidden_layers"])
EPS = float(tc.get("rms_norm_eps", 1e-6))
try:
    hq = json.load(open(os.path.join(CK, "hf_quant_config.json")))
    print("hf_quant_config:", json.dumps(hq.get("quantization", hq))[:400], flush=True)
except Exception as e:  # noqa: BLE001
    print("hf_quant_config unreadable", e)
N, K2 = 2 * I, H // 2
NKT_IN = (H // 16 + 3) // 4
SF_IN_BLK = NKT_IN * 512  # SF bytes per 128 rows of the gate_up input
NKT_ACT = (I // 16 + 3) // 4
print(f"hidden_size={H} intermediate_size={I} layers={NL} eps={EPS} N={N} K={H} "
      f"nkt_in={NKT_IN} sf_in_per128={SF_IN_BLK} nkt_act={NKT_ACT}", flush=True)

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops import gelu_tanh_mul_fp4_quant as gm  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import (  # noqa: E402
    rmsnorm_fp4_quant, rmsnorm_fp4_quant_available)

dev = "cuda"
BACKENDS = ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
OP = "trtllm::nvfp4_gemm::gemm"
MS = [2000, 4800, 8000, 8100]
CS = [1024, 1280, 2560]
print("gpu:", torch.cuda.get_device_name(), torch.cuda.get_device_capability(),
      "cute rmsnorm_fp4:", rmsnorm_fp4_quant_available(), flush=True)

# ---------------------------------------------------------------- act kernel copy
@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _act_kernel_ord(x_ptr, out_ptr, sf_ptr, gs_ptr, M, SX, NKT, IDIM: tl.constexpr,
                    BM: tl.constexpr, BK: tl.constexpr, REV: tl.constexpr):
    # verbatim copy of main's _gelu_tanh_mul_fp4_kernel except the block order
    # (REV walks the launch's linear block order backwards).
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    if REV:
        p0 = tl.num_programs(0) - 1 - p0
        p1 = tl.num_programs(1) - 1 - p1
    rows = p0 * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = p1 * BK + tl.arange(0, BK)
    mask = rmask[:, None] & (cols[None, :] < IDIM)
    gate = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + rows[:, None] * SX + IDIM + cols[None, :], mask=mask, other=0.0).to(
        tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner], dtype=tl.float32, is_pure=True, pack=1)
    act = gate * (0.5 * (1.0 + t))
    v = (act * up).to(tl.bfloat16).to(tl.float32)
    v16 = tl.reshape(v, (BM, BK // 16, 16))
    vmax = tl.max(tl.abs(v16), axis=2)
    gs = tl.load(gs_ptr)
    rcp6 = _rcp_approx(tl.full((BM, BK // 16), 6.0, tl.float32))
    sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
    rcpgs = _rcp_approx(tl.full((BM, BK // 16), 0.0, tl.float32) + gs)
    oscale = _rcp_approx(sf8.to(tl.float32) * rcpgs)
    oscale = tl.where(vmax != 0.0, oscale, 0.0)
    kvec = (p1 * (BK // 16) + tl.arange(0, BK // 16)).to(tl.int64)
    kvmask = rmask[:, None] & (kvec[None, :] * 16 < IDIM)
    m2 = rows[:, None]
    k2 = kvec[None, :]
    sfoff = ((m2 // 128) * (NKT * 512) + (k2 // 4) * 512 + (m2 % 32) * 16
             + ((m2 % 128) // 32) * 4 + (k2 % 4))
    tl.store(sf_ptr + sfoff, sf8.to(tl.uint8, bitcast=True), mask=kvmask)
    y = tl.reshape(v16 * oscale[:, :, None], (BM, BK))
    lo, hi = tl.split(tl.reshape(y, (BM, BK // 2, 2)))
    byte = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        "=h,f,f", [lo, hi], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)
    ocols = p1 * (BK // 2) + tl.arange(0, BK // 2)
    omask = rmask[:, None] & (ocols[None, :] * 2 < IDIM)
    tl.store(out_ptr + rows[:, None] * (IDIM // 2) + ocols[None, :], byte, mask=omask)


ACT_CFGS = {"main": (8, 512, 8), "b1k1024w4": (1, 1024, 4)}


def act_into(x, gs, out, sf, cfgname, rev):
    """Act on x [m, 2I] writing into out [m, I/2] and sf (pad128(m) rows)."""
    bm, bk, nw = ACT_CFGS[cfgname]
    m = x.shape[0]
    if not rev and "out" in inspect.signature(gm.gelu_tanh_mul_fp4_quant).parameters:
        # post-landing-7 tree: the production wrapper (its own SM120 config)
        gm.gelu_tanh_mul_fp4_quant(x, gs, out=out, sf_out=sf)
    elif cfgname == "main" and not rev:
        gm._gelu_tanh_mul_fp4_kernel[(triton.cdiv(m, bm), triton.cdiv(I, bk))](
            x, out, sf, gs, m, x.stride(0), NKT_ACT, IDIM=I, BM=bm, BK=bk, num_warps=nw)
    else:
        _act_kernel_ord[(triton.cdiv(m, bm), triton.cdiv(I, bk))](
            x, out, sf, gs, m, x.stride(0), NKT_ACT, IDIM=I, BM=bm, BK=bk, REV=rev, num_warps=nw)


def sf_act_size(m):
    return ((m + 127) // 128) * 128 * NKT_ACT * 4


# ---------------------------------------------------------------- weights / inputs
def load_real(li):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]

    def key(sub):
        ks = [k for k in idx if k.endswith(f".layers.{li}.{sub}")]
        lm = [k for k in ks if "language_model" in k]
        return (lm or ks)[0]

    def get(sub):
        k = key(sub)
        with safe_open(os.path.join(CK, idx[k]), framework="pt", device="cpu") as f:
            return f.get_tensor(k)
    wg, wu = get("mlp.gate_proj.weight"), get("mlp.up_proj.weight")
    sg, su = get("mlp.gate_proj.weight_scale"), get("mlp.up_proj.weight_scale")
    g2, u2 = get("mlp.gate_proj.weight_scale_2").float(), get("mlp.up_proj.weight_scale_2").float()
    ig = get("mlp.gate_proj.input_scale").float()
    idn = get("mlp.down_proj.input_scale").float()
    nw = get("pre_feedforward_layernorm.weight").to(torch.bfloat16)
    print(f"layer {li}: key={key('mlp.gate_proj.weight')} w {tuple(wg.shape)} {wg.dtype} "
          f"ws {tuple(sg.shape)} {sg.dtype} ws2 {g2.item():.4g}/{u2.item():.4g} in {ig.item():.4g} "
          f"down_in {idn.item():.4g}", flush=True)
    w = torch.cat([wg, wu], 0).view(torch.uint8).contiguous().to(dev)
    ws = torch.ops.trtllm.block_scale_interleave(torch.cat([sg, su], 0).view(torch.uint8).contiguous().to(dev))
    ws2 = torch.maximum(g2, u2)
    alpha = (ig * ws2).reshape(1).to(dev)
    return dict(name=f"real_l{li}", w=w, ws=ws, alpha=alpha,
                gs_in=(1.0 / ig).reshape(1).to(dev), gs_down=(1.0 / idn).reshape(1).to(dev),
                nw=nw.to(dev))


def synth(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (N, K2), dtype=torch.uint8, device=dev, generator=g)
    sl = (torch.rand(N, H // 16, device=dev, generator=g) * 3.5 + 0.25).to(torch.float8_e4m3fn)
    ws = torch.ops.trtllm.block_scale_interleave(sl.view(torch.uint8))
    nw = (torch.rand(H, device=dev, generator=g) * 2 + 0.05).to(torch.bfloat16)
    return dict(name=f"synth{seed}", w=w, ws=ws, alpha=torch.tensor([3e-4], device=dev),
                gs_in=torch.tensor([448 * 6 / 30.0], device=dev),
                gs_down=torch.tensor([448 * 6 / 8.0], device=dev), nw=nw)


def make_x(m, seed, kind, lw):
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + m)
    x = torch.randn(m, H, device=dev, generator=g)
    if kind == "heavy":
        x = x * torch.exp2(torch.rand(m, 1, device=dev, generator=g) * 16 - 8)
        x = x * torch.exp(torch.randn(H, device=dev, generator=g))
    x = x.to(torch.bfloat16)
    if rmsnorm_fp4_quant_available():
        fp4, sf = rmsnorm_fp4_quant(x, lw["nw"], EPS, lw["gs_in"])
    else:
        fp4, sf = torch.ops.trtllm.fp4_quantize(x, lw["gs_in"], 16, False)
    return fp4.view(torch.uint8), sf.view(torch.uint8).reshape(-1)


# ---------------------------------------------------------------- gemm
RUNNER = NVFP4GemmUnifiedRunner(0, torch.bfloat16, BACKENDS)


def tune(lw, fp4, sf):
    with autotune():
        torch.ops.trtllm.nvfp4_gemm(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16,
                                    allowed_backends=",".join(BACKENDS))
    torch.cuda.synchronize()


def full_pick(lw, m, sf):
    shapes = (torch.Size([m, K2]), lw["w"].size(), sf.size(), lw["ws"].size(), lw["alpha"].size())
    hit, rid, tac, t = AutoTuner.get().profiling_cache.search_cache(
        OP, [RUNNER], shapes, NVFP4GemmUnifiedRunner.tuning_config)
    return hit, tac


def main_gemm(lw, fp4, sf):
    return torch.ops.trtllm.nvfp4_gemm(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16,
                                       allowed_backends=",".join(BACKENDS))


def chunk_gemm(lw, fp4, sf, r0, r1, tac):
    sfc = sf[(r0 // 128) * SF_IN_BLK: ((r1 + 127) // 128) * SF_IN_BLK]
    return RUNNER.forward([fp4[r0:r1], lw["w"], sfc, lw["ws"], lw["alpha"]], tactic=tac)


def main_chain(lw, fp4, sf):
    h1 = main_gemm(lw, fp4, sf)
    return gm.gelu_tanh_mul_fp4_quant(h1, lw["gs_down"]), h1


def chunk_chain(lw, fp4, sf, C, tac, cfg="main", rev=False):
    m = fp4.shape[0]
    out = torch.empty((m, I // 2), dtype=torch.uint8, device=dev)
    sfo = torch.zeros((sf_act_size(m),), dtype=torch.uint8, device=dev)
    for r0 in range(0, m, C):
        r1 = min(m, r0 + C)
        h1 = chunk_gemm(lw, fp4, sf, r0, r1, tac)
        act_into(h1, lw["gs_down"], out[r0:r1], sfo[(r0 // 128) * NKT_ACT * 512:
                                                     ((r1 + 127) // 128) * NKT_ACT * 512], cfg, rev)
        del h1
    return out, sfo


# ---------------------------------------------------------------- check
def check():
    res = {"cases": 0, "gemm_rows_equal": 0, "act_equal": 0, "fail": []}
    picks = {}
    lws = [synth(1)]
    for li in sorted({0, NL // 2, NL - 1}):
        try:
            lws.append(load_real(li))
        except Exception as e:  # noqa: BLE001
            print("real weights unavailable:", repr(e), flush=True)
            break
    for lw in lws:
        for m in MS:
            for seed, kind in ((1, "randn"), (2, "randn"), (3, "heavy")):
                fp4, sf = make_x(m, seed, kind, lw)
                if seed == 1:
                    tune(lw, fp4, sf)
                hit, tac = full_pick(lw, m, sf)
                picks[f"{lw['name']}@{m}"] = [hit, str(tac)]
                (o_ref, s_ref), h_ref = main_chain(lw, fp4, sf)
                h_ref16 = h_ref.view(torch.int16)
                for C in CS:
                    ok_g = True
                    for r0 in range(0, m, C):
                        r1 = min(m, r0 + C)
                        hc = chunk_gemm(lw, fp4, sf, r0, r1, tac)
                        if not torch.equal(hc.view(torch.int16), h_ref16[r0:r1]):
                            d = (hc.view(torch.int16) != h_ref16[r0:r1]).nonzero()
                            ok_g = False
                            res["fail"].append(f"gemm {lw['name']} m={m} s={seed} C={C} rows {r0}: "
                                               f"{d.shape[0]} diffs first {d[0].tolist()}")
                            break
                    for cfg in ("main", "b1k1024w4"):
                        for rev in (False, True):
                            o, s = chunk_chain(lw, fp4, sf, C, tac, cfg, rev)
                            ok_a = torch.equal(o, o_ref) and torch.equal(s, s_ref)
                            if not ok_a:
                                res["fail"].append(
                                    f"act {lw['name']} m={m} s={seed} C={C} cfg={cfg} rev={rev}: "
                                    f"payload_eq={torch.equal(o, o_ref)} sf_eq={torch.equal(s, s_ref)}")
                            res["cases"] += 1
                            res["gemm_rows_equal"] += ok_g
                            res["act_equal"] += ok_a
                print(f"{lw['name']} m={m} seed={seed}/{kind} pick={tac} hit={hit} cases={res['cases']} "
                      f"gemm_eq={res['gemm_rows_equal']} act_eq={res['act_equal']}", flush=True)
                del h_ref, h_ref16, o_ref, s_ref
        torch.cuda.empty_cache()
    res["picks"] = picks
    res["fail"] = res["fail"][:20]
    res["passed"] = res["cases"] == res["gemm_rows_equal"] == res["act_equal"] and res["cases"] > 0
    print("CHECK:", json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------- L2 window
try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 12.x
except Exception:  # noqa: BLE001
    try:
        from cuda import cudart
    except Exception:  # noqa: BLE001
        cudart = None
print("cudart for L2 window:", cudart is not None, flush=True)
SETASIDE = 80 * 1024 * 1024
if cudart is not None:
    _ATTR_ENUM = getattr(cudart, "cudaStreamAttrID", None) or cudart.cudaLaunchAttributeID
    APW = (getattr(_ATTR_ENUM, "cudaStreamAttributeAccessPolicyWindow", None)
           or getattr(_ATTR_ENUM, "cudaLaunchAttributeAccessPolicyWindow"))


def l2_setup():
    if cudart is None:
        return False
    try:
        r = cudart.cudaDeviceSetLimit(cudart.cudaLimit.cudaLimitPersistingL2CacheSize, SETASIDE)
        r2 = cudart.cudaDeviceGetLimit(cudart.cudaLimit.cudaLimitPersistingL2CacheSize)
        print("persisting L2 limit set:", r, r2, flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print("L2 setup failed", repr(e), flush=True)
        return False


def set_window(ptr, nbytes, hit_ratio, persist=True):
    st = torch.cuda.current_stream().cuda_stream
    v = cudart.cudaStreamAttrValue()
    v.accessPolicyWindow.base_ptr = ptr
    v.accessPolicyWindow.num_bytes = nbytes
    v.accessPolicyWindow.hitRatio = hit_ratio
    v.accessPolicyWindow.hitProp = (cudart.cudaAccessProperty.cudaAccessPropertyPersisting if persist
                                    else cudart.cudaAccessProperty.cudaAccessPropertyNormal)
    v.accessPolicyWindow.missProp = cudart.cudaAccessProperty.cudaAccessPropertyStreaming
    r = cudart.cudaStreamSetAttribute(st, APW, v)
    if r[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamSetAttribute: {r}")
    return r


def clear_window(reset=True):
    st = torch.cuda.current_stream().cuda_stream
    v = cudart.cudaStreamAttrValue()
    v.accessPolicyWindow.num_bytes = 0
    cudart.cudaStreamSetAttribute(st, APW, v)
    if reset:
        cudart.cudaCtxResetPersistingL2Cache()


def timing_win():
    """Window-focused arms: fwd order only, C in {1280,1536,2048}; window modes
    none / reset (persist on gemm+act, ctx reset after) / compliant (persist on
    the gemm write, hitProp=Normal on the act read so hits demote the lines,
    num_bytes=0 after the MLP, never a reset)."""
    lw = synth(11)
    l2_setup()
    try:
        mw = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize, 0)
        print("max access policy window bytes:", mw, flush=True)
        maxw = int(mw[1])
    except Exception as e:  # noqa: BLE001
        print("max window query failed", repr(e), flush=True)
        maxw = 128 * 1024 * 1024
    table = []
    for m in MS:
        fp4, sf = make_x(m, 1, "randn", lw)
        tune(lw, fp4, sf)
        hit, tac = full_pick(lw, m, sf)
        out = torch.empty((m, I // 2), dtype=torch.uint8, device=dev)
        sfo = torch.zeros((sf_act_size(m),), dtype=torch.uint8, device=dev)
        for cfg in ("b1k1024w4", "main"):
            def base(rec, cfg=cfg):
                h1 = main_gemm(lw, fp4, sf)
                rec("g")
                act_into(h1, lw["gs_down"], out, sfo, cfg, False)
                rec("a")
            s = run_timed(base)
            row = {"M": m, "arm": "base", "C": m, "cfg": cfg, "window": "none",
                   "chain_us": s["chain"]["med_us"], "chain_p10": s["chain"]["p10"],
                   "chain_p90": s["chain"]["p90"], "gemm_us": s["g"]["med_us"], "act_in_us": s["a"]["med_us"]}
            print(json.dumps(row), flush=True)
            table.append(row)
            for C in (1280, 1536, 2048):
                if C >= m:
                    continue
                ptr_holder = {}
                for win in ("none", "reset_fit", "reset_hr1", "compliant_fit", "compliant_hr1"):
                    if cfg == "main" and win.endswith("hr1"):
                        continue
                    mism = [0]

                    def arm(rec, C=C, cfg=cfg, win=win):
                        for r0 in range(0, m, C):
                            r1 = min(m, r0 + C)
                            nb = min((r1 - r0) * N * 2, maxw)
                            hr = 1.0 if win.endswith("hr1") else min(1.0, SETASIDE / nb)
                            on = win != "none" and "ptr" in ptr_holder
                            if on:
                                set_window(ptr_holder["ptr"], nb, hr, True)
                            h1 = chunk_gemm(lw, fp4, sf, r0, r1, tac)
                            if "ptr" not in ptr_holder:
                                ptr_holder["ptr"] = h1.data_ptr()
                            elif h1.data_ptr() != ptr_holder["ptr"]:
                                mism[0] += 1
                            rec("g")
                            if on and win.startswith("compliant"):
                                set_window(ptr_holder["ptr"], nb, hr, False)
                            act_into(h1, lw["gs_down"], out[r0:r1],
                                     sfo[(r0 // 128) * NKT_ACT * 512: ((r1 + 127) // 128) * NKT_ACT * 512],
                                     cfg, False)
                            rec("a")
                            del h1
                        if win != "none":
                            clear_window(reset=win.startswith("reset"))
                    try:
                        s = run_timed(arm)
                    except Exception as e:  # noqa: BLE001
                        print(f"arm M={m} C={C} cfg={cfg} win={win} failed: {e!r}", flush=True)
                        continue
                    row = {"M": m, "arm": "chunk", "C": C, "cfg": cfg, "window": win,
                           "chain_us": s["chain"]["med_us"], "chain_p10": s["chain"]["p10"],
                           "chain_p90": s["chain"]["p90"], "gemm_us": s["g"]["med_us"],
                           "act_in_us": s["a"]["med_us"], "ptr_mismatch": mism[0]}
                    print(json.dumps(row), flush=True)
                    table.append(row)
        del fp4, sf, out, sfo
        torch.cuda.empty_cache()
    return table


# ---------------------------------------------------------------- time
FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
WARM, ITERS = 5, 20
L2_RATIO = 1.5015 / 6.106


def stats(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2], v[int(0.1 * (n - 1))], v[int(0.9 * (n - 1))]


def run_timed(fn):
    """fn(rec) runs the chain, calling rec(tag) after each kernel."""
    rows = []
    for it in range(WARM + ITERS):
        FLUSH.fill_(1.0)
        evs = []

        def rec(tag):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            evs.append((tag, e))
        rec("start")
        fn(rec)
        torch.cuda.synchronize()
        if it >= WARM:
            t = {"g": [], "a": []}
            for (tag, e), (_, e0) in zip(evs[1:], evs[:-1]):
                t[tag].append(e0.elapsed_time(e) * 1000.0)
            t["chain"] = [evs[0][1].elapsed_time(evs[-1][1]) * 1000.0]
            rows.append(t)
    out = {}
    for key in ("g", "a", "chain"):
        med, p10, p90 = stats([sum(r[key]) for r in rows])
        out[key] = {"med_us": round(med, 1), "p10": round(p10, 1), "p90": round(p90, 1)}
    return out


def act_cold(h1s, gs, cfg, rev, m):
    """Act alone per chunk after a flush (cold), summed over chunks."""
    out = torch.empty((m, I // 2), dtype=torch.uint8, device=dev)
    sfo = torch.zeros((sf_act_size(m),), dtype=torch.uint8, device=dev)
    tot = []
    for it in range(3 + 10):
        s = 0.0
        r0 = 0
        for h in h1s:
            r1 = r0 + h.shape[0]
            FLUSH.fill_(1.0)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            act_into(h, gs, out[r0:r1], sfo[(r0 // 128) * NKT_ACT * 512: ((r1 + 127) // 128) * NKT_ACT * 512],
                     cfg, rev)
            e1.record()
            torch.cuda.synchronize()
            s += e0.elapsed_time(e1) * 1000.0
            r0 = r1
        if it >= 3:
            tot.append(s)
    return stats(tot)


def timing():
    lw = synth(11)
    have_l2 = l2_setup()
    table = []
    for m in MS:
        fp4, sf = make_x(m, 1, "randn", lw)
        tune(lw, fp4, sf)
        hit, tac = full_pick(lw, m, sf)
        for C in CS:
            fpc, sfc = make_x(C, 1, "randn", lw)
            tune(lw, fpc, sfc)
        cpicks = {C: str(full_pick(lw, C, sf)[1]) for C in CS}
        print(f"M={m}: main full-M pick={tac} (hit={hit}); own picks at C: {cpicks}", flush=True)
        out = torch.empty((m, I // 2), dtype=torch.uint8, device=dev)
        sfo = torch.zeros((sf_act_size(m),), dtype=torch.uint8, device=dev)

        for cfg in ACT_CFGS:
            def base(rec, cfg=cfg):
                h1 = main_gemm(lw, fp4, sf)
                rec("g")
                act_into(h1, lw["gs_down"], out, sfo, cfg, False)
                rec("a")
            s = run_timed(base)
            h1 = main_gemm(lw, fp4, sf)
            cold = act_cold([h1], lw["gs_down"], cfg, False, m)
            del h1
            t_in = s["a"]["med_us"]
            H_ = (cold[0] - t_in) / (cold[0] * (1 - L2_RATIO))
            row = {"M": m, "arm": "base", "C": m, "cfg": cfg, "order": "fwd", "window": "none",
                   "chain_us": s["chain"]["med_us"], "chain_p10": s["chain"]["p10"],
                   "chain_p90": s["chain"]["p90"], "gemm_us": s["g"]["med_us"], "act_in_us": t_in,
                   "act_cold_us": round(cold[0], 1), "H": round(H_, 3)}
            print(json.dumps(row), flush=True)
            table.append(row)

        for C in CS:
            if C >= m:
                continue
            ptr_holder = {}
            for cfg in ACT_CFGS:
                for rev in (False, True):
                    for win in (("none", "persist_hr1", "persist_fit") if have_l2 else ("none",)):
                        mism = [0]

                        def arm(rec, C=C, cfg=cfg, rev=rev, win=win):
                            for r0 in range(0, m, C):
                                r1 = min(m, r0 + C)
                                if win != "none" and "ptr" in ptr_holder:
                                    nb = (r1 - r0) * N * 2
                                    hr = 1.0 if win == "persist_hr1" else min(1.0, SETASIDE / nb)
                                    set_window(ptr_holder["ptr"], nb, hr)
                                h1 = chunk_gemm(lw, fp4, sf, r0, r1, tac)
                                if "ptr" not in ptr_holder:
                                    ptr_holder["ptr"] = h1.data_ptr()
                                elif h1.data_ptr() != ptr_holder["ptr"]:
                                    mism[0] += 1
                                rec("g")
                                act_into(h1, lw["gs_down"], out[r0:r1],
                                         sfo[(r0 // 128) * NKT_ACT * 512: ((r1 + 127) // 128) * NKT_ACT * 512],
                                         cfg, rev)
                                rec("a")
                                del h1
                            if win != "none":
                                clear_window()
                        try:
                            s = run_timed(arm)
                        except Exception as e:  # noqa: BLE001
                            print(f"arm C={C} cfg={cfg} rev={rev} win={win} failed: {e!r}", flush=True)
                            if win != "none":
                                have_l2 = False
                            continue
                        h1s = [chunk_gemm(lw, fp4, sf, r0, min(m, r0 + C), tac) for r0 in range(0, m, C)]
                        cold = act_cold(h1s, lw["gs_down"], cfg, rev, m)
                        del h1s
                        t_in = s["a"]["med_us"]
                        H_ = (cold[0] - t_in) / (cold[0] * (1 - L2_RATIO))
                        row = {"M": m, "arm": "chunk", "C": C, "cfg": cfg, "order": "rev" if rev else "fwd",
                               "window": win, "chain_us": s["chain"]["med_us"], "chain_p10": s["chain"]["p10"],
                               "chain_p90": s["chain"]["p90"], "gemm_us": s["g"]["med_us"],
                               "act_in_us": t_in, "act_cold_us": round(cold[0], 1), "H": round(H_, 3),
                               "ptr_mismatch": mism[0]}
                        print(json.dumps(row), flush=True)
                        table.append(row)
        del fp4, sf, out, sfo
        torch.cuda.empty_cache()
    return table


if __name__ == "__main__":
    result = {}
    if MODE in ("check", "all"):
        result["check"] = check()
        if not result["check"]["passed"]:
            print("BITWISE GATE FAILED -> stop", flush=True)
            print(json.dumps({"passed": False, "check": result["check"]}))
            sys.exit(0)
    if MODE in ("time", "all"):
        result["table"] = timing()
    if MODE == "win":
        result["table"] = timing_win()
    print(json.dumps({"passed": result.get("check", {}).get("passed", None),
                      "n_rows": len(result.get("table", []))}))
