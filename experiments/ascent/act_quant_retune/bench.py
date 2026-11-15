"""gelu_tanh_mul_fp4_quant: SM120 launch retune + in-kernel SF padding fill.

usage: bench.py time|check <checkpoint>
  time : CUDA-event timing of this tree's gelu_tanh_mul_fp4_quant wrapper
         (main's includes its torch.zeros SF memset):
           cold_<M>  standalone, L2 flushed (256 MB write) before every call
           hot_<M>   standalone, input L2-resident (same input every call)
           chain_<M> gate_up nvfp4_gemm (autotuned pick, 3 rotated weight
                     copies = cold weights) then the act wrapper; the act
                     part (event between the two) is reported.
         If the tree has _launch_config (candidate), the chain is also timed
         with the SM120 hot config and the default config forced at every M,
         to place the M threshold.
  check: untimed; tree's wrapper vs an in-script copy of main's kernel and
         wrapper (torch.zeros SF, BM=8/BK=512/warps=8): payload and the full
         SF buffer (padding included) bitwise, from 0xA5-sentinel-filled
         allocator blocks; repeated sequences A,B,A (8000,300,8000) and
         (1000,1280,1000) with the timing run's call counts; FP32 reference.
"""
import json
import os
import subprocess
import sys

import torch
import triton
import triton.language as tl

MODE = sys.argv[1]
CK = sys.argv[2] if len(sys.argv) > 2 else None
dev = "cuda"


def log(*a):
    print(*a, flush=True)


H, I = 5376, 21504
NL = 60
if CK and os.path.exists(os.path.join(CK, "config.json")):
    cfg = json.load(open(os.path.join(CK, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = int(tc["hidden_size"])
    I = int(tc["intermediate_size"])
    NL = int(tc["num_hidden_layers"])
log(f"hidden_size={H} intermediate_size={I} layers={NL}")
try:
    log("gpu:", subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
         "clocks_throttle_reasons.active", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20).stdout.strip())
except Exception as e:  # noqa: BLE001
    log("nvidia-smi unavailable", repr(e))

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.kernels.fused_ops import gelu_tanh_mul_fp4_quant as mod  # noqa: E402

TREE = mod.gelu_tanh_mul_fp4_quant
IS_CAND = hasattr(mod, "_launch_config")
log("tree has _launch_config (candidate):", IS_CAND, "cap:", torch.cuda.get_device_capability())
if IS_CAND:
    log("SM120 config", mod._SM120_CONFIG, "max M", mod._SM120_CONFIG_MAX_M,
        "default", mod._DEFAULT_CONFIG)

WARMUP, ITERS = 5, 25


# ----------------------------------------------------------------------------
# Verbatim copy of main's (357e111f) kernel + wrapper: the protected reference.
@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _main_kernel(x_ptr, out_ptr, sf_ptr, gs_ptr, M, SX, NKT,
                 IDIM: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = rmask[:, None] & (cols[None, :] < IDIM)
    gate = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + rows[:, None] * SX + IDIM + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner], dtype=tl.float32, is_pure=True, pack=1
    )
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
    kvec = (tl.program_id(1) * (BK // 16) + tl.arange(0, BK // 16)).to(tl.int64)
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
        "=h,f,f", [lo, hi], dtype=tl.uint16, is_pure=True, pack=1,
    ).to(tl.uint8)
    ocols = tl.program_id(1) * (BK // 2) + tl.arange(0, BK // 2)
    omask = rmask[:, None] & (ocols[None, :] * 2 < IDIM)
    tl.store(out_ptr + rows[:, None] * (IDIM // 2) + ocols[None, :], byte, mask=omask)


def main_ref(x, gs):
    m, two_i = x.shape
    i = two_i // 2
    nkt = triton.cdiv(i // 16, 4)
    out = torch.empty((m, i // 2), dtype=torch.uint8, device=x.device)
    sf = torch.zeros((((m + 127) // 128) * 128 * nkt * 4,), dtype=torch.uint8, device=x.device)
    if m == 0:
        return out, sf
    grid = (triton.cdiv(m, 8), triton.cdiv(i, 512))
    _main_kernel[grid](x, out, sf, gs, m, x.stride(0), nkt, IDIM=i, BM=8, BK=512, num_warps=8)
    return out, sf


# ----------------------------------------------------------------------------
REAL_GS = {}


def real_gs():
    if REAL_GS or not CK:
        return REAL_GS
    try:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]
        for li in sorted({0, NL // 2, NL - 1}):
            k = [k for k in idx if k.endswith(f"layers.{li}.mlp.down_proj.input_scale")][0]
            with safe_open(os.path.join(CK, idx[k]), framework="pt", device="cpu") as f:
                iv = f.get_tensor(k).float().item()
            REAL_GS[li] = torch.tensor([1.0 / iv], device=dev, dtype=torch.float32)
        log("real down_proj global scales:", {k: v.item() for k, v in REAL_GS.items()})
    except Exception as e:  # noqa: BLE001
        log("real scales unavailable:", repr(e))
    return REAL_GS


def default_gs():
    r = real_gs()
    return r[sorted(r)[len(r) // 2]] if r else torch.tensor([448.0 * 6.0 / 30.0], device=dev)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return med, p10, p90


flush_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)


def time_standalone(fn, x, gs, cold):
    for _ in range(WARMUP):
        fn(x, gs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        if cold:
            flush_buf.fill_(1.0)
        torch.cuda._sleep(100000)  # keep the host ahead of the GPU
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(x, gs)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return stats(ts)


class GateUp:
    """gate_up nvfp4_gemm [M,H] x [2I,H]^T, autotuned, 3 rotated weight copies."""

    def __init__(self):
        from tensorrt_llm._torch.autotuner import autotune  # noqa: F401
        g = torch.Generator(device=dev).manual_seed(7)
        self.gs_a = torch.tensor([448.0 * 6.0 / 8.0], device=dev)
        self.ws = []
        for c in range(3):
            w = (torch.randn(2 * I, H, device=dev, generator=g) * 0.03).to(torch.bfloat16)
            gs_w = (448.0 * 6.0 / w.float().abs().max()).reshape(1).float()
            wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs_w, 16, False)
            del w
            alpha = (1.0 / (self.gs_a * gs_w)).float()
            self.ws.append((wq, wsf, alpha))
        torch.cuda.empty_cache()
        self.tuned = set()

    def act_in(self, m, seed=11):
        g = torch.Generator(device=dev).manual_seed(seed + m)
        a = (torch.randn(m, H, device=dev, generator=g) * 2.0).to(torch.bfloat16)
        return torch.ops.trtllm.fp4_quantize(a, self.gs_a, 16, False)

    def __call__(self, aq, asf, c):
        wq, wsf, alpha = self.ws[c % 3]
        return torch.ops.trtllm.nvfp4_gemm(aq, wq, asf, wsf, alpha, torch.bfloat16,
                                           allowed_backends="cutlass,cublaslt,cuda_core")

    def tune(self, aq, asf):
        m = aq.shape[0]
        if m in self.tuned:
            return
        from tensorrt_llm._torch.autotuner import autotune
        with autotune():
            self(aq, asf, 0)
        torch.cuda.synchronize()
        self.tuned.add(m)


def time_chain(gu, fn, aq, asf, gs):
    for w in range(WARMUP):
        fn(gu(aq, asf, w), gs)
    torch.cuda.synchronize()
    ts_act, ts_all = [], []
    for it in range(ITERS):
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        y = gu(aq, asf, it)
        e1.record()
        fn(y, gs)
        e2.record()
        torch.cuda.synchronize()
        ts_act.append(e1.elapsed_time(e2))
        ts_all.append(e0.elapsed_time(e2))
        del y
    return stats(ts_act), stats(ts_all)


def make_x(m, kind="randn", seed=1):
    g = torch.Generator(device=dev).manual_seed(seed * 100003 + m)
    x = torch.randn(m, 2 * I, device=dev, generator=g)
    if kind == "randn":
        x = x * 1.5
    elif kind == "heavy":
        e = torch.randint(-8, 9, (m, 1), device=dev, generator=g).float()
        x = x * torch.exp2(e)
        if m >= 4:
            x[m // 3] = 0.0  # an all-zero row (vmax == 0 everywhere)
            x[m // 2, : I // 2] = 0.0
    return x.to(torch.bfloat16)


# ----------------------------------------------------------------------------
if MODE == "sweep":  # candidate tree only: chain act time per forced config
    assert IS_CAND
    gs = default_gs()
    gu = GateUp()
    CFGS = [(8, 512, 8), (1, 1024, 4), (1, 512, 4), (2, 1024, 4), (1, 2048, 8), (4, 1024, 8),
            (1, 1024, 8), (2, 512, 4), (2, 2048, 8), (4, 512, 4)]
    ITERS = 40
    res = {}
    mod._SM120_CONFIG_MAX_M = 1 << 40
    for m in [300, 1000, 1280, 1536, 2000, 4000, 8000]:
        aq, asf = gu.act_in(m)
        gu.tune(aq, asf)
        for cfg_ in CFGS:
            mod._SM120_CONFIG = cfg_
            ts = []
            for w in range(WARMUP):
                TREE(gu(aq, asf, w), gs)
            for it in range(ITERS):
                e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                y = gu(aq, asf, it)
                e1.record()
                TREE(y, gs)
                e2.record()
                torch.cuda.synchronize()
                ts.append(e1.elapsed_time(e2) * 1e3)
                del y
            med, p10, p90 = stats(ts)
            mean = sum(ts) / len(ts)
            res.setdefault(str(m), {})[str(cfg_)] = [round(med, 2), round(p10, 2), round(p90, 2), round(mean, 2)]
            log(f"M={m:5d} cfg={cfg_}: median {med:8.2f} p10 {p10:8.2f} p90 {p90:8.2f} mean {mean:8.2f} us")
        del aq, asf
    print(json.dumps({"passed": True, "sweep_chain_act_us_median_p10_p90_mean": res}))
    sys.exit(0)

if MODE == "time":
    gs = default_gs()
    out = {}
    extra = {}
    MS_STAND = [300, 1000, 1200, 1280, 2000, 4000, 8000]
    for m in MS_STAND:
        x = make_x(m)
        for cold in (True, False):
            med, p10, p90 = time_standalone(TREE, x, gs, cold)
            key = f"{'cold' if cold else 'hot'}_{m}"
            out[key] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90}
            log(f"{key:>10}: median {med*1e3:8.2f} us  p10 {p10*1e3:8.2f}  p90 {p90*1e3:8.2f}")
        del x
    torch.cuda.empty_cache()
    gu = GateUp()
    log("gate_up weights ready")
    MS_CHAIN = [300, 1000, 1200, 1280, 1536, 2000, 2560, 8000]
    for m in MS_CHAIN:
        aq, asf = gu.act_in(m)
        gu.tune(aq, asf)
        (med, p10, p90), (amed, _, _) = time_chain(gu, TREE, aq, asf, gs)
        key = f"chain_{m}"
        out[key] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90,
                    "gemm_plus_act_ms": amed}
        log(f"{key:>10}: act {med*1e3:8.2f} us  p10 {p10*1e3:8.2f}  p90 {p90*1e3:8.2f}  "
            f"(gemm+act {amed*1e3:8.1f} us)")
        if IS_CAND:
            saved = mod._SM120_CONFIG_MAX_M
            for name, thr in (("forced_sm120_1_1024_4", 1 << 40), ("forced_default_8_512_8", -1)):
                mod._SM120_CONFIG_MAX_M = thr
                (fm, f10, f90), _ = time_chain(gu, TREE, aq, asf, gs)
                extra.setdefault(name, {})[str(m)] = {"median_us": fm * 1e3, "p10_us": f10 * 1e3,
                                                      "p90_us": f90 * 1e3}
                log(f"           {name}: act {fm*1e3:8.2f} us  p10 {f10*1e3:8.2f}  p90 {f90*1e3:8.2f}")
            mod._SM120_CONFIG_MAX_M = saved
        del aq, asf
    try:
        log("gpu end:", subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps({"batch": out, "output": {"class": "exact"}, "extra": extra}))
    sys.exit(0)

# ---------------------------- correctness -----------------------------------
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)


def sf_offsets(m, nvec):
    nkt = (nvec + 3) // 4
    mm = torch.arange(m, device=dev, dtype=torch.int64)[:, None]
    kk = torch.arange(nvec, device=dev, dtype=torch.int64)[None, :]
    return ((mm // 128) * (nkt * 512) + (kk // 4) * 512 + (mm % 32) * 16
            + ((mm % 128) // 32) * 4 + (kk % 4)).reshape(-1)


def dequant(fp4, sf, gs, m):
    vals = torch.stack([E2M1[(fp4 & 0xF).long()], E2M1[(fp4 >> 4).long()]], -1).reshape(m, I)
    s = sf[sf_offsets(m, I // 16)].view(torch.float8_e4m3fn).float().reshape(m, I // 16)
    return (vals.reshape(m, I // 16, 16) * s[:, :, None]).reshape(m, I) / gs.item()


def ref_fp32(x):
    xf = x.float()
    gate, up = xf[:, :I], xf[:, I:]
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


def sentinel_call(fn, x, gs):
    """Fill the allocator blocks the wrapper will get with 0xA5, then call."""
    m = x.shape[0]
    nkt = triton.cdiv(I // 16, 4)
    a = torch.full((m, I // 2), 0xA5, dtype=torch.uint8, device=dev)
    b = torch.full((((m + 127) // 128) * 128 * nkt * 4,), 0xA5, dtype=torch.uint8, device=dev)
    pa, pb = a.data_ptr(), b.data_ptr()
    del a, b
    o, s = fn(x, gs)
    return o, s, (o.data_ptr() == pa and s.data_ptr() == pb)


def compare(o, s, ro, rs):
    pay = (o.shape == ro.shape) and torch.equal(o, ro)
    sff = (s.shape == rs.shape) and torch.equal(s, rs)
    first = None
    if not pay and o.shape == ro.shape:
        idx = (o != ro).nonzero()[:1].tolist()
        first = ("payload", idx)
    elif not sff and s.shape == rs.shape:
        idx = (s != rs).nonzero()[:1].tolist()
        first = ("sf", idx, s[idx[0][0]].item(), rs[idx[0][0]].item())
    return pay, sff, first


allok = True
report = {}
gsr = real_gs() or {None: default_gs()}
# padding SF bytes of main are 0x00: check the e4m3 zero byte is +0 (0x00)
z = torch.zeros(1, device=dev).to(torch.float8_e4m3fn).view(torch.uint8).item()
log("e4m3(+0) byte:", hex(z))

CASES = [1, 27, 127, 128, 129, 300, 1000, 1200, 1280, 1536, 1537, 2000, 2560, 4000, 8000]
for m in CASES:
    res = {}
    for kind, seed in (("randn", 1), ("randn", 2), ("heavy", 3)):
        x = make_x(m, kind, seed)
        snap = x.clone()
        for li, gs in gsr.items():
            ro, rs = main_ref(x, gs)
            o, s, same_blocks = sentinel_call(TREE, x, gs)
            torch.cuda.synchronize()
            pay, sff, first = compare(o, s, ro, rs)
            r = {"payload_bitwise": pay, "sf_full_bitwise": sff, "sentinel_blocks_reused": same_blocks}
            if first:
                r["first_mismatch"] = first
            # FP32 reference sanity (b)
            if m <= 2000:
                ref = ref_fp32(x)
                dc, dm = dequant(o, s, gs, m), dequant(ro, rs, gs, m)
                d1, d2 = dc - ref, dm - ref
                r.update(cand_err_maxabs=d1.abs().max().item(), main_err_maxabs=d2.abs().max().item(),
                         cand_err_relfro=(d1.norm() / ref.norm().clamp_min(1e-30)).item(),
                         main_err_relfro=(d2.norm() / ref.norm().clamp_min(1e-30)).item(),
                         cand_vs_main_maxabs=(dc - dm).abs().max().item(),
                         finite=bool(torch.isfinite(dc).all().item()))
                ok_ref = r["finite"] and r["cand_err_maxabs"] <= 1.10 * r["main_err_maxabs"] + 1e-30
                r["ref_ok"] = ok_ref
                del ref, dc, dm, d1, d2
            else:
                ok_ref = True
            ok = pay and sff and ok_ref
            res[f"{kind}{seed}_L{li}"] = r
            allok &= ok
            if not ok:
                log("FAIL", m, kind, seed, li, r)
            del ro, rs, o, s
        allok &= torch.equal(x, snap)
        del x, snap
    # strided input (row stride > 2I)
    if m in (300, 1537):
        big = make_x(m, "randn", 5)
        xs = torch.cat([big, big[:, :64]], 1)[:, : 2 * I]
        ro, rs = main_ref(xs, default_gs())
        o, s = TREE(xs, default_gs())
        pay, sff, first = compare(o, s, ro, rs)
        res["strided"] = {"payload_bitwise": pay, "sf_full_bitwise": sff}
        allok &= pay and sff
        del big, xs
    # gate_up-like rows produced by the chain's GEMM at the same M
    report[str(m)] = res
    log(m, "all bitwise:", all(v.get("payload_bitwise") and v.get("sf_full_bitwise") for v in res.values()))
    torch.cuda.empty_cache()

# GEMM-produced inputs (the chain's gate_up output) at a few M
gu = GateUp()
for m in (300, 1280, 2000):
    aq, asf = gu.act_in(m, seed=91)
    y = gu(aq, asf, 0)
    ro, rs = main_ref(y, default_gs())
    o, s, _ = sentinel_call(TREE, y, default_gs())
    pay, sff, first = compare(o, s, ro, rs)
    report[f"gemm_out_{m}"] = {"payload_bitwise": pay, "sf_full_bitwise": sff, "first_mismatch": first}
    allok &= pay and sff
    log("gemm_out", m, pay, sff)
    del aq, asf, y
del gu
torch.cuda.empty_cache()

# Repeated-execution sequences A,B,A across the threshold, WARMUP+ITERS calls
# per shape exactly as the timing run, every call vs call 1 and vs main.
seq_report = {}
for seq in ((8000, 300, 8000), (1000, 1280, 1000)):
    gs = default_gs()
    xs = {m: make_x(m, "heavy", 4) for m in set(seq)}
    snaps = {m: x.clone() for m, x in xs.items()}
    refs = {m: main_ref(xs[m], gs) for m in set(seq)}
    torch.cuda.synchronize()
    firsts = {}
    ok_seq = True
    nbad = 0
    for m in seq:
        for c in range(WARMUP + ITERS):
            o, s, _ = sentinel_call(TREE, xs[m], gs)
            if m not in firsts:
                firsts[m] = (o.clone(), s.clone())
            p1, s1, _ = compare(o, s, *firsts[m])
            p2, s2, _ = compare(o, s, *refs[m])
            good = p1 and s1 and p2 and s2
            nbad += 0 if good else 1
            ok_seq &= good
    torch.cuda.synchronize()
    inputs_ok = all(torch.equal(xs[m], snaps[m]) for m in xs)
    seq_report[str(seq)] = {"ok": ok_seq, "bad_calls": nbad, "inputs_unchanged": inputs_ok,
                            "calls": len(seq) * (WARMUP + ITERS)}
    allok &= ok_seq and inputs_ok
    log("sequence", seq, seq_report[str(seq)])
    del xs, snaps, refs, firsts
    torch.cuda.empty_cache()

# Launch-config gating: fixed per arch, keyed only on M. Other archs keep
# main's (8,512,8); that path (with the in-kernel padding fill) is also
# checked bitwise by faking a non-SM120 capability.
arch_report = {}
if IS_CAND:
    idx = torch.cuda.current_device()
    real_cap = mod._DEVICE_CAPABILITY.get(idx, torch.cuda.get_device_capability(idx))
    for cap in ((12, 0), (12, 1), (10, 0), (9, 0), (8, 9)):
        mod._DEVICE_CAPABILITY[idx] = cap
        cfgs = {m: mod._launch_config(m, torch.device("cuda", idx)) for m in (1, 300, 1280, 2000, 8000, 8192)}
        want = mod._SM120_CONFIG if cap == (12, 0) else (8, 512, 8)
        ok = all(v == want for v in cfgs.values())
        arch_report[str(cap)] = {"configs": {str(k): v for k, v in cfgs.items()}, "ok": ok}
        allok &= ok
        if cap == (10, 0):
            for m in (1, 300, 1537, 8000):
                x = make_x(m, "heavy", 6)
                ro, rs = main_ref(x, default_gs())
                o, s, _ = sentinel_call(TREE, x, default_gs())
                pay, sff, _ = compare(o, s, ro, rs)
                arch_report[str(cap)][f"bitwise_{m}"] = pay and sff
                allok &= pay and sff
                del x, ro, rs, o, s
    mod._DEVICE_CAPABILITY[idx] = real_cap
    log("arch gating:", arch_report)

# M == 0
o0, s0 = TREE(torch.empty(0, 2 * I, device=dev, dtype=torch.bfloat16), default_gs())
m0_ok = o0.shape == (0, I // 2) and s0.numel() == 0
allok &= m0_ok

print(json.dumps({"passed": bool(allok), "class": "exact", "e4m3_zero_byte": z, "m0_ok": m0_ok,
                  "compared": "tree gelu_tanh_mul_fp4_quant vs verbatim copy of main 357e111f kernel+wrapper "
                              "(torch.zeros SF, 8/512/8): payload + full SF buffer (padding rows/cols) "
                              "bitwise from 0xA5-sentinel blocks; 2 randn seeds + heavy-tailed 2^U(-8,8) rows "
                              "with zero rows, real down_proj input scales (layers 0/30/59), strided input, "
                              "gate_up nvfp4_gemm outputs; A,B,A repeated sequences (8000,300,8000) and "
                              "(1000,1280,1000) with 30 calls per shape; FP32 gelu*up reference; inputs unchanged",
                  "arch_gating": arch_report, "sequences": seq_report, "cases": report}))
