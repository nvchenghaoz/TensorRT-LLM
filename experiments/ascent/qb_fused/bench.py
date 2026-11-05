"""S1 prototype bench for idea 232 (branch 127): SM120 decode attention-input chain.

main chain per layer : A = trtllm.csa2_mxfp8_low_m_gemm wq_a||wkv [1792,5120] -> flashinfer rmsnorm (q_norm, rmsnorm_cute)
                       -> trtllm.mxfp8_quantize(qr) -> flashinfer_mm_mxfp8 (b12x) wq_b [8192,1280]
                       (indexer layers: [12288,1280] at M=1, two GEMMs at M>1) -> glue
proto chain per layer: A' (proto copy of A, trigger position trigA) -> B (q_norm + MXFP8 quant + wq_b(||index_wq_b),
                       one launch, trigger trigB) -> glue
glue = 1-CTA PDL kernel (wait at start, trigger at end) standing in for the rest of the layer.

Modes: norm | check | chain | all  (argv[1]); argv[2] = checkpoint dir (eps from config.json).
Prints one JSON object last.
"""

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1234
L2_BYTES = 128 * 1024 * 1024
MAX_OPTIN = 101376
DRAM_GBPS = 1507.0
Q_RANK = 1280
QB_N = 8192
IDX_N = 4096
HIDDEN = 5120
A_N = 1792
LAYERS = 40
A_DEFAULT = {1: (8, 2, 2), 2: (8, 2, 2), 3: (8, 4, 1), 4: (8, 4, 1), 5: (8, 4, 1), 6: (8, 4, 1), 7: (8, 4, 1),
             8: (8, 4, 1)}
B_CFGS = [(4, 1, 2), (4, 1, 1), (4, 2, 1), (4, 2, 2), (8, 1, 2), (4, 1, 4), (8, 1, 4)]
A_ALTS = [(8, 2, 2), (8, 4, 1), (4, 4, 1), (4, 4, 2), (4, 8, 1), (4, 8, 2), (4, 2, 2)]
ORDER = int(os.environ.get("QB_ORDER", "0"))


def log(*a):
    print(*a, flush=True)


def load_ext():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    bd = "/tmp/qb_proto_build_%d" % os.getpid()
    os.makedirs(bd, exist_ok=True)
    t = time.time()
    ext = load(name="qb_proto", sources=[os.path.join(HERE, "proto.cu")], build_directory=bd,
               extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"], verbose=False)
    log(f"built ext in {time.time() - t:.1f}s")
    return ext


def read_eps(ckpt):
    try:
        with open(os.path.join(ckpt, "config.json")) as f:
            cfg = json.load(f)
        log("config:", {k: cfg.get(k) for k in ("rms_norm_eps", "q_lora_rank", "hidden_size", "num_attention_heads",
                                                 "head_dim", "index_n_heads", "index_head_dim")})
        return float(cfg.get("rms_norm_eps", 1e-6))
    except Exception as e:  # noqa: BLE001
        log("config.json not read:", e)
        return 1e-6


def make_weight(n, k, gen):
    w = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 128, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    sw = torch.ops.trtllm.block_scale_interleave(raw.contiguous()).reshape(-1)
    return w, sw, raw


def unswizzle(sf, rows, k):
    cols = k // 32
    pc = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    idx = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * pc)
    return sf.reshape(-1).view(torch.uint8)[idx]


def dequant_ref(x, w, raw):
    m, k = x.shape
    xq, xs = torch.ops.trtllm.mxfp8_quantize(x, True)
    xsf = unswizzle(xs, m, k)
    xd = (xq.float().view(m, k // 32, 32) * torch.exp2(xsf.float() - 127).unsqueeze(-1)).view(m, k)
    n = w.shape[0]
    wd = (w.float().view(n, k // 32, 32) * torch.exp2(raw.float() - 127).unsqueeze(-1)).view(n, k)
    return (xd.double() @ wd.double().t())


def rel_err(out, ref):
    d = (out.double() - ref).abs()
    return d.max().item(), d.max().item() / max(ref.abs().max().item(), 1e-30)


def valid(cfg, n, k, m, norm=False):
    warps, ck, rt = cfg
    kc = k // ck
    if n % (16 * rt) or k % (ck * 128) or (kc // 32) % warps or (kc // 32) % 4:
        return False
    if norm and k != 1280:
        return False
    smem = (16 * rt * (kc + 32) + rt * (kc // 32 // 4) * 256 + m * kc + ((m * (kc // 32) + 15) // 16) * 16
            + (ck - 1) * 16 * rt * m * 4)
    red = warps * 16 * rt * 8 * 4
    if red > 16 * rt * (kc + 32):
        smem += red
    return smem + 64 <= MAX_OPTIN


def rmsnorm_ref(x, gamma, eps):
    return torch.ops.trtllm.flashinfer_rmsnorm(x, gamma, eps)


# ----------------------------------------------------------------------------------------------------------------
def norm_check(ext, eps):
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    rows = 8192 + 256
    base = torch.randn((rows, A_N), generator=gen, device="cuda")
    scale = torch.logspace(-3, 2, rows, device="cuda")[torch.randperm(rows, generator=gen, device="cuda")]
    base = base * scale.view(-1, 1)
    base[0:16] = 0.0                                              # zero rows
    base[16:48, 64:96] *= 30.0                                     # one 30x group
    base[48:80] = torch.randn((32, A_N), generator=gen, device="cuda") * 1e-39   # bf16 subnormal range
    base[80:112] = torch.randn((32, A_N), generator=gen, device="cuda") * 3e4      # large
    base[112:144, :640] = 0.0                                      # half-zero rows
    base[144:176] = torch.randn((32, A_N), generator=gen, device="cuda") * 1e-20   # tiny normals (sumsq underflow)
    full = base.to(torch.bfloat16)
    xs = full[:, :Q_RANK]  # strided view, row stride 1792
    gamma = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    ref = rmsnorm_ref(xs, gamma, eps)
    qref, sref = torch.ops.trtllm.mxfp8_quantize(ref.contiguous(), True)
    sref_u = unswizzle(sref, rows, Q_RANK)
    torch.cuda.synchronize()
    res = {}
    best = None
    for order in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        qr = torch.empty((rows, Q_RANK), dtype=torch.bfloat16, device="cuda")
        xq = torch.empty((rows, Q_RANK), dtype=torch.uint8, device="cuda")
        xc = torch.empty((rows, Q_RANK // 32), dtype=torch.uint8, device="cuda")
        ext.norm_test(xs, gamma, eps, order, qr, xq, xc)
        torch.cuda.synchronize()
        a = qr.view(torch.int16).int()
        b = ref.view(torch.int16).int()
        bad_rows = (a != b).any(1).sum().item()
        ulp = (a - b).abs().max().item()
        # quant bytes vs mxfp8_quantize of the reference qr and of our own qr
        q_eq_ref = torch.equal(xq, qref.view(torch.uint8)) and torch.equal(xc, sref_u)
        qo, so = torch.ops.trtllm.mxfp8_quantize(qr, True)
        q_eq_own = torch.equal(xq, qo.view(torch.uint8)) and torch.equal(xc, unswizzle(so, rows, Q_RANK))
        res[order] = {"qr_bad_rows": bad_rows, "qr_max_ulp": ulp, "quant_eq_ref": q_eq_ref, "quant_eq_own": q_eq_own}
        log(f"NORM order={order} {res[order]}")
        if bad_rows == 0 and q_eq_ref and best is None:
            best = order
    return {"rows": rows, "orders": res, "bitwise_order": best}


# ----------------------------------------------------------------------------------------------------------------
def gemm_check(ext, eps, order):
    """B vs (rmsnorm_cute + low_m_mm) bitwise; qr bitwise; q error vs FP64 beside b12x; proto A vs trtllm A bitwise;
    live-overlap PDL test with a skip-wait negative control."""
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    wa, sa, rawa = make_weight(A_N, HIDDEN, gen)
    wb, sb, rawb = make_weight(QB_N + IDX_N, Q_RANK, gen)
    wq, sq, rawq = wb[:QB_N], sb[: QB_N * 40], rawb[:QB_N]
    gamma = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    ok = True
    out = {}
    for m in range(1, 9):
        x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
        cfg_a = A_DEFAULT[m]
        rows_ref = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, wa, sa, *cfg_a, 0, False)[0]
        # proto A bitwise vs trtllm A for every alt config and trigger
        a_ok = True
        for ca in A_ALTS:
            if not valid(ca, A_N, HIDDEN, m):
                continue
            r_trt = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, wa, sa, *ca, 0, False)[0]
            for trig in (0, 1, 2, 3):
                r = torch.empty((m, A_N), dtype=torch.bfloat16, device="cuda")
                ext.gemm(x, wa, sa, r, None, 0, *ca, False, None, 0.0, None, 0, trig, 0, True)
                a_ok &= torch.equal(r, r_trt)
        rows = rows_ref
        qr_in = rows[:, :Q_RANK]
        qr_ref = rmsnorm_ref(qr_in, gamma, eps)
        ref64 = dequant_ref(qr_ref.contiguous(), wq, rawq)
        a_, s_ = torch.ops.trtllm.mxfp8_quantize(qr_ref.contiguous(), True)
        b12x = torch.ops.trtllm.flashinfer_mm_mxfp8(a_, s_, wq, sq, torch.bfloat16)
        b12x_rel = rel_err(b12x, ref64)[1]
        ref64_i = dequant_ref(qr_ref.contiguous(), wb, rawb)
        row = {"A_proto_bitwise": a_ok, "b12x_rel": b12x_rel, "cfgs": {}}
        ok &= a_ok
        for cb in B_CFGS:
            if not valid(cb, QB_N, Q_RANK, m, norm=True):
                continue
            lowm = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(qr_ref.contiguous(), wq, sq, *cb, 0, False)[0]
            lowm_i = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(qr_ref.contiguous(), wb, sb, *cb, QB_N, False)
            c = {}
            for trig in (0, 1, 3):
                q = torch.full((m, QB_N), float("nan"), dtype=torch.bfloat16, device="cuda")
                qr = torch.full((m, Q_RANK), float("nan"), dtype=torch.bfloat16, device="cuda")
                ext.gemm(qr_in, wq, sq, q, None, 0, *cb, True, gamma, eps, qr, order, trig, 0, True)
                qi = torch.full((m, QB_N), float("nan"), dtype=torch.bfloat16, device="cuda")
                ii = torch.full((m, IDX_N), float("nan"), dtype=torch.bfloat16, device="cuda")
                ext.gemm(qr_in, wb, sb, qi, ii, QB_N, *cb, True, gamma, eps, None, order, trig, 0, True)
                torch.cuda.synchronize()
                c[trig] = {
                    "qr_bitwise": torch.equal(qr.view(torch.int16), qr_ref.view(torch.int16)),
                    "q_bitwise_vs_lowm": torch.equal(q, lowm),
                    "idx_bitwise_vs_lowm": torch.equal(qi, lowm_i[0]) and torch.equal(ii, lowm_i[1]),
                    "q_rel_fp64": rel_err(q, ref64)[1],
                    "idx_rel_fp64": rel_err(torch.cat([qi, ii], 1), ref64_i)[1],
                }
                cc = c[trig]
                ok &= cc["qr_bitwise"] and cc["q_bitwise_vs_lowm"] and cc["idx_bitwise_vs_lowm"] and \
                    cc["q_rel_fp64"] < 1.6e-2
            row["cfgs"][",".join(map(str, cb))] = c
        out[m] = row
        log(f"CHECK M={m} {json.dumps(row)}")

    # Live-overlap PDL test: the producer triggers at once, NaN-poisons the rows, spins 5 us, then writes them.
    live = {}
    for m in (1, 4, 8):
        x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
        src = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, wa, sa, *A_DEFAULT[m], 0, False)[0]
        qr_ref = rmsnorm_ref(src[:, :Q_RANK], gamma, eps)
        ref = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(qr_ref.contiguous(), wq, sq, 4, 1, 2, 0, False)[0]
        dst = torch.zeros_like(src)
        res = {}
        for flags in (0, 1):
            bad = 0
            for it in range(20):
                dst.zero_()
                q = torch.full((m, QB_N), float("nan"), dtype=torch.bfloat16, device="cuda")
                ext.poison_producer(src, dst, 5000)
                ext.gemm(dst[:, :Q_RANK], wq, sq, q, None, 0, 4, 1, 2, True, gamma, eps, None, order, 0, flags, True)
                torch.cuda.synchronize()
                bad += 0 if torch.equal(q, ref) else 1
            res["skip_wait" if flags else "wait"] = bad
        live[m] = res
        ok &= res["wait"] == 0 and res["skip_wait"] > 0
        log(f"LIVE M={m} bad replays of 20: {res}")
    return {"passed": bool(ok), "per_m": out, "live_overlap": live}


# ----------------------------------------------------------------------------------------------------------------
def sm_clock():
    try:
        import pynvml

        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUUID(h)
            u = u.decode() if isinstance(u, bytes) else u
            if u.replace("GPU-", "") == uuid.replace("GPU-", ""):
                return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"
    return "n/a"


def capture(fn, layers):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for l in range(2):
            fn(l)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for l in range(layers):
            fn(l)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g


def time_arms(graphs, layers, rounds):
    names = list(graphs)
    t = {n: [] for n in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            b.synchronize()
            t[n].append(a.elapsed_time(b) * 1000.0 / layers)
    out = {}
    for n in names:
        v = sorted(t[n])
        med = v[len(v) // 2]
        p10 = v[int(0.1 * (len(v) - 1))]
        p90 = v[int(0.9 * (len(v) - 1))]
        out[n] = {"us": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                  "noise": round((p90 - p10) / med, 4)}
    return out


def chain(ext, eps, order, ms=(1, 2, 4, 8), kinds=("plain", "indexer"), rounds=30):
    gen = torch.Generator(device="cuda").manual_seed(SEED + 2)
    layers = LAYERS
    A = [make_weight(A_N, HIDDEN, gen)[:2] for _ in range(layers)]
    gamma = (1.0 + 0.2 * torch.randn(Q_RANK, generator=gen, device="cuda")).to(torch.bfloat16)
    sink = torch.zeros(1, dtype=torch.float32, device="cuda")
    table = {}
    for kind in kinds:
        nb = QB_N if kind == "plain" else QB_N + IDX_N
        B = [make_weight(nb, Q_RANK, gen)[:2] for _ in range(layers)]
        wbytes = A_N * HIDDEN * (1 + 1 / 32) + nb * Q_RANK * (1 + 1 / 32)
        log(f"chain {kind}: per-layer weight bytes {wbytes / 1e6:.2f} MB, DRAM floor {wbytes / (DRAM_GBPS * 1e3):.2f} us, "
            f"total {layers * wbytes / 1e6:.0f} MB")
        for m in ms:
            x = (torch.randn((m, HIDDEN), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            rows = torch.empty((m, A_N), dtype=torch.bfloat16, device="cuda")
            q = torch.empty((m, QB_N), dtype=torch.bfloat16, device="cuda")
            idx = torch.empty((m, IDX_N), dtype=torch.bfloat16, device="cuda")
            qr = torch.empty((m, Q_RANK), dtype=torch.bfloat16, device="cuda")
            cfg_a = A_DEFAULT[m]

            def main_fn(l, proto_a=False):
                wa, sa = A[l]
                if proto_a:
                    ext.gemm(x, wa, sa, rows, None, 0, *cfg_a, False, None, 0.0, None, 0, 0, 0, True)
                    r = rows
                else:
                    r = torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, wa, sa, *cfg_a, 0, True)[0]
                qn = rmsnorm_ref(r[:, :Q_RANK], gamma, eps)
                a_, s_ = torch.ops.trtllm.mxfp8_quantize(qn, True)
                wb, sb = B[l]
                if kind == "plain" or m == 1:
                    o = torch.ops.trtllm.flashinfer_mm_mxfp8(a_, s_, wb, sb, torch.bfloat16)
                else:
                    o = torch.ops.trtllm.flashinfer_mm_mxfp8(a_, s_, wb[:QB_N], sb[: QB_N * 40], torch.bfloat16)
                    torch.ops.trtllm.flashinfer_mm_mxfp8(a_, s_, wb[QB_N:], sb[QB_N * 40:], torch.bfloat16)
                ext.glue(o, sink)

            def proto_fn(l, ca, ta, cb, tb):
                wa, sa = A[l]
                wb, sb = B[l]
                ext.gemm(x, wa, sa, rows, None, 0, *ca, False, None, 0.0, None, 0, ta, 0, True)
                if kind == "plain":
                    ext.gemm(rows[:, :Q_RANK], wb, sb, q, None, 0, *cb, True, gamma, eps, qr, order, tb, 0, True)
                else:
                    ext.gemm(rows[:, :Q_RANK], wb, sb, q, idx, QB_N, *cb, True, gamma, eps, qr, order, tb, 0, True)
                ext.glue(q, sink)

            arms = {"main": lambda l: main_fn(l), "main_protoA": lambda l: main_fn(l, True)}
            bcfgs = [cb for cb in B_CFGS if valid(cb, nb, Q_RANK, m, norm=True)]
            for cb in bcfgs:
                tag = ",".join(map(str, cb))
                arms[f"A{cfg_a}e>B{tag}e"] = (lambda cb_: lambda l: proto_fn(l, cfg_a, 0, cb_, 0))(cb)
                arms[f"A{cfg_a}s>B{tag}e"] = (lambda cb_: lambda l: proto_fn(l, cfg_a, 1, cb_, 0))(cb)
            for cb in bcfgs[:3]:
                tag = ",".join(map(str, cb))
                arms[f"A{cfg_a}s>B{tag}s"] = (lambda cb_: lambda l: proto_fn(l, cfg_a, 1, cb_, 1))(cb)
                arms[f"A{cfg_a}w>B{tag}e"] = (lambda cb_: lambda l: proto_fn(l, cfg_a, 2, cb_, 0))(cb)
            for ca in A_ALTS:
                if ca == cfg_a or not valid(ca, A_N, HIDDEN, m):
                    continue
                for cb in bcfgs[:3]:
                    tag = ",".join(map(str, cb))
                    arms[f"A{ca}s>B{tag}e"] = (lambda ca_, cb_: lambda l: proto_fn(l, ca_, 1, cb_, 0))(ca, cb)
            graphs = {}
            for n, fn in arms.items():
                try:
                    graphs[n] = capture(fn, layers)
                except Exception as e:  # noqa: BLE001
                    log(f"capture failed {kind} M={m} {n}: {e}")
            clk0 = sm_clock()
            res = time_arms(graphs, layers, rounds)
            clk1 = sm_clock()
            base = res["main"]["us"]
            best = min((v["us"], n) for n, v in res.items() if n not in ("main", "main_protoA"))
            log(f"== {kind} M={m} main {base:.2f} us/layer (noise {res['main']['noise']}) best {best[1]} {best[0]:.2f} "
                f"({(base - best[0]):.2f} us saved) sm_clock {clk0}/{clk1}")
            for n, v in sorted(res.items(), key=lambda kv: kv[1]["us"]):
                log(f"   {kind} M={m} {n:32s} {v['us']:7.2f} [{v['p10']:.2f}, {v['p90']:.2f}] noise {v['noise']}")
            table[f"{kind}/M{m}"] = {"arms": res, "sm_clock": [clk0, clk1], "best": best[1]}
            del graphs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del B
        torch.cuda.empty_cache()
    return table


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else ""
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    log("device", torch.cuda.get_device_name(0), "sm_clock", sm_clock())
    ext = load_ext()
    eps = read_eps(ckpt) if ckpt else 1e-6
    result = {"mode": mode, "eps": eps}
    order = ORDER
    if mode in ("norm", "all", "check"):
        nr = norm_check(ext, eps)
        result["norm"] = nr
        if nr["bitwise_order"] is not None:
            order = nr["bitwise_order"]
        log("NORM bitwise order:", nr["bitwise_order"])
    result["order"] = order
    passed = result.get("norm", {}).get("bitwise_order", 0) is not None
    if mode in ("check", "all"):
        ck = gemm_check(ext, eps, order)
        result["check"] = {"passed": ck["passed"], "live_overlap": ck["live_overlap"]}
        passed &= ck["passed"]
    if mode in ("chain", "all"):
        tb = chain(ext, eps, order)
        result["chain"] = {k: {"best": v["best"], "main": v["arms"]["main"], "best_arm": v["arms"][v["best"]],
                               "sm_clock": v["sm_clock"]} for k, v in tb.items()}
        os.makedirs("/tmp/qb_out", exist_ok=True)
        with open("/tmp/qb_out/chain_table.json", "w") as f:
            json.dump(tb, f)
    result["passed"] = bool(passed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
