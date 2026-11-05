"""Idea 233 S1: SM120 decode output projection on the low-M MXFP8 kernel family (TP4 rank shapes, 1 GPU, no model).

main  = b12x_wo chain: Triton K1 (_inv_rope_quant128) -> b12x grouped wo_a -> b12x wo_b fused-quant (production
        calls of csa2/b12x_wo.py, incl. the wide fq_pipeline at M in {2,3,4}).
new   = K2' (lowm_wo_ext.wo_a: inverse RoPE + 1x128 UE8M0 quant + grouped wo_a, bf16 [M, 2048])
        -> K3' (csa2_mxfp8_low_m_gemm on wo_b [5120, 2048]; landed op, or the prototype copy with the bit-1 hunk).

Modes: check (correctness gates) | sweep (config sweep) | table (ABBA chain table) | all.
Timing: 40-layer CUDA graph (PDL), weights rotated over 40 copies (~780 MB > 128 MB L2, so every layer is cold),
CUDA events per replay, us/layer = replay / 40, median and p10/p90 over all samples, ABBA visit order.
"""
import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b12x_wo"))
import common as C  # noqa: E402  (sets TRTLLM_ENABLE_PDL=1)
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: E402

NL = 40
MS = [1, 2, 4, 8]
DRAM_GBPS = 1508.0


def log(*a):
    print(*a, flush=True)


_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        t = time.time()
        bdir = tempfile.mkdtemp(prefix="lowm_wo_ext_")
        _EXT = load(name="lowm_wo_ext", sources=[os.path.join(HERE, "lowm_wo_ext.cu")],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo"],
                    extra_cflags=["-O2", "-std=c++20"], build_directory=bdir, verbose=False)
        log(f"ext built in {time.time() - t:.1f}s")
    return _EXT


# ------------------------------------------------------------------------------------------------------------------
class Layer:
    pass


def make_layer(c, dims):
    L = Layer()
    L.c = c
    L.wa = c.wa_main  # [G, R, GW] e4m3 (same bytes b12x wo_a views)
    L.ws = (torch.log2(c.sa_pow2.float()) + 127).round().to(torch.uint8).contiguous()  # [G, R/128, GW/128]
    L.wb = c.wb_q
    L.wbsf = c.wb_sf.reshape(-1)
    return L


class Env:
    def __init__(self, ckpt, n_layers=NL):
        from b12x.gemm._shared import wo_mxfp8 as W
        self.W = W
        self.dims = C.dims_from_ckpt(ckpt)
        d = self.dims
        log("dims", json.dumps(d))
        self.G, self.R, self.GW, self.H = d["groups_local"], d["rank"], d["group_width"], d["hidden"]
        self.HD, self.NOPE, self.ROPE = d["head_dim"], d["nope"], d["rope"]
        self.base = C.make_weights(d, seed=1234)
        self.cs = C.make_cos_sin(d)
        t = time.time()
        self.layers = [make_layer(self.base, d)] + [make_layer(C.clone_weights(self.base, d), d)
                                                    for _ in range(n_layers - 1)]
        torch.cuda.synchronize()
        wa_b = self.G * self.R * self.GW
        wb_b = self.H * self.G * self.R * (1 + 1 / 32)
        self.floor_us = (wa_b + wb_b) / (DRAM_GBPS * 1e3)
        log(f"{n_layers} layer copies in {time.time() - t:.1f}s; per-layer DRAM floor {self.floor_us:.2f} us")

    # main: production b12x_wo calls
    def main_layer(self, L, o3, pos, sc, M):
        W = self.W
        B.inv_rope_quant128(o3, pos, self.cs, sc.x_q, groups=self.G, group_width=self.GW, nope_dim=self.NOPE,
                            rope_dim=self.ROPE)
        W.wo_a_dense_gemm_mxfp8(sc.x_q, L.c.wa_b12x, out=sc.tmp, alpha=sc.alpha, expected_m=M,
                                sfb_k_replicated=True)
        pipe = (False, True, 0) if M in B._wo_b_wide_rows() else (False, False, 0)
        W.wo_b_dense_gemm_fused_quant_mxfp8(sc.tmp, L.c.wb_b12x, out=sc.out, expected_m=M, sfb_k_replicated=False,
                                            fq_pipeline=pipe)
        return sc.out[:, :, 0]

    def k2(self, L, o2, pos, lora, cfg, flags=0, qv=0, xq=None, xs=None, pf=None, ts=None):
        ext().wo_a(o2, pos, self.cs, L.wa, L.ws, lora, self.G, self.HD, self.NOPE, cfg[0], cfg[1], cfg[2], qv,
                   flags, True, xq, xs, pf, ts)
        return lora

    def k3(self, L, lora, out, cfg, impl="copy", flags=0, xq=None, xsf=None):
        if impl == "landed":
            return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(lora, L.wb, L.wbsf, cfg[0], cfg[1], cfg[2], 0, True,
                                                         None, None, xq, xsf)[0]
        ext().port_mm(lora, L.wb, L.wbsf, out, cfg[0], cfg[1], cfg[2], flags, True, xq, xsf)
        return out


def unswizzle(sf, rows, k):
    cols = k // 32
    pc = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    idx = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * pc)
    return sf.reshape(-1).view(torch.uint8)[idx]


def k1_ref(env, o3, pos, M):
    """K1's e4m3 bytes [M, G, GW] and UE8M0 scale bytes [M, G, GW/128] (all four replicated slots checked)."""
    sc = C.Scratch(env.dims, M)
    B.inv_rope_quant128(o3, pos, env.cs, sc.x_q, groups=env.G, group_width=env.GW, nope_dim=env.NOPE,
                        rope_dim=env.ROPE)
    torch.cuda.synchronize()
    vals = sc.x_q.values.view(torch.uint8).permute(0, 2, 1).contiguous()  # [M, G, GW]
    sm = sc.x_q.scale_mma.view(torch.uint8)
    m = torch.arange(M, device="cuda")
    s = sm[m % 32, (m // 32) % 4, m // 128]  # [M, 4, GW/128, G]
    rep_ok = bool((s == s[:, :1]).all())
    return vals, s[:, 0].permute(0, 2, 1).contiguous(), rep_ok


def stress_inputs(env, M, seed):
    """Random o3 with magnitude sweeps, exact 448*2^k amax blocks, zero blocks and tiny blocks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    d = env.dims
    o3 = torch.randn(M, d["heads_tp"], d["head_dim"], device="cuda", generator=g)
    scale = torch.exp2(torch.randint(-20, 8, (M, d["heads_tp"], d["head_dim"] // 128, 1), device="cuda",
                                     generator=g).float())
    o3 = (o3.view(M, d["heads_tp"], -1, 128) * scale).view_as(o3)
    o3 = o3.to(torch.bfloat16)
    for m in range(M):
        h = (seed + m) % d["heads_tp"]
        o3[m, h, :128] = 0
        o3[m, (h + 1) % d["heads_tp"], 128:256] *= 1e-30
        for k in range(3):
            hh = (h + 2 + k) % d["heads_tp"]
            blk = o3[m, hh, 128 * k:128 * (k + 1)]
            blk.clamp_(-1, 1)
            blk[5] = 448.0 * 2.0 ** (k * 7 - 9)
            blk.mul_(1.0)
    pos = torch.randint(0, 65536, (M,), device="cuda", generator=g, dtype=torch.int32)
    return o3, pos


# ------------------------------------------------------------------------------------------------------------------
def check(env):
    E = ext()
    res = {"passed": True}
    L = env.layers[0]
    # (1) K2' quant bytes vs K1, per PTX variant, over stress inputs
    qv_bad = {qv: 0 for qv in (0, 1)}
    n_cases = 0
    for M in MS:
        for seed in range(6):
            o3, pos = stress_inputs(env, M, 100 * M + seed) if seed else C.make_inputs(env.dims, M, 7 + M)
            ref_q, ref_s, rep_ok = k1_ref(env, o3, pos, M)
            if not rep_ok:
                res["passed"] = False
                log("K1 scale slots not replicated?!")
            n_cases += 1
            for qv in qv_bad:
                xq = torch.zeros(M, env.G, env.GW, dtype=torch.uint8, device="cuda")
                xs = torch.zeros(M, env.G, env.GW // 128, dtype=torch.uint8, device="cuda")
                lora = torch.empty(M, env.G * env.R, dtype=torch.bfloat16, device="cuda")
                env.k2(L, o3.view(M, -1), pos, lora, (4, 4, 1), qv=qv, xq=xq, xs=xs)
                torch.cuda.synchronize()
                nb = int((xq != ref_q).sum()) + int((xs != ref_s).sum())
                qv_bad[qv] += nb
    res["k2_quant_bytes_mismatch_by_variant"] = qv_bad
    res["k2_quant_cases"] = n_cases
    log("K2' vs K1 quant byte mismatches per variant:", qv_bad)
    if qv_bad[0] != 0:
        res["passed"] = False
    # (2) every K2' config: quant bytes equal + output vs b12x tmp and FP64; K3' quant bytes vs mxfp8_quantize;
    #     chain output vs main and FP64; the prototype copy (flags 0/2) bitwise == landed op.
    rows = []
    k2_cfgs = [(4, 4, 1), (4, 8, 1), (4, 4, 2), (4, 2, 1), (4, 8, 2), (8, 4, 1), (8, 8, 2), (4, 2, 2), (8, 4, 2),
               (4, 1, 1)]
    k3_cfgs = [(4, 1, 2), (4, 2, 2), (4, 2, 1), (4, 4, 1), (4, 4, 2), (8, 2, 2), (4, 1, 1), (8, 1, 2), (8, 2, 4),
               (8, 4, 2), (4, 8, 1), (8, 4, 1), (4, 2, 4)]
    for M in MS:
        o3, pos = C.make_inputs(env.dims, M, 11 + M)
        o_orig = o3.clone()
        ref_out, ref_lora = C.fp64_reference(o_orig, pos, env.cs, env.base, env.dims)
        sc = C.Scratch(env.dims, M)
        main_out = env.main_layer(L, o3, pos, sc, M).clone()
        main_lora = sc.tmp.permute(0, 2, 1).reshape(M, -1).clone()
        torch.cuda.synchronize()
        assert torch.equal(o3, o_orig)
        ref_q, ref_s, _ = k1_ref(env, o3, pos, M)
        row = {"M": M, "main_lora_err": C.err(main_lora, ref_lora), "main_out_err": C.err(main_out, ref_out)}
        k2r = {}
        for cfg in k2_cfgs:
            if env.G * env.R % (16 * cfg[2]) or (env.GW // cfg[1] // 32) % cfg[0]:
                continue
            if E.wo_a_smem(cfg[0], cfg[1], cfg[2], M, env.GW) > 101376 - 64:
                continue
            xq = torch.zeros(M, env.G, env.GW, dtype=torch.uint8, device="cuda")
            xs = torch.zeros(M, env.G, env.GW // 128, dtype=torch.uint8, device="cuda")
            outs = []
            for flags in (0, 2, 0):
                lora = torch.full((M, env.G * env.R), float("nan"), dtype=torch.bfloat16, device="cuda")
                env.k2(L, o3.view(M, -1), pos, lora, cfg, flags=flags, xq=xq, xs=xs)
                outs.append(lora)
            torch.cuda.synchronize()
            det = all(torch.equal(outs[0], x) for x in outs[1:])
            qok = torch.equal(xq, ref_q) and torch.equal(xs, ref_s)
            e = C.err(outs[0], ref_lora)
            em = C.err(outs[0], main_lora)
            # Reordered class: within main's own (b12x) error vs FP64 (1.25x), and close to main's output.
            ok = (det and qok and bool(torch.isfinite(outs[0]).all())
                  and e["max_rel"] <= 1.25 * row["main_lora_err"]["max_rel"] and em["max_rel"] < 1.6e-2)
            res["passed"] &= ok
            k2r["%d,%d,%d" % cfg] = {"ok": ok, "det": det, "quant_bytes_equal": qok, "vs_fp64": e, "vs_main": em}
        row["k2"] = k2r
        lora = torch.empty(M, env.G * env.R, dtype=torch.bfloat16, device="cuda")
        env.k2(L, o3.view(M, -1), pos, lora, (4, 4, 1))
        q_ref, s_ref = torch.ops.trtllm.mxfp8_quantize(lora, True)
        k3r = {}
        for cfg in k3_cfgs:
            K = env.G * env.R
            if env.H % (16 * cfg[2]) or (K // cfg[1] // 32) % cfg[0] or (K // cfg[1] // 32) % 4:
                continue
            if E.port_smem(cfg[0], cfg[1], cfg[2], M, K) > 101376 - 64:
                continue
            xq = torch.zeros(M, K, dtype=torch.uint8, device="cuda")
            xsf = torch.zeros_like(s_ref.view(torch.uint8).reshape(-1))
            landed = env.k3(L, lora, None, cfg, impl="landed", xq=xq, xsf=xsf).clone()
            torch.cuda.synchronize()
            qok = torch.equal(xq, q_ref.view(torch.uint8)) and torch.equal(unswizzle(xsf, M, K),
                                                                          unswizzle(s_ref, M, K))
            xq2 = torch.zeros_like(xq)
            xsf2 = torch.zeros_like(xsf)
            cp0 = torch.full((M, env.H), float("nan"), dtype=torch.bfloat16, device="cuda")
            cp2 = torch.full((M, env.H), float("nan"), dtype=torch.bfloat16, device="cuda")
            env.k3(L, lora, cp0, cfg, impl="copy", flags=0, xq=xq2, xsf=xsf2)
            env.k3(L, lora, cp2, cfg, impl="copy", flags=2)
            torch.cuda.synchronize()
            qok2 = torch.equal(xq2, xq) and torch.equal(xsf2, xsf)
            same = torch.equal(cp0, landed) and torch.equal(cp2, landed)
            ok = qok and qok2 and same
            res["passed"] &= ok
            k3r["%d,%d,%d" % cfg] = {"ok": ok, "quant_bytes_equal": qok, "copy_quant_equal": qok2,
                                     "copy_bitwise_landed": same}
        row["k3"] = k3r
        chain = env.k3(L, lora, None, (4, 2, 2), impl="landed").clone()
        row["new_out_err"] = C.err(chain, ref_out)
        row["new_vs_main"] = C.err(chain, main_out)
        res["passed"] &= row["new_out_err"]["max_rel"] <= max(1.6e-2, 1.5 * row["main_out_err"]["max_rel"])
        log(f"CHECK M={M}: main lora {row['main_lora_err']['max_rel']:.3e} out {row['main_out_err']['max_rel']:.3e}"
            f" | new out {row['new_out_err']['max_rel']:.3e} vs main {row['new_vs_main']['max_rel']:.3e} | "
            f"k2 ok {sum(v['ok'] for v in k2r.values())}/{len(k2r)} k3 ok {sum(v['ok'] for v in k3r.values())}/"
            f"{len(k3r)}")
        rows.append(row)
    res["rows"] = rows
    return res


# ------------------------------------------------------------------------------------------------------------------
def nvml_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUUID(h)
            u = u.decode() if isinstance(u, bytes) else u
            if u.replace("GPU-", "") == uuid.replace("GPU-", ""):
                return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception as e:  # pragma: no cover
        return f"nvml error {e!r}"
    return "uuid not found"


def build_graph(fns):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_graph(g, reps, n=NL):
    out = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / n)
    return out


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)
    p10 = xs[int(0.1 * (n - 1))]
    p90 = xs[int(0.9 * (n - 1))]
    return {"med": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3), "noise": round((p90 - p10) / med, 4),
            "n": n}


class Bufs:
    def __init__(self, env, M):
        self.o3, self.pos = C.make_inputs(env.dims, M, 21 + M)
        self.o3_src = self.o3.clone()
        self.o2 = self.o3.view(M, -1)
        self.lora = torch.empty(M, env.G * env.R, dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty(M, env.H, dtype=torch.bfloat16, device="cuda")
        self.ar = torch.empty(M, env.H, dtype=torch.bfloat16, device="cuda")
        self.sc = C.Scratch(env.dims, M)


def arm_fns(env, bufs, M, arm):
    """arm: dict(kind=main|new, k2=cfg, k3=cfg, k2f, k3f, impl, pf, attn=None|'end'|'start', ar=None|'end')."""
    E = ext()
    fns = []
    attn_spin = arm.get("attn_spin", 9000)
    ar_spin = arm.get("ar_spin", 4000)
    for i in range(NL):
        L = env.layers[i]
        steps = []
        if arm.get("attn"):
            af = 2 if arm["attn"] == "start" else 0
            steps.append(lambda af=af: E.standin(bufs.o3_src, bufs.o3, attn_spin, af, 16, True))
        if arm["kind"] == "main":
            steps.append(lambda L=L: env.main_layer(L, bufs.o3, bufs.pos, bufs.sc, M))
            src = lambda: bufs.sc.out  # noqa: E731
        else:
            pf = None
            if arm.get("pf"):
                frac = arm["pf"]
                flat = L.wb.view(torch.uint8).view(-1)
                pf = flat[int(flat.numel() * (1 - frac)) // 16 * 16:]
            steps.append(lambda L=L, pf=pf: env.k2(L, bufs.o2, bufs.pos, bufs.lora, arm["k2"], flags=arm["k2f"],
                                                    pf=pf))
            steps.append(lambda L=L: env.k3(L, bufs.lora, bufs.out, arm["k3"], impl=arm.get("impl", "copy"),
                                            flags=arm.get("k3f", 0)))
            src = lambda: bufs.out  # noqa: E731
        if arm.get("ar"):
            steps.append(lambda: E.standin(src(), bufs.ar, ar_spin, 0, 8, True))
        fns.append(lambda steps=steps: [s() for s in steps])
    return fns


def sweep(env, quick_reps=8):
    res = {}
    k2_cfgs = [(4, 4, 1), (4, 8, 1), (4, 4, 2), (4, 2, 1), (4, 8, 2), (8, 4, 1), (8, 8, 2), (4, 2, 2), (8, 4, 2)]
    k3_cfgs = [(4, 1, 2), (4, 2, 2), (4, 2, 1), (4, 4, 1), (4, 4, 2), (8, 2, 2), (4, 1, 1), (8, 1, 2), (8, 2, 4),
               (8, 4, 2), (4, 8, 1), (8, 4, 1)]
    E = ext()
    K = env.G * env.R
    for M in MS:
        bufs = Bufs(env, M)
        r = {}
        best3 = (4, 2, 2)
        for c3 in k3_cfgs:
            if E.port_smem(*c3, M, K) > 101376 - 64 or (K // c3[1] // 32) % c3[0]:
                continue
            arm = {"kind": "new", "k2": (4, 4, 1), "k3": c3, "k2f": 2, "k3f": 0}
            g = build_graph(arm_fns(env, bufs, M, arm))
            t = stats(time_graph(g, quick_reps))["med"]
            del g
            r["k2=4,4,1 k3=%d,%d,%d" % c3] = t
            log(f"SWEEP M={M} k2=(4,4,1) k3={c3}: {t:.2f} us/layer")
        best3 = min(((k, v) for k, v in r.items() if k.startswith("k2=4,4,1")), key=lambda kv: kv[1])[0]
        best3 = tuple(int(x) for x in best3.split("k3=")[1].split(","))
        for c2 in k2_cfgs:
            if E.wo_a_smem(*c2, M, env.GW) > 101376 - 64 or (env.GW // c2[1] // 32) % c2[0]:
                continue
            arm = {"kind": "new", "k2": c2, "k3": best3, "k2f": 2, "k3f": 0}
            g = build_graph(arm_fns(env, bufs, M, arm))
            t = stats(time_graph(g, quick_reps))["med"]
            del g
            r["k2=%d,%d,%d k3=%d,%d,%d" % (c2 + best3)] = t
            log(f"SWEEP M={M} k2={c2} k3={best3}: {t:.2f} us/layer")
        best = min(r.items(), key=lambda kv: kv[1])
        log(f"SWEEP M={M} best {best[0]} {best[1]:.2f}")
        res[M] = {"all": r, "best": best[0]}
        torch.cuda.empty_cache()
    return res


def parse_best(s):
    a, b = s.split(" ")
    return tuple(int(x) for x in a[3:].split(",")), tuple(int(x) for x in b[3:].split(","))


def table(env, best, rounds=2, reps=12):
    out = {}
    for M in MS:
        c2, c3 = best.get(M, ((4, 4, 1), (4, 2, 2)))
        bufs = Bufs(env, M)
        arms = {
            "main": {"kind": "main"},
            "new_end_end_landed": {"kind": "new", "k2": c2, "k3": c3, "k2f": 0, "impl": "landed"},
            "new_end_end": {"kind": "new", "k2": c2, "k3": c3, "k2f": 0, "k3f": 0},
            "new_start_end": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 0},
            "new_start_start": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 2},
            "new_start_end_pf50": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 0, "pf": 0.5},
            "new_start_end_pf100": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 0, "pf": 1.0},
            # behind a ~9 us latency-bound attention-merge stand-in (+ a 4 us plain-AR stand-in after K3/K3')
            "main_S_attnEnd": {"kind": "main", "attn": "end", "ar": "end"},
            "main_S_attnStart": {"kind": "main", "attn": "start", "ar": "end"},
            "new_S_attnEnd_k3End": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 0, "attn": "end",
                                    "ar": "end"},
            "new_S_attnEnd_k3Start": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 2, "attn": "end",
                                      "ar": "end"},
            "new_S_attnStart_k3End": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 0, "attn": "start",
                                      "ar": "end"},
            "new_S_attnStart_k3Start": {"kind": "new", "k2": c2, "k3": c3, "k2f": 2, "k3f": 2, "attn": "start",
                                        "ar": "end"},
            "S_only": {"kind": "none", "attn": "end", "ar": "end"},
        }
        graphs = {}
        for name, arm in arms.items():
            if arm["kind"] == "none":
                E = ext()
                fns = [lambda: (E.standin(bufs.o3_src, bufs.o3, 9000, 0, 16, True),
                                E.standin(bufs.out, bufs.ar, 4000, 0, 8, True)) for _ in range(NL)]
            else:
                fns = arm_fns(env, bufs, M, arm)
            graphs[name] = build_graph(fns)
        names = list(arms)
        samples = {n: [] for n in names}
        clocks = []
        for rnd in range(rounds):
            order = names + names[::-1]
            for n in order:
                time_graph(graphs[n], 2)
                samples[n] += time_graph(graphs[n], reps)
            clocks.append(nvml_clock())
            log(f"TABLE M={M} round {rnd} done, SM clock {clocks[-1]}")
        r = {n: stats(v) for n, v in samples.items()}
        r["_cfg"] = {"k2": c2, "k3": c3}
        r["_sm_clock_mhz"] = clocks
        r["_floor_us"] = round(env.floor_us, 2)
        for n in names:
            log(f"TABLE M={M} {n:28s} {r[n]['med']:7.2f} us/layer (p10 {r[n]['p10']:.2f} p90 {r[n]['p90']:.2f}, "
                f"noise {r[n]['noise']:.3f})")
        out[M] = r
        del graphs
        torch.cuda.empty_cache()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CKPT")
    torch.manual_seed(0)
    env = Env(ckpt)
    ext()
    log("SM clock at start", nvml_clock())
    result = {"mode": mode}
    passed = True
    if mode in ("check", "all"):
        c = check(env)
        passed &= c["passed"]
        result["check"] = c
        log("CHECK passed:", c["passed"])
    if mode in ("sweep", "all", "table"):
        if mode == "table" and len(sys.argv) > 3:
            best = {int(k): parse_best(v) for k, v in json.loads(sys.argv[3]).items()}
        else:
            s = sweep(env)
            result["sweep"] = s
            best = {M: parse_best(v["best"]) for M, v in s.items()}
        result["table"] = table(env, best)
    result["passed"] = bool(passed)
    art = os.environ.get("AUTO_DEV_ARTIFACTS", tempfile.gettempdir())
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, f"lowm_wo_{mode}.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    summ = {"passed": bool(passed), "mode": mode}
    if "table" in result:
        summ["table"] = {M: {n: v["med"] for n, v in r.items() if not n.startswith("_")} for M, r in
                         result["table"].items()}
    print(json.dumps(summ, default=str))


if __name__ == "__main__":
    main()
