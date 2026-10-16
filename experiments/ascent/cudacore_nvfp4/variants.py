# EXPERIMENT (candidate tree only, needs the TRTLLM_CCNVFP4_VARIANT build):
# compare K-loop structures of cudaCoreGemmFp4 in one process, cold L2, CUDA graphs, PDL on.
#   V0 = main's structure (no pre-PDL loads), V1 = first-step weight loads before the PDL wait,
#   V2 = V1 with the first step peeled (no branch in the loop), V3 = V1 + next-step weight
#   prefetch while the current step computes.
# Each is timed as chain (fp4_quantize -> GEMV with swizzled act-SF read) and as lin
# (fp4_quantize -> block_scale_interleave_reverse -> GEMV with linear act-SF read).
# Also: DRAM probe (>=1 GB read, 1 GB D2D copy, nvidia-smi line), FP8 cudaCoreGemmV2 rate,
# register usage of the cudaCoreGemmFp4 instantiations (cuobjdump).
# argv[1] = JSON of main's output hashes (baseline arm of the pair) for the bitwise check.
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}
COPIES = {"gate_up": 4, "down": 8}
ROUNDS = 2
WARMUP, ITERS = 10, 60
TIME_MS = [1, 2, 4]
CHECK_MS = [1, 2, 3, 4, 8]
VARS = [0, 1, 2, 3]
FP8_MAX, E2M1_MAX = 448.0, 6.0
EXPECTED = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}


def sha(t):
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def make_weight(n, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    w = torch.randn(n, k, device=dev, generator=g).to(torch.bfloat16)
    w.view(n, k // 16, 16)[::5, ::11, :] = 0
    gs = (FP8_MAX * E2M1_MAX / w.float().abs().amax()).reshape(1).float()
    wq, wsf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return wq, wsf, gs


def make_act(m, k, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = (torch.randn(m, k, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    x.view(m, k // 16, 16)[:, 3::7, :] = 0
    gs = (FP8_MAX * E2M1_MAX / x.float().abs().amax()).reshape(1).float()
    return x, gs


def quant(x, gsa):
    return torch.ops.trtllm.fp4_quantize(x, gsa, 16, False, True)


def v_chain(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, asf, wsf, alpha, None, torch.bfloat16,
                                                 act_sf_swizzled=True)


def v_lin(x, gsa, wq, wsf, alpha):
    aq, asf = quant(x, gsa)
    m = aq.shape[0]
    lin = torch.ops.trtllm.block_scale_interleave_reverse(asf.view((m + 127) // 128 * 128, -1))
    return torch.ops.trtllm.cuda_core_nvfp4_gemm(aq, wq, lin, wsf, alpha, None, torch.bfloat16)


def v_quant(x, gsa, wq, wsf, alpha):
    return quant(x, gsa)[0]


def set_variant(v):
    os.environ["TRTLLM_CCNVFP4_VARIANT"] = str(v)


def capture(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            fn(*c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(*c) for c in calls]
    torch.cuda.synchronize()
    return g, outs


def stats(ts):
    s = sorted(ts)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return q(0.5), q(0.9), q(0.1)


def time_graphs(graphs, ncall):
    ts = {vn: [] for vn in graphs}
    for _ in range(WARMUP):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    for _ in range(ITERS):
        for vn, g in graphs.items():
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            ts[vn].append(e0.elapsed_time(e1) / ncall)
    return ts


def dram_probe():
    res = {}
    try:
        q = ("name,ecc.mode.current,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,"
             "clocks_throttle_reasons.active,temperature.gpu,power.draw")
        res["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
            text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        res["nvidia_smi"] = repr(e)
    nbytes = 2 << 30
    buf = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    ts = []
    for it in range(30):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        buf.sum()
        e1.record()
        e1.synchronize()
        if it >= 5:
            ts.append(e0.elapsed_time(e1))
    med, p90, p10 = stats(ts)
    res["read_2GB_TBps_med"] = round(nbytes / (med * 1e-3) / 1e12, 3)
    res["read_2GB_TBps_p90time"] = round(nbytes / (p90 * 1e-3) / 1e12, 3)
    del buf
    src = torch.ones((1 << 30) // 4, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    ts = []
    for it in range(30):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        dst.copy_(src)
        e1.record()
        e1.synchronize()
        if it >= 5:
            ts.append(e0.elapsed_time(e1))
    med, p90, p10 = stats(ts)
    res["d2d_1GB_rd_plus_wr_TBps_med"] = round(2 * (1 << 30) / (med * 1e-3) / 1e12, 3)
    del src, dst
    torch.cuda.empty_cache()
    log("[probe] " + json.dumps(res))
    return res


def fp8_gemv_rate():
    res = {}
    for n, k, copies in ((14336, 5120, 6), (16384, 5120, 6)):
        w = (torch.randn(n, k, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = [w] + [w.clone() for _ in range(copies - 1)]
        sa = torch.tensor([0.01], device=dev)
        sb = torch.tensor([0.02], device=dev)
        a = (torch.randn(1, k, device=dev)).to(torch.float8_e4m3fn)
        calls = [(a, ws[i % copies]) for i in range(copies * ROUNDS)]

        def fn(a_, w_):
            return torch.ops.trtllm.cuda_scaled_mm(a_, w_.t(), sa, sb, None, torch.bfloat16)

        g, _ = capture(fn, calls)
        ts = time_graphs({"fp8": g}, len(calls))["fp8"]
        med, p90, p10 = stats(ts)
        res[f"fp8_{n}x{k}_M1"] = {"us": round(med * 1e3, 2),
                                   "TBps": round(n * k / (med * 1e-3) / 1e12, 3)}
        del g, ws, w
        torch.cuda.empty_cache()
    log("[fp8] " + json.dumps(res))
    return res


def reg_usage():
    out = {}
    try:
        libdir = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs")
        libs = sorted(glob.glob(os.path.join(libdir, "*.so*")),
                      key=lambda p: (os.path.basename(p) != "libtensorrt_llm.so", p))
        for lib in libs:
            log(f"[regs] cuobjdump -res-usage {lib} ...")
            p = subprocess.run(["cuobjdump", "-res-usage", lib], capture_output=True, text=True, timeout=600)
            lines = p.stdout.splitlines()
            for i, line in enumerate(lines):
                if "cudaCoreGemmFp4" in line and "Function" in line:
                    name = line.split("Function", 1)[1].strip().rstrip(":")
                    dem = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
                    if "__nv_bfloat16" not in dem:
                        continue
                    mm = re.search(r"cudaCoreGemmFp4<[^>]*?, (\d+), 2, 128, (true|false)(?:, (\d+))?>", dem)
                    nxt = lines[i + 1] if i + 1 < len(lines) else ""
                    reg = re.search(r"REG:(\d+)", nxt)
                    stack = re.search(r"STACK:(\d+)", nxt)
                    key = (f"M{mm.group(1)}/{'swz' if mm.group(2) == 'true' else 'lin'}/V{mm.group(3)}"
                           if mm else dem[:120])
                    if mm and int(mm.group(1)) > 8:
                        continue
                    out[key] = f"REG:{reg.group(1) if reg else '?'} STACK:{stack.group(1) if stack else '?'}"
            if out:
                break
    except Exception as e:  # pragma: no cover
        out["err"] = repr(e)[:200]
    log("[regs] " + json.dumps(out, sort_keys=True))
    return out


def main():
    probe = dram_probe()
    fp8 = fp8_gemv_rate()
    data = {}
    for name, (n, k) in SHAPES.items():
        wq, wsf, gsw = make_weight(n, k, seed=17 if name == "gate_up" else 29)
        copies = [(wq, wsf)] + [(wq.clone(), wsf.clone()) for _ in range(COPIES[name] - 1)]
        data[name] = dict(n=n, k=k, copies=copies, gsw=gsw, wbytes=wq.numel() + wsf.numel())
        log(f"{name}: {COPIES[name]} copies x {data[name]['wbytes'] / 1e6:.1f} MB")

    def calls_for(d, m):
        out = []
        for i in range(len(d["copies"]) * ROUNDS):
            x, gsa = make_act(m, d["k"], seed=1000 * m + i)
            alpha = (1.0 / (gsa * d["gsw"])).float()
            wq, wsf = d["copies"][i % len(d["copies"])]
            out.append((x, gsa, wq, wsf, alpha))
        return out

    exact = {}
    all_ok = True
    for name, d in data.items():
        for m in CHECK_MS:
            calls = calls_for(d, m)
            for v in VARS:
                set_variant(v)
                for kind, fn in (("swz", v_chain), ("lin", v_lin)):
                    h = sha(torch.stack([fn(*c) for c in calls]))
                    ok = EXPECTED.get(f"{name}/M{m}") == h
                    all_ok = all_ok and ok
                    exact[f"{name}/M{m}/V{v}/{kind}"] = ok
    log(f"[exact] all variants bitwise == main: {all_ok}; " + json.dumps(exact))

    detail = {}
    for name, d in data.items():
        for m in TIME_MS:
            calls = calls_for(d, m)
            graphs = {}
            for v in VARS:
                set_variant(v)
                graphs[f"V{v}/swz"], _ = capture(v_chain, calls)
                graphs[f"V{v}/lin"], _ = capture(v_lin, calls)
            graphs["quant"], _ = capture(v_quant, calls)
            ts = time_graphs(graphs, len(calls))
            rec = {}
            for vn, t in ts.items():
                med, p90, p10 = stats(t)
                rec[vn] = {"us": round(med * 1e3, 2), "nf": round((p90 - p10) / med, 4)}
                if vn != "quant":
                    tb = d["wbytes"] / (med * 1e-3) / 1e12
                    rec[vn]["TBps"] = round(tb, 3)
                    rec[vn]["pct_read"] = round(100 * tb / probe["read_2GB_TBps_med"], 1)
            detail[f"{name}/M{m}"] = rec
            log(f"{name} M={m}: " + json.dumps(rec))
            del graphs
            torch.cuda.synchronize()
    set_variant(1)
    regs = reg_usage()
    log(json.dumps({"passed": all_ok, "probe": probe, "fp8": fp8, "detail": detail, "regs": regs}))


main()
