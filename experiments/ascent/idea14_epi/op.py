"""Idea 14 op check / bench: fused QKV epilogue (flux3_qkv_epi_fp8) vs main's chain
(strided.gemm_qkv3 + qknorm_rope_fp8 VEC/FOLD/STATIC_QK, discard_qk) on synthetic NVFP4 inputs.

check: qkv8 bytes, scales[8] and the gemm buffer's v plane bitwise vs main, M in MS; eager calls
       0..3 vs main and vs call 0 (outputs re-prefilled 0xFF-ish by fresh allocation each call),
       CUDA-graph capture + 3 replays vs main, inputs unchanged; keep_v False and True.
bench: graph replay (R calls per graph), interleaved main vs candidate, median of ITERS.
Last stdout line: JSON.
"""

import json
import os
import sys
import time

import torch

N = 3072
K = 3072
SM = 128 ** -0.5


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sf_rows(m):
    return (m + 127) // 128 * 128


def make_inputs(m, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    a = torch.randint(0, 256, (m, K // 2), generator=g, dtype=torch.uint8).to(dev)
    w = torch.randint(0, 256, (3 * N, K // 2), generator=g, dtype=torch.uint8).to(dev)
    sfa = torch.randint(0x30, 0x40, (sf_rows(m) * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    sfb = torch.randint(0x30, 0x40, (3 * N * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    alpha = torch.tensor([0.0075, 0.0125, 0.005], dtype=torch.float32, device=dev)
    qn = (1.0 + 0.3 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    kn = (0.8 + 0.4 * torch.randn(128, generator=g)).to(torch.bfloat16).to(dev)
    ang = torch.rand(m, 64, generator=g) * 6.28
    cs = torch.cat([ang.cos(), ang.sin()], -1).float().contiguous().to(dev)
    pos = torch.arange(m, dtype=torch.int64, device=dev)
    return a, w, sfa, sfb, alpha, qn, kn, cs, pos


def main_chain(strided, qr8, inp, keep_v):
    a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
    m = a.shape[0]
    out = torch.empty((m, 3 * N), dtype=torch.bfloat16, device=a.device)
    strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)
    p = out.view(3, m, N)
    q8, sc = qr8.qknorm_rope_fp8(p[0], p[1], p[2], qn, kn, cs, pos, SM, discard_qk=True,
                                 keep_v=keep_v)
    return q8, sc, out


def cand(epi, inp, keep_v):
    a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
    return epi.qkv_rope_fp8(a, w, sfa, sfb, alpha, qn, kn, cs, SM, keep_v)


def u8(t):
    return t.view(torch.uint8)


def compare(tag, ref, got, fails, check_v=True):
    r8, rs, rout = ref
    g8, gs, gout = got
    m = r8.shape[1]
    for p, nm in enumerate("qkv"):
        eq = torch.equal(u8(r8[p]), u8(g8[p]))
        if not eq:
            diff = (u8(r8[p]) != u8(g8[p]))
            n = int(diff.sum())
            idx = diff.nonzero()[:4].tolist()
            fails.append(f"{tag} qkv8[{nm}] {n} bytes differ, first {idx}")
    if not torch.equal(rs.view(torch.int32), gs.view(torch.int32)):
        fails.append(f"{tag} scales differ ref={rs.tolist()} got={gs.tolist()}")
    if check_v:
        rv = rout.view(3, m, N)[2]
        gv = gout.view(3, m, N)[2]
        if not torch.equal(rv.view(torch.int16), gv.view(torch.int16)):
            n = int((rv.view(torch.int16) != gv.view(torch.int16)).sum())
            fails.append(f"{tag} v plane {n} elems differ")


def check(strided, qr8, epi, ms):
    dev = torch.device("cuda")
    fails = []
    info = {"stages": int(epi.module().qkv3_epi_stages()), "smem": int(epi.module().qkv3_epi_smem()),
            "main_stages": int(strided.qkv_pingpong_module().stages_pingpong()),
            "main_smem": int(strided.qkv_pingpong_module().smem_pingpong())}
    log("info", info)
    for m in ms:
        inp = make_inputs(m, dev)
        pristine = [t.clone() for t in inp]
        for keep_v in (True, False):
            # main reference; v plane compared only when keep_v (else main discards v's lines).
            ref = main_chain(strided, qr8, inp, True)
            ref = tuple(t.clone() for t in ref)
            torch.cuda.synchronize()
            call0 = None
            for i in range(4):
                got = cand(epi, inp, keep_v)
                torch.cuda.synchronize()
                compare(f"M{m} kv{int(keep_v)} eager{i}", ref, got, fails, check_v=keep_v)
                if call0 is None:
                    call0 = tuple(t.clone() for t in got)
                else:
                    compare(f"M{m} kv{int(keep_v)} eager{i} vs call0", call0, got, fails,
                            check_v=keep_v)
            # graph capture + replays
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                cand(epi, inp, keep_v)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = cand(epi, inp, keep_v)
            for r in range(3):
                for t in gout:
                    t.view(torch.uint8).fill_(0xFF) if t.dtype != torch.float32 else t.fill_(-1.0)
                g.replay()
                torch.cuda.synchronize()
                compare(f"M{m} kv{int(keep_v)} replay{r}", ref, gout, fails, check_v=keep_v)
            del g
            for t, p0 in zip(inp, pristine):
                if not torch.equal(t, p0):
                    fails.append(f"M{m} input mutated")
        # stats for scale
        q8 = ref[0]
        log(f"M{m}: fails so far {len(fails)}; scales {ref[1].tolist()}; "
            f"q8 nonzero frac {float((u8(q8[0]) != 0).float().mean()):.3f}")
        del inp, pristine, ref
        torch.cuda.empty_cache()
    return {"passed": not fails, "info": info, "fails": fails[:30],
            "compared": "fused QKV epilogue (qkv8 q/k/v bytes, scales[8], gemm v plane) vs main "
                        "gemm_qkv3 + qknorm_rope_fp8 (VEC, STATIC_QK, discard_qk), bitwise; eager x4 "
                        "vs call0, graph capture + 3 replays, keep_v True/False, inputs unchanged",
            "Ms": ms}


def killswitch(strided, qr8, epi, ms):
    """Run under TRTLLM_FLUX3_QKV_EPI_FP8=0: the route never fires, the fused module is never
    built or called, and the routed op equals main's chain byte for byte."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    dev = torch.device("cuda")
    fails = []
    calls = {"fused": 0}
    orig = epi.qkv_rope_fp8

    def counted(*a, **k):
        calls["fused"] += 1
        return orig(*a, **k)

    epi.qkv_rope_fp8 = counted
    env = os.environ.get("TRTLLM_FLUX3_QKV_EPI_FP8")
    if env != "0" or epi.enabled():
        fails.append(f"kill switch not active: env={env!r} enabled={epi.enabled()}")
    for m in ms:
        inp = make_inputs(m, dev)
        a, w, sfa, sfb, alpha, qn, kn, cs, pos = inp
        if fp._qkv_epi_fp8_route(a, w, qn, kn, cs):
            fails.append(f"M{m}: route fires with the kill switch")
        for keep_v in (True, False):
            ref = tuple(t.clone() for t in main_chain(strided, qr8, inp, True))
            got = fp.nvfp4_qkv_rope_fp8_impl(a, w, sfa, sfb, qn, kn, cs, pos, alpha, SM, keep_v)
            torch.cuda.synchronize()
            compare(f"M{m} kv{int(keep_v)} killswitch", ref, got, fails, check_v=keep_v)
    if calls["fused"]:
        fails.append(f"fused route called {calls['fused']} times")
    built = epi.module.cache_info().currsize
    return {"passed": not fails, "fails": fails, "fused_calls": calls["fused"],
            "fused_module_built": bool(built), "Ms": ms,
            "compared": "TRTLLM_FLUX3_QKV_EPI_FP8=0: routed nvfp4_qkv_rope_fp8_impl vs main's "
                        "gemm_qkv3 + qknorm_rope_fp8 chain, qkv8/scales/v bitwise; route predicate "
                        "false; fused module never called"}


def bench(strided, qr8, epi, ms):
    dev = torch.device("cuda")
    R = int(os.environ.get("B_R", "4"))
    ITERS = int(os.environ.get("B_ITERS", "50"))
    keep_v = os.environ.get("B_KEEPV", "0") == "1"
    res = {}
    for m in ms:
        inp = make_inputs(m, dev)
        fns = {"main": lambda: main_chain(strided, qr8, inp, keep_v),
               "cand": lambda: cand(epi, inp, keep_v),
               "gemm_only": lambda: strided.gemm_qkv3(inp[0], inp[1], inp[2], inp[3], inp[4],
                                                      torch.empty((m, 3 * N), dtype=torch.bfloat16,
                                                                  device=dev))}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        graphs = {}
        for name, f in fns.items():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                f()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    f()
            graphs[name] = g
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        times = {n: [] for n in graphs}
        for _ in range(ITERS):
            for name, g in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) / R)
        row = {}
        for name, t in times.items():
            t = sorted(t)
            med = t[len(t) // 2]
            row[name] = {"ms": round(med, 4),
                         "nf": round((t[int(0.9 * len(t)) - 1] - t[int(0.1 * len(t))]) / med, 4)}
        log(f"M{m}", json.dumps(row))
        res[m] = row
        del graphs, inp
        torch.cuda.empty_cache()
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    ms = [int(x) for x in os.environ.get("OP_MS", "16896,4176,33280").split(",")]
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (
        flux3_sm120_qknorm_rope_fp8 as qr8,
    )
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    dev = torch.device("cuda")
    t0 = time.time()
    strided.preload(dev)
    epi.preload(dev)
    log(f"modules ready {time.time() - t0:.1f}s VEC={qr8.VEC} STATIC={qr8.STATIC_QK}")
    if mode == "check":
        res = check(strided, qr8, epi, ms)
    elif mode == "killswitch":
        res = killswitch(strided, qr8, epi, ms)
    else:
        res = bench(strided, qr8, epi, ms)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
