"""SM120 port of MiniMax-M3 mxfp8LowMGemm for the CSA2 decode input projections (DeepSeek-V4.1-Flash TP4).

Per-rank shapes: wq_a||wkv [1792, 5120], wq_b [8192, 1280], wq_b||index_wq_b [12288, 1280] (M=1 path),
index_wq_b [4096, 1280].

Modes (argv[1]):
  sweep   - every built (warps, cluster_k, row_tiles) per shape x M in 1/2/4/8: correctness vs the FP32 dequant
            reference + cold-L2 PDL CUDA-graph chain timing; also main (quantize + flashinfer_mm_mxfp8 / b12x) and
            the port with a pre-quantized input. Prints a table (CTAs/SM, smem/CTA, floor).
  smoke   - quick correctness only.
"""

import json
import math
import sys

import torch

SHAPES = [(1792, 5120), (8192, 1280), (12288, 1280), (4096, 1280)]
MS = [1, 2, 4, 8]
SEED = 1234
CONFIGS = [(4, 8, 1), (4, 8, 2), (4, 4, 1), (4, 4, 2), (4, 2, 1), (4, 2, 2), (4, 2, 4), (4, 1, 1), (4, 1, 2),
           (4, 1, 4), (8, 4, 1), (8, 4, 2), (8, 2, 2), (8, 2, 4), (8, 1, 2), (8, 1, 4)]
DRAM_GBPS = 1507.0
SMEM_PER_SM = 102400
SMEM_PER_CTA_RESERVED = 1024
MAX_OPTIN = 101376
L2_BYTES = 128 * 1024 * 1024


def log(*a):
    print(*a, flush=True)


def smem_bytes(warps, ck, rt, m, k):
    kc = k // ck
    kcb = kc // 32
    rows = 16 * rt
    w = rows * (kc + 32)
    red = warps * rows * 8 * 4
    aliased = red <= w
    return (w + rt * (kcb // 4) * 256 + m * kc + ((m * kcb + 15) // 16) * 16 + (ck - 1) * rows * m * 4
            + (0 if aliased else red))


def valid(cfg, n, k, m):
    warps, ck, rt = cfg
    kc = k // ck
    if n % (16 * rt) or k % (ck * 128) or (kc // 32) % warps:
        return False
    return smem_bytes(warps, ck, rt, m, k) + 64 <= MAX_OPTIN


def ctas_per_sm(cfg, m, k):
    warps, ck, rt = cfg
    s = smem_bytes(warps, ck, rt, m, k) + 32 + SMEM_PER_CTA_RESERVED
    return min(SMEM_PER_SM // s, 1536 // (warps * 32), 32)


def make_weight(n, k, gen):
    w = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 128, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    sw = torch.ops.trtllm.block_scale_interleave(raw.contiguous()).reshape(-1)
    return w, sw, raw


def dequant_ref(x, w, raw):
    """FP32 product of mxfp8(x) (as trtllm::mxfp8_quantize) and the dequantized weight."""
    m, k = x.shape
    xq, xs = torch.ops.trtllm.mxfp8_quantize(x, True)
    xsf = unswizzle(xs, m, k)
    xd = (xq.float().view(m, k // 32, 32) * torch.exp2(xsf.float() - 127).unsqueeze(-1)).view(m, k)
    n = w.shape[0]
    wd = (w.float().view(n, k // 32, 32) * torch.exp2(raw.float() - 127).unsqueeze(-1)).view(n, k)
    return (xd.double() @ wd.double().t()).float()


def unswizzle(sf, rows, k):
    cols = k // 32
    pc = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device).view(-1, 1)
    c = torch.arange(cols, device=sf.device).view(1, -1)
    idx = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * pc)
    return sf.reshape(-1).view(torch.uint8)[idx]


def err(out, ref):
    d = (out.float() - ref).abs()
    scale = ref.abs().max().item()
    return d.max().item(), d.max().item() / max(scale, 1e-30)


def main_mm(x, w, sw):
    q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
    return torch.ops.trtllm.flashinfer_mm_mxfp8(q, s, w, sw, torch.bfloat16)


def port_mm(x, w, sw, cfg, n_split=0):
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, w, sw, cfg[0], cfg[1], cfg[2], n_split, True)


def port_preq_mm(x, w, sw, cfg):
    q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
    return torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, w, sw, cfg[0], cfg[1], cfg[2], 0, True, q, s)


def chain_time(fn, copies, reps=15):
    """Cold-L2 chain: one graph runs fn over every weight copy (copies exceed L2), per-call us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for wc, sc in copies[:2]:
            fn(wc, sc)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for wc, sc in copies:
            fn(wc, sc)
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / len(copies))
    times.sort()
    med = times[len(times) // 2]
    p10 = times[int(0.1 * (len(times) - 1))]
    p90 = times[int(0.9 * (len(times) - 1))]
    del g
    return med, (p90 - p10) / med


def check_quant(x, w, sw, cfg):
    m, k = x.shape
    q_ref, s_ref = torch.ops.trtllm.mxfp8_quantize(x, True)
    xq = torch.zeros((m, k), dtype=torch.uint8, device="cuda")
    xsf = torch.zeros_like(s_ref.view(torch.uint8).reshape(-1))
    torch.ops.trtllm.csa2_mxfp8_low_m_gemm(x, w, sw, cfg[0], cfg[1], cfg[2], 0, False, None, None, xq, xsf)
    torch.cuda.synchronize()
    ok_q = torch.equal(xq, q_ref.view(torch.uint8))
    ok_s = torch.equal(unswizzle(xsf, m, k), unswizzle(s_ref, m, k))
    return ok_q and ok_s


def run(mode):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    import tensorrt_llm  # noqa: F401

    assert hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm"), "op not built"
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    rows = []
    all_ok = True
    for (n, k) in SHAPES:
        w, sw, raw = make_weight(n, k, gen)
        wbytes = n * k + sw.numel()
        floor_us = wbytes / (DRAM_GBPS * 1e3)
        ncopy = max(4, math.ceil(2 * L2_BYTES / wbytes))
        copies = [(w.clone(), sw.clone()) for _ in range(ncopy)] if mode == "sweep" else None
        for m in MS:
            x = (torch.randn((m, k), generator=gen, device="cuda") * 2).to(torch.bfloat16)
            ref = dequant_ref(x, w, raw)
            base = main_mm(x, w, sw)
            torch.cuda.synchronize()
            be, brel = err(base, ref)
            row = {"n": n, "k": k, "m": m, "floor_us": round(floor_us, 2), "main_err": brel}
            if mode == "sweep":
                row["main_us"], row["main_noise"] = chain_time(lambda wc, sc: main_mm(x, wc, sc), copies)
            cands = {}
            for cfg in CONFIGS:
                if not valid(cfg, n, k, m):
                    continue
                (out,) = port_mm(x, w, sw, cfg)
                (outp,) = port_preq_mm(x, w, sw, cfg)
                torch.cuda.synchronize()
                ae, arel = err(out, ref)
                same_preq = torch.equal(out, outp)
                vs_main = (out.float() - base.float()).abs().max().item()
                okq = check_quant(x, w, sw, cfg)
                ok = (torch.isfinite(out).all().item() and arel < 1.6e-2 and same_preq and okq)
                all_ok &= ok
                c = {"rel_err": arel, "max_abs": ae, "vs_main_abs": vs_main, "preq_bitwise": same_preq,
                     "xq_sf_bitwise": okq, "ok": ok, "grid": n // (16 * cfg[2]) * cfg[1],
                     "ctas_per_sm": ctas_per_sm(cfg, m, k), "smem": smem_bytes(*cfg, m, k)}
                if mode == "sweep":
                    c["us"], c["noise"] = chain_time(lambda wc, sc: port_mm(x, wc, sc, cfg), copies)
                    c["preq_us"], c["preq_noise"] = chain_time(lambda wc, sc: port_preq_mm(x, wc, sc, cfg), copies)
                cands["%d,%d,%d" % cfg] = c
                log(f"N={n} K={k} M={m} cfg={cfg} {json.dumps(c)}")
            row["cands"] = cands
            if mode == "sweep":
                best = min(cands.items(), key=lambda kv: kv[1]["us"])
                log(f"== N={n} K={k} M={m} main {row['main_us']:.2f}us floor {floor_us:.2f}us best {best[0]} "
                    f"{best[1]['us']:.2f}us (preq {best[1]['preq_us']:.2f}us)")
            rows.append(row)
        del copies
        torch.cuda.empty_cache()
    print(json.dumps({"passed": bool(all_ok), "mode": mode}))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "smoke")
