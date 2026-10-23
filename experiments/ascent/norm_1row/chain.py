"""Step 0 / pair microbench: decode layer chain with both gemma add+RMSNorm sites.

Per layer (M rows, L distinct layers in one CUDA graph, cold weights):
  out_proj Marlin W4A16 (5120 x 6144) -> post_attention norm -> fp4_quantize
  -> gate_up nvfp4_gemm -> fused_silu_mul_nvfp4 -> down nvfp4_gemm
  -> next_layer norm -> qkvz (Marlin W4A16 at M <= cap, else fp4_quantize +
  nvfp4_gemm), as main routes them (base 8b7eaf6c: both decode norm sites call
  trtllm::flashinfer_gemma_fused_add_rmsnorm).

Arms (env CHAIN_ARMS, comma list): S / S2 = stock FlashInfer (A/A pair);
"<arm>:<tpr>" = the 1-row module at both sites; "OP" = the custom op as this
tree routes it (pair mode). Each arm its own graph; rotating order, residual
reset + 256 MB READ flush before every replay, outside the events.
Last line: evidence JSON for the arm named by CHAIN_REPORT (default OP).
"""
import json
import os
import statistics
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402
from flashinfer.norm import gemma_fused_add_rmsnorm  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import fused_silu_mul_nvfp4 as _fs  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as LIN  # noqa: E402

try:
    from tensorrt_llm._torch.modules import gemma_add_rmsnorm_1row as G
except ImportError:  # base tree
    G = None

dev = torch.device("cuda")
H, I, VD, QKVZ = 5120, 17408, 6144, 16384
L = int(os.environ.get("S0_LAYERS", "10"))
WARM = int(os.environ.get("S0_WARM", "10"))
ITERS = int(os.environ.get("S0_ITERS", "100"))
EPS = 1e-6
ARMS = tuple(os.environ.get("CHAIN_ARMS", "OP,OP2").split(","))
REPORT = os.environ.get("CHAIN_REPORT", ARMS[0])


def log(*a):
    print(*a, flush=True)


def cap():
    f = getattr(LIN, "fp8_nvfp4_decode_max_rows", None)
    if f is not None:
        try:
            return int(f(QKVZ, H))
        except Exception:
            pass
    return LIN._FP8_NVFP4_DECODE_MAX_ROWS


def marlin_copy(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    s = (w.abs().amax() / 448.0).float()
    wv = (w / s).to(torch.float8_e4m3fn).to(torch.bfloat16)
    del w
    amax = wv.abs().amax().float() * s
    gg = torch.exp2(torch.ceil(torch.log2(amax / 2688.0)))
    fp4, sf = torch.ops.trtllm.fp4_quantize(wv, (s / gg).reshape(1), 16, False, True)
    del wv
    return LIN._pack_nvfp4_weight_for_marlin(fp4, sf, gg.reshape(1).to(torch.float32), n, k, 16)


def nvfp4_w(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    wsf = torch.randint(0x28, 0x38, (n * k // 16, ), dtype=torch.uint8, device=dev, generator=g)
    return w, wsf


def marlin(x, c, n, k, mode):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, c[0], scale_a=None, scale_b=c[1], alpha=None,
                                              weight_global_scale=c[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k,
                                              workspace_mode=mode)


def build_weights():
    W = dict(
        alpha=torch.tensor([0.002], dtype=torch.float32, device=dev),
        gs_x=torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev),
        gs_h=torch.tensor([30.0], dtype=torch.float32, device=dev),
    )
    W["wo"] = [marlin_copy(H, VD, 1000 + i) for i in range(L)]
    W["wq"] = [marlin_copy(QKVZ, H, 2000 + i) for i in range(L)]
    W["wq4"] = [nvfp4_w(QKVZ, H, 2500 + i) for i in range(L)]
    W["wg"] = [nvfp4_w(2 * I, H, 3000 + i) for i in range(L)]
    W["wd"] = [nvfp4_w(H, I, 4000 + i) for i in range(L)]
    W["g1"] = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    W["g2"] = [(torch.randn(H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
    W["scrub"] = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev).normal_()
    W["sink"] = torch.empty((), dtype=torch.float32, device=dev)
    log(f"weights built (L={L})")
    return W


def norm_fn(arm):
    if arm.startswith("OP"):
        return lambda x, r, w: torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)
    if arm in ("S", "S2"):
        return lambda x, r, w: gemma_fused_add_rmsnorm(x, r, w, EPS, enable_pdl=True)
    a, t = arm.split(":")
    t = int(t)
    return lambda x, r, w: G.gemma_fused_add_rmsnorm_1row(x, r, w, EPS, True, tpr=t, arm=a)


class Chain:

    def __init__(self, m, W):
        self.m, self.W = m, W
        c = cap()
        self.mode = 2 if m > LIN._FP8_NVFP4_DECODE_MAX_ROWS else 0
        self.qkvz_marlin = m <= c
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        self.xa = (torch.randn((L, m, VD), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        self.r0 = (torch.randn((m, H), generator=gen, device=dev) * 3.0).to(torch.bfloat16)
        self.res = self.r0.clone()
        with autotune():
            self.reset()
            self.layer(0, norm_fn("S"))
        torch.cuda.synchronize()

    def reset(self):
        self.res.copy_(self.r0)

    def layer(self, i, nf):
        W = self.W
        y = marlin(self.xa[i], W["wo"][i], H, VD, self.mode)
        nf(y, self.res, W["g1"][i])
        q, sf = torch.ops.trtllm.fp4_quantize(y, W["gs_x"], 16, False)
        h = torch.ops.trtllm.nvfp4_gemm(q, W["wg"][i][0], sf, W["wg"][i][1], W["alpha"], torch.bfloat16)
        hq, hsf = torch.ops.trtllm.fused_silu_mul_nvfp4(h, W["gs_h"])
        d = torch.ops.trtllm.nvfp4_gemm(hq, W["wd"][i][0], hsf, W["wd"][i][1], W["alpha"], torch.bfloat16)
        nf(d, self.res, W["g2"][i])
        if self.qkvz_marlin:
            o = marlin(d, W["wq"][i], QKVZ, H, 0)
        else:
            dq, dsf = torch.ops.trtllm.fp4_quantize(d, W["gs_x"], 16, False)
            o = torch.ops.trtllm.nvfp4_gemm(dq, W["wq4"][i][0], dsf, W["wq4"][i][1], W["alpha"],
                                            torch.bfloat16)
        return y, d, o

    def capture(self, arm):
        nf = norm_fn(arm)

        def body():
            return [self.layer(i, nf) for i in range(L)]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.reset()
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = body()
        torch.cuda.synchronize()
        return gr, outs


def stats(v):
    v = sorted(v)
    n = len(v)
    med = statistics.median(v)
    return {"med": round(med, 3), "p10": round(v[int(0.1 * (n - 1))], 3),
            "p90": round(v[int(0.9 * (n - 1))], 3),
            "noise": round((v[int(0.9 * (n - 1))] - v[int(0.1 * (n - 1))]) / med, 4)}


def time_arms(ch, graphs):
    arms = tuple(graphs)
    W = ch.W
    ts = {a: [] for a in arms}
    for it in range(WARM + ITERS):
        r = it % len(arms)
        order = list(arms[r:] + arms[:r])
        if it % 2:
            order = order[::-1]
        evs = {}
        for arm in order:
            ch.reset()
            torch.sum(W["scrub"], dim=0, out=W["sink"])
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm][0].replay()
            e1.record()
            evs[arm] = (e0, e1)
        torch.cuda.synchronize()
        if it >= WARM:
            for arm in arms:
                ts[arm].append(evs[arm][0].elapsed_time(evs[arm][1]) * 1e3 / L)
    return ts


def main():
    ms = [int(a) for a in sys.argv[1:]] or [8, 16, 24, 32]
    log(f"chain L={L} warm={WARM} iters={ITERS} ms={ms} arms={ARMS} 1row_module={G is not None} "
        f"pdl={os.environ.get('TRTLLM_ENABLE_PDL')} cap={cap()}")
    if G is not None:
        log(f"routing: enabled={G._ENABLED} rows={G._MIN_ROWS}..{G._MAX_ROWS} tpr/arm={G.routed_tpr_arm()}")
    W = build_weights()
    results = {}
    for m in ms:
        ch = Chain(m, W)
        graphs = {arm: ch.capture(arm) for arm in ARMS}
        log(f"M={m} captured (qkvz {'marlin' if ch.qkvz_marlin else 'w4a4'})")
        ts = time_arms(ch, graphs)
        outs = {}
        for arm in ARMS:
            ch.reset()
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            outs[arm] = ([t.clone() for tr in graphs[arm][1] for t in tr], ch.res.clone())
        base = ARMS[0]

        def same(a, b):
            return (all(torch.equal(x, y) for x, y in zip(outs[a][0], outs[b][0]))
                    and torch.equal(outs[a][1], outs[b][1]))

        row = {a: stats(ts[a]) for a in ARMS}
        for a in ARMS[1:]:
            row[f"d_{a}_us"] = round(statistics.median([x - y for x, y in zip(ts[a], ts[base])]), 3)
            row[f"bitwise_{a}"] = same(a, base)
        results[str(m)] = row
        log("M", m, json.dumps(row))
        del graphs, outs, ch
        torch.cuda.empty_cache()
    rep = REPORT if REPORT in ARMS else ARMS[0]
    log(json.dumps({
        "batch": {k: {"kernel_ms": v[rep]["med"] / 1e3, "noise_floor": v[rep]["noise"]}
                  for k, v in results.items()},
        "output": {"class": "exact"},
        "unit": "per-layer chain time (out_proj..qkvz), us/1e3",
        "chain": results,
    }))


if __name__ == "__main__":
    main()
