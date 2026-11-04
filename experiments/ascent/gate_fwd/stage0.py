"""Stage 0 sizing for gate_forward_kernel<384,6,false> (idea 162).

Modes:
  chain  - 40-layer CUDA graphs at M=1,2,4,8,32: (A) router only, (B) router -> gate_forward,
           (C) router -> null kernel (Triton, same grid ceil(M/4) x 128 threads, writes the 6 outputs from constants),
           (D) gate_forward only (40 distinct score buffers, no router).
           In-graph per-node cost: gate = (B-A)/40, null = (C-A)/40. Read-only 512 MB flush before every replay.
  ncu    - eager router->gate chain for 40 layers x a few iterations (for profile_kernels launch_skip).
"""
import json
import sys

import torch
import triton
import triton.language as tl

HIDDEN, EXPERTS, LAYERS, TOPK = 5120, 384, 40, 6
SEED = 4242
WARMUP, ITERS = 5, 40


def log(*a):
    print(*a, flush=True)


@triton.jit
def _null_gate(w_ptr, i_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * 6
    tl.store(w_ptr + offs, (offs % 6).to(tl.float32) * 0.25, mask=mask)
    tl.store(i_ptr + offs, offs % 6, mask=mask)


def null_gate(scores, w, i):
    m = scores.shape[0]
    # 4 tokens (24 outputs) per CTA of 4 warps = same grid as gate_forward
    _null_gate[(triton.cdiv(m, 4),)](w, i, m, BLOCK=32, num_warps=4)


def router(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def gate(scores, bias, ow, oi):
    torch.ops.trtllm.gate_forward(scores, bias, EMPTY_I, EMPTY_I, ow, oi, TOPK, 1.5, False)


def time_graph(step, flush):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        step()
        step()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    ts = []
    for it in range(WARMUP + ITERS):
        flush.sum()  # read-only flush (no dirty lines)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        if it >= WARMUP:
            ts.append(a.elapsed_time(b) * 1000.0)
    del g
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med


def main():
    global EMPTY_I
    import tensorrt_llm  # noqa: F401
    mode = sys.argv[1] if len(sys.argv) > 1 else "chain"
    EMPTY_I = torch.empty(0, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    ws = [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(LAYERS)]
    bias = (torch.randn((EXPERTS,), generator=gen, device="cuda") * 0.05).float()
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda").fill_(1.0)
    if mode == "ncu":
        for m in (1, 4):
            xs = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
            ow = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
            oi = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
            for it in range(3):
                for li in range(LAYERS):
                    sc = router(xs[li], ws[li])
                    gate(sc, bias, ow, oi)
                    null_gate(sc, ow, oi)
            torch.cuda.synchronize()
            log("ncu M", m, "done")
        return
    res = {}
    for m in (1, 2, 4, 8, 32):
        xs = [torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16) for _ in range(LAYERS)]
        scs = [router(x, w) for x, w in zip(xs, ws)]
        ows = [torch.empty((m, TOPK), dtype=torch.float32, device="cuda") for _ in range(LAYERS)]
        ois = [torch.empty((m, TOPK), dtype=torch.int32, device="cuda") for _ in range(LAYERS)]

        def a():
            for li in range(LAYERS):
                router(xs[li], ws[li])

        def b():
            for li in range(LAYERS):
                gate(router(xs[li], ws[li]), bias, ows[li], ois[li])

        def c():
            for li in range(LAYERS):
                null_gate(router(xs[li], ws[li]), ows[li], ois[li])

        def d():
            for li in range(LAYERS):
                gate(scs[li], bias, ows[li], ois[li])

        def e():
            for li in range(LAYERS):
                null_gate(scs[li], ows[li], ois[li])

        r = {}
        for name, fn in (("router", a), ("router_gate", b), ("router_null", c), ("gate_only", d), ("null_only", e)):
            meds = [time_graph(fn, flush) for _ in range(2)]
            r[name] = min(x[0] for x in meds)
            r[name + "_nf"] = max(x[1] for x in meds)
        out = {
            "router_us_layer": r["router"] / LAYERS,
            "gate_in_chain_us": (r["router_gate"] - r["router"]) / LAYERS,
            "null_in_chain_us": (r["router_null"] - r["router"]) / LAYERS,
            "gate_only_us": r["gate_only"] / LAYERS,
            "null_only_us": r["null_only"] / LAYERS,
            "nf": max(v for k, v in r.items() if k.endswith("_nf")),
        }
        out["body_us"] = out["gate_in_chain_us"] - out["null_in_chain_us"]
        log(f"M={m}", json.dumps({k: round(v, 3) for k, v in out.items()}))
        res[str(m)] = out
    print(json.dumps({"stage0": res}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
