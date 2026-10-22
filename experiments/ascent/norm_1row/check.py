"""Correctness of the 1-row gemma add+RMSNorm arms vs stock FlashInfer (H=5120).

For each M and arm: residual bitwise vs stock, y bitwise / max ulp vs stock and
error vs an FP32 reference; 10 warm + 50 repeated calls from a pristine copy
(each compared to stock and to call 1); CUDA graph capture + replay at
M=1..32 for the routed arm. Last line JSON {"passed": ...}.
"""
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from flashinfer.norm import gemma_fused_add_rmsnorm  # noqa: E402

from tensorrt_llm._torch.modules import gemma_add_rmsnorm_1row as G  # noqa: E402

dev = torch.device("cuda")
H, EPS = 5120, 1e-6
ARMS = [(a.split(":")[0], int(a.split(":")[1])) for a in (
    sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "op" else
    ["E:160", "E:128", "E:320", "E:64", "stock:64", "stock:128", "stock:160", "stock:320"])]
MS = [1, 2, 4, 8, 16, 17, 24, 32]


def log(*a):
    print(*a, flush=True)


def ulps(a, b):
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    return (ai - bi).abs().max().item()


def stock(x, r, w):
    x, r = x.clone(), r.clone()
    gemma_fused_add_rmsnorm(x, r, w, EPS, enable_pdl=True)
    return x, r


def arm_call(x, r, w, arm, tpr):
    x, r = x.clone(), r.clone()
    G.gemma_fused_add_rmsnorm_1row(x, r, w, EPS, True, tpr=tpr, arm=arm)
    return x, r


def ref32(x, r, w):
    h = x.float() + r.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * (w.float() + 1)


def op_mode():
    """trtllm::flashinfer_gemma_fused_add_rmsnorm as this tree routes it vs
    stock FlashInfer, 10 warm + 50 calls from a pristine copy, prefill sizes
    included; CUDA graph capture/replay of the op at M = 1..32."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops  # noqa: F401
    op = torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm
    log("routing", G._ENABLED, G._MIN_ROWS, G._MAX_ROWS, G.routed_tpr_arm())
    ok = True
    rows = {}
    g = torch.Generator(device=dev).manual_seed(5)
    w = (torch.randn(H, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    for m in [1, 4, 8, 15, 16, 17, 20, 24, 31, 32, 33, 64, 2048]:
        x = (torch.randn((m, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
        r = (torch.randn((m, H), generator=g, device=dev) * 3.0).to(torch.bfloat16)
        ys, rs = stock(x, r, w)
        first, bit, drift = None, True, False
        for it in range(60):
            xb, rb = x.clone(), r.clone()
            op(xb, rb, w, EPS)
            if first is None:
                first = (xb, rb)
            elif not (torch.equal(first[0], xb) and torch.equal(first[1], rb)):
                drift = True
            bit &= torch.equal(xb, ys) and torch.equal(rb, rs)
        torch.cuda.synchronize()
        rows[m] = dict(bitwise_vs_stock=bit, drift=drift, routed_1row=G.use_1row(x, r))
        ok &= bit and not drift
        log("OP M", m, rows[m])
    graph_ok = True
    for m in range(1, 33):
        x = (torch.randn((m, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
        r = (torch.randn((m, H), generator=g, device=dev) * 3.0).to(torch.bfloat16)
        ys, rs = stock(x, r, w)
        xb, rb = x.clone(), r.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(xb, rb, w, EPS)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            op(xb, rb, w, EPS)
        for _ in range(3):
            xb.copy_(x)
            rb.copy_(r)
            gr.replay()
            torch.cuda.synchronize()
            graph_ok &= torch.equal(xb, ys) and torch.equal(rb, rs)
    log("graph op 1..32", graph_ok)
    ok &= graph_ok
    print(json.dumps({"passed": bool(ok), "graph_ok_1_32": graph_ok,
                      "compared": "trtllm::flashinfer_gemma_fused_add_rmsnorm (routed) vs stock FlashInfer gemma_fused_add_rmsnorm, y+residual bitwise, 10+50 calls",
                      "rows": {str(k): v for k, v in rows.items()}}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "op":
        return op_mode()
    ok = True
    res = {}
    g = torch.Generator(device=dev).manual_seed(11)
    w = (torch.randn(H, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    for arm, tpr in ARMS:
        key = f"{arm}:{tpr}"
        rows = {}
        for m in MS:
            x = (torch.randn((m, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
            r = (torch.randn((m, H), generator=g, device=dev) * 3.0).to(torch.bfloat16)
            ys, rs = stock(x, r, w)
            ref = ref32(x, r, w)
            first = None
            max_u, r_bitwise, y_bitwise, drift = 0, True, True, False
            for it in range(60):
                y1, r1 = arm_call(x, r, w, arm, tpr)
                if first is None:
                    first = (y1.clone(), r1.clone())
                elif not (torch.equal(first[0], y1) and torch.equal(first[1], r1)):
                    drift = True
                r_bitwise &= torch.equal(r1, rs)
                y_bitwise &= torch.equal(y1, ys)
                max_u = max(max_u, ulps(y1, ys))
            torch.cuda.synchronize()
            e_arm = (first[0].float() - ref).abs()
            e_stk = (ys.float() - ref).abs()
            n_diff = (first[0] != ys).sum().item()
            row = dict(r_bitwise=r_bitwise, y_bitwise=y_bitwise, max_ulp=max_u,
                       n_y_diff=n_diff, drift=drift,
                       maxabs_arm=e_arm.max().item(), maxabs_stock=e_stk.max().item(),
                       meanabs_arm=e_arm.mean().item(), meanabs_stock=e_stk.mean().item())
            exact = arm == "E" or tpr == 64
            good = r_bitwise and not drift and (y_bitwise if exact else max_u <= 1)
            row["pass"] = good
            ok &= good
            rows[m] = row
            log(key, "M", m, json.dumps(row))
        res[key] = rows
    # CUDA graph capture/replay of the first arm at M = 1..32
    arm, tpr = ARMS[0]
    graph_ok = True
    for m in range(1, 33):
        x = (torch.randn((m, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
        r = (torch.randn((m, H), generator=g, device=dev) * 3.0).to(torch.bfloat16)
        ys, rs = arm_call(x, r, w, arm, tpr)
        xb, rb = x.clone(), r.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            G.gemma_fused_add_rmsnorm_1row(xb, rb, w, EPS, True, tpr=tpr, arm=arm)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            G.gemma_fused_add_rmsnorm_1row(xb, rb, w, EPS, True, tpr=tpr, arm=arm)
        for _ in range(3):
            xb.copy_(x)
            rb.copy_(r)
            gr.replay()
            torch.cuda.synchronize()
            graph_ok &= torch.equal(xb, ys) and torch.equal(rb, rs)
    log("graph", arm, tpr, graph_ok)
    ok &= graph_ok
    print(json.dumps({"passed": bool(ok), "graph_ok": graph_ok, "compared": "stock flashinfer gemma_fused_add_rmsnorm (base tree kernel) + FP32 ref",
                      "summary": {k: {m: [v["r_bitwise"], v["y_bitwise"], v["max_ulp"], v["n_y_diff"]] for m, v in rows.items()} for k, rows in res.items()}}))


if __name__ == "__main__":
    main()
