"""Idea 254 kernel tuning: shared_decode_mlp alone, synthetic FP8 weights of one rank's
shared half (gate_up [2304, 5120], down [5120, 1152], fp32 block-scale tables), 40
distinct layer copies in one CUDA graph, read-only 512 MB flush per replay, CUDA events.
Also a quick fp64 block-math reference check per config (one layer, M=1..8 set).
Env CFGS: ';'-separated 'GU_BN,GU_BK,GU_WARPS,GU_STAGES,DN_BN,DN_WARPS,DN_STAGES'.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NL = 40
REPS = int(os.environ.get("REPS", "20"))
MS = [int(v) for v in os.environ.get("MS", "1,2,4,8").split(",")]
RANGES = [tuple(int(t) for t in r.split("-")) for r in os.environ.get("RANGES", "0-5,0-9").split(",")]
CFGS = os.environ.get("CFGS", "16,512,4,2,32,4,3")


def log(*a):
    print(*a, flush=True)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.modules import fp8_shared_decode as fsd
    from check_kernel import errs, ref_mlp

    H, I = 5120, 1152
    g = torch.Generator(device="cpu").manual_seed(1)
    layers = []
    for _ in range(NL):
        w1 = (torch.randn(2 * I, H, generator=g) * 0.3).to(torch.float8_e4m3fn).cuda()
        w2 = (torch.randn(H, I, generator=g) * 0.3).to(torch.float8_e4m3fn).cuda()
        s1 = torch.exp2(torch.randint(-12, -6, (2 * I // 128, H // 128), generator=g).float()).cuda()
        s2 = torch.exp2(torch.randint(-12, -6, (H // 128, I // 128), generator=g).float()).cuda()
        layers.append((w1, s1, w2, s2))
    flush = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    limit = 10.0
    rows = []
    for cfg in CFGS.split(";"):
        gbn, gbk, gw, gs, dbn, dw, ds = (int(v) for v in cfg.split(","))
        fsd._GU_BN, fsd._GU_BK, fsd._GU_WARPS, fsd._GU_STAGES = gbn, gbk, gw, gs
        fsd._DN_BN, fsd._DN_WARPS, fsd._DN_STAGES = dbn, dw, ds
        # correctness (one layer)
        ok = True
        worst = 0.0
        for m in (1, 3, 8):
            x = (torch.randn(m, H, generator=g) * 2.0).to(torch.bfloat16).cuda()
            for (b0, b1) in ((0, 0), (0, 1), (8, 9), (0, 5), (5, 9), (0, 9), (2, 7)):
                rng = torch.tensor([b0, b1], dtype=torch.int32, device="cuda")
                out = fsd.shared_decode_mlp(x, *layers[0], rng, limit)
                ref, _ = ref_mlp(x, *layers[0], b0, b1, limit)
                e = errs(out, ref)
                if b0 == b1:
                    e["ok"] = e["ok"] and bool((out == 0).all().item())
                ok = ok and e["ok"]
                worst = max(worst, e["max_abs"])
        log(f"CAND cfg {cfg}: ref check ok={ok} worst_abs={worst:.4g}")
        for (b0, b1) in RANGES:
            rng = torch.tensor([b0, b1], dtype=torch.int32, device="cuda")
            for m in MS:
                xs = [(torch.randn(m, H, generator=g)).to(torch.bfloat16).cuda() for _ in range(NL)]

                def step():
                    return [fsd.shared_decode_mlp(x, *lw, rng, limit) for x, lw in zip(xs, layers)]

                step()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    outs = step()
                for _ in range(10):
                    flush.sum()
                    gr.replay()
                torch.cuda.synchronize()
                ref0 = [o.clone() for o in outs]
                evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(REPS)]
                for a, b in evs:
                    flush.sum()
                    a.record()
                    gr.replay()
                    b.record()
                torch.cuda.synchronize()
                v = [a.elapsed_time(b) * 1000 / NL for a, b in evs]
                same = all(torch.equal(o, r) for o, r in zip(outs, ref0))
                med = pct(v, 0.5)
                noise = (pct(v, 0.9) - pct(v, 0.1)) / med
                log(f"CAND cfg {cfg} range [{b0},{b1}) M={m}: {med:.2f} us/layer noise {noise:.3f} "
                    f"replays_bitwise {same} kernels _shared_gate_up_kernel+_shared_down_kernel")
                rows.append({"cfg": cfg, "range": [b0, b1], "M": m, "us": round(med, 3),
                             "noise": round(noise, 4), "bitwise": same, "ref_ok": ok})
                del gr, outs
    best = {}
    for r in rows:
        k = f"{r['range'][0]}-{r['range'][1]}_M{r['M']}"
        if k not in best or r["us"] < best[k][1]:
            best[k] = (r["cfg"], r["us"])
    log("BEST", json.dumps(best))
    print(json.dumps({"rows": rows, "best": best}))


if __name__ == "__main__":
    main()
