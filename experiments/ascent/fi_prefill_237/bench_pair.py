"""Idea 237 paired microbench: the routed-MoE leaf's GRAPH-CAPTURED call at prefill M, base vs candidate.

Runs in both trees. Inside a CUDA-graph capture the leaf (w4a8_mxfp4_mxfp8.py _takes_fi_chain) sends
a BF16 call to the FlashInfer SM12x chain when the tree has the prefill gate (candidate, default
TRTLLM_SM12X_W4A8_FI_PREFILL=graph) and to CUTLASS otherwise (base: mxfp8_quantize + fused_moe below
the fold minimum, fused_moe(bf16) with in-kernel quant above it). This script makes exactly that
choice per tree: dec.fi_prefill_takes(m) evaluated inside the capture.

One EP2xTP2 rank (192 local experts x inter 1152, H 5120, top-6), synthetic MXFP4 weights.
TARGETED rows: M128 (the L20-39 tail MoE window graph, 128 rows): D66 and D166.
SECONDARY rows: M48, M512, M2048 (graph-captured prefill sizes), and M4 (decode control, same
kernels in both trees). Timing: graph replays, each after a READ-only 512 MiB L2 flush; median over
40 replays, noise = (p90 - p10) / median.
"""

import argparse
import hashlib
import json

import torch

import stage0 as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", default="128:66,128:166,4:6,48:101,512:192,2048:120")
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    has_gate = hasattr(S.dec, "fi_prefill_takes")
    tree = "CAND" if has_gate else "BASE"
    print(f"{tree}: TARGETED rows M128_D66, M128_D166; SECONDARY M4_D6 (decode control), M48_D101, "
          f"M512_D192, M2048_D120", flush=True)
    H, I_full, E_glob, topk, limit = S.read_cfg(args.ckpt)
    E, I = E_glob // 2, I_full // 2
    p = S.Problem(H, I, E, E_glob, topk, limit)
    if has_gate:
        p.st.warmup(topk)
        p.st.warmup_prefill(topk)
    else:
        p.st.warmup(topk)
    S.FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device="cuda")
    cases = [tuple(int(v) for v in c.split(":")) for c in args.cases.split(",")]
    inputs = {c: p.inputs(*c) for c in cases}
    with S.autotune():
        for c in cases:
            x, sel, sc, _, _ = inputs[c]
            if c[0] > S.dec.FI_DECODE_MAX_TOKENS:
                p.cutlass(x, sel, sc)
    torch.cuda.synchronize()
    taken = {}

    def leaf(x, sel, sc):
        m = x.shape[0]
        if m <= S.dec.FI_DECODE_MAX_TOKENS:
            taken[m] = "fi_decode"
            return p.st.run(x, sel, sc)
        if has_gate and S.dec.fi_prefill_takes(m):
            taken[m] = "fi_prefill"
            return p.st.run(x, sel, sc)
        taken[m] = "cutlass"
        return p.cutlass(x, sel, sc)

    res, shas = {}, {}
    for m, d in cases:
        x, sel, sc, n_act, n_loc = inputs[(m, d)]
        key = f"M{m}_D{d}"
        st, out = S.time_graph(lambda: leaf(x, sel, sc), args.reps)
        ok = bool(torch.isfinite(out.float()).all())
        res[key] = {"kernel_ms": st["median_us"] / 1e3, "noise_floor": st["noise"]}
        shas[key] = hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
        print(f"{tree} {key} path={taken.get(m)} act={n_act} loc={n_loc}: {st['median_us']:.1f} us "
              f"(p10 {st['p10_us']:.1f} p90 {st['p90_us']:.1f} noise {st['noise']:.4f}) finite={ok}", flush=True)
        torch.cuda.empty_cache()
    # Another backend of the same MXFP8 x MXFP4 mode (FlashInfer SM12x, as decode already runs it):
    # errors vs the FP32 reference and vs main's CUTLASS are reported by check.py.
    out_class = "mode_switch" if has_gate else "exact"
    print(json.dumps({"batch": res, "output": {"class": out_class}, "tree": tree, "paths": taken,
                      "sha12": shas}))


if __name__ == "__main__":
    main()
