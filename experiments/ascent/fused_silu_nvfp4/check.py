"""Bitwise check: fused SwiGLU+NVFP4 vs silu_and_mul + fp4_quantize (main).

Untimed repeated-execution protocol mirroring bench: 10 warmup + 50 iterations,
outputs NaN/0xAB-poisoned before every call, every call (eager and CUDA-graph
replay) compared bitwise to iteration 1 and to the reference.
"""
import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_silu_mul_nvfp4 import (
    _fused_silu_mul_nvfp4_impl, _launch)

I = 17408
dev = "cuda"
WARM, ITERS = 10, 50


def make_h1(m, seed, subnormal=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn((m, 2 * I), generator=g, device=dev) * 2.0
    r = torch.rand((m, 2 * I), generator=g, device=dev)
    h = torch.where(r < 0.01, torch.zeros_like(h), h)
    h = torch.where((r >= 0.01) & (r < 0.02), torch.full_like(h, -0.0), h)
    h = torch.where((r >= 0.02) & (r < 0.025), h * 60.0, h)
    h = h.to(torch.bfloat16)
    # Whole zero / negative-zero 16-blocks in 'up' (vecMax == 0 path).
    nb = I // 16
    hb = h.view(m, 2, nb, 16)
    sel = torch.rand((m, nb), generator=g, device=dev)
    hb[:, 1][sel < 0.02] = 0.0
    hb[:, 1][(sel >= 0.02) & (sel < 0.04)] = -0.0
    # gate = -0 blocks
    hb[:, 0][(sel >= 0.04) & (sel < 0.05)] = -0.0
    if subnormal:
        tiny = (torch.randn((m, nb, 16), generator=g, device=dev) *
                1e-38).to(torch.bfloat16)
        msk = (sel >= 0.05) & (sel < 0.10)
        hb[:, 1][msk] = tiny[msk]
        hb[:, 0][msk] = torch.full_like(tiny[msk], 2.0)
    return h


def reference(h1, scale):
    h = torch.ops.trtllm.silu_and_mul(h1)
    return torch.ops.trtllm.fp4_quantize(h, scale, 16, False)


def first_mismatch(a, b):
    idx = (a != b).nonzero()
    return idx[0].tolist() if idx.numel() else None


def main():
    results = {}
    passed = True
    ms = [1, 2, 3, 4, 8, 64, 512, 2048, 8192]
    scales = [30.0, 448.0 * 6.0 / 80.0, 0.37]
    for m in ms:
        for si, sv in enumerate(scales):
            scale = torch.tensor([sv], dtype=torch.float32, device=dev)
            h1 = make_h1(m, 1000 * m + si)
            rq, rsf = reference(h1, scale)
            rq = rq.view(torch.uint8)
            rsf = rsf.view(torch.uint8)
            fq, fsf = _fused_silu_mul_nvfp4_impl(h1, scale)
            ok_q = torch.equal(fq, rq)
            ok_sf = torch.equal(fsf, rsf)
            key = f"M{m}_s{si}"
            results[key] = {"fp4_eq": ok_q, "sf_eq": ok_sf,
                            "sf_numel": [fsf.numel(), rsf.numel()]}
            if not (ok_q and ok_sf):
                passed = False
                fm = first_mismatch(fq, rq) if not ok_q else None
                results[key]["first_fp4_mismatch"] = fm
                if fm is not None:
                    r_, c_ = fm
                    results[key]["fp4_vals"] = [int(fq[r_, c_]), int(rq[r_, c_])]
                    blk = (2 * c_) // 16 * 16
                    results[key]["gate"] = h1[r_, blk:blk + 16].float().tolist()
                    results[key]["up"] = h1[r_, I + blk:I + blk + 16].float().tolist()
                if not ok_sf:
                    results[key]["first_sf_mismatch"] = first_mismatch(fsf, rsf)
                    results[key]["n_sf_mismatch"] = int((fsf != rsf).sum())
            print(key, results[key], flush=True)

    # Subnormal edge case (reported separately).
    sub = {}
    for m in [4, 128]:
        scale = torch.tensor([30.0], dtype=torch.float32, device=dev)
        h1 = make_h1(m, 77 + m, subnormal=True)
        rq, rsf = reference(h1, scale)
        fq, fsf = _fused_silu_mul_nvfp4_impl(h1, scale)
        sub[f"M{m}"] = {"fp4_eq": torch.equal(fq, rq.view(torch.uint8)),
                        "sf_eq": torch.equal(fsf, rsf.view(torch.uint8))}
        if not sub[f"M{m}"]["fp4_eq"]:
            fm = first_mismatch(fq, rq.view(torch.uint8))
            sub[f"M{m}"]["first"] = fm
            sub[f"M{m}"]["n"] = int((fq != rq.view(torch.uint8)).sum())
    print("subnormal", sub, flush=True)
    results["subnormal"] = sub
    for v in sub.values():
        passed = passed and v["fp4_eq"] and v["sf_eq"]

    # Repeated execution: eager with poisoned preallocated outputs, then graph.
    rep = {}
    for m in [1, 2, 4, 8, 512, 2048, 8192]:
        scale = torch.tensor([30.0], dtype=torch.float32, device=dev)
        h1 = make_h1(m, 5 + m)
        h1_copy = h1.clone()
        rq, rsf = reference(h1, scale)
        rq = rq.view(torch.uint8)
        rsf = rsf.view(torch.uint8)
        fq = torch.empty_like(rq)
        fsf = torch.empty_like(rsf)
        first = None
        bad = 0
        for it in range(WARM + ITERS):
            fq.fill_(0xAB)
            fsf.fill_(0xAB)
            _launch(h1, scale, fq, fsf)
            torch.cuda.synchronize()
            if first is None:
                first = (fq.clone(), fsf.clone())
            if not (torch.equal(fq, first[0]) and torch.equal(fsf, first[1])
                    and torch.equal(fq, rq) and torch.equal(fsf, rsf)):
                bad += 1
        # CUDA graph replay with poisoned outputs between replays.
        gq = torch.empty_like(rq)
        gsf = torch.empty_like(rsf)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _launch(h1, scale, gq, gsf)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _launch(h1, scale, gq, gsf)
        gbad = 0
        for it in range(WARM + ITERS):
            gq.fill_(0xAB)
            gsf.fill_(0xAB)
            g.replay()
            torch.cuda.synchronize()
            if not (torch.equal(gq, rq) and torch.equal(gsf, rsf)
                    and torch.equal(gq, first[0])):
                gbad += 1
        inputs_ok = torch.equal(h1, h1_copy)
        rep[f"M{m}"] = {"eager_bad": bad, "graph_bad": gbad,
                        "inputs_unchanged": inputs_ok}
        passed = passed and bad == 0 and gbad == 0 and inputs_ok
        print("repeat", m, rep[f"M{m}"], flush=True)
    results["repeat"] = rep
    print(json.dumps({"passed": passed,
                      "compared": "fused_silu_mul_nvfp4 FP4+SF bytes vs "
                                  "trtllm.silu_and_mul + trtllm.fp4_quantize "
                                  "(main kernels), bitwise",
                      "results": results}))


if __name__ == "__main__":
    sys.exit(main())
