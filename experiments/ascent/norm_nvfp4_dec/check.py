"""Bitwise check: decode gemma add+RMSNorm with a fused NVFP4 epilogue
(FlashInfer overlay flashinfer/norm/kernels/fused_add_rmsnorm_nvfp4.py,
TRT-LLM op trtllm::flashinfer_gemma_fused_add_rmsnorm_nvfp4) vs the stock
FlashInfer gemma_fused_add_rmsnorm + trtllm fp4_quantize.

Modes (argv[1]):
  ref   - runs in ANY tree (base included): stock norm + fp4_quantize only;
          prints the sha256 of residual / FP4 / SF (padding included) per case
          and one digest over all cases.
  cand  - candidate tree: same cases through the fused op, compared bitwise
          against the in-process stock reference, plus the repeated-execution
          sequence (10 warmup + 50 iterations, pristine residual restore,
          poisoned outputs) eagerly and as CUDA-graph replays; prints the same
          digest so it can be matched with the base tree's ref digest.
Self-contained so the base arm can run it with python -c.
"""
import hashlib
import json
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import torch_custom_ops  # noqa: F401

H = 5120
EPS = 1e-6
WARM, ITERS = 10, 50
dev = "cuda"
MS = list(range(1, 33)) + [48, 64, 127, 128, 129]
SCALES = [448.0 * 6.0 / 20.0, 100.0, 1.0, 0.05]
FAMS = ["plain", "rowscale", "blocks"]


def make_inputs(m, seed, family):
    g = torch.Generator(device=dev).manual_seed(seed)
    if family == "plain":
        x = torch.randn((m, H), generator=g, device=dev) * 0.7
        r = torch.randn((m, H), generator=g, device=dev) * 3.0
    elif family == "rowscale":
        sc = torch.exp(torch.empty((m, 1), device=dev).uniform_(
            -9.0, 7.0, generator=g))
        x = torch.randn((m, H), generator=g, device=dev) * sc
        r = torch.randn((m, H), generator=g, device=dev) * sc * 4.0
        u = torch.rand((m, H), generator=g, device=dev)
        x = torch.where(u < 0.002, x * 150.0, x)
        x = torch.where((u >= 0.002) & (u < 0.01), torch.zeros_like(x), x)
        r = torch.where((u >= 0.005) & (u < 0.01), torch.zeros_like(r), r)
        x = torch.where((u >= 0.01) & (u < 0.015), torch.full_like(x, -0.0),
                        x)
    else:
        x = torch.randn((m, H), generator=g, device=dev)
        r = torch.randn((m, H), generator=g, device=dev) * 2.0
        nb = H // 16
        sel = torch.rand((m, nb), generator=g, device=dev)
        xb, rb = x.view(m, nb, 16), r.view(m, nb, 16)
        xb[sel < 0.03] = 0.0
        rb[sel < 0.03] = 0.0
        xb[(sel >= 0.03) & (sel < 0.05)] = -0.0
        rb[(sel >= 0.03) & (sel < 0.05)] = -0.0
        if m >= 2:
            x[1] = 0.0
            r[1] = 0.0
        if m >= 3:
            x[2] *= 1e-20
            r[2] *= 1e-20
        if m >= 4:
            x[3] *= 1e-36
            r[3] *= 1e-36
    return x.to(torch.bfloat16), r.to(torch.bfloat16)


def make_weight(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn((H, ), generator=g, device=dev) * 0.3
    u = torch.rand((H, ), generator=g, device=dev)
    w = torch.where(u < 0.01, torch.full_like(w, -1.0), w)
    w = torch.where((u >= 0.01) & (u < 0.02), w * 20.0, w)
    return w.to(torch.bfloat16)


def reference(x, r, w, gs):
    x2, r2 = x.clone(), r.clone()
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x2, r2, w, EPS)
    q, sf = torch.ops.trtllm.fp4_quantize(x2, gs, 16, False)
    return r2, q.view(torch.uint8), sf.view(torch.uint8)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()
                          .tobytes()).hexdigest()[:16]


def cases():
    for m in MS:
        for fi, fam in enumerate(FAMS):
            for si, sv in enumerate(SCALES):
                yield m, fi, fam, si, sv


def case_inputs(m, fi, fam, si, sv):
    gs = torch.tensor([sv], dtype=torch.float32, device=dev)
    x, r = make_inputs(m, 7919 * m + 31 * fi + si, fam)
    w = make_weight(17 * m + fi)
    return x, r, w, gs


def triple_hash(t3):
    return "/".join(sha(t) for t in t3)


def run_ref():
    hashes = {}
    for m, fi, fam, si, sv in cases():
        x, r, w, gs = case_inputs(m, fi, fam, si, sv)
        hashes[f"M{m}_{fam}_s{si}"] = triple_hash(reference(x, r, w, gs))
    dig = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()
                         ).hexdigest()
    print("REF_HASHES", json.dumps(hashes, sort_keys=True), flush=True)
    return {"passed": True, "mode": "ref", "n_cases": len(hashes),
            "digest": dig}


def run_cand():
    from tensorrt_llm._torch.modules import fused_gemma_add_rmsnorm_nvfp4 as fg
    op = torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm_nvfp4

    def fused(x, r, w, gs):
        r2 = r.clone()
        q, sf = op(x, r2, w, EPS, gs)
        return r2, q.view(torch.uint8), sf.view(torch.uint8)

    passed = True
    hashes, bad_cases = {}, []
    for m, fi, fam, si, sv in cases():
        x, r, w, gs = case_inputs(m, fi, fam, si, sv)
        x0 = x.clone()
        ref = reference(x, r, w, gs)
        fu = fused(x, r, w, gs)
        key = f"M{m}_{fam}_s{si}"
        hashes[key] = triple_hash(fu)
        eq = [torch.equal(a, b) for a, b in zip(fu, ref)]
        ok = all(eq) and torch.equal(x, x0)
        if not ok:
            passed = False
            nd = [int((a != b).sum()) if a.shape == b.shape else -1
                  for a, b in zip(fu, ref)]
            bad_cases.append({key: {"res_fp4_sf_eq": eq, "ndiff": nd,
                                    "x_unchanged": torch.equal(x, x0)}})
            print("A MISMATCH", key, eq, nd, flush=True)
    print("A done", len(hashes), "cases, bad", len(bad_cases), flush=True)
    dig = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()
                         ).hexdigest()

    # B. repeated execution: 10 + 50, pristine residual restore, poisoned
    # outputs, eager and CUDA-graph replay (interleaved with eager calls).
    rep = {}
    gs = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device=dev)
    for m in [1, 2, 3, 4, 7, 8, 16, 24, 31, 32]:
        x, r_pristine = make_inputs(m, 4242 + m, "rowscale")
        w = make_weight(99 + m)
        x0 = x.clone()
        ref = reference(x, r_pristine, w, gs)
        r = r_pristine.clone()
        q = torch.empty_like(ref[1])
        sf = torch.empty_like(ref[2])
        first, bad, maxd = None, 0, 0
        for it in range(WARM + ITERS):
            r.copy_(r_pristine)
            q.fill_(0xFF)
            sf.fill_(0xFF)
            fg.flashinfer_nvfp4_launch(x, r, w, EPS, gs, q, sf)
            torch.cuda.synchronize()
            cur = (r.clone(), q.clone(), sf.clone())
            if first is None:
                first = cur
            d1 = sum(int((a != b).sum()) for a, b in zip(cur, first))
            d2 = sum(int((a != b).sum()) for a, b in zip(cur, ref))
            maxd = max(maxd, d1, d2)
            if d1 or d2:
                bad += 1
        gr = r_pristine.clone()
        gq = torch.empty_like(ref[1])
        gsf = torch.empty_like(ref[2])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fg.flashinfer_nvfp4_launch(x, gr, w, EPS, gs, gq, gsf)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fg.flashinfer_nvfp4_launch(x, gr, w, EPS, gs, gq, gsf)
        gbad, ebad = 0, 0
        for it in range(WARM + ITERS):
            gr.copy_(r_pristine)
            gq.fill_(0xFF)
            gsf.fill_(0xFF)
            g.replay()
            torch.cuda.synchronize()
            cur = (gr.clone(), gq.clone(), gsf.clone())
            if any(not torch.equal(a, b) for a, b in zip(cur, ref)):
                gbad += 1
            fu = fused(x, r_pristine, w, gs)
            if any(not torch.equal(a, b) for a, b in zip(fu, ref)):
                ebad += 1
        del g
        rep[f"M{m}"] = {"eager_bad": bad, "graph_bad": gbad,
                        "interleaved_eager_bad": ebad, "max_ndiff": maxd,
                        "x_unchanged": torch.equal(x, x0)}
        print("B", m, rep[f"M{m}"], flush=True)
        passed = (passed and bad == 0 and gbad == 0 and ebad == 0
                  and torch.equal(x, x0))
    print("CAND_HASHES", json.dumps(hashes, sort_keys=True), flush=True)
    return {"passed": passed, "mode": "cand", "n_cases": len(hashes),
            "digest": dig, "bad_cases": bad_cases[:8], "repeated": rep,
            "compared": "fused op vs stock flashinfer gemma_fused_add_rmsnorm"
                        " + trtllm fp4_quantize (residual, FP4, full SF incl."
                        " padding rows), bitwise"}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cand"
    out = run_ref() if mode == "ref" else run_cand()
    print(json.dumps(out), flush=True)
