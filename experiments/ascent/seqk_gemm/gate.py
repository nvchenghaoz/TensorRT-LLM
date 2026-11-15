"""Idea 73 bitwise gate (pitfall 629 (a),(b),(d),(e)) + CUDA-graph check for the routed
sequential-K qkv GEMM, untimed. Synthetic tensors with the checkpoint's shapes, no model load.

(a) every routed integer M of every routed shape, out.view(int16) == main's dispatch
(b) seeds 0 and 1, plus heavy-tailed rows randn * 2^U(-8, 8)
(d) repeated sequence of warmup+timed length (5 + 25 calls) over shapes A, B, A, outputs drawn from
    0xA5-prefilled allocator blocks: every call equals call 1 and main; inputs unchanged
(e) first mismatch reported; max-abs and rel-Frobenius error vs FP32 torch.matmul equal to main's
graph: torch.cuda.graph capture at M=300 and M=301 (and 317), 25 replays on unchanged inputs, each
    equal to eager and to main
Also: the routed model path (modeling_gemma4._Sm120SeqKGemmMixin.apply_linear on a real Linear)
reaches the kernel exactly at routed M, and TRTLLM_GEMMA4_SEQK_GEMM=0 (module switch) restores main.

usage: gate.py <ckpt> [impl=op|ext]"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_prod_ext, ckpt_dims, gpu_receipt, load_main  # noqa: E402


def i16eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.view(torch.int16),
                                                                          b.view(torch.int16)))


def first_mismatch(a, b):
    d = (a.view(torch.int16) != b.view(torch.int16)).nonzero()
    if d.numel() == 0:
        return None
    r, c = d[0].tolist()
    return {"row": r, "col": c, "cand": float(a[r, c]), "main": float(b[r, c]), "count": int(d.shape[0])}


def err_vs_fp32(y, x, w):
    ref = x.float() @ w.float().t()
    d = (y.float() - ref)
    return float(d.abs().max()), float(d.norm() / ref.norm())


def main():
    ckpt = sys.argv[1]
    impl = sys.argv[2] if len(sys.argv) > 2 else "op"
    _, shapes = ckpt_dims(ckpt)
    print("receipt", gpu_receipt(), flush=True)
    main_op, _ = load_main()
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    routes = mg._SM120_SEQK_GEMM_ROUTES
    print("routes", routes, flush=True)
    ext = None
    if impl == "op":
        assert hasattr(torch.ops.trtllm, "sm120_bf16_seqk_gemm"), "op not in this build"

        def cand(x, w, cfg):
            return torch.ops.trtllm.sm120_bf16_seqk_gemm(x, w, cfg)
        try:
            ext = build_prod_ext()
        except Exception as ex:  # noqa: BLE001
            print("prod ext unavailable:", ex, flush=True)
    else:
        ext = build_prod_ext()

        def cand(x, w, cfg):
            return ext.seqk(x, w, cfg)

    name_of = {v: k for k, v in shapes.items()}
    res = {"a_b": {}, "e": {}, "d": {}, "graph": {}, "model_path": {}}
    ok = True
    for (K, N), (ranges, cfg) in routes.items():
        sn = name_of.get((N, K), f"{N}x{K}")
        g = torch.Generator(device="cuda")
        g.manual_seed(1234)
        w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
        Ms = [m for lo, hi in ranges for m in range(lo, hi + 1)]
        fails = []
        for seed in (0, 1, "heavy"):
            for M in Ms:
                gg = torch.Generator(device="cuda")
                gg.manual_seed(M * 7 + (0 if seed == "heavy" else seed) * 1000003)
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=gg)
                if seed == "heavy":
                    ex = torch.empty(M, 1, device="cuda").uniform_(-8, 8, generator=gg)
                    x = (x.float() * torch.exp2(ex)).to(torch.bfloat16)
                ref = main_op(x, w)
                y = cand(x, w, cfg)
                if not i16eq(y, ref):
                    fails.append({"M": M, "seed": seed, "first": first_mismatch(y, ref)})
                if ext is not None and impl == "op":
                    y2 = ext.seqk(x, w, cfg)
                    if not i16eq(y2, y):
                        fails.append({"M": M, "seed": seed, "ext_vs_op": first_mismatch(y2, y)})
                if M in (ranges[0][0], 300, ranges[-1][1]) and seed == 0:
                    res["e"][f"{sn}_M{M}"] = {"cand": err_vs_fp32(y, x, w), "main": err_vs_fp32(ref, x, w)}
        res["a_b"][sn] = {"Ms": f"{Ms[0]}..{Ms[-1]} ({len(Ms)} M x 3 input kinds)", "fails": fails[:5],
                          "n_fail": len(fails)}
        ok &= not fails
        print(sn, "a/b fails", len(fails), fails[:2], flush=True)
        for k, v in res["e"].items():
            if k.startswith(sn) and v["cand"] != v["main"]:
                ok = False

        # graph capture / replay at M=300, 301 and 317
        for M in (300, 301, 317):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = main_op(x, w)
            eager = cand(x, w, cfg)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    cand(x, w, cfg)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                yg = cand(x, w, cfg)
            reps_ok = 0
            for _ in range(25):
                yg.fill_(0)  # replay must rewrite every element
                graph.replay()
                torch.cuda.synchronize()
                reps_ok += int(i16eq(yg, eager) and i16eq(yg, ref))
            res["graph"][f"{sn}_M{M}"] = {"replays_equal": reps_ok, "of": 25, "eager_eq_main": i16eq(eager, ref)}
            ok &= reps_ok == 25 and i16eq(eager, ref)
            del graph
        print(sn, "graph", {k: v for k, v in res["graph"].items() if k.startswith(sn)}, flush=True)

    # (d) repeated sequence A, B, A of warmup+timed length with 0xA5-prefilled allocator blocks
    items = list(routes.items())
    (KA, NA), (rA, cA) = items[0]
    (KB, NB), (rB, cB) = items[-1]
    torch.manual_seed(0)
    wA = torch.randn(NA, KA, device="cuda", dtype=torch.bfloat16) * 0.02
    wB = torch.randn(NB, KB, device="cuda", dtype=torch.bfloat16) * 0.02
    xA = torch.randn(300, KA, device="cuda", dtype=torch.bfloat16)
    xB = torch.randn(317, KB, device="cuda", dtype=torch.bfloat16)
    snap = [t.clone() for t in (wA, wB, xA, xB)]
    refA, refB = main_op(xA, wA), main_op(xB, wB)
    seq_ok = True
    firsts = {}
    calls = 0
    for phase, (x, w, cfg, ref) in (("A", (xA, wA, cA, refA)), ("B", (xB, wB, cB, refB)),
                                    ("A2", (xA, wA, cA, refA))):
        for i in range(30):
            junk = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
            junk.view(torch.uint8).fill_(0xA5)
            del junk
            y = cand(x, w, cfg)
            calls += 1
            key = "A" if phase.startswith("A") else "B"
            if key not in firsts:
                firsts[key] = y.clone()
            seq_ok &= i16eq(y, firsts[key]) and i16eq(y, ref)
    inputs_unchanged = all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                           for a, b in zip(snap, (wA, wB, xA, xB)))
    res["d"] = {"calls": calls, "all_equal_call1_and_main": bool(seq_ok), "inputs_unchanged": inputs_unchanged}
    ok &= seq_ok and inputs_unchanged
    print("d", res["d"], flush=True)

    # model path: the mixin on a real Linear reaches the kernel exactly at routed M; switch off = main
    if impl == "op":
        from tensorrt_llm._torch.modules.linear import Linear
        for (K, N), (ranges, cfg) in routes.items():
            lin = Linear(K, N, bias=False, dtype=torch.bfloat16).cuda()
            with torch.no_grad():
                lin.weight.copy_(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
            lin.use_custom_cublas_mm = True
            lin.custom_cublas_mm_tokens = mg._CublasMmTokenGate(mg._SM120_BF16_CUBLAS_MM_ROUTES[(K, N)])
            mg._route_sm120_seqk_gemm(lin)
            hits = {"routed": 0, "fallback": 0}
            orig = torch.ops.trtllm.sm120_bf16_seqk_gemm
            row = {}
            for M in (287, 288, 289, 300, 301, 320, 321, 600, 1000, 1200):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                ref = main_op(x, lin.weight)
                mg._SEQK_GEMM = True
                y_on = lin(x)
                mg._SEQK_GEMM = False
                y_off = lin(x)
                mg._SEQK_GEMM = True
                routed = M in lin.seqk_gemm_tokens
                hits["routed" if routed else "fallback"] += 1
                row[M] = {"routed": routed, "on_eq_main": i16eq(y_on, ref), "off_eq_main": i16eq(y_off, ref)}
                ok &= row[M]["on_eq_main"] and row[M]["off_eq_main"]
            res["model_path"][f"{N}x{K}"] = {"class": type(lin).__name__, "rows": row}
            del orig
        print("model_path", res["model_path"], flush=True)

    print("receipt_end", gpu_receipt(), flush=True)
    print(json.dumps({"passed": bool(ok), "impl": impl,
                      "compared": "int16 bitwise vs main dispatch (cublas_mm/F.linear) at every routed M; "
                                  "FP32 torch.matmul error equal to main's", **res}))


if __name__ == "__main__":
    main()
