"""Step 1: cold-L2 CUDA-graph timing of FlashInfer SM120 MXFP8 GEMM backends.

cutlass (every tactic) vs b12x (every valid tile/prefetch tactic + default plan)
at DeepSeek-V4.1 TP4 per-rank CSA2 shapes, M in {1,2,4,8,2048}.
Weights are rotated over >= 256 MB of copies inside one CUDA graph so every
call streams its weight from DRAM. Bitwise comparison vs cutlass tactic 0.
"""
import json
import math
import os
import sys
import time

import torch

import tensorrt_llm  # noqa: F401  registers torch.ops.trtllm
from flashinfer.gemm.gemm_base import (DEFAULT_WORKSPACE_SIZE, _get_cache_buf,
                                       get_cutlass_mxfp8_gemm_module)
from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_runner

SHAPES = [
    ("wq_a", 1280, 5120),
    ("wkv", 512, 5120),
    ("wq_b", 8192, 1280),
    ("o_b_proj", 5120, 2048),
    ("index_wq_b", 4096, 1280),
    ("fused_wqa_wkv", 1792, 5120),
    ("fused_wqb_idx", 12288, 1280),
]
MS = [int(m) for m in os.environ.get("MS", "1,2,4,8,2048").split(",")]
ONLY = os.environ.get("ONLY")
ROT_BYTES = 256 * 1024 * 1024
REPLAYS = int(os.environ.get("REPLAYS", "10"))
DRAM_BW = 1.5e12

dev = torch.device("cuda")
torch.manual_seed(1234)
cut = get_cutlass_mxfp8_gemm_module(12).cutlass_mxfp8_gemm_runner()
b12 = _b12x_gemm_mxfp8_runner(12, 0, True, torch.bfloat16)
ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)
n_cut = len(cut.get_valid_tactics(None, None))


def tname(backend, t):
    if backend == "cutlass":
        return f"cutlass:{t}"
    if t is None:
        return "b12x:default"
    return f"b12x:{t[0][0]}x{t[0][1]}{'_pf' if t[3] else ''}"


def run(backend, inputs, t):
    if backend == "cutlass":
        return cut.forward(inputs, tactic=t)
    return b12.forward(inputs, tactic=t)


results = []
for name, N, K in SHAPES:
    if ONLY and name not in ONLY.split(","):
        continue
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    del w
    wbytes = wq.numel() + wsf.numel()
    ncopy = max(2, math.ceil(ROT_BYTES / wbytes))
    wqs = [wq] + [wq.clone() for _ in range(ncopy - 1)]
    wsfs = [wsf] + [wsf.clone() for _ in range(ncopy - 1)]
    print(f"[{name}] N={N} K={K} ncopy={ncopy}", flush=True)
    for M in MS:
        torch.manual_seed(M * 7 + N)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        base_inputs = [aq, wq.t(), asf, wsf, torch.bfloat16, out, ws]
        if os.environ.get("TILES"):
            b12_tacs = [((int(s.split("x")[0]), int(s.split("x")[1])), (1, 1), False, False, "sm120", None)
                        for s in os.environ["TILES"].split(",")]
        else:
            b12_tacs = b12.get_valid_tactics(base_inputs, None)
        cands = [("cutlass", t) for t in range(n_cut)] + [("b12x", None)] + [("b12x", t) for t in b12_tacs]
        # reference: cutlass tactic 0
        ref = run("cutlass", base_inputs, 0).clone()
        floor_us = (wbytes + aq.numel() + asf.numel() + M * N * 2) / DRAM_BW * 1e6
        row = {"shape": name, "N": N, "K": K, "M": M, "floor_us": round(floor_us, 2), "cands": {}}
        for backend, t in cands:
            nm = tname(backend, t)
            try:
                t0 = time.time()
                o = run(backend, base_inputs, t).clone()
                torch.cuda.synchronize()
                first_s = time.time() - t0
                o2 = run(backend, base_inputs, t).clone()
                bitwise = bool(torch.equal(o, ref))
                maxabs = float((o.float() - ref.float()).abs().max())
                rel = float(((o.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)))
                det = bool(torch.equal(o, o2))
                ins = [[aq, wqs[i].t(), asf, wsfs[i], torch.bfloat16, out, ws] for i in range(ncopy)]
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for i in range(2):
                        run(backend, ins[i], t)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(ncopy):
                        run(backend, ins[i], t)
                for _ in range(2):
                    g.replay()
                ts = []
                for _ in range(REPLAYS):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000.0 / ncopy)
                ts.sort()
                med = ts[len(ts) // 2]
                p10 = ts[max(0, int(0.1 * len(ts)))]
                p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))]
                # after-graph output check (last call wrote copy ncopy-1, same values)
                post_bitwise = bool(torch.equal(out, o))
                row["cands"][nm] = {"us": round(med, 3), "noise": round((p90 - p10) / med, 4),
                                    "bitwise": bitwise, "det": det, "post_graph_same": post_bitwise,
                                    "maxabs": maxabs, "maxrel": rel, "first_call_s": round(first_s, 3)}
                del g
            except Exception as e:  # noqa
                row["cands"][nm] = {"error": repr(e)[:300]}
        ok = {k: v for k, v in row["cands"].items() if "us" in v}
        cutl = {k: v for k, v in ok.items() if k.startswith("cutlass")}
        b12d = {k: v for k, v in ok.items() if k.startswith("b12x")}
        bc = min(cutl, key=lambda k: cutl[k]["us"]) if cutl else None
        bb = min(b12d, key=lambda k: b12d[k]["us"]) if b12d else None
        row["best_cutlass"] = bc
        row["best_b12x"] = bb
        summ = (f"  M={M:5d} floor={floor_us:7.2f}us " +
                " ".join(f"{k}={v['us']:.2f}{'' if v['bitwise'] else '(!=)'}" for k, v in ok.items()) +
                f" | best cut {bc} {cutl[bc]['us'] if bc else None} best b12x {bb} {b12d[bb]['us'] if bb else None}")
        errs = {k: v for k, v in row["cands"].items() if "error" in v}
        if errs:
            summ += f" ERR {errs}"
        print(summ, flush=True)
        results.append(row)
    del wqs, wsfs
    torch.cuda.empty_cache()

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/b12x_table.json", "w") as f:
    json.dump(results, f, indent=1)
compact = {f"{r['shape']}:M{r['M']}": {
    "floor": r["floor_us"],
    "cut": [r["best_cutlass"], r["cands"].get(r["best_cutlass"], {}).get("us")],
    "cut0": r["cands"].get("cutlass:0", {}).get("us"),
    "b12x": [r["best_b12x"], r["cands"].get(r["best_b12x"], {}).get("us"),
             r["cands"].get(r["best_b12x"], {}).get("bitwise")],
    "all_bitwise": all(v.get("bitwise", False) for v in r["cands"].values())} for r in results}
print(json.dumps({"passed": True, "table": compact}))
