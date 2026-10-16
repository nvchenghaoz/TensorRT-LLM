# Stage sweep for the full-K dense Marlin launch (idea 136), diagnostic only (candidate tree).
# Same method as bench.py "mm": 16 cold-L2 weight copies (~283 MB) back to back in one graph, PDL on,
# arms captured in one process and replayed round-robin; every arm checked bitwise against arm A.
import json
import os
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
WARM, ITERS, COPIES = 10, 60, 16
N, K = 5120, 6144
MS = [1, 2, 4, 8]
ARMS = [("A", "0", "12"), ("F8", "1", "8"), ("F10", "1", "10"), ("F12", "1", "12"), ("F14", "1", "14")]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def set_arm(a):
    os.environ["TRTLLM_MARLIN_FULLK"] = a[1]
    os.environ["TRTLLM_MARLIN_FULLK_STAGES"] = a[2]


g = torch.Generator(device=dev).manual_seed(11)
w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
base = _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), N, K, 16)
ws = [tuple(t.clone() for t in base) for _ in range(COPIES)]
del w, base


def marlin(x, wt):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=N, size_k=K)


table = {}
ok_all = True
for m in MS:
    x = (torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    graphs, outs, eager = {}, {}, {}
    for a in ARMS:
        set_arm(a)
        eager[a[0]] = torch.cat([marlin(x, wt).reshape(-1) for wt in ws])
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                [marlin(x, wt) for wt in ws]
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o = [marlin(x, wt) for wt in ws]
        graphs[a[0]], outs[a[0]] = gr, o
        for _ in range(WARM):
            gr.replay()
    torch.cuda.synchronize()
    evs = {a[0]: [] for a in ARMS}
    for it in range(ITERS):
        order = ARMS if it % 2 == 0 else ARMS[::-1]
        for a in order:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[a[0]].replay()
            e1.record()
            evs[a[0]].append((e0, e1))
    torch.cuda.synchronize()
    row = {}
    for a in ARMS:
        t = sorted(e0.elapsed_time(e1) * 1e3 / COPIES for e0, e1 in evs[a[0]])
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        st = torch.cat([y.reshape(-1) for y in outs[a[0]]])
        eq = bool(torch.equal(st, eager["A"]) and torch.equal(eager[a[0]], eager["A"]))
        ok_all = ok_all and eq
        row[a[0]] = dict(med=round(med, 3), p10=round(p10, 3), p90=round(p90, 3), bitwise_eq_A=eq)
    a_med = row["A"]["med"]
    for a in ARMS[1:]:
        row[a[0]]["vs_A_pct"] = round(100.0 * (row[a[0]]["med"] - a_med) / a_med, 2)
    table[f"M{m}"] = row
    log(f"M={m} " + " | ".join(f"{k} {v['med']:.2f} ({v['p10']:.2f}-{v['p90']:.2f})"
                                + (f" {v['vs_A_pct']:+.2f}%" if 'vs_A_pct' in v else "") + f" eq={v['bitwise_eq_A']}"
                                for k, v in row.items()))
    del graphs, outs
    torch.cuda.synchronize()
print(json.dumps(dict(table=table, bitwise_all=ok_all)))
