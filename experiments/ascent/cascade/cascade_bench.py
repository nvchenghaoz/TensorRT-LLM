#!/usr/bin/env python3
# Candidate arm for idea 20: kernel_mha_prefix + kernel_mha (suffix + merge) at B=8/16/32 over the serving decode
# shape (24 q / 4 kv heads, d256, e4m3 paged KV, 32768-token shared prefix pages + 2048 + 128 unique tokens), with
# kernel_mha's sub-sequence count as main's runner picks it (one-wave cap: 5/2/1). Timed exactly like
# experiments/ascent/xqa_merge/xqa_evidence.py --mode bench (the baseline arm): one CUDA graph per batch (here holding
# both launches), output NaN-poisoned and L2 flushed with a 384 MiB read sweep before every replay, CUDA events,
# 10-90% trimmed mean over rounds x iters. --diag also times each kernel alone and prints errors vs the FP32
# reference and vs main's kernel_mha (compiled from the main commit) on the same inputs.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade_lib as cl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def time_graph(call, flush, a, label):
    all_ts = []
    for k in range(a.rounds):
        n = a.warmup + a.iters
        st = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            call.inp["out"].fill_(float("nan"))
            flush()
            st[i].record()
            call.graph.replay()
            en[i].record()
        torch.cuda.synchronize()
        ts = np.array([st[i].elapsed_time(en[i]) for i in range(a.warmup, n)])
        cl.log(f"  {label} round {k}: trimmed mean {cl.stats(ts)['kernel_ms'] * 1e3:.2f} us")
        all_ts.append(ts)
    return cl.stats(np.concatenate(all_ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="8,16,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--flush-mb", type=int, default=384)
    p.add_argument("--diag", action="store_true")
    p.add_argument("--sub-seq", default="", help="override kernel_mha counts, e.g. 8:5,16:2,32:1")
    a = p.parse_args()
    cl.init_cuda()
    kern = cl.Kernels(os.path.join(os.getcwd(), "cpp/kernels/xqa"))
    assert kern.prefix is not None, "candidate mha.cu has no kernel_mha_prefix"
    batches = [int(x) for x in a.batches.split(",")]
    override = dict(tuple(int(v) for v in kv.split(":")) for kv in a.sub_seq.split(",") if kv)
    inp = cl.build_inputs(max(batches))
    flush = cl.xe.Flusher(a.flush_mb)
    main_k = cl.Kernels(cl.extract_tree(cl.MAIN_COMMIT)) if a.diag else None
    res = {"batch": {}, "diag": {}}
    for b in batches:
        m = override.get(b, cl.main_nb_sub_seq(b))
        call = cl.Call(kern, inp, b, m, cascade=True).capture()
        s = time_graph(call, flush, a, f"B={b} m={m} C={call.c} G={call.g}")
        s.update(nb_sub_seq=m, nb_prefix_chunks=call.c, nb_row_groups=call.g,
                 output_finite=bool(torch.isfinite(call.out().float()).all().item()))
        res["batch"][str(b)] = s
        cl.log(f"B={b}: cascade {s['kernel_ms'] * 1e3:.2f} us (median {s['median_ms'] * 1e3:.2f}, "
               f"noise {s['noise_floor']:.3f})")
        if a.diag:
            d = {}
            out_c = call.out().clone()
            info = torch.empty(1 + call.c, dtype=torch.int32, device="cuda")
            cl.ck(cl.cu.cuMemcpyDtoD(info.data_ptr(), inp["scratch_ptr"], 4 * (1 + call.c)), "dtod")
            d["cascade_info"] = info.cpu().tolist()
            ref = cl.reference(inp, b)
            mk = cl.Call(main_k, inp, b, m, cascade=False).capture()
            mk.graph.replay()
            torch.cuda.synchronize()
            out_m = mk.out().clone()
            for name, o in (("cascade", out_c), ("main", out_m)):
                e = (o.float() - ref).abs()
                d[f"{name}_max_abs"] = float(e.max())
                d[f"{name}_mean_abs"] = float(e.mean())
            d["cascade_vs_main_max_abs"] = float((out_c.float() - out_m.float()).abs().max())
            d["main_us"] = time_graph(mk, flush, a, f"main B={b} m={m}")["kernel_ms"] * 1e3
            # each kernel alone
            pf = cl.Call(kern, inp, b, m, cascade=True)
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                cl.Kernels._launch(kern.prefix, (pf.c, pf.g, cl.NB_K), (kern.pcta, 1, 1), kern.psmem, pf.pfx_args)
            pf.graph = g1
            d["prefix_only_us"] = time_graph(pf, flush, a, f"prefix B={b}")["kernel_ms"] * 1e3
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                cl.Kernels._launch(kern.mha, (m, cl.NB_K, b), (128, 1, 2), kern.smem, pf.mha_args)
            pf.graph = g2
            d["suffix_mha_only_us"] = time_graph(pf, flush, a, f"suffix+merge B={b}")["kernel_ms"] * 1e3
            res["diag"][str(b)] = d
            cl.log(f"diag B={b}: {d}")
    res["output"] = {"class": "reordered", "basis": "prefix partials merged in kernel_mha's multi-block merge; "
                     "tolerance evidence from cascade_check.py (FP32 reference)"}
    res["config"] = dict(source_sha256=kern.src_sha, smem=kern.smem, prefix_smem=kern.psmem, prefix_regs=kern.pregs,
                         warmup=a.warmup, iters=a.iters, rounds=a.rounds,
                         flush=f"{a.flush_mb} MiB read sweep + NaN poison before every replay",
                         timing="CUDA graph replay (prefix + kernel_mha) + CUDA events; 10-90% trimmed mean",
                         pdl="off (as the baseline arm)")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
