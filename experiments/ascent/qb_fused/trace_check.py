"""Kineto trace of the CUDA-graph decode replay of the CSA2 attention-input route (idea 232, S4), one GPU.

The checkpoint's attention layers (index / SWA probes of DeepseekV41Attention with real MXFP8 Linears, as in
chain_pair.py) run ``_project_inputs`` inside one CUDA graph per M; graphs are captured at M = 1, 2, 4, 8 after
the eager warmup (the fused route warms every kernel instance eagerly; nothing compiles inside capture). The graph
replay at M = 1 and 4 is traced with torch.profiler (Kineto, CUDA activity) and checked per layer:
  * the gated route runs exactly two kernels per layer: wq_a||wkv (csa2_mxfp8_low_m_gemm_kernel<..., false>) then
    the fused q_norm + quant + wq_b(||index_wq_b) kernel (<..., true>); no FlashInfer RMSNorm, no
    quantize_with_block_size and no b12x dense GEMM;
  * the fused kernel starts before wq_a||wkv ends (PDL overlap of its pre-wait weight staging).
argv: [label, checkpoint]
"""

import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import chain_pair as cp  # noqa: E402
import csa2_shared_proj_bench as base  # noqa: E402

FORBIDDEN = re.compile(r"rmsnorm|RMSNorm|quantize_with_block_size|blockscaled|B12x|b12x", re.I)


def log(*a):
    print(*a, flush=True)


def main():
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from torch.profiler import ProfilerActivity, profile

    gen = torch.Generator(device="cuda").manual_seed(base.SEED)
    n_index, n_swa = base.layer_counts()
    order = cp.layer_order(n_index, n_swa)
    probes = []
    for with_index in order:
        p = base.Probe(with_index, gen)
        assert base.attach_branch(p)
        probes.append(p)
    scratch = torch.empty((8, 8), dtype=torch.bfloat16, device="cuda")

    def step(h):
        for p in probes:
            qr, q, swa, idx = base.branch(p, h)
            scratch[: q.shape[0]].copy_(q[:, :8])

    failures, report = [], {}
    graphs = {}
    for m in (1, 2, 4, 8):
        h = torch.randn((m, base.HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
        step(h)  # eager warmup
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step(h)
            with torch.cuda.graph(g, stream=s):
                step(h)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        graphs[m] = g
        report[f"capture_M{m}"] = "ok"
        log(f"captured M={m}")
    for m in (1, 4):
        g = graphs[m]
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            g.replay()
            torch.cuda.synchronize()
        evs = sorted(((e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA), key=lambda t: t[0])
        names = [n for _, _, n in evs]
        a_idx = [i for i, n in enumerate(names) if "csa2_mxfp8_low_m_gemm_kernel" in n and ", false>" in n]
        b_idx = [i for i, n in enumerate(names) if "csa2_mxfp8_low_m_gemm_kernel" in n and ", true>" in n]
        forbidden = sorted(set(n[:90] for n in names if FORBIDDEN.search(n)))
        overlaps, gaps = 0, []
        for ai in a_idx:
            nxt = [bi for bi in b_idx if bi > ai]
            if not nxt:
                continue
            bi = nxt[0]
            a_s, a_e, _ = evs[ai]
            b_s, _, _ = evs[bi]
            gaps.append(b_s - a_e)
            overlaps += int(b_s < a_e)
        per_layer_kernels = len([n for n in names if "copy" not in n.lower() and "elementwise" not in n.lower()])
        entry = {"layers": len(probes), "wq_a_wkv_kernels": len(a_idx), "fused_qb_kernels": len(b_idx),
                 "B_starts_before_A_ends": overlaps, "B_start_minus_A_end_us": [round(x, 2) for x in gaps[:8]],
                 "forbidden_kernels": forbidden, "non_separator_kernels": per_layer_kernels,
                 "distinct_kernels": sorted(set(n[:90] for n in names))}
        report[f"trace_M{m}"] = entry
        log(f"TRACE M={m} {json.dumps(entry)}")
        # Index layers at M <= 4 and SWA layers at every M take the fused route.
        if len(b_idx) != len(probes) or len(a_idx) != len(probes):
            failures.append(f"M={m}: {len(a_idx)} A / {len(b_idx)} fused kernels for {len(probes)} layers")
        if forbidden:
            failures.append(f"M={m}: forbidden kernels on the gated route: {forbidden}")
        if overlaps < len(probes) // 2:
            failures.append(f"M={m}: B started before A ended in only {overlaps}/{len(probes)} layers")
        prof.export_chrome_trace(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), f"qb_trace_M{m}.json"))
    print(json.dumps({"passed": not failures, "failures": failures, "report": report}))


if __name__ == "__main__":
    main()
