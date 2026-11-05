"""Paired timing of one decode step's CSA2 attention-input projections over ALL attention layers (idea 232).

Each attention layer is a probe of DeepseekV41Attention's projection fields (experiments/ascent/
csa2_shared_proj_bench.py Probe: real MXFP8 Linears, q_norm RMSNorm, own random weights) running the tree's
``_project_inputs``. The checkpoint's layers with / without index_wq_b are interleaved in one CUDA graph per M, with a
non-PDL separator kernel between layers (the rest of the layer), so every layer's weights stream cold from DRAM
(~1 GB per step > 128 MB L2). Base tree: wq_a||wkv low-M port -> rmsnorm_cute -> mxfp8_quantize -> b12x wq_b
(||index_wq_b). This branch: wq_a||wkv (trigger at start) -> fused q_norm + quant + wq_b(||index_wq_b).
kernel_ms = median graph replay time (one decode step's projections). 5 warmup + 30 timed replays.
argv: [label, checkpoint]  (csa2_shared_proj_bench.layer_counts reads the checkpoint from argv[2])
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import csa2_shared_proj_bench as base  # noqa: E402

MS = (1, 2, 4)
WARMUP, ITERS = 5, 30


def log(*a):
    print(*a, flush=True)


def layer_order(n_index, n_swa):
    """Interleave index / SWA layers evenly (the checkpoint mixes them)."""
    order, acc = [], 0.0
    total = n_index + n_swa
    for i in range(total):
        acc += n_index / total
        if acc >= 1.0 - 1e-9:
            order.append(True)
            acc -= 1.0
        else:
            order.append(False)
    assert sum(order) == n_index, (sum(order), n_index)
    return order


def main():
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    gen = torch.Generator(device="cuda").manual_seed(base.SEED)
    n_index, n_swa = base.layer_counts()
    order = layer_order(n_index, n_swa)
    log(f"layers with index_wq_b: {n_index}, without: {n_swa}; fused q-path op in tree:",
        hasattr(torch.ops.trtllm, "csa2_qnorm_low_m_gemm"))
    probes = []
    for with_index in order:
        p = base.Probe(with_index, gen)
        assert base.attach_branch(p), "DeepseekV41Attention._project_inputs missing"
        probes.append(p)
    scratch = torch.empty((8, 8), dtype=torch.bfloat16, device="cuda")

    def step(h):
        for p in probes:
            qr, q, swa, idx = base.branch(p, h)
            scratch[: q.shape[0]].copy_(q[:, :8])  # non-PDL separator: the rest of the layer

    batch = {}
    for m in MS:
        h = torch.randn((m, base.HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
        for p in (probes[order.index(True)], probes[order.index(False)]):
            base.autotune_run(lambda x, p=p: base.branch(p, x), h)
        for _ in range(2):
            step(h)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step(h)
            with torch.cuda.graph(graph, stream=stream):
                step(h)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        times = []
        for i in range(WARMUP + ITERS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if i >= WARMUP:
                times.append(start.elapsed_time(end))
        times.sort()
        med = times[len(times) // 2]
        p10 = times[int(0.1 * (len(times) - 1))]
        p90 = times[int(0.9 * (len(times) - 1))]
        batch[str(m)] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90,
                         "per_layer_us": med * 1000 / len(probes)}
        log(f"M={m}: step {med * 1000:.1f} us ({med * 1000 / len(probes):.2f} us/layer) p10 {p10 * 1000:.1f} "
            f"p90 {p90 * 1000:.1f}")
        del graph
        torch.cuda.empty_cache()
    print(json.dumps({"batch": batch, "output": {"class": "reordered"},
                      "layers": {"index": n_index, "no_index": n_swa}}))


if __name__ == "__main__":
    main()
