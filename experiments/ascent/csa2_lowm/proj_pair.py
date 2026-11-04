"""Paired timing of one decode step's CSA2 input projections (DeepseekV41Attention._project_inputs).

Same harness as experiments/ascent/csa2_shared_proj_bench.py (its Probe, weights and cold-L2 graph timing), at the
campaign's decode batch sizes M = 1, 2, 4. In the base tree the projections run mxfp8_quantize + flashinfer
mm_mxfp8 (b12x); in this branch wq_a||wkv runs the fused SM120 low-M kernel (trtllm::csa2_mxfp8_low_m_gemm).
kernel_ms = sum over the checkpoint's attention layers (with / without index_wq_b) of the per-layer graph time.

argv: [checkpoint]
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import csa2_shared_proj_bench as base  # noqa: E402

MS = (1, 2, 4)
WARMUP, ITERS = 5, 30


def log(*a):
    print(*a, flush=True)


def main():
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    gen = torch.Generator(device="cuda").manual_seed(base.SEED)
    l2_flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    fused = hasattr(torch.ops.trtllm, "csa2_mxfp8_low_m_gemm")
    log("tree has csa2_mxfp8_low_m_gemm:", fused)
    probes = {}
    for with_index in (True, False):
        probe = base.Probe(with_index, gen)
        assert base.attach_branch(probe), "DeepseekV41Attention._project_inputs missing"
        probes[with_index] = (probe, (lambda p: (lambda x: base.branch(p, x)))(probe))
    results, noises = {}, {}
    for m in MS:
        for with_index in (True, False):
            probe, fn = probes[with_index]
            h = torch.randn((m, base.HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
            base.autotune_run(fn, h)
            for _ in range(3):
                fn(h)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn(h)
                with torch.cuda.graph(graph, stream=stream):
                    fn(h)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            times = []
            for i in range(WARMUP + ITERS):
                l2_flush.zero_()
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
            key = f"{m}_{'index' if with_index else 'swa'}"
            results[key] = med
            noises[key] = (p90 - p10) / med
            log(f"M={m} {key}: median {med * 1000:.2f} us  p10 {p10 * 1000:.2f}  p90 {p90 * 1000:.2f}")
            del graph
    n_index, n_swa = base.layer_counts()
    log(f"layers with index_wq_b: {n_index}, without: {n_swa}")
    batch = {}
    for m in MS:
        ki, ks = f"{m}_index", f"{m}_swa"
        batch[str(m)] = {
            "kernel_ms": n_index * results[ki] + n_swa * results[ks],
            "noise_floor": max(noises[ki], noises[ks]),
            "index_layer_us": results[ki] * 1000,
            "swa_layer_us": results[ks] * 1000,
        }
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "fused_kernel_in_tree": fused,
                      "layers": {"index": n_index, "no_index": n_swa}}))


if __name__ == "__main__":
    main()
