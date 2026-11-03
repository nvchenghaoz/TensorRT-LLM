"""Time FA2 hd512 BF16 CSA2 attention with K and V passed as ONE tensor (as the tree does).

decode (labels 1/2/4/8): one 40-layer decode step of CSA2FlashInfer._forward_pools (stage rows
+ mask, FA2 ragged run(q, rows, rows), sink epilogue), 40 separate adapters, CUDA-graph replay,
L2 flushed before every replay.
prefill (label 2048): the run_shared paged FA2 call wrapper.run(q, (pages, pages)) for 2048
queries x 640 one-row pages, eager, L2 flushed before every call.
Baseline tree = pinned FlashInfer, candidate = overlay. Also reports, as a diagnostic, the same
calls with V cloned (non-alias path) so the overlay's effect is visible inside one arm.
Prints {"batch": {label: {"kernel_ms", "noise_floor"}}, "output": {"class": "exact"}}.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (  # noqa: E402
    SCALE,
    L2Flush,
    log,
    log_flashinfer,
    log_jit_modules,
    make_prefill,
    new_adapter,
    prefill_plan,
    start_heartbeat,
)

import common as fa  # noqa: E402

LAYERS = 40
COUNTS = [int(c) for c in os.environ.get("FA2_ALIAS_COUNTS", "1,2,4,8").split(",")]
SAMPLES = 30


def stats(times):
    times = sorted(times)
    n = len(times)
    med = times[n // 2]
    p10 = times[int(0.1 * (n - 1))]
    p90 = times[int(0.9 * (n - 1))]
    return med, (p90 - p10) / med


def time_graph(fn, flush):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(SAMPLES):
        flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    del graph
    return stats(times)


def time_eager(fn, flush):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(SAMPLES):
        flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return stats(times)


def main():
    start_heartbeat()
    torch.cuda.set_device(0)
    log("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    log_flashinfer()
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "glue_common", os.path.join(os.path.dirname(HERE), "csa2_fi_glue", "common.py")
    )
    glue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glue)

    flush = L2Flush()
    results = {}
    diag = {}
    adapters = [new_adapter() for _ in range(LAYERS)]
    for count in COUNTS:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234 + count)
        layers = [glue.make_layer(count, gen, glue.EXTRA) for _ in range(LAYERS)]
        outs = [torch.empty(count, 16 * 512, dtype=torch.bfloat16, device="cuda") for _ in layers]

        def step():
            for inst, layer, out in zip(adapters, layers, outs):
                glue.fused_call(inst, layer, out)

        med, nf = time_graph(step, flush)
        results[str(count)] = {"kernel_ms": round(med, 5), "noise_floor": round(nf, 5),
                               "per_layer_us": round(med * 1000 / LAYERS, 3)}
        log(f"decode count {count}: step {med:.4f} ms ({med * 1000 / LAYERS:.2f} us/layer) nf {nf:.4f}")

        # Diagnostic: FA2 ragged call alone, alias vs cloned V, 40 calls in a graph.
        qs, rowss, rowsc, plans = [], [], [], []
        for i in range(LAYERS):
            q, rows, valid, _ = fa.make_decode(count, fa.WIDTH, 777 + 31 * i + count)
            plan = fa.decode_plan(adapters[i], count, fa.WIDTH, valid)
            qs.append(q)
            rowss.append(rows)
            rowsc.append(rows.clone())
            plans.append(plan)

        def fa2_alias():
            for p, q, r in zip(plans, qs, rowss):
                p.wrapper.run(q, r, r, return_lse=True)

        def fa2_clone():
            for p, q, r, rc in zip(plans, qs, rowss, rowsc):
                p.wrapper.run(q, r, rc, return_lse=True)

        ma, na = time_graph(fa2_alias, flush)
        mc, nc = time_graph(fa2_clone, flush)
        diag[f"decode{count}"] = {"fa2_alias_us_per_call": round(ma * 1000 / LAYERS, 3),
                                  "fa2_clone_us_per_call": round(mc * 1000 / LAYERS, 3),
                                  "nf_alias": round(na, 4), "nf_clone": round(nc, 4)}
        log(f"  fa2 ragged alias {ma * 1000 / LAYERS:.2f} us/call, clone-V {mc * 1000 / LAYERS:.2f} us/call")
        del layers, outs, qs, rowss, rowsc, plans
        torch.cuda.synchronize()

    # Prefill: run_shared's paged FA2 call.
    adapter = new_adapter()
    q, bank, indices, sink = make_prefill(4242)
    plan, pages = prefill_plan(adapter, q, bank, indices, sink)
    pages_c = pages.clone()
    qc = q.contiguous()
    log_jit_modules("after plans")
    med, nf = time_eager(lambda: plan.wrapper.run(qc, (pages, pages), return_lse=True), flush)
    results[str(fa.PREFILL_Q)] = {"kernel_ms": round(med, 5), "noise_floor": round(nf, 5)}
    log(f"prefill 2048x640 paged alias: {med:.4f} ms nf {nf:.4f}")
    mc, nc = time_eager(lambda: plan.wrapper.run(qc, (pages, pages_c), return_lse=True), flush)
    diag["prefill2048"] = {"alias_ms": round(med, 5), "clone_ms": round(mc, 5), "nf_clone": round(nc, 4)}
    log(f"prefill 2048x640 paged clone-V: {mc:.4f} ms nf {nc:.4f}")
    log(json.dumps({"diagnostics": diag}))
    print(json.dumps({"batch": results, "output": {"class": "exact"}, "diagnostics": diag}))


if __name__ == "__main__":
    main()
