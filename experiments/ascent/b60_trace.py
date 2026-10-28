"""Branch 60 (idea 96): CUPTI (torch.profiler) trace of one steady served-config request,
switch ON (step-batched image vec rows) vs OFF (TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH=0 = main's
per-step M=1 GEMVs), in one process. Not timing evidence (in-process toggles).

Pipeline built from experiments/ascent-config/engine.json as serve_launcher's trtllm-serve run
applies it (VisualGenArgs overlay incl. cuda_graph_config), warmup [[1024,1024]] t2i.
Per config {graphs on (served), graphs off (eager forward, record_shapes)} x {ON, OFF}:
  1 warm request, then 1 profiled request: kernel count / sum per request, top kernels, kernels
  whose per-request count differs ON vs OFF, and (graphs off) the BF16 GEMM ops at M<=16 by
  shape (the image vec GEMVs) with their device time.
Also: Dynamo unique-graph count after warmup and per phase; A/B/A per mode not here (see
b60_served_check.py).

usage: python b60_trace.py <checkpoint>
"""
import collections
import json
import os
import subprocess
import sys
import time
import types

import torch

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = "TRTLLM_FLUX3_IMAGE_VEC_STEP_BATCH"


def log(*a):
    print(*a, flush=True)


def counters():
    from torch._dynamo.utils import counters as c

    return c["stats"].get("unique_graphs", 0)


import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

assert hasattr(tf.Flux3Model, "prepare_flow_map_image"), "branch 60 symbol missing"
assert tf.FLUX3_TEXT_STEP_BROADCAST and tf.FLUX3_BROADCAST_ADALN
engine = json.load(open(os.path.join(HERE, "..", "ascent-config", "engine.json")))
log("engine overlay", json.dumps(engine))
log("tensorrt_llm", tensorrt_llm.__file__)
try:
    log("git", subprocess.run(["git", "-C", HERE, "log", "--oneline", "-3"], capture_output=True,
                              text=True).stdout)
except Exception as exc:  # noqa: BLE001
    log("git unavailable", exc)
pipeline_config = dict(engine.get("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
kwargs = {k: v for k, v in engine.items() if k != "pipeline_config"}
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **kwargs,
)
t0 = time.time()
os.environ[ENV] = "1"
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s; graph runner: {runner is not None and runner.enabled}"
    f"; graphs captured: {len(runner.graphs) if runner else 0}")
warm_graphs = counters()


def gen(prompt="A red fox standing in fresh snow at sunrise, photorealistic", seed=1000):
    from tensorrt_llm.visual_gen.params import VisualGenParams

    params = VisualGenParams(height=1024, width=1024, seed=seed, num_inference_steps=16)
    req = types.SimpleNamespace(prompt=prompt, params=params)
    out = pipe.infer(req)
    torch.cuda.synchronize()
    return out


def profile_one(shapes):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CUDA] + ([ProfilerActivity.CPU] if shapes else [])
    with profile(activities=acts, record_shapes=shapes) as prof:
        out = gen()
    kern = collections.defaultdict(lambda: [0, 0.0])
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA:
            k = kern[ev.name[:120]]
            k[0] += 1
            k[1] += ev.time_range.end - ev.time_range.start
    small = {}
    if shapes:
        for ev in prof.key_averages(group_by_input_shape=True):
            dev = getattr(ev, "self_device_time_total", 0)
            if ev.key not in ("aten::mm", "aten::addmm") or dev <= 0:
                continue
            a = ev.input_shapes[0] if ev.key == "aten::mm" else ev.input_shapes[1]
            if len(a) >= 2 and a[0] <= 16:
                small[f"{ev.key} {ev.input_shapes}"] = {"calls": ev.count,
                                                        "us": round(dev, 1)}
    return {
        "denoise_ms": round(out.denoise * 1e3, 2) if getattr(out, "denoise", None) else None,
        "kernel_launches": sum(v[0] for v in kern.values()),
        "kernel_sum_ms": round(sum(v[1] for v in kern.values()) / 1e3, 3),
        "kernels": {n: [v[0], round(v[1], 1)] for n, v in kern.items()},
        "gemm_m_le_16": small,
        "gemm_m_le_16_calls": sum(v["calls"] for v in small.values()),
        "gemm_m_le_16_ms": round(sum(v["us"] for v in small.values()) / 1e3, 3),
    }


res = {"warmup_unique_graphs": warm_graphs}
for graphs in (True, False):
    if runner is not None:
        runner.enabled = graphs
    for sw in ("1", "0"):
        os.environ[ENV] = sw
        before = counters()
        gen()
        p = profile_one(shapes=not graphs)
        p["new_dynamo_graphs_warm+profile"] = counters() - before
        p["cuda_graphs_captured_total"] = len(runner.graphs) if runner else 0
        key = f"graphs_{'on' if graphs else 'off'}_switch_{'on' if sw == '1' else 'off'}"
        res[key] = p
        log(key, json.dumps({k: v for k, v in p.items() if k != "kernels"}))
os.environ[ENV] = "1"
if runner is not None:
    runner.enabled = True

summary = {}
for g in ("on", "off"):
    on, off = res[f"graphs_{g}_switch_on"], res[f"graphs_{g}_switch_off"]
    names = set(on["kernels"]) | set(off["kernels"])
    diff = []
    for n in names:
        a, b = on["kernels"].get(n, [0, 0.0]), off["kernels"].get(n, [0, 0.0])
        if a[0] != b[0]:
            diff.append({"kernel": n, "calls_off": b[0], "calls_on": a[0],
                         "us_off": b[1], "us_on": a[1]})
    diff.sort(key=lambda d: -(d["us_off"] - d["us_on"]))
    top = sorted(on["kernels"].items(), key=lambda kv: -kv[1][1])[:12]
    summary[f"graphs_{g}"] = {
        "kernel_sum_ms_per_request": {"off": off["kernel_sum_ms"], "on": on["kernel_sum_ms"]},
        "kernel_launches_per_request": {"off": off["kernel_launches"], "on": on["kernel_launches"]},
        "denoise_ms_informational": {"off": off["denoise_ms"], "on": on["denoise_ms"]},
        "gemm_m_le_16": {"off": off["gemm_m_le_16"], "on": on["gemm_m_le_16"]},
        "gemm_m_le_16_calls": {"off": off["gemm_m_le_16_calls"], "on": on["gemm_m_le_16_calls"]},
        "gemm_m_le_16_ms": {"off": off["gemm_m_le_16_ms"], "on": on["gemm_m_le_16_ms"]},
        "kernels_with_changed_count": diff[:30],
        "changed_kernels_ms": {"off": round(sum(d["us_off"] for d in diff) / 1e3, 3),
                               "on": round(sum(d["us_on"] for d in diff) / 1e3, 3)},
        "top_kernels_on": [[n, v[0], v[1]] for n, v in top],
        "new_dynamo_graphs": {"off": off["new_dynamo_graphs_warm+profile"],
                              "on": on["new_dynamo_graphs_warm+profile"]},
    }
    log(f"SUMMARY graphs_{g}", json.dumps(summary[f"graphs_{g}"]))
print(json.dumps({"passed": True, "note": "trace only, not timing evidence",
                  "warmup_unique_graphs": warm_graphs, **summary}))
