"""Branch 113 tail census: compiled bf16 FLUX.3 VAE decode (real weights, 1x128x64x64).

Profiles 5 warm decodes with torch.profiler and prints the LAST 10 kernels of a decode in
launch order (ms/call): the decoder tail rows (norm_out stats, norm_out apply+swish,
conv_out convertTensor, conv_out fprop, bias / the fused Triton tail kernel). Also checks
Dynamo counters: one frame, no graph break, counters unchanged across warmup + 30 timed
+ 5 profiled decodes.
usage: python experiments/ascent/b113_census.py <checkpoint_root>
Last stdout line: JSON.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

DUMP = os.environ.get("B113_CODE") == "1"
if DUMP:
    ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b113")
    os.makedirs(ART, exist_ok=True)
    LOG = os.path.join(ART, "output_code.txt")
    os.environ["TORCH_LOGS"] = "output_code"
    os.environ["TORCH_LOGS_OUT"] = LOG

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    set_vae_decoder_trunk_dtype,
)


def log(*a):
    print(*a, flush=True)


def frames():
    import torch._dynamo.utils as du

    return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "recompiles", "graph_break")}


root = sys.argv[1]
vae, dt = gn_check.build_vae(root)
set_vae_decoder_trunk_dtype(vae.decoder, torch.bfloat16)
vae.compile_decoder()
g = torch.Generator(device="cuda").manual_seed(0)
Z = torch.randn(1, 128, 64, 64, device="cuda", generator=g).to(getattr(torch, dt))


@torch.inference_mode()
def dec():
    return vae.decode(Z)


for i in range(3):
    dec()
    torch.cuda.synchronize()
    log("warm", i)
f_warm = frames()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
ts = []
for _ in range(30):
    s.record()
    dec()
    e.record()
    e.synchronize()
    ts.append(s.elapsed_time(e))
ts.sort()
log(f"census timing 30 decodes: median {ts[15]:.3f} p10 {ts[3]:.3f} p90 {ts[26]:.3f} ms")

from torch.profiler import ProfilerActivity, profile  # noqa: E402

NDEC = 5
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(NDEC):
        dec()
    torch.cuda.synchronize()
f_end = frames()
evs = sorted(
    [ev for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA],
    key=lambda ev: ev.time_range.start,
)
evs = [ev for ev in evs if "Memcpy" not in ev.name and "Memset" not in ev.name]
per = len(evs) // NDEC
dur, cnt = defaultdict(float), Counter()
for ev in evs:
    dur[ev.name] += ev.device_time_total / 1e3 / NDEC
    cnt[ev.name] += 1
# per-position mean over the 5 decodes for the last 10 launches
tail = []
for k in range(10, 0, -1):
    ts_k = [evs[d * per + per - k].device_time_total / 1e3 for d in range(NDEC)]
    tail.append({"pos": per - k, "name": evs[per - k].name[:120], "ms": round(sum(ts_k) / NDEC, 4)})
for t in tail:
    log(f"  {t['pos']:4d} {t['ms']:8.4f}  {t['name']}")
if DUMP:
    txt = open(LOG).read()
    src = {m.group(1): m.group(2) for m in re.finditer(r"async_compile\.triton\('(\w+)', '''(.*?)'''", txt, re.S)}
    pat = re.compile(r"(size_hints|xnumel =|rnumel|r0_numel|tl\.load|tl\.store|def triton_)")
    for t in tail:
        body = src.get(t["name"])
        if body is None:
            continue
        log(f"==== {t['name']} ({t['ms']} ms)")
        for line in body.splitlines():
            if pat.search(line):
                log("   " + line.strip()[:220])
    # the call sequence around the custom op in the wrapper
    j = txt.rfind("def call(")
    if j >= 0:
        seg = txt[j:]
        log("==== wrapper tail")
        lines = seg.splitlines()
        k = max((n for n, l in enumerate(lines) if "_conv3x3_to3_kernel" in l), default=len(lines))
        for line in lines[max(0, k - 60): k + 5]:
            log("   " + line[:220])
gpu_sum = sum(dur[n] for n in set(ev.name for ev in evs[-per:]))
fr = f_end.get("frames", {})
dyn = {"frames_after_warmup": f_warm, "frames_after_timed_and_profiled": f_end,
       "counters_unchanged": f_warm == f_end, "one_frame": fr.get("total", 0) == 1 and fr.get("ok", 0) == 1,
       "no_graph_break": not f_end.get("graph_break")}
log(json.dumps({"census": True, "kernels_per_decode": per, "median_ms": ts[15], "p10_ms": ts[3],
                "p90_ms": ts[26], "gpu_sum_ms": round(gpu_sum, 3), "tail_last10": tail, "dynamo": dyn,
                "passed": all(v for k, v in dyn.items() if isinstance(v, bool))}))
