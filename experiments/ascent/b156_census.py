"""Branch 156 census: compiled bf16 FLUX.3 VAE decode (1x128x64x64 latent, real weights).

TORCH_LOGS=output_code -> $AUTO_DEV_ARTIFACTS/output_code.txt; 5 warm decodes under
torch.profiler. Prints kernel classes per decode (conv, dgrad, triton_poi, triton_red,
nchwToNhwc, nhwcToNchw, convertTensor), and for every upsample (main: the sub-pixel
fprop conv followed by the 'stack' interleave; branch: the strided dgrad) the window of
the next kernels in launch order (ms/call), with the load/store lines of their Triton
source. Usable in either arm of run_gpu_pair.
usage: python experiments/ascent/b156_census.py <checkpoint_root>
"""

import os
import re
import subprocess
import sys

ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b156")
os.makedirs(ART, exist_ok=True)
LOG = os.path.join(ART, "output_code.txt")
os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_LOGS_OUT"] = LOG

import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

root = sys.argv[1]


def log(*a):
    print(*a, flush=True)


log("HEAD", subprocess.run(["git", "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip())

from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    set_vae_decoder_trunk_dtype,
)

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

import torch._dynamo.utils as du  # noqa: E402

log("dynamo counters", json.dumps({k: dict(v) for k, v in du.counters.items()
                                   if k in ("frames", "graph_break", "recompiles")}))

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

evs = sorted([ev for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA],
             key=lambda ev: ev.time_range.start)
evs = [ev for ev in evs if "Memcpy" not in ev.name and "Memset" not in ev.name]
per = len(evs) // NDEC
dur, cnt = defaultdict(float), Counter()
for ev in evs:
    dur[ev.name] += ev.device_time_total / 1e3 / NDEC
    cnt[ev.name] += 1
last = evs[-per:]
seq = [ev.name for ev in last]
seq_ms = [ev.device_time_total / 1e3 for ev in last]
log(f"kernels per decode: {per}")


def cls(n):
    lo = n.lower()
    for k, c in (("nchwtonhwc", "nchwToNhwc"), ("nhwctonchw", "nhwcToNchw"), ("converttensor", "convertTensor")):
        if k in lo:
            return c
    if n.startswith("triton_poi"):
        return "triton_poi"
    if n.startswith("triton_red") or n.startswith("triton_per"):
        return "triton_red"
    if "dgrad" in lo:
        return "dgrad"
    if "nvjet" in lo:
        return "nvjet"
    if any(k in lo for k in ("xmma", "conv", "cutlass", "cudnn", "implicit", "sm80", "fprop")):
        return "conv"
    return "other"


classes = defaultdict(lambda: [0, 0.0])
for n in set(seq):
    c = cls(n)
    classes[c][0] += cnt[n] // NDEC
    classes[c][1] += dur[n]
log("classes (launches/decode, ms/decode):")
for c, (k, t) in sorted(classes.items(), key=lambda kv: -kv[1][1]):
    log(f"  {c:14s} {k:4d} {t:8.3f}")
log(f"GPU sum ms/decode: {sum(dur[n] for n in set(seq)):.3f}")

# windows: main = sub-pixel 'stack' interleave (start at the conv before it);
# branch = strided dgrad main kernel (not the helper stages)
windows, interest = [], []
for i, n in enumerate(seq):
    lo = n.lower()
    is_main_up = n.startswith("triton_poi") and "stack" in n
    is_dgrad = "dgrad" in lo and "kernel_helper" not in lo
    if is_main_up or is_dgrad:
        lo_i = i - 1 if is_main_up else i
        w = list(zip(seq[lo_i: i + 9], seq_ms[lo_i: i + 9]))
        windows.append([(k[:120], round(t, 4)) for k, t in w])
        interest += [k for k, _ in w if k.startswith("triton_")]
log("upsample windows (ms/call in the last profiled decode, name):")
for w in windows:
    for k, t in w:
        log(f"  {t:8.4f}  {k}")
    log("  --")

src = {}
try:
    txt = open(LOG).read()
    for m in re.finditer(r"async_compile\.triton\('(\w+)', '''(.*?)'''", txt, re.S):
        src.setdefault(m.group(1), m.group(2))
    log(f"output_code: {len(txt)} bytes, {len(src)} triton kernels")
except Exception as ex:  # noqa: BLE001
    log("output_code read failed:", ex)

pat = re.compile(r"(size_hints|numel =|tl\.load|tl\.store|def triton_)")
done = set()
for k in interest:
    if k in done:
        continue
    done.add(k)
    body = src.get(k)
    log(f"==== {k}")
    if body is None:
        log("  (source not found)")
        continue
    for line in body.splitlines():
        if pat.search(line):
            log("   " + line.strip()[:220])

with open(os.path.join(ART, "census_seq.json"), "w") as f:
    json.dump(list(zip(seq, seq_ms)), f)
top = sorted(((round(dur[n], 4), cnt[n] // NDEC, n[:110]) for n in set(seq)), reverse=True)[:30]
log(json.dumps({"census": True, "passed": True, "kernels_per_decode": per, "median_ms": ts[15],
                "classes": {c: [k, round(t, 4)] for c, (k, t) in classes.items()},
                "windows": windows, "top": top}))
