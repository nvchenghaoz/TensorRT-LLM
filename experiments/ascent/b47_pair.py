"""Branch 47 paired step-time benchmark: the server's b64_json encode
base64.b64encode(image_to_bytes(img, "PNG")) on the saved served FLUX.3 tensors
(12 distinct timed images, served layout), 60 rounds. Runs identically in the base
and candidate trees. Last line: evidence JSON (step_ms = median, noise (p90-p10)/med).
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b47_common import find_dump, load_images, log  # noqa: E402

from tensorrt_llm.media.encoding import image_to_bytes  # noqa: E402

os.environ.pop("TRTLLM_VISUALGEN_DUMP_DIR", None)
dump = find_dump()
log(f"dump {dump}")
imgs = load_images(dump)
timed = [x for x in imgs if x[1] == "timed"][:12]
log(f"{len(timed)} timed images, stride {timed[0][4]}")


def step(g):
    return base64.b64encode(image_to_bytes(g, format="PNG"))


for _, _, _, g, _ in timed:
    step(g)
per = [[] for _ in timed]  # per image: repeated timings of ONE configuration
sizes = []
for it in range(60):
    for i, (_, _, _, g, _) in enumerate(timed):
        torch.cuda.synchronize()
        t = time.perf_counter()
        b = step(g)
        per[i].append((time.perf_counter() - t) * 1e3)
        if it == 0:
            sizes.append(len(b))
    if it % 10 == 0:
        log(f"round {it}: last {per[-1][-1]:.2f} ms")


def q(v, f):
    s = sorted(v)
    return s[min(len(s) - 1, int(f * len(s)))]


ok = all(np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(step(g))))),
                        (src[0] if src.dim() == 4 else src).numpy()) for _, _, src, g, _ in timed)
img_med = [q(v, 0.5) for v in per]
img_noise = [(q(v, 0.9) - q(v, 0.1)) / q(v, 0.5) for v in per]
# step = mean over the 12 workload images of each image's median encode+b64 time
step_ms = float(np.mean(img_med))
noise = float(np.median(img_noise))
allts = [x for v in per for x in v]
print(json.dumps({"per_image_median_ms": [round(x, 3) for x in img_med],
                  "per_image_noise": [round(x, 4) for x in img_noise],
                  "pooled_median": round(q(allts, 0.5), 3), "pooled_p90": round(q(allts, 0.9), 3),
                  "png_mb_mean": round(float(np.mean(sizes)) * 0.75 / 1e6, 4),
                  "decoded_equal_source": ok}), flush=True)
print(json.dumps({"batch": {"1": {"step_ms": round(step_ms, 3), "noise_floor": round(noise, 4)}},
                  "output": {"class": "exact"}, "decoded_equal_source": ok}))
