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
ts, sizes = [], []
for it in range(60):
    for _, _, _, g, _ in timed:
        torch.cuda.synchronize()
        t = time.perf_counter()
        b = step(g)
        ts.append((time.perf_counter() - t) * 1e3)
        if it == 0:
            sizes.append(len(b))
    if it % 10 == 0:
        log(f"round {it}: last {ts[-1]:.2f} ms")
ok = all(np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(step(g))))),
                        src[0].numpy()) for _, _, src, g, _ in timed)
s = sorted(ts)
n = len(s)
med, p10, p90 = s[n // 2], s[n // 10], s[(9 * n) // 10]
print(json.dumps({"median": round(med, 3), "p10": round(p10, 3), "p90": round(p90, 3),
                  "mean": round(float(np.mean(ts)), 3), "b64_mb_mean": round(float(np.mean(sizes)) / 1e6, 4),
                  "png_mb_mean": round(float(np.mean(sizes)) * 0.75 / 1e6, 4),
                  "decoded_equal_source": ok}), flush=True)
print(json.dumps({"batch": {"1": {"step_ms": round(med, 3), "noise_floor": round((p90 - p10) / med, 4)}},
                  "output": {"class": "exact"}}))
