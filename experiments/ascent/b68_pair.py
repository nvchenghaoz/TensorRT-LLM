"""Branch 68 paired step benchmark: server b64_json encode (image_to_bytes PNG + b64encode)
on the 12 saved served FLUX.3 timed tensors (b47 dump, served layout), as
openai_image_generation runs it. Self-contained: runs unchanged in the base tree.
Per image: 5 warmup + 60 timed calls, images interleaved per round. Reports the mean
over images of per-image medians; noise_floor = median over images of (p90-p10)/median.
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

os.environ.pop("TRTLLM_VISUALGEN_DUMP_DIR", None)
from tensorrt_llm.media.encoding import image_to_bytes  # noqa: E402

try:  # the tree's b64_json call site: image_to_b64 when the tree has it
    from tensorrt_llm.media.encoding import image_to_b64  # noqa: E402
except ImportError:
    image_to_b64 = None

imgs = [x for x in load_images(find_dump()) if x[1] == "timed"][:12]
log(f"{len(imgs)} timed served tensors; image_to_b64 {'present' if image_to_b64 else 'absent'}")


def step(g):
    if image_to_b64 is not None and os.environ.get("B68_FORCE_A") != "1":
        return image_to_b64(g, format="PNG")
    return base64.b64encode(image_to_bytes(g, format="PNG"))


for _ in range(5):
    for x in imgs:
        step(x[3])
ts = [[] for _ in imgs]
outs = [None] * len(imgs)
for r in range(60):
    for i, x in enumerate(imgs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs[i] = step(x[3])
        ts[i].append((time.perf_counter() - t0) * 1e3)
    if r % 20 == 0:
        log(f"round {r}: mean of medians so far {np.mean([np.median(t) for t in ts]):.3f} ms")
meds, nf, p90s = [], [], []
for t in ts:
    s = np.sort(t)
    med, p10, p90 = np.median(s), s[len(s) // 10], s[(9 * len(s)) // 10]
    meds.append(med)
    p90s.append(p90)
    nf.append((p90 - p10) / med)
ok = all(np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(o)))),
                        (x[2][0] if x[2].dim() == 4 else x[2]).numpy())
         for o, x in zip(outs, imgs))
mb = float(np.mean([len(o) * 0.75 / 1e6 for o in outs]))
print(json.dumps({"per_image_median_ms": [round(m, 3) for m in meds],
                  "mean_p90_ms": round(float(np.mean(p90s)), 3), "png_mb_mean": round(mb, 5),
                  "decoded_equal_source": ok}), flush=True)
print(json.dumps({"batch": {"1": {"step_ms": round(float(np.mean(meds)), 3),
                                  "noise_floor": round(float(np.median(nf)), 4)}},
                  "output": {"class": "exact"}}), flush=True)
