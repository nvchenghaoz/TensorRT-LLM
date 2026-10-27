# Paired step-time benchmark for the server's b64_json PNG response encode:
# image_to_bytes(cuda uint8 1024x1024x3, "PNG") + base64 encode, exactly what
# openai_image_generation runs per image. Self-contained so the identical code
# runs in the base tree via `python -c`.
import base64, glob, io, json, time  # noqa: E401
import numpy as np
import torch
from PIL import Image

print("importing tensorrt_llm.media.encoding", flush=True)
from tensorrt_llm.media.encoding import image_to_bytes  # noqa: E402

paths = sorted(glob.glob("docs/source/blogs/media/tech_blog28_visual_comparison_*.jpg"))
arr = np.asarray(Image.open(paths[0]).convert("RGB").resize((1024, 1024), Image.LANCZOS)).copy()
g = torch.from_numpy(arr).cuda()


def step():
    return base64.b64encode(image_to_bytes(g, format="PNG")).decode("utf-8")


for _ in range(5):
    out = step()
print("warm", flush=True)
ts = []
for i in range(60):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = step()
    ts.append((time.perf_counter() - t) * 1e3)
    if i % 20 == 0:
        print(f"iter {i}: {ts[-1]:.2f} ms", flush=True)
s = sorted(ts)
med = s[len(s) // 2]
p10, p90 = s[len(s) // 10], s[(9 * len(s)) // 10]
dec = np.asarray(Image.open(io.BytesIO(base64.b64decode(out))))
print(json.dumps({"png_b64_bytes": len(out), "decoded_equal_input": bool(np.array_equal(dec, arr)),
                  "p10": round(p10, 3), "p90": round(p90, 3)}), flush=True)
print(json.dumps({"batch": {"1": {"step_ms": round(med, 3), "noise_floor": round((p90 - p10) / med, 4)}},
                  "output": {"class": "exact"}}))
