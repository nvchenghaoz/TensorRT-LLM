"""Step 0 for the fast PNG writer: time the pieces of the current path and the
new one on real photos resized to 1024x1024, and verify exactness.

Run: {python} experiments/ascent/png_step0.py
"""

import base64
import glob
import io
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
from PIL import Image, features

t0 = time.time()
print("importing tensorrt_llm.media.encoding", flush=True)
from tensorrt_llm.media import encoding as enc  # noqa: E402

print(f"import took {time.time() - t0:.1f}s", flush=True)

info = {
    "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
    "pil_zlib": features.version("zlib"),
    "py_zlib": zlib.ZLIB_RUNTIME_VERSION,
    "affinity": len(os.sched_getaffinity(0)),
    "cpu_count": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
}
try:
    import PIL

    info["pillow"] = PIL.__version__
except Exception:
    pass
print(json.dumps(info), flush=True)

paths = sorted(glob.glob("docs/source/blogs/media/tech_blog28_visual_comparison_*.jpg"))[:4]
imgs = []
for p in paths:
    im = Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    imgs.append(np.asarray(im).copy())
print(f"loaded {len(imgs)} images", flush=True)


def pil_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def png_idat(png):
    pos = 8
    idat = b""
    while pos < len(png):
        n = int.from_bytes(png[pos : pos + 4], "big")
        tag = png[pos + 4 : pos + 8]
        if tag == b"IDAT":
            idat += png[pos + 8 : pos + 8 + n]
        pos += 12 + n
    return idat


def timeit(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"med": round(ts[len(ts) // 2], 3), "p10": round(ts[n // 10], 3), "p90": round(ts[(9 * n) // 10], 3)}


results = []
for idx, arr in enumerate(imgs):
    g = torch.from_numpy(arr).cuda()
    r = {"img": os.path.basename(paths[idx])}
    r["cpu_copy"] = timeit(lambda: g.cpu().numpy())
    ref = pil_png(arr)
    r["pil_png"] = timeit(lambda: pil_png(g.cpu().numpy()))
    r["b64"] = timeit(lambda: base64.b64encode(ref).decode("utf-8"))
    r["filter_gpu"] = timeit(lambda: enc._png_filter_scanlines(g).cpu())
    new = enc._encode_png_tensor(g)
    r["new_png"] = timeit(lambda: enc._encode_png_tensor(g))
    r["new_png_cpu_input"] = timeit(lambda: enc._encode_png_tensor(torch.from_numpy(arr)), n=10)
    # exactness
    ref_filtered = zlib.decompress(png_idat(ref))
    new_filtered = zlib.decompress(png_idat(new))
    rb = 1 + 1024 * 3
    ref_codes = bytes(ref_filtered[i * rb] for i in range(1024))
    new_codes = bytes(new_filtered[i * rb] for i in range(1024))
    r["filter_codes_equal"] = ref_codes == new_codes
    r["filtered_stream_equal"] = ref_filtered == new_filtered
    r["ref_code_hist"] = {k: ref_codes.count(bytes([k])) for k in range(5)}
    dec_ref = Image.open(io.BytesIO(ref))
    dec_new = Image.open(io.BytesIO(new))
    r["decoded_equal"] = (
        dec_ref.mode == dec_new.mode
        and dec_ref.size == dec_new.size
        and np.array_equal(np.asarray(dec_ref), np.asarray(dec_new))
    )
    r["ref_bytes"] = len(ref)
    r["new_bytes"] = len(new)
    r["size_ratio"] = round(len(new) / len(ref), 4)
    r["deterministic"] = all(enc._encode_png_tensor(g) == new for _ in range(3))
    print(json.dumps(r), flush=True)
    results.append(r)

ok = all(r["decoded_equal"] and r["filter_codes_equal"] for r in results)
print(json.dumps({"passed": ok, "info": info}))
sys.exit(0)
