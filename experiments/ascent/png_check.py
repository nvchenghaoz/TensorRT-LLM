"""Untimed correctness for the fast PNG writer (tensorrt_llm/media/encoding.py).

Reference = main's path: Image.fromarray(t.cpu().numpy()).save(PNG, compress_level=1).
Checks, per case: decoded pixels/mode/size bitwise equal to the reference PNG's,
filtered scanline stream identical to Pillow's (same per-row filters), chunk CRCs
and zlib adler32 valid, repeated encodes byte-identical; plus the timed sequence
(5 warmup + 60 iterations) re-run untimed with every output compared to the
reference and to iteration 0, the CPU / disabled fallbacks and save_image.
"""

import base64
import glob
import io
import json
import os
import tempfile
import zlib

import numpy as np
import torch
from PIL import Image

print("importing tensorrt_llm.media.encoding", flush=True)
from tensorrt_llm.media import encoding as enc  # noqa: E402


def main_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def chunks(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, out = 8, []
    while pos < len(png):
        n = int.from_bytes(png[pos : pos + 4], "big")
        tag = png[pos + 4 : pos + 8]
        payload = png[pos + 8 : pos + 8 + n]
        crc = int.from_bytes(png[pos + 8 + n : pos + 12 + n], "big")
        assert zlib.crc32(tag + payload) == crc, f"bad CRC in {tag}"
        out.append((tag, payload))
        pos += 12 + n
    assert out[-1][0] == b"IEND"
    return out


def filtered(png):
    return zlib.decompress(b"".join(p for t, p in chunks(png) if t == b"IDAT"))


def decoded(png):
    im = Image.open(io.BytesIO(png))
    im.load()
    return im.mode, im.size, np.asarray(im)


def same_decode(a, b):
    ma, sa, pa = decoded(a)
    mb, sb, pb = decoded(b)
    return ma == mb and sa == sb and pa.dtype == pb.dtype and np.array_equal(pa, pb)


failures = []
results = {}


def check(name, arr, reps=3):
    t = torch.from_numpy(arr).cuda()
    ref = main_png(arr)
    new = enc.image_to_bytes(t, format="PNG")
    ok = {
        "decoded_equal": same_decode(new, ref),
        "filtered_equal": filtered(new) == filtered(ref),
        "deterministic": all(enc.image_to_bytes(t, format="PNG") == new for _ in range(reps)),
    }
    if arr.ndim == 3:  # (1, H, W, C) batched form; main only unbatches 4-D inputs
        ok["batched_input_equal"] = enc.image_to_bytes(t.unsqueeze(0), format="png") == new
    results[name] = dict(ok, ref_bytes=len(ref), new_bytes=len(new), ratio=round(len(new) / len(ref), 4))
    if not all(ok.values()):
        failures.append(name)
    print(name, json.dumps(results[name]), flush=True)
    return ref, new


rng = np.random.default_rng(0)
shapes = [(1, 1), (17, 33), (33, 17), (1, 2), (2, 1), (64, 64), (65, 130), (129, 7), (200, 301)]
for h, w in shapes:
    for c in (1, 3, 4):
        arr = rng.integers(0, 256, size=(h, w) if c == 1 else (h, w, c), dtype=np.uint8)
        check(f"rand_{h}x{w}x{c}", arr)
# structured content: ties and zero-cost rows
for c in (1, 3, 4):
    shp = (130, 77) if c == 1 else (130, 77, c)
    check(f"zeros_x{c}", np.zeros(shp, np.uint8))
    check(f"const_x{c}", np.full(shp, 200, np.uint8))
    hg = np.broadcast_to(np.arange(77, dtype=np.uint8)[None, :], (130, 77))
    vg = np.broadcast_to(np.arange(130, dtype=np.uint8)[:, None], (130, 77))
    if c > 1:
        hg = np.repeat(hg[..., None], c, axis=2)
        vg = np.repeat(vg[..., None], c, axis=2)
    check(f"hgrad_x{c}", np.ascontiguousarray(hg))
    check(f"vgrad_x{c}", np.ascontiguousarray(vg))
    smooth = (np.cumsum(rng.integers(-2, 3, size=shp), axis=1) % 256).astype(np.uint8)
    check(f"smooth_x{c}", smooth)

# real photos at the benchmark resolution (RGB) + RGBA/L variants
paths = sorted(glob.glob("docs/source/blogs/media/tech_blog28_visual_comparison_*.jpg"))
real = []
for p in paths:
    arr = np.asarray(Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)).copy()
    real.append(arr)
    check("real_" + os.path.basename(p), arr, reps=5)
check("real_L_odd", np.asarray(Image.fromarray(real[0]).convert("L"))[:1001, :777].copy())
rgba = np.concatenate([real[1], real[2][..., :1]], axis=2)[:999, :1023].copy()
check("real_RGBA_odd", rgba)

# untimed copy of the timed sequence (5 warmup + 60 iterations, same op)
arr = real[0]
g = torch.from_numpy(arr).cuda()
g_pristine = g.clone()
ref = main_png(arr)
outs = []
for i in range(65):
    outs.append(base64.b64encode(enc.image_to_bytes(g, format="PNG")).decode("utf-8"))
    assert torch.equal(g, g_pristine), "input mutated"
seq_ok = all(o == outs[0] for o in outs) and same_decode(base64.b64decode(outs[0]), ref)
results["timed_sequence_65"] = seq_ok
if not seq_ok:
    failures.append("timed_sequence_65")
print("timed_sequence_65", seq_ok, flush=True)

# fallbacks: CPU tensors and TRTLLM_FAST_PNG=0 keep Pillow's exact bytes
cpu_ok = enc.image_to_bytes(torch.from_numpy(arr), format="PNG") == ref
os.environ["TRTLLM_FAST_PNG"] = "0"
off_ok = enc.image_to_bytes(g, format="PNG") == ref
del os.environ["TRTLLM_FAST_PNG"]
jpeg_ok = enc.image_to_bytes(g, format="JPEG")[:2] == b"\xff\xd8"
with tempfile.TemporaryDirectory() as d:
    p = enc.save_image(g, os.path.join(d, "x.png"))
    with open(p, "rb") as f:
        save_ok = same_decode(f.read(), ref)
    p2 = enc.save_image(g, os.path.join(d, "y.jpg"))
    save_jpg_ok = open(p2, "rb").read()[:2] == b"\xff\xd8"
for k, v in {"cpu_fallback_bytes_equal": cpu_ok, "disabled_bytes_equal": off_ok, "jpeg_ok": jpeg_ok,
             "save_image_png_equal": save_ok, "save_image_jpg_ok": save_jpg_ok}.items():
    results[k] = v
    if not v:
        failures.append(k)
    print(k, v, flush=True)

real_ratios = [results["real_" + os.path.basename(p)]["ratio"] for p in paths]
print(json.dumps({"passed": not failures, "failures": failures, "cases": len(results),
                  "compared_against": "main path Pillow PNG compress_level=1 of the same array",
                  "real_1024_size_ratio": real_ratios}))
