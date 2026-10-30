"""Idea 63 test helpers: frozen reference lookup and a synthetic PNG encoder.

make_png writes an 8-bit RGB PNG with chosen per-row filter types (0-4 or a cycle), zlib
level, IDAT chunk size and extra chunks, so every filter and the multi-IDAT / level paths
of png_fast are exercised against PIL.
"""

import glob
import hashlib
import math
import os
import struct
import time
import zlib
from io import BytesIO

import numpy as np
import torch

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {  # experiments/ascent-benchmark/inputs.json
    "A": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",  # check00
    "B": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",  # check10
}


def log(*a):
    print(f"[idea63 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_refs():
    want = {v: k for k, v in REFS.items()}
    found = {}
    files = [f for f in sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True))
             if os.path.isfile(f)]
    named = [f for f in files if any(s in os.path.basename(f) for s in want)]
    for f in named + [f for f in files if f not in named]:
        if len(found) == len(want):
            break
        data = open(f, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        if h in want:
            found[want[h]] = data
    if len(found) != len(want):
        raise SystemExit(f"references not found in {STORE}: have {list(found)}")
    return found


def chunk(cid: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + cid + body
            + struct.pack(">I", zlib.crc32(cid + body) & 0xFFFFFFFF))


def _paeth(a, b, c):
    a, b, c = a.astype(np.int32), b.astype(np.int32), c.astype(np.int32)
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def filter_rows(px: np.ndarray, filters) -> bytes:
    """px [H, W, 3] uint8 -> filtered scanlines with filters[r] (int or list)."""
    h, w, _ = px.shape
    flat = px.reshape(h, w * 3).astype(np.int32)
    out = []
    for r in range(h):
        ft = filters[r % len(filters)]
        cur = flat[r]
        up = flat[r - 1] if r > 0 else np.zeros_like(cur)
        left = np.concatenate([np.zeros(3, np.int32), cur[:-3]])
        ul = np.concatenate([np.zeros(3, np.int32), up[:-3]])
        if ft == 0:
            f = cur
        elif ft == 1:
            f = cur - left
        elif ft == 2:
            f = cur - up
        elif ft == 3:
            f = cur - ((left + up) >> 1)
        else:
            f = cur - _paeth(left, up, ul)
        out.append(bytes([ft]) + (f & 255).astype(np.uint8).tobytes())
    return b"".join(out)


def make_png(px: np.ndarray, filters=(0, 1, 2, 3, 4), level=6, idat_size=None,
             pre_chunks=(), post_chunks=(), raw_override=None, color_type=2, depth=8,
             interlace=0) -> bytes:
    h, w = px.shape[:2]
    raw = filter_rows(px, list(filters)) if raw_override is None else raw_override
    z = zlib.compress(raw, level)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, interlace)
    parts = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", ihdr)]
    parts += [chunk(c, b) for c, b in pre_chunks]
    if idat_size is None:
        parts.append(chunk(b"IDAT", z))
    else:
        for i in range(0, len(z), idat_size):
            parts.append(chunk(b"IDAT", z[i:i + idat_size]))
    parts += [chunk(c, b) for c, b in post_chunks]
    parts.append(chunk(b"IEND", b""))
    return b"".join(parts)


def pil_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(data)) as im:
        im.load()
        return np.asarray(im.convert("RGB") if im.mode != "RGB" else im).copy()


def ref_load_image_pixels(source, *, device):
    """Main's _load_image_pixels (c73df91a), copied verbatim."""
    from PIL import ExifTags, Image, ImageOps

    device = torch.device(device)
    with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as original:
        original.load()
        image = original
        if original.getexif().get(ExifTags.Base.Orientation, 1) != 1:
            image = ImageOps.exif_transpose(original)
        if image.mode != "RGB":
            image = image.convert("RGB")
        scale = min(1.0, 4096 / math.sqrt(image.width * image.height))
        if scale < 1:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
        width, height = image.width // 16 * 16, image.height // 16 * 16
        if min(width, height) == 0:
            raise ValueError("FLUX.3 reference images must be at least 16 pixels on each axis")
        left, top = (image.width - width) // 2, (image.height - height) // 2
        box = (left, top, left + width, top + height)
        if box != (0, 0, image.width, image.height):
            image = image.crop(box)
        pixels = np.asarray(image)
    hwc = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=device.type == "cuda")
    np.copyto(hwc.numpy(), pixels)
    if device.type == "cuda":
        hwc = hwc.to(device, non_blocking=True)
    else:
        hwc = hwc.to(device)
    return hwc.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 127.5 - 1
