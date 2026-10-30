"""Idea 74 H(a): valid-CRC malformed-ancillary PNG corpus vs PIL.

Each case is an 80x64 RGB8 PNG with one allowlisted ancillary chunk (sRGB, gAMA, cHRM,
pHYs, tIME, iCCP) carrying a well-formed or malformed body, with a correct CRC, placed
before or after the IDAT. For every case:
  - tree pipeline_flux3._load_image_pixels (cuda) vs main's (copied verbatim in
    idea63/pngtest.ref_load_image_pixels = pure PIL): same pixels+strides bitwise, or the
    same exception (type, message);
  - TRTLLM_FLUX3_PNG_FAST=0 arm equal to that too;
  - fast path taken exactly for the spec-exact bodies (expect_fast).
"""

import json
import os
import re
import struct
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "idea63"))
from pngtest import find_refs, make_png, pil_rgb, ref_load_image_pixels  # noqa: E402


def log(*a):
    print(f"[idea74-ha {time.strftime('%H:%M:%S')}]", *a, flush=True)


def corpus():
    icc = zlib.compress(bytes(range(256)) * 12, 9)
    big = zlib.compress(bytes(2 << 20), 9)
    cases = {
        # (cid, body, expect_fast)
        "sRGB_ok": (b"sRGB", b"\x00", True),
        "sRGB_ok_val7": (b"sRGB", b"\x07", True),
        "sRGB_empty": (b"sRGB", b"", False),
        "sRGB_len2": (b"sRGB", b"\x00\x00", False),
        "gAMA_ok": (b"gAMA", struct.pack(">I", 45455), True),
        "gAMA_empty": (b"gAMA", b"", False),
        "gAMA_len3": (b"gAMA", b"\x00\xb1\x8f", False),
        "gAMA_len5": (b"gAMA", struct.pack(">I", 45455) + b"\x00", False),
        "gAMA_zero": (b"gAMA", bytes(4), True),
        "cHRM_ok": (b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000,
                                         15000, 6000), True),
        "cHRM_empty": (b"cHRM", b"", False),
        "cHRM_len31": (b"cHRM", bytes(31), False),
        "cHRM_len33": (b"cHRM", bytes(33), False),
        "cHRM_len36": (b"cHRM", bytes(36), False),
        "cHRM_len28": (b"cHRM", bytes(28), False),
        "pHYs_ok_m": (b"pHYs", struct.pack(">IIB", 2835, 2835, 1), True),
        "pHYs_ok_unit7": (b"pHYs", struct.pack(">IIB", 1, 2, 7), True),
        "pHYs_empty": (b"pHYs", b"", False),
        "pHYs_len8": (b"pHYs", bytes(8), False),
        "pHYs_len10": (b"pHYs", bytes(10), False),
        "tIME_ok": (b"tIME", struct.pack(">HBBBBB", 2026, 10, 30, 4, 30, 0), True),
        "tIME_empty": (b"tIME", b"", False),
        "tIME_len6": (b"tIME", bytes(6), False),
        "tIME_len8": (b"tIME", bytes(8), False),
        "iCCP_ok": (b"iCCP", b"ICC profile\x00\x00" + icc, True),
        "iCCP_empty": (b"iCCP", b"", False),
        "iCCP_no_nul": (b"iCCP", b"ICC profile", False),
        "iCCP_no_nul_leading0": (b"iCCP", b"\x00" , False),
        "iCCP_nul_at_end": (b"iCCP", b"ICC profile\x00", False),
        "iCCP_empty_name": (b"iCCP", b"\x00\x00" + icc, False),
        "iCCP_name80": (b"iCCP", b"n" * 80 + b"\x00\x00" + icc, False),
        "iCCP_method1": (b"iCCP", b"ICC\x00\x01" + icc, False),
        "iCCP_method255": (b"iCCP", b"ICC\x00\xff" + icc, False),
        "iCCP_bad_zlib": (b"iCCP", b"ICC\x00\x00garbage-not-zlib", False),
        "iCCP_truncated_zlib": (b"iCCP", b"ICC\x00\x00" + icc[:-7], False),
        "iCCP_trailing": (b"iCCP", b"ICC\x00\x00" + icc + b"tail", False),
        "iCCP_no_profile": (b"iCCP", b"ICC\x00\x00", False),
        "iCCP_over_1MiB": (b"iCCP", b"ICC\x00\x00" + big, False),
        "iCCP_bad_adler": (b"iCCP", b"ICC\x00\x00" + icc[:-1] + bytes([icc[-1] ^ 0x55]), False),
    }
    return cases


def main():
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3
    from tensorrt_llm._torch.visual_gen.models.flux3 import png_fast

    dev = torch.device("cuda")
    refs = find_refs()
    px = np.ascontiguousarray(pil_rgb(refs["A"])[:64, :80])

    def outcome(fn, data):
        try:
            t = fn(data, device=dev)
            torch.cuda.synchronize()
            return ("px", t.shape, t.stride(), t.cpu())
        except Exception as exc:  # noqa: BLE001
            # UnidentifiedImageError names the BytesIO object's address; drop addresses.
            return ("exc", type(exc).__name__, re.sub(r"0x[0-9a-f]+", "0x?", str(exc)))

    def same(x, y):
        if x[0] != y[0]:
            return False
        if x[0] == "exc":
            return x == y
        return x[1] == y[1] and x[2] == y[2] and torch.equal(x[3], y[3])

    ok = True
    rows = {}
    n_pil_exc = 0
    for name, (cid, body, expect_fast) in corpus().items():
        for where in ("pre", "post"):
            kw = {"pre_chunks": [(cid, body)]} if where == "pre" else {"post_chunks": [(cid, body)]}
            data = make_png(px, **kw)
            fast = png_fast.decode_png_rgb8_to_device(data, dev) is not None
            ref = outcome(ref_load_image_pixels, data)
            cand = outcome(flux3._load_image_pixels, data)
            cand2 = outcome(flux3._load_image_pixels, data)
            os.environ["TRTLLM_FLUX3_PNG_FAST"] = "0"
            try:
                kill = outcome(flux3._load_image_pixels, data)
            finally:
                os.environ.pop("TRTLLM_FLUX3_PNG_FAST")
            good = same(ref, cand) and same(cand, cand2) and same(ref, kill) and fast == expect_fast
            n_pil_exc += ref[0] == "exc"
            rows[f"{name}/{where}"] = {"ok": good, "fast": fast,
                                       "pil": ref[0] if ref[0] == "px" else f"{ref[1]}: {ref[2]}"}
            if not good:
                log("FAIL", name, where, rows[f"{name}/{where}"], cand[:3] if cand[0] == "exc" else "px")
            ok &= good
    for k in "AB":
        f = png_fast.decode_png_rgb8_to_device(refs[k], dev) is not None
        rows[f"ref_{k}_fast"] = {"ok": f}
        ok &= f
    for k, v in rows.items():
        log(k, v)
    print(json.dumps({"passed": bool(ok), "cases": len(rows), "failed": [k for k, v in rows.items()
                      if not v["ok"]], "pil_exception_cases": n_pil_exc,
                      "compared": "tree _load_image_pixels vs main's (PIL, copied verbatim) and vs "
                                  "TRTLLM_FLUX3_PNG_FAST=0: pixels+strides bitwise or same "
                                  "exception (type, message); fast path exactly for spec-exact "
                                  "bodies; frozen refs A/B on the fast path", "rows": rows}))


if __name__ == "__main__":
    main()
