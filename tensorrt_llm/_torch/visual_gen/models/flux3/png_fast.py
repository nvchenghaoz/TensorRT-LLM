# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast decode of 8-bit RGB non-interlaced PNG references straight to the GPU.

A PNG decode is an inflate of the concatenated IDAT payloads followed by a per-row
unfilter (None/Sub/Up/Avg/Paeth, PNG spec section 9). Here:

1. The host inflates the zlib stream with libdeflate (ctypes, written straight into
   pinned memory; ~2.4x faster than zlib at 2048^2) or, without libdeflate, with zlib.
   The stream must decompress to exactly H * (1 + 3W) bytes with no trailing input.
2. The filtered scanlines go to the device (non_blocking H2D) and three Triton kernels
   unfilter them:
   - skew: raw_skew[c, s, r] = raw[r, 1 + 3 (s - r) + c], so wavefront step s reads one
     contiguous run;
   - wavefront: one CTA per channel (the channels of an RGB8 PNG never mix: bpp = 3).
     At step s, lane r handles pixel x = s - r, which only depends on pixel x - 1 of the
     same row (its own value from step s - 1), pixel x of row r - 1 (lane r - 1's value
     from step s - 1, shifted in with tl.gather) and pixel x - 1 of row r - 1 (lane r's
     shifted value from step s - 1). W + H - 1 steps;
   - unskew: out[r, x, c] = out_skew[c, r + x, r].
   The arithmetic is the spec's integer arithmetic, so the pixels are bitwise PIL's.

decode_png_rgb8_to_device returns None (the caller uses PIL) for anything outside the
fast path's scope: no PNG signature, not bit depth 8 / colour type 2 / interlace 0,
sizes outside 16..4096, any chunk outside an allowlist of chunks that cannot change the
pixels PIL returns (no eXIf / text chunks, which PIL reads EXIF orientation from), an
allowlisted ancillary chunk whose body is not the spec's exact layout (sRGB 1, gAMA 4,
cHRM 32, pHYs 9, tIME 7 bytes; iCCP = 1-79 byte name, NUL, method 0, one complete zlib
stream of at most 1 MiB), so PIL raises or decodes those itself, bad
CRCs on non-IDAT chunks (PIL checks those; it does not check IDAT CRCs), non-consecutive
IDATs, a missing IEND, a zlib stream that is not exactly the scanline size, or a filter
byte > 4.
"""

import ctypes
import ctypes.util
import os
import struct
import zlib

import numpy as np
import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ALLOWED_CHUNKS = frozenset(
    (b"IHDR", b"IDAT", b"IEND", b"sRGB", b"gAMA", b"cHRM", b"pHYs", b"iCCP", b"tIME")
)
_MIN_SIDE = 16
_MAX_SIDE = 4096
# Wavefront lanes (rows) per CTA: 2048 for H <= 2048, else 4096. Both are compiled by
# warmup(), so no request compiles.
_LANE_BLOCKS = (2048, 4096)
# Swept at 2048^2 on RTX PRO 6000 SM120 (branch 59 job 720): wavefront 4/8/16/32 warps
# -> 3.95/2.68/2.32/2.10 ms for skew + wavefront + unskew.
CFG = {2048: {"num_warps": 32}, 4096: {"num_warps": 32}}
_COPY_BLOCK = 1024


def enabled() -> bool:
    """Kill switch, read per call: TRTLLM_FLUX3_PNG_FAST=0 keeps the PIL path."""
    return os.environ.get("TRTLLM_FLUX3_PNG_FAST", "1") != "0"


# ----------------------------------------------------------------------------- inflate
class _LibDeflate:
    def __init__(self):
        self.lib = None
        self.dec = None
        name = ctypes.util.find_library("deflate")
        for cand in ([name] if name else []) + ["libdeflate.so.0", "libdeflate.so"]:
            try:
                lib = ctypes.CDLL(cand)
            except OSError:
                continue
            if not hasattr(lib, "libdeflate_zlib_decompress_ex"):
                continue
            lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
            lib.libdeflate_alloc_decompressor.argtypes = []
            lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
            lib.libdeflate_zlib_decompress_ex.argtypes = [
                ctypes.c_void_p,  # decompressor
                ctypes.c_void_p,  # in
                ctypes.c_size_t,  # in_nbytes
                ctypes.c_void_p,  # out
                ctypes.c_size_t,  # out_nbytes_avail
                ctypes.POINTER(ctypes.c_size_t),  # actual_in_nbytes_ret
                ctypes.POINTER(ctypes.c_size_t),  # actual_out_nbytes_ret
            ]
            self.lib = lib
            break

    def decompress_into(self, src: bytes, out_ptr: int, out_nbytes: int) -> bool:
        """True when src is one zlib stream of exactly out_nbytes bytes, consuming all of src."""
        if self.dec is None:
            self.dec = self.lib.libdeflate_alloc_decompressor()
            if not self.dec:
                self.dec = None
                return False
        in_used = ctypes.c_size_t(0)
        out_used = ctypes.c_size_t(0)
        rc = self.lib.libdeflate_zlib_decompress_ex(
            self.dec,
            ctypes.cast(ctypes.c_char_p(src), ctypes.c_void_p),
            len(src),
            out_ptr,
            out_nbytes,
            ctypes.byref(in_used),
            ctypes.byref(out_used),
        )
        return rc == 0 and in_used.value == len(src) and out_used.value == out_nbytes


_LIBDEFLATE = None


def _libdeflate():
    global _LIBDEFLATE
    if _LIBDEFLATE is None:
        _LIBDEFLATE = _LibDeflate()
    return _LIBDEFLATE if _LIBDEFLATE.lib is not None else None


def _zlib_header_ok(z: bytes) -> bool:
    # What zlib's inflate itself requires of the 2-byte header (method 8, window <= 32K,
    # header check, no preset dictionary), checked up front so both inflaters agree.
    if len(z) < 2:
        return False
    cmf, flg = z[0], z[1]
    return (cmf & 0x0F) == 8 and (cmf >> 4) <= 7 and (cmf * 256 + flg) % 31 == 0 and not flg & 0x20


def inflate_scanlines_into(z: bytes, out: Tensor, *, use_libdeflate: bool = True) -> bool:
    """Inflate z into the uint8 CPU tensor out (exact size, nothing left over)."""
    if not _zlib_header_ok(z):
        return False
    lib = _libdeflate() if use_libdeflate else None
    if lib is not None:
        return lib.decompress_into(z, out.data_ptr(), out.numel())
    d = zlib.decompressobj()
    try:
        raw = d.decompress(z)
    except zlib.error:
        return False
    if not d.eof or d.unused_data or len(raw) != out.numel():
        return False
    np.copyto(out.numpy(), np.frombuffer(raw, dtype=np.uint8))
    return True


# ----------------------------------------------------------------------------- parse
def _ancillary_body_ok(cid: bytes, body) -> bool:
    """True only for ancillary bodies PIL's PngStream reads without raising.

    PIL raises on a short sRGB/gAMA/pHYs (ValueError / struct.error / IndexError), a
    cHRM whose length is not a multiple of 4 (struct.error), an iCCP without its
    separator and method byte (IndexError), a non-zero method (SyntaxError) or a profile
    over MAX_TEXT_CHUNK (1 MiB, ValueError). Only the spec's exact layouts are taken here
    (stricter than PIL); anything else returns False and PIL decides, so the outcome is
    PIL's exception or PIL's pixels.
    """
    n = len(body)
    if cid == b"sRGB":
        return n == 1
    if cid == b"gAMA":
        return n == 4
    if cid == b"cHRM":
        return n == 32
    if cid == b"pHYs":
        return n == 9
    if cid == b"tIME":
        return n == 7
    if cid == b"iCCP":
        raw = bytes(body)
        i = raw.find(b"\0")
        if not (1 <= i <= 79) or i + 2 > n or raw[i + 1] != 0:
            return False
        d = zlib.decompressobj()
        try:
            d.decompress(raw[i + 2 :], 1 << 20)
        except zlib.error:
            return False
        return d.eof and not d.unconsumed_tail and not d.unused_data
    return False


def parse_png_rgb8(data) -> tuple[int, int, list] | None:
    """(width, height, IDAT payload memoryviews) when the fast path applies, else None."""
    mv = memoryview(data)
    n = len(mv)
    if n < 8 + 25 + 12 or mv[:8] != PNG_SIGNATURE:
        return None
    pos = 8
    width = height = None
    idat = []
    idat_done = False
    seen_iend = False
    while pos + 12 <= n:
        length, cid = struct.unpack_from(">I4s", mv, pos)
        end = pos + 12 + length
        if end > n or cid not in _ALLOWED_CHUNKS:
            return None
        body = mv[pos + 8 : pos + 8 + length]
        if cid == b"IDAT":
            if width is None or idat_done:
                return None
            idat.append(body)
        else:
            if idat:
                idat_done = True
            (crc,) = struct.unpack_from(">I", mv, pos + 8 + length)
            if zlib.crc32(mv[pos + 4 : pos + 8 + length]) != crc:
                return None
            if cid == b"IHDR":
                if width is not None or pos != 8 or length != 13:
                    return None
                width, height, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", body)
                if (depth, ctype, comp, filt, interlace) != (8, 2, 0, 0, 0):
                    return None
                if not (_MIN_SIDE <= width <= _MAX_SIDE and _MIN_SIDE <= height <= _MAX_SIDE):
                    return None
            elif width is None:
                return None
            elif cid == b"IEND":
                seen_iend = True
                break
            elif not _ancillary_body_ok(cid, body):
                return None
        pos = end
    if not seen_iend or not idat:
        return None
    return width, height, idat


# ----------------------------------------------------------------------------- kernels
if triton is not None:

    @triton.jit(do_not_specialize=["H", "W", "n"])
    def _png_skew_kernel(raw_ptr, skew_ptr, H, W, n, BLOCK: tl.constexpr):
        # raw [H, 1 + 3W] -> skew [3, W + H - 1, H]: skew[c, r + x, r] = raw[r, 1 + 3x + c]
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < n
        row_bytes = 3 * W
        r = idx // row_bytes
        j = idx - r * row_bytes
        x = j // 3
        c = j - 3 * x
        v = tl.load(raw_ptr + r * (row_bytes + 1) + 1 + j, mask=m)
        S = W + H - 1
        tl.store(skew_ptr + (c * S + r + x) * H + r, v, mask=m)

    @triton.jit(do_not_specialize=["H", "W", "n"])
    def _png_unskew_kernel(skew_ptr, out_ptr, H, W, n, BLOCK: tl.constexpr):
        # out [H, W, 3]: out[r, x, c] = skew[c, r + x, r]
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = idx < n
        row_bytes = 3 * W
        r = idx // row_bytes
        j = idx - r * row_bytes
        x = j // 3
        c = j - 3 * x
        S = W + H - 1
        v = tl.load(skew_ptr + (c * S + r + x) * H + r, mask=m)
        tl.store(out_ptr + idx, v, mask=m)

    @triton.jit(do_not_specialize=["H", "W"])
    def _png_wavefront_kernel(raw_ptr, skew_in_ptr, skew_out_ptr, H, W, BLOCK_R: tl.constexpr):
        c = tl.program_id(0)
        r = tl.arange(0, BLOCK_R)
        rv = r < H
        S = W + H - 1
        ft = tl.load(raw_ptr + r.to(tl.int64) * (3 * W + 1), mask=rv, other=0).to(tl.int32)
        base_in = skew_in_ptr + c.to(tl.int64) * S * H
        base_out = skew_out_ptr + c.to(tl.int64) * S * H
        up_idx = tl.where(r > 0, r - 1, 0)
        v = tl.zeros([BLOCK_R], dtype=tl.int32)  # out[r, x - 1]: this lane's last value
        cprev = tl.zeros([BLOCK_R], dtype=tl.int32)  # out[r - 1, x - 1]
        # Two-step-ahead prefetch of the contiguous skewed input rows.
        n0 = tl.load(base_in + r, mask=rv, other=0)
        n1 = tl.load(base_in + H + r, mask=rv & (S > 1), other=0)
        for s in range(0, S):
            cur = n0.to(tl.int32)
            n0 = n1
            n1 = tl.load(base_in + (s + 2) * H + r, mask=rv & (s + 2 < S), other=0)
            b = tl.gather(v, up_idx, 0)
            b = tl.where(r > 0, b, 0)  # out[r - 1, x]
            a = v
            cc = cprev
            x = s - r
            active = rv & (x >= 0) & (x < W)
            pa = tl.abs(b - cc)
            pb = tl.abs(a - cc)
            pc = tl.abs(a + b - 2 * cc)
            paeth = tl.where((pa <= pb) & (pa <= pc), a, tl.where(pb <= pc, b, cc))
            pred = tl.where(
                ft == 4,
                paeth,
                tl.where(ft == 3, (a + b) >> 1, tl.where(ft == 2, b, tl.where(ft == 1, a, 0))),
            )
            val = (cur + pred) & 255
            tl.store(base_out + s * H + r, val.to(tl.uint8), mask=active)
            v = tl.where(active, val, v)
            cprev = b


def unfilter_rgb8_on_device(raw_dev: Tensor, width: int, height: int) -> Tensor:
    """Unfilter H * (1 + 3W) filtered bytes on the device into [H, W, 3] uint8."""
    dev = raw_dev.device
    steps = width + height - 1
    skew_in = torch.empty((3, steps, height), dtype=torch.uint8, device=dev)
    skew_out = torch.empty((3, steps, height), dtype=torch.uint8, device=dev)
    out = torch.empty((height, width, 3), dtype=torch.uint8, device=dev)
    n = height * width * 3
    grid = (triton.cdiv(n, _COPY_BLOCK),)
    _png_skew_kernel[grid](raw_dev, skew_in, height, width, n, BLOCK=_COPY_BLOCK, num_warps=4)
    lanes = _LANE_BLOCKS[0] if height <= _LANE_BLOCKS[0] else _LANE_BLOCKS[1]
    _png_wavefront_kernel[(3,)](
        raw_dev, skew_in, skew_out, height, width, BLOCK_R=lanes, **CFG[lanes]
    )
    _png_unskew_kernel[grid](skew_out, out, height, width, n, BLOCK=_COPY_BLOCK, num_warps=4)
    return out


def decode_png_rgb8_to_device(data, device) -> Tensor | None:
    """[H, W, 3] uint8 pixels on `device` (bitwise PIL's), or None to use PIL."""
    device = torch.device(device)
    if triton is None or device.type != "cuda":
        return None
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return None
    parsed = parse_png_rgb8(data)
    if parsed is None:
        return None
    width, height, idat = parsed
    stride = 1 + 3 * width
    z = idat[0].tobytes() if len(idat) == 1 else b"".join(idat)
    staged = torch.empty(height * stride, dtype=torch.uint8, pin_memory=True)
    if not inflate_scanlines_into(z, staged):
        return None
    if int(staged.numpy()[::stride].max()) > 4:
        return None
    raw_dev = staged.to(device, non_blocking=True)
    with torch.cuda.device(device):
        return unfilter_rgb8_on_device(raw_dev, width, height)


def warmup(device) -> None:
    """Compile the kernels outside any timed request: one compile per lane block for all
    sizes (shape arguments are not specialized)."""
    device = torch.device(device)
    if triton is None or device.type != "cuda":
        return
    rng = np.random.default_rng(0)
    for h in _LANE_BLOCKS:
        w = 16
        rows = [
            bytes([r % 5]) + rng.integers(0, 256, 3 * w, dtype=np.uint8).tobytes()
            for r in range(h)
        ]
        raw = torch.frombuffer(bytearray(b"".join(rows)), dtype=torch.uint8).to(device)
        with torch.cuda.device(device):
            unfilter_rgb8_on_device(raw, w, h)
    _libdeflate()
    torch.cuda.synchronize(device)
