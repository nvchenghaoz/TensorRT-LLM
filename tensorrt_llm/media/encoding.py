# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor → image/video encoding free functions."""

import os
import shutil
import struct
import subprocess  # nosec B404
import tempfile
import threading
import wave
import zlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import heapq

import numpy as np
import torch
from PIL import Image

from tensorrt_llm.logger import logger

# Video encoder availability flags (cached after first check).
_FFMPEG_PATH: Optional[str] = None
_VIDEO_ENCODER: Optional["_VideoEncoder"] = None


def _check_ffmpeg_available() -> bool:
    """Return True if ffmpeg CLI is installed; cache its path.

    Only a successful probe is cached. A negative result is re-probed on the
    next call: ffmpeg may be installed after import (e.g. a test fixture or
    deployment step running ``apt-get install ffmpeg``), and permanently
    caching the miss silently downgrades every later video encode in the
    process to the fallback encoder.
    """
    global _FFMPEG_PATH
    if not _FFMPEG_PATH:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None:
            try:
                result = subprocess.run(
                    [ffmpeg_path, "-version"],
                    capture_output=True,
                    text=True,
                )
                _FFMPEG_PATH = ffmpeg_path if result.returncode == 0 else ""
            except (FileNotFoundError, OSError):
                _FFMPEG_PATH = ""
        else:
            _FFMPEG_PATH = ""
    return bool(_FFMPEG_PATH)


def _get_ffmpeg_path() -> str:
    """Return cached ffmpeg path (after :func:`_check_ffmpeg_available`)."""
    if _FFMPEG_PATH is None:
        _check_ffmpeg_available()
    return _FFMPEG_PATH or ""


# ---------------------------------------------------------------------------
# Encoder backends (private)
# ---------------------------------------------------------------------------


class _VideoEncoder(ABC):
    """Internal abstract base for video encoders."""

    @abstractmethod
    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str: ...

    @staticmethod
    def _validate_video_tensor(video: torch.Tensor) -> None:
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for video, got {type(video)}")

    @staticmethod
    def _validate_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to ``(samples, channels)`` int16."""
        if not isinstance(audio, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for audio, got {type(audio)}")

        audio_tensor = audio
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor[:, None]
        elif audio_tensor.ndim == 2:
            if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
                audio_tensor = audio_tensor.T
            if audio_tensor.shape[1] > 2:
                audio_tensor = audio_tensor[:, :2]
        elif audio_tensor.ndim == 3:
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            else:
                audio_tensor = audio_tensor[0]
            if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
                audio_tensor = audio_tensor.T
            if audio_tensor.shape[1] > 2:
                audio_tensor = audio_tensor[:, :2]
        else:
            raise ValueError(
                f"Unsupported audio tensor shape: {audio_tensor.shape}. "
                "Expected 1D, 2D, or 3D tensor."
            )

        if audio_tensor.shape[1] > 2:
            audio_tensor = audio_tensor[:, :2]

        if audio_tensor.dtype != torch.int16:
            audio_tensor = torch.clip(audio_tensor, -1.0, 1.0)
            audio_tensor = (audio_tensor * 32767.0).to(torch.int16)
        return audio_tensor


class _FfmpegCliEncoder(_VideoEncoder):
    """Pipe raw RGB frames to ffmpeg for H.264/MP4 encoding (with optional audio)."""

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str:
        self._validate_video_tensor(video)

        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_input_args: list = []
            audio_output_args: list = []
            if audio is not None:
                audio_tensor = self._validate_audio_tensor(audio)
                audio_np = audio_tensor.cpu().numpy()
                tmp_audio_path = os.path.join(tmp_dir, "audio.wav")
                self._write_wav(tmp_audio_path, audio_np, sample_rate=audio_sample_rate)
                audio_input_args = ["-i", tmp_audio_path]
                audio_output_args = ["-c:a", "aac", "-shortest"]

            ffmpeg_path = _get_ffmpeg_path()
            cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(frame_rate),
                "-i",
                "-",
                *audio_input_args,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                *audio_output_args,
                output_path,
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                raw_frames = video_np.tobytes()
                _, stderr = process.communicate(input=raw_frames)
                if process.returncode != 0:
                    raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode()}")
            except FileNotFoundError:
                raise RuntimeError("ffmpeg not found. Install ffmpeg for video encoding.")

        logger.info(f"Saved video{' with audio' if audio is not None else ''} to {output_path}")
        return output_path

    def _write_wav(self, path: str, audio: Any, sample_rate: int) -> None:
        import numpy as np

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio, audio])

        audio_interleaved = audio.flatten().astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_interleaved.tobytes())


class _PurePythonEncoder(_VideoEncoder):
    """Fallback MJPEG/AVI encoder (video only, no audio); requires only PIL."""

    def encode_video(
        self,
        video: torch.Tensor,
        output_path: str,
        frame_rate: float,
        audio: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 24000,
    ) -> str:
        if audio is not None:
            logger.warning(
                "Pure-Python video encoder does not support audio; audio will be ignored."
            )
        self._validate_video_tensor(video)

        video_np = video.cpu().numpy()
        num_frames, height, width, channels = video_np.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video, got {channels} channels")

        jpeg_frames: List[bytes] = []
        for frame in video_np:
            pil_image = Image.fromarray(frame)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            jpeg_frames.append(buffer.getvalue())

        self._write_mjpeg_avi(output_path, jpeg_frames, width, height, frame_rate)
        logger.info(f"Saved video to {output_path}")
        return output_path

    def _write_mjpeg_avi(
        self,
        output_path: str,
        jpeg_frames: List[bytes],
        width: int,
        height: int,
        frame_rate: float,
    ) -> None:
        num_frames = len(jpeg_frames)
        usec_per_frame = int(1000000 / frame_rate)

        movi_data = b""
        frame_index: List[Tuple[int, int]] = []
        for jpeg_data in jpeg_frames:
            original_size = len(jpeg_data)
            frame_index.append((len(movi_data) + 4, original_size))
            chunk = b"00dc" + struct.pack("<I", original_size) + jpeg_data
            if original_size % 2:
                chunk += b"\x00"
            movi_data += chunk

        idx1_data = b""
        for offset, size in frame_index:
            idx1_data += b"00dc"
            idx1_data += struct.pack("<I", 0x10)
            idx1_data += struct.pack("<I", offset)
            idx1_data += struct.pack("<I", size)

        movi_list_size = 4 + len(movi_data)
        idx1_size = len(idx1_data)

        avih = struct.pack(
            "<IIIIIIIIIIIIII",
            usec_per_frame,
            0,
            0,
            0x10,
            num_frames,
            0,
            1,
            max(len(f) for f in jpeg_frames),
            width,
            height,
            0,
            0,
            0,
            0,
        )
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIHHHH",
            b"vids",
            b"MJPG",
            0,
            0,
            0,
            0,
            1,
            int(frame_rate),
            0,
            num_frames,
            max(len(f) for f in jpeg_frames),
            0,
            0,
            0,
            0,
            width,
            height,
        )
        strf = struct.pack(
            "<IiiHHIIiiII",
            40,
            width,
            height,
            1,
            24,
            0x47504A4D,
            width * height * 3,
            0,
            0,
            0,
            0,
        )

        strh_chunk = b"strh" + struct.pack("<I", len(strh)) + strh
        strf_chunk = b"strf" + struct.pack("<I", len(strf)) + strf
        strl_data = strh_chunk + strf_chunk
        strl_list = b"LIST" + struct.pack("<I", 4 + len(strl_data)) + b"strl" + strl_data

        avih_chunk = b"avih" + struct.pack("<I", len(avih)) + avih
        hdrl_data = avih_chunk + strl_list
        hdrl_list = b"LIST" + struct.pack("<I", 4 + len(hdrl_data)) + b"hdrl" + hdrl_data

        movi_list = b"LIST" + struct.pack("<I", movi_list_size) + b"movi" + movi_data
        idx1_chunk = b"idx1" + struct.pack("<I", idx1_size) + idx1_data
        riff_data = hdrl_list + movi_list + idx1_chunk
        riff_size = 4 + len(riff_data)

        with open(output_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", riff_size))
            f.write(b"AVI ")
            f.write(riff_data)


def _get_video_encoder() -> _VideoEncoder:
    """Return the best available video encoder (cached singleton).

    The pure-Python fallback is not cached permanently: if ffmpeg becomes
    available later in the process lifetime, upgrade to the ffmpeg encoder.
    """
    global _VIDEO_ENCODER
    if _VIDEO_ENCODER is None or (
        isinstance(_VIDEO_ENCODER, _PurePythonEncoder) and _check_ffmpeg_available()
    ):
        _VIDEO_ENCODER = _FfmpegCliEncoder() if _check_ffmpeg_available() else _PurePythonEncoder()
        logger.info(f"Using {_VIDEO_ENCODER.__class__.__name__} for video encoding")
    return _VIDEO_ENCODER


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def resolve_video_format(output_format) -> Tuple[str, str]:
    """Resolve a requested format to a concrete ``(format, extension)``.

    Args:
        output_format: Either a logical format string (``'mp4'``/``'avi'``/
            ``'auto'``) or a :class:`pathlib.Path` whose suffix indicates the
            desired container.

    Returns:
        Tuple ``(format, extension)`` such as ``('mp4', '.mp4')``.

    Raises:
        RuntimeError: ``'mp4'`` resolved but ffmpeg is not available.
        ValueError: Unsupported value.
    """
    if isinstance(output_format, Path):
        suffix = output_format.suffix.lower().lstrip(".")
        if not suffix:
            raise ValueError(
                f"Cannot resolve video format from path without suffix: {output_format}"
            )
        return resolve_video_format(suffix)

    if output_format == "mp4":
        if _check_ffmpeg_available():
            return "mp4", ".mp4"
        raise RuntimeError(
            "MP4 (H.264) format requires ffmpeg to be installed. "
            "Install ffmpeg (e.g., 'apt-get install ffmpeg' on Ubuntu/Debian) "
            "or use output_format='avi' for MJPEG encoding (no ffmpeg required). "
            "See https://ffmpeg.org/download.html for installation instructions."
        )
    elif output_format == "avi":
        return "avi", ".avi"
    elif output_format == "auto":
        if _check_ffmpeg_available():
            return "mp4", ".mp4"
        return "avi", ".avi"
    else:
        raise ValueError(
            f"Unsupported video format: {output_format}. Please use 'auto' if you "
            "want to automatically choose the best format based on availability."
        )


# ---------------------------------------------------------------------------
# Pixel/PIL helpers
# ---------------------------------------------------------------------------


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    """Convert a uint8 image tensor (``(H, W, C)`` or ``(B, H, W, C)``) to PIL."""
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(image)}")
    if image.dim() == 4:
        image = image[0]
    image_np = image.cpu().numpy()
    return Image.fromarray(image_np)


# ---------------------------------------------------------------------------
# Fast exact PNG writer (uint8 L/RGB/RGBA CUDA tensors)
# ---------------------------------------------------------------------------
#
# Produces a PNG whose decoded pixels are bitwise identical to Pillow's PNG
# output for the same array (and whose filtered scanline stream is identical
# to Pillow's). Per-row filters are chosen with Pillow's own heuristic
# (ZipEncode.c: None, Up, Sub, Paeth in that order, keep a filter only when
# its sum of min(v, 256 - v) is strictly smaller), computed with tensor ops on
# the GPU, so only the filtered scanlines cross to the host. Set
# TRTLLM_FAST_PNG=0 to use Pillow instead. The filtered stream is then
# deflated in fixed-size row strips on a thread pool (zlib releases the GIL);
# each strip is primed with the previous 32 KiB as a preset dictionary and
# ends with a sync flush, the last with Z_FINISH, so the strips concatenate
# into one valid zlib stream. Strip boundaries depend only on the image
# height, so output bytes are deterministic regardless of the core count.

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# channels -> PNG color type (L, RGB, RGBA)
_PNG_COLOR_TYPES = {1: 0, 3: 2, 4: 6}
_PNG_STRIP_ROWS = 64
_PNG_WINDOW = 32768
# Deflate configuration of each strip (any config decodes to the same pixels).
# Huffman-only on the filtered scanlines is both faster and smaller than
# level-1 Z_FILTERED with a preset dictionary on generated 1024^2 images
# (5.6 vs 7.5 ms encode+b64, 1.715 vs 1.820 MB on the 12 workload prompts).
_PNG_STRATEGY = zlib.Z_HUFFMAN_ONLY
_PNG_MEM_LEVEL = 9
# Bytes of the previous strip given as preset dictionary (0: none; Huffman-only
# emits no back-references, so a dictionary would only cost a copy).
_PNG_DICT_BYTES = 0
_PNG_MAX_WORKERS = 16
_PNG_POOLS = {}  # max workers -> ThreadPoolExecutor
_PNG_POOL_LOCK = threading.Lock()


def _fast_png_enabled() -> bool:
    return os.environ.get("TRTLLM_FAST_PNG", "1") != "0"


def _png_pool():
    key = _PNG_MAX_WORKERS
    pool = _PNG_POOLS.get(key)
    if pool is None:
        with _PNG_POOL_LOCK:
            pool = _PNG_POOLS.get(key)
            if pool is None:
                try:
                    ncpu = len(os.sched_getaffinity(0))
                except (AttributeError, OSError):
                    ncpu = os.cpu_count() or 1
                workers = max(1, min(key, ncpu))
                pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="png_enc")
                _PNG_POOLS[key] = pool
    return pool


def _png_filter_scanlines(img: torch.Tensor) -> torch.Tensor:
    """Filter a uint8 ``(H, W, C)`` image into PNG scanlines ``(H, 1 + W*C)``.

    Each row gets the filter Pillow's encoder would pick for it.
    """
    h, w, c = img.shape
    x = img.reshape(h, w * c).to(torch.int16)
    up = torch.zeros_like(x)
    up[1:] = x[:-1]
    left = torch.zeros_like(x)
    left[:, c:] = x[:, :-c]
    upleft = torch.zeros_like(x)
    upleft[1:, c:] = x[:-1, :-c]

    pa = (up - upleft).abs()
    pb = (left - upleft).abs()
    pc = (left + up - 2 * upleft).abs()
    paeth_pred = torch.where((pa <= pb) & (pa <= pc), left, torch.where(pb <= pc, up, upleft))
    del pa, pb, pc

    # Candidate order matches Pillow's evaluation order: None, Up, Sub, Paeth.
    cand = torch.stack((x, x - up, x - left, x - paeth_pred)).bitwise_and_(0xFF)
    del up, left, upleft, paeth_pred
    cost = torch.minimum(cand, 256 - cand).sum(dim=2, dtype=torch.int32)  # (4, H)

    # Sequential strict-less selection == Pillow's tie-breaking.
    best_cost = cost[0]
    choice = torch.zeros(h, dtype=torch.int64, device=x.device)
    for k in (1, 2, 3):
        better = cost[k] < best_cost
        choice = torch.where(better, torch.full_like(choice, k), choice)
        best_cost = torch.where(better, cost[k], best_cost)

    rows = cand.gather(0, choice.view(1, h, 1).expand(1, h, w * c))[0]
    codes = torch.tensor([0, 2, 1, 4], dtype=torch.int16, device=x.device)[choice]
    out = torch.empty((h, 1 + w * c), dtype=torch.uint8, device=x.device)
    out[:, 0] = codes.to(torch.uint8)
    out[:, 1:] = rows.to(torch.uint8)
    return out


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(payload, zlib.crc32(tag)))
    )


# ---------------------------------------------------------------------------
# GPU Huffman-only deflate of the filtered scanlines
# ---------------------------------------------------------------------------
#
# Exact: any valid deflate stream inflates to the same filtered scanlines, so
# decoded pixels are unchanged. The filtered scanlines stay on the GPU. Rows
# are split into fixed blocks (a function of the height only); one bincount
# gives every block's byte histogram, which (with the Adler-32 partial sums)
# is all that crosses to the host before the stream itself. The host builds a
# dynamic Huffman code per block (BTYPE=2, literals + EOB only, like zlib's
# Z_HUFFMAN_ONLY; two 1-bit distance codes as zlib writes them), and the GPU
# writes every symbol's bit-reversed code at its exclusive-cumsum bit offset
# with integer scatter-adds (bits never overlap, so add == OR and the result
# is deterministic). Two host syncs per image: the histogram and the stream.
# Set TRTLLM_PNG_GPU_DEFLATE=0 to use the zlib strip writer instead.

# Rows per deflate block. One Huffman table per 1024 rows: each extra block
# costs ~0.2 ms of host Huffman building, more than its size saving buys
# (1024^2 FLUX images: 1.772 MB at 1024 rows, 1.732 at 256, 1.715 at 64).
_PNG_GPU_BLOCK_ROWS = 1024
_PNG_GPU_MAX_BLOCK_BYTES = 1 << 24  # keeps Huffman weights small
_PNG_GPU_MAX_BYTES = 1 << 27  # keeps the int64 Adler-32 sums and int bit offsets exact
_PNG_IDAT_PIECES = 4  # IDAT chunks whose CRCs run in parallel
_DEFLATE_CL_ORDER = np.array(
    (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15), dtype=np.int64
)
_DEFLATE_REV16 = None


def _gpu_deflate_enabled() -> bool:
    return os.environ.get("TRTLLM_PNG_GPU_DEFLATE", "1") != "0"


def _rev16() -> np.ndarray:
    global _DEFLATE_REV16
    if _DEFLATE_REV16 is None:
        x = np.arange(1 << 16, dtype=np.int64)
        r = np.zeros_like(x)
        for i in range(16):
            r |= ((x >> i) & 1) << (15 - i)
        _DEFLATE_REV16 = r
    return _DEFLATE_REV16


def _huffman_lengths(counts: List[int], limit: int) -> List[int]:
    """Code lengths (<= limit) of a complete Huffman code for ``counts``.

    Zero counts get length 0. Fewer than two used symbols are padded with
    count-1 dummies (as zlib's build_tree does), so the code is complete.
    Ties break on (weight, node id), so the result is deterministic. A tree
    deeper than ``limit`` is flattened on its per-length counts (JPEG Annex
    K.3 adjust_bits, which keeps the code complete) and the lengths are
    re-dealt shortest-first in (count desc, symbol) order.
    """
    n = len(counts)
    c = list(counts)
    used = sum(1 for x in c if x)
    for s in range(n):
        if used >= 2:
            break
        if not c[s]:
            c[s] = 1
            used += 1
    pop, push = heapq.heappop, heapq.heappush
    heap = [(w, s) for s, w in enumerate(c) if w]
    heapq.heapify(heap)
    parent = [0] * (2 * n)
    nxt = n
    while len(heap) > 1:
        w1, a = pop(heap)
        w2, b = pop(heap)
        parent[a] = nxt
        parent[b] = nxt
        push(heap, (w1 + w2, nxt))
        nxt += 1
    depth = [0] * (2 * n)
    for node in range(nxt - 2, n - 1, -1):
        depth[node] = depth[parent[node]] + 1
    lengths = [depth[parent[s]] + 1 if c[s] else 0 for s in range(n)]
    mx = max(lengths)
    if mx <= limit:
        return lengths
    bl = [0] * (mx + 1)
    for x in lengths:
        if x:
            bl[x] += 1
    for i in range(mx, limit, -1):
        while bl[i] > 0:
            j = i - 2
            while j > 0 and bl[j] == 0:
                j -= 1
            if j <= 0:
                raise RuntimeError("Huffman length limit failed")
            bl[i] -= 2
            bl[i - 1] += 1
            bl[j + 1] += 2
            bl[j] -= 1
    if any(bl[limit + 1 :]) or sum(bl[x] << (limit - x) for x in range(1, limit + 1)) != (
        1 << limit
    ):
        raise RuntimeError("Huffman length limit produced an incomplete code")
    syms = sorted((s for s in range(n) if c[s]), key=lambda s: (-c[s], s))
    out = [0] * n
    k = 0
    for length in range(1, limit + 1):
        for _ in range(bl[length]):
            out[syms[k]] = length
            k += 1
    return out


def _canonical_rev_codes(lengths: np.ndarray, max_bits: int) -> np.ndarray:
    """RFC 1951 3.2.2 canonical codes, bit-reversed for LSB-first packing."""
    full = np.bincount(lengths, minlength=max_bits + 1)
    fl = full.tolist()
    next_code = [0] * (max_bits + 1)
    code = 0
    for bits in range(1, max_bits + 1):
        code = (code + (fl[bits - 1] if bits > 1 else 0)) << 1
        next_code[bits] = code
    order = np.argsort(lengths, kind="stable")
    sl = lengths[order]
    first = np.cumsum(full) - full
    codes = np.empty(len(lengths), dtype=np.int64)
    codes[order] = np.asarray(next_code, dtype=np.int64)[sl] + (
        np.arange(len(lengths), dtype=np.int64) - first[sl]
    )
    return _rev16()[codes & 0xFFFF] >> (16 - lengths)


class _GpuPngCache(threading.local):
    def __init__(self):
        self.shape_key = None
        self.shape_val = None
        self.pins = {}


_GPU_PNG_CACHE = _GpuPngCache()


def _png_pinned(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Thread-local pinned staging buffer (grown, never shrunk)."""
    buf = _GPU_PNG_CACHE.pins.get(name)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(max(1, numel + (numel >> 2)), dtype=dtype, pin_memory=True)
        _GPU_PNG_CACHE.pins[name] = buf
    return buf[:numel]


def _png_gpu_deflate(scan: torch.Tensor):
    """Huffman-only raw deflate of ``scan`` (H, rowbytes) uint8 CUDA.

    Returns (stream bytes-like, adler32). Every GPU read of ``scan`` is
    finished when this returns.
    """
    h, rb = scan.shape
    n = h * rb
    dev = scan.device
    br = max(1, min(_PNG_GPU_BLOCK_ROWS, _PNG_GPU_MAX_BLOCK_BYTES // rb))
    nb = (h + br - 1) // br
    key = (dev, h, rb, br)
    cache = _GPU_PNG_CACHE
    if cache.shape_key != key:
        rowblk = torch.arange(h, device=dev, dtype=torch.int64) // br
        cache.shape_val = (
            rowblk,
            (rowblk * 257).to(torch.int32).view(h, 1),
            torch.arange(h, device=dev, dtype=torch.int64),
            torch.arange(rb, device=dev, dtype=torch.int64),
        )
        cache.shape_key = key
    rowblk, rowoff, ar_h, ar_rb = cache.shape_val
    stream = torch.cuda.current_stream(dev)

    # --- histogram per block + Adler-32 partial sums (sync 1)
    ids = rowoff + scan  # int32 (h, rb): block * 257 + byte
    counts = torch.bincount(ids.view(-1), minlength=nb * 257)
    rs = scan.sum(dim=1, dtype=torch.int64)
    cs = scan.sum(dim=0, dtype=torch.int64)
    sums = torch.stack((rs.sum(), (rs * ar_h).sum(), (cs * ar_rb).sum()))
    pin_c = _png_pinned("counts", nb * 257 + 3, torch.int64)
    pin_c[: nb * 257].copy_(counts, non_blocking=True)
    pin_c[nb * 257 :].copy_(sums, non_blocking=True)
    stream.synchronize()
    host = pin_c.numpy()
    s_sum, t_row, t_col = (int(v) for v in host[nb * 257 :])
    # Adler-32: A = 1 + sum d_i; B = n + sum (n - i) d_i  (mod 65521)
    a_mod = (1 + s_sum) % 65521
    b_mod = (n + n * s_sum - (rb * t_row + t_col)) % 65521
    adler = (b_mod << 16) | a_mod
    cnt = host[: nb * 257].reshape(nb, 257).copy()
    cnt[:, 256] = 1  # end-of-block

    # --- per-block dynamic Huffman headers (host)
    tab = np.empty((nb, 257), dtype=np.int64)
    vals = np.zeros((nb, 280), dtype=np.int64)
    wids = np.zeros((nb, 280), dtype=np.int64)
    body = np.empty(nb, dtype=np.int64)
    ar19 = np.arange(19)
    for b in range(nb):
        row = cnt[b]
        lit_len = np.asarray(_huffman_lengths(row.tolist(), 15), dtype=np.int64)
        lit_code = _canonical_rev_codes(lit_len, 15)
        tab[b] = lit_code | (lit_len << 16)
        seq = np.concatenate((lit_len, np.ones(2, dtype=np.int64)))
        cl_len = np.asarray(
            _huffman_lengths(np.bincount(seq, minlength=19).tolist(), 7), dtype=np.int64
        )
        cl_code = _canonical_rev_codes(cl_len, 7)
        ordered = cl_len[_DEFLATE_CL_ORDER]
        nz = np.flatnonzero(ordered)
        hclen = max(4, int(nz[-1]) + 1) - 4
        bfinal = 1 if b == nb - 1 else 0
        # BFINAL, BTYPE=2, HLIT=257-257, HDIST=2-1, HCLEN
        vals[b, 0] = bfinal | (2 << 1) | (0 << 3) | (1 << 8) | (hclen << 13)
        wids[b, 0] = 17
        vals[b, 1:20] = ordered
        wids[b, 1:20] = np.where(ar19 < hclen + 4, 3, 0)
        vals[b, 20:279] = cl_code[seq]
        wids[b, 20:279] = cl_len[seq]
        vals[b, 279] = lit_code[256]
        wids[b, 279] = lit_len[256]
        body[b] = int(row[:256] @ lit_len[:256])
    hw = wids[:, :279]
    hbits = hw.sum(axis=1)
    tbits = hbits + body + wids[:, 279]
    start = np.cumsum(tbits) - tbits
    offs = np.empty_like(wids)
    offs[:, :279] = start[:, None] + np.cumsum(hw, axis=1) - hw
    offs[:, 279] = start + hbits + body
    extra = start + hbits - (np.cumsum(body) - body)
    total_bits = int(tbits.sum())

    ne = nb * 280
    pin_u = _png_pinned("upload", nb * 257 + nb + 2 * ne, torch.int64)
    u = pin_u.numpy()
    u[: nb * 257] = tab.reshape(-1)
    u[nb * 257 : nb * 258] = extra
    u[nb * 258 : nb * 258 + ne] = offs.reshape(-1)
    u[nb * 258 + ne :] = vals.reshape(-1)
    up = pin_u.to(dev, non_blocking=True)
    tab_g = up[: nb * 257]
    extra_g = up[nb * 257 : nb * 258]
    hoff = up[nb * 258 : nb * 258 + ne]
    hval = up[nb * 258 + ne :]

    # --- GPU bit packing (sync 2)
    packed = tab_g[ids]  # (h, rb): code | len << 16
    del ids
    lens = packed >> 16
    off = torch.cumsum(lens.view(-1), 0).view(h, rb)
    off.sub_(lens).add_(extra_g[rowblk].view(h, 1))
    del lens
    v = packed.bitwise_and_(0xFFFF).bitwise_left_shift_(off & 31)
    widx = off.bitwise_right_shift_(5).view(-1)
    v = v.view(-1)
    words = torch.zeros((total_bits >> 5) + 2, dtype=torch.int64, device=dev)
    words.scatter_add_(0, widx, v & 0xFFFFFFFF)
    words.scatter_add_(0, widx + 1, v >> 32)
    del v, widx, off, packed
    hv = hval << (hoff & 31)
    hwi = hoff >> 5
    words.scatter_add_(0, hwi, hv & 0xFFFFFFFF)
    words.scatter_add_(0, hwi + 1, hv >> 32)
    nbytes = (total_bits + 7) >> 3
    out = words.view(torch.uint8).view(-1, 8)[:, :4].reshape(-1)[:nbytes]
    pin_o = _png_pinned("out", nbytes, torch.uint8)
    pin_o.copy_(out, non_blocking=True)
    stream.synchronize()
    return memoryview(pin_o.numpy()), adler


def _png_assemble(w: int, h: int, c: int, stream, adler: int) -> bytes:
    """PNG file around a raw deflate ``stream``: zlib header 78 01 + Adler-32."""
    nbytes = len(stream)
    k = max(1, min(_PNG_IDAT_PIECES, nbytes >> 16))
    cuts = [nbytes * i // k for i in range(k + 1)]
    zhdr = b"\x78\x01"  # 32 KiB window, FLEVEL 0, FCHECK ok
    tail = struct.pack(">I", adler)
    init = zlib.crc32(b"IDAT")
    pieces = [stream[cuts[i] : cuts[i + 1]] for i in range(k)]
    if k > 1:
        pool = _png_pool()
        futs = [
            pool.submit(zlib.crc32, p, zlib.crc32(zhdr, init) if i == 0 else init)
            for i, p in enumerate(pieces)
        ]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _PNG_COLOR_TYPES[c], 0, 0, 0)
    parts = [_PNG_SIGNATURE, _png_chunk(b"IHDR", ihdr)]
    for i, p in enumerate(pieces):
        if k > 1:
            crc = futs[i].result()
        else:
            crc = zlib.crc32(p, zlib.crc32(zhdr, init))
        plen = len(p)
        if i == 0:
            plen += 2
        if i == k - 1:
            plen += 4
            crc = zlib.crc32(tail, crc)
        parts.append(struct.pack(">I", plen))
        parts.append(b"IDAT")
        if i == 0:
            parts.append(zhdr)
        parts.append(p)
        if i == k - 1:
            parts.append(tail)
        parts.append(struct.pack(">I", crc))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


def _encode_png_tensor(image: Any, compress_level: int = 1) -> Optional[bytes]:
    """Encode a uint8 L/RGB/RGBA image tensor as PNG; None when not eligible."""
    if not _fast_png_enabled() or not isinstance(image, torch.Tensor):
        return None
    # The filter step is a handful of device kernels on CUDA; on CPU tensors
    # torch's elementwise ops are slower than Pillow's C filter loop, so
    # host tensors keep the Pillow path.
    if not image.is_cuda:
        return None
    if image.dim() == 4:
        image = image[0]
    if image.dtype != torch.uint8:
        return None
    if image.dim() == 2:
        image = image.unsqueeze(-1)
    if image.dim() != 3:
        return None
    h, w, c = image.shape
    if c not in _PNG_COLOR_TYPES or h == 0 or w == 0:
        return None

    scan_g = _png_filter_scanlines(image)
    if _gpu_deflate_enabled() and scan_g.numel() <= _PNG_GPU_MAX_BYTES:
        try:
            stream, adler = _png_gpu_deflate(scan_g)
        except RuntimeError as e:  # Huffman limit failure: use the zlib writer
            logger.warning(f"GPU PNG deflate failed ({e}); using zlib")
        else:
            return _png_assemble(w, h, c, stream, adler)
    scan = scan_g.cpu().numpy()
    data = memoryview(scan.reshape(-1))
    row_bytes = 1 + w * c

    strip_rows = _PNG_STRIP_ROWS
    strategy = _PNG_STRATEGY
    mem_level = _PNG_MEM_LEVEL
    dict_bytes = _PNG_DICT_BYTES
    bounds = []
    for r0 in range(0, h, strip_rows):
        r1 = min(h, r0 + strip_rows)
        bounds.append((r0 * row_bytes, r1 * row_bytes))
    last = len(bounds) - 1

    def _deflate(i: int):
        start, end = bounds[i]
        if start > 0 and dict_bytes > 0:
            comp = zlib.compressobj(
                compress_level,
                zlib.DEFLATED,
                -15,
                mem_level,
                strategy,
                bytes(data[max(0, start - dict_bytes) : start]),
            )
        else:
            comp = zlib.compressobj(compress_level, zlib.DEFLATED, -15, mem_level, strategy)
        out = comp.compress(data[start:end])
        out += comp.flush(zlib.Z_FINISH if i == last else zlib.Z_SYNC_FLUSH)
        if i == 0:
            # zlib header for a 32 KiB window; FLEVEL as zlib writes it.
            flevel = 0 if (compress_level < 2 or strategy >= zlib.Z_HUFFMAN_ONLY) else (
                1 if compress_level < 6 else (2 if compress_level == 6 else 3))
            cmf = 0x78
            flg = flevel << 6
            flg += 31 - ((cmf * 256 + flg) % 31)
            out = bytes((cmf, flg)) + out
        if i == last:
            return out, None
        return out, zlib.crc32(out, zlib.crc32(b"IDAT"))

    pool = _png_pool()
    adler_future = pool.submit(zlib.adler32, data)
    futures = [pool.submit(_deflate, i) for i in range(len(bounds))]

    ihdr = struct.pack(">IIBBBBB", w, h, 8, _PNG_COLOR_TYPES[c], 0, 0, 0)
    parts = [_PNG_SIGNATURE, _png_chunk(b"IHDR", ihdr)]
    for i, fut in enumerate(futures):
        payload, crc = fut.result()
        if i == last:
            payload += struct.pack(">I", adler_future.result())
            crc = zlib.crc32(payload, zlib.crc32(b"IDAT"))
        parts.append(struct.pack(">I", len(payload)))
        parts.append(b"IDAT")
        parts.append(payload)
        parts.append(struct.pack(">I", crc))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


def _save_pil_image(
    pil_image: Image.Image,
    output: Any,
    format: str,
    quality: int,
    png_compress_level: int = 1,
) -> None:
    """Save a PIL image to a path or BytesIO with format-specific defaults."""
    format_upper = format.upper()
    if format_upper in ("JPEG", "JPG"):
        if pil_image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            if pil_image.mode == "P":
                pil_image = pil_image.convert("RGBA")
            background.paste(
                pil_image, mask=pil_image.split()[-1] if pil_image.mode == "RGBA" else None
            )
            pil_image = background
        pil_image.save(output, format="JPEG", quality=quality, optimize=True)
    elif format_upper == "WEBP":
        pil_image.save(output, format="WEBP", quality=quality)
    else:
        pil_image.save(output, format="PNG", compress_level=png_compress_level)


def _save_encoded_video(
    video: torch.Tensor,
    audio: Optional[torch.Tensor],
    output_path: str,
    frame_rate: float,
    audio_sample_rate: int = 24000,
) -> str:
    """Encode video (and optional audio) to ``output_path`` via the best encoder."""
    encoder = _get_video_encoder()
    if isinstance(encoder, _PurePythonEncoder) and output_path.lower().endswith(".mp4"):
        raise RuntimeError(
            "MP4 format requires ffmpeg to be installed. Please install ffmpeg "
            "(e.g., 'apt-get install ffmpeg' on Ubuntu/Debian) or use AVI format instead. "
            "See https://ffmpeg.org/download.html for installation instructions."
        )
    try:
        return encoder.encode_video(video, output_path, frame_rate, audio, audio_sample_rate)
    except Exception as e:
        logger.error(f"Error encoding video: {e}")
        raise e


# ---------------------------------------------------------------------------
# Public free functions
# ---------------------------------------------------------------------------


def save_image(
    image: torch.Tensor,
    output_path: Any,
    format: Optional[str] = None,
    quality: int = 95,
) -> str:
    """Encode and save an image tensor to disk.

    Args:
        image: Image as ``torch.Tensor`` ``(H, W, C)`` or ``(B, H, W, C)``
            with ``B == 1``, dtype ``uint8``. Batched inputs with
            ``B > 1`` are rejected — pass a single tensor or use
            :func:`save_images` with one path per batch item.
        output_path: Output file path (``str`` or :class:`pathlib.Path`).
        format: Image format (``'png'``/``'jpg'``/``'webp'``). If ``None``,
            inferred from the path extension; defaults to PNG when unknown.
        quality: Quality for lossy formats (1-100, higher is better).

    Returns:
        Path string where the image was actually saved.

    Raises:
        ValueError: When ``image`` is a 4-D tensor with ``B > 1``.
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)

    if hasattr(image, "dim") and image.dim() == 4:
        if image.shape[0] > 1:
            raise ValueError(
                f"save_image received a batched tensor of size {image.shape[0]}; "
                "pass a single (H, W, C) tensor, or use save_images(images, paths) "
                "with one path per batch item."
            )
        image = image[0]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            format = "PNG"
        elif ext in (".jpg", ".jpeg"):
            format = "JPEG"
        elif ext == ".webp":
            format = "WEBP"
        else:
            logger.warning(f"Unknown image extension {ext}, defaulting to PNG")
            format = "PNG"
            output_path = output_path.rsplit(".", 1)[0] + ".png"

    png_bytes = None
    if format.upper() not in ("JPEG", "JPG", "WEBP"):
        png_bytes = _encode_png_tensor(image)
    if png_bytes is not None:
        with open(output_path, "wb") as f:
            f.write(png_bytes)
    else:
        _save_pil_image(_to_pil_image(image), output_path, format, quality)
    logger.info(f"Saved image to {output_path} (format={format})")
    return output_path


_DUMP_SEQ = [0]


def _dump_image_to_bytes(dump_dir: str, image: Any, format: str, quality: int) -> bytes:
    """Debug only (TRTLLM_VISUALGEN_DUMP_DIR): time the encode, save the tensor."""
    import base64
    import json
    import time

    if isinstance(image, torch.Tensor) and image.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    data = _image_to_bytes(image, format, quality)
    t1 = time.perf_counter()
    b64 = base64.b64encode(data)
    t2 = time.perf_counter()
    seq = _DUMP_SEQ[0]
    _DUMP_SEQ[0] += 1
    os.makedirs(dump_dir, exist_ok=True)
    rec = {"seq": seq, "encode_ms": (t1 - t0) * 1e3, "b64_ms": (t2 - t1) * 1e3,
           "png_bytes": len(data), "format": format}
    if isinstance(image, torch.Tensor):
        rec.update(shape=list(image.shape), stride=list(image.stride()),
                   storage_offset=int(image.storage_offset()), dtype=str(image.dtype),
                   device=str(image.device))
        torch.save({"data": image.detach().contiguous().cpu(), **rec},
                   os.path.join(dump_dir, f"img_{seq:04d}.pt"))
    with open(os.path.join(dump_dir, "stamps.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return data


def image_to_bytes(image: torch.Tensor, format: str = "PNG", quality: int = 95) -> bytes:
    """Encode an image tensor to in-memory bytes."""
    dump_dir = os.environ.get("TRTLLM_VISUALGEN_DUMP_DIR")
    if dump_dir:
        return _dump_image_to_bytes(dump_dir, image, format, quality)
    return _image_to_bytes(image, format, quality)


def _image_to_bytes(image: torch.Tensor, format: str, quality: int) -> bytes:
    if format.upper() not in ("JPEG", "JPG", "WEBP"):
        # Same branch as _save_pil_image's PNG default.
        data = _encode_png_tensor(image)
        if data is not None:
            return data
    pil_image = _to_pil_image(image)
    buffer = BytesIO()
    _save_pil_image(pil_image, buffer, format, quality)
    return buffer.getvalue()


def save_video(
    video: torch.Tensor,
    output_path: Any,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: Optional[str] = None,
    audio_sample_rate: int = 24000,
) -> str:
    """Encode and save a video tensor (with optional audio) to disk.

    Args:
        video: Video as ``torch.Tensor`` ``(T, H, W, C)`` or ``(B, T, H, W, C)``
            with ``B == 1``, dtype ``uint8``. Batched inputs with
            ``B > 1`` are rejected — pass a single tensor or use
            :func:`save_videos` with one path per batch item.
        output_path: Output file path (``str`` or :class:`pathlib.Path`).
        audio: Optional audio tensor; ignored by the pure-Python AVI fallback.
        frame_rate: Frames per second.
        format: Container (``'mp4'``/``'avi'``). If ``None``, inferred from
            the path extension; defaults to MP4 when unknown.
        audio_sample_rate: Sample rate (Hz) used when muxing audio into the
            output container.

    Returns:
        Path string where the video was actually saved.

    Raises:
        ValueError: When ``video`` is a 5-D tensor with ``B > 1``.
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)
    if hasattr(video, "dim") and video.dim() == 5:
        if video.shape[0] > 1:
            raise ValueError(
                f"save_video received a batched tensor of size {video.shape[0]}; "
                "pass a single (T, H, W, C) tensor, or use save_videos(videos, paths) "
                "with one path per batch item."
            )
        video = video[0]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        format = ext[1:] if ext else "mp4"
    format = format.lower()

    if format in ("mp4", "avi"):
        return _save_encoded_video(video, audio, output_path, frame_rate, audio_sample_rate)
    logger.warning(f"Unsupported video format: {format}, defaulting to mp4")
    output_path = output_path.rsplit(".", 1)[0] + ".mp4"
    return _save_encoded_video(video, audio, output_path, frame_rate, audio_sample_rate)


def video_to_bytes(
    video: torch.Tensor,
    audio: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: str = "mp4",
    audio_sample_rate: int = 24000,
) -> bytes:
    """Encode a video tensor (with optional audio) to in-memory bytes."""
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    actual_path = None
    try:
        actual_path = save_video(video, tmp_path, audio, frame_rate, format, audio_sample_rate)
        with open(actual_path, "rb") as f:
            return f.read()
    finally:
        for path in (tmp_path, actual_path):
            if path and os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Batch save helpers (used by serve endpoints when n > 1)
# ---------------------------------------------------------------------------

_IMAGE_EXT_MAP = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
}


def _image_ext_for_format(format: Optional[str]) -> str:
    """Return the file extension (including leading dot) for an image format."""
    if format is None:
        return ".png"
    return _IMAGE_EXT_MAP.get(format.upper(), ".png")


def _resolve_batch_paths(
    output_paths: Union[str, List[str]],
    batch_size: int,
    ext: str,
) -> List[str]:
    """Build per-item output paths for a batch save operation.

    When *output_paths* is a string it is used as a prefix and each item is
    written to ``{prefix}_{i}{ext}``. When it is a list, each entry is used
    as-is; missing extensions are filled in from *ext*. The list length
    must equal *batch_size*.
    """
    if isinstance(output_paths, list):
        if len(output_paths) != batch_size:
            raise ValueError(
                f"Length of output_paths ({len(output_paths)}) does not "
                f"match batch size ({batch_size})"
            )
        resolved = []
        for p in output_paths:
            if not os.path.splitext(p)[1]:
                p = p + ext
            resolved.append(p)
        return resolved
    return [f"{output_paths}_{i}{ext}" for i in range(batch_size)]


def save_images(
    images: torch.Tensor,
    output_paths: Union[str, List[str]],
    format: Optional[str] = None,
    quality: int = 95,
) -> List[str]:
    """Save a batch of images to individual files.

    Args:
        images: ``torch.Tensor`` of shape ``(B, H, W, C)`` or ``(H, W, C)``,
            dtype ``uint8``.
        output_paths: Either a path prefix string (each image is saved as
            ``{prefix}_{i}.{ext}``) or an explicit list of per-image paths.
        format: Image format (``'png'``/``'jpg'``/``'webp'``). Defaults to PNG.
        quality: Quality for lossy formats (1-100).

    Returns:
        List of paths where the images were saved.
    """
    ext = _image_ext_for_format(format)

    if hasattr(images, "dim") and images.dim() == 3:
        images = images.unsqueeze(0)

    batch_size = images.shape[0]
    resolved = _resolve_batch_paths(output_paths, batch_size, ext)

    paths: List[str] = []
    for i in range(batch_size):
        save_image(images[i], resolved[i], format, quality)
        paths.append(resolved[i])
    return paths


def save_videos(
    videos: torch.Tensor,
    output_paths: Union[str, List[str]],
    audios: Optional[torch.Tensor] = None,
    frame_rate: float = 24.0,
    format: Optional[str] = None,
    audio_sample_rate: int = 24000,
) -> List[str]:
    """Save a batch of videos to individual files.

    Args:
        videos: ``torch.Tensor`` of shape ``(B, T, H, W, C)`` or
            ``(T, H, W, C)``, dtype ``uint8``.
        output_paths: Either a path prefix string (each video is saved as
            ``{prefix}_{i}.{ext}``) or an explicit list of per-video paths.
        audios: Optional audio tensor. When batched with a matching leading
            dimension, each slice is paired with the corresponding video;
            an unbatched audio tensor is attached to the first video only.
        frame_rate: Frames per second.
        format: Container (``'mp4'``/``'avi'``). Defaults to ``mp4``.
        audio_sample_rate: Audio sample rate (Hz).

    Returns:
        List of paths where the videos were actually saved.
    """
    ext = f".{format}" if format else ".mp4"

    if hasattr(videos, "dim") and videos.dim() == 4:
        videos = videos.unsqueeze(0)

    batch_size = videos.shape[0]
    resolved = _resolve_batch_paths(output_paths, batch_size, ext)

    paths: List[str] = []
    for i in range(batch_size):
        audio_i = None
        if audios is not None:
            if hasattr(audios, "dim") and audios.dim() >= 2 and audios.shape[0] == batch_size:
                audio_i = audios[i]
            elif i == 0:
                audio_i = audios
        actual_path = save_video(
            videos[i],
            resolved[i],
            audio_i,
            frame_rate,
            format,
            audio_sample_rate,
        )
        paths.append(actual_path)
    return paths
