# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for :mod:`tensorrt_llm.media.decoding` and
:mod:`tensorrt_llm.media.png_decode`, plus the rank convergence protocol that
guards them (``tensorrt_llm._torch.visual_gen.utils.synchronize_media_prepare_status``).

The decode tests run on the checked-in H.264 fixtures (see
``test_data/README.md`` for provenance). Each fixture frame encodes its own
display index three ways — red-channel ramp, green horizontal bar, blue
vertical bar — so ordering, channel layout, and surface ownership are all
observable from content alone.

GPU (NVDEC) tests skip only on CUDA-less platforms; a missing PyNvVideoCodec
on a supported platform fails them (it is a declared dependency).
"""

import math
import struct
import subprocess
import sys
import time
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.utils import synchronize_media_prepare_status
from tensorrt_llm.media import png_decode
from tensorrt_llm.media.decoding import (
    _lanczos_taps,
    decode_video_reference_window,
    resize_center_crop_uint8,
    resize_fit_pad_uint8,
)

_TEST_DATA = Path(__file__).parent / "test_data"
_MP4 = _TEST_DATA / "cosmos3_v2v_ref_9f_bframes.mp4"
_AVI = _TEST_DATA / "cosmos3_v2v_ref_9f_bframes.avi"


def _frame_indices(frames: torch.Tensor) -> list[int]:
    """Recover each frame's display index from the red-channel ramp."""
    return [round((f[:, :, 0].float().mean().item() - 20) / 25) for f in frames]


class TestResizeFitPad:
    """The action counterpart to cover-scale + center-crop: contain-scale, then
    pad. A gripper works at the frame edge, so cropping costs the policy the
    evidence it acts on."""

    def test_contains_the_whole_source(self):
        """Cover-scale would crop the wide source; fit keeps all of it."""
        frames = torch.full((2, 100, 400, 3), 200, dtype=torch.uint8)
        out = resize_fit_pad_uint8(frames, target_h=200, target_w=400)
        assert out.shape == (2, 200, 400, 3)
        # 400x100 contained in 400x200 scales by 1.0 and pads the bottom half.
        assert (out[:, :100] == 200).all()

    def test_never_enlarges_a_small_source(self):
        """min(..., 1.0): a small clip keeps its own pixels and a wider border
        rather than being upscaled."""
        frames = torch.full((1, 10, 20, 3), 255, dtype=torch.uint8)
        out = resize_fit_pad_uint8(frames, target_h=64, target_w=64)
        assert out.shape == (1, 64, 64, 3)
        assert (out[0, :10, :20] == 255).all()

    def test_native_resolution_is_identity(self):
        frames = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        assert resize_fit_pad_uint8(frames, 32, 32) is frames

    def test_aspect_ratio_is_preserved(self):
        """The source is contained, not stretched: a square stays square."""
        frames = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
        frames[:, :, :, 0] = 255
        out = resize_fit_pad_uint8(frames, target_h=64, target_w=128)
        assert out.shape == (1, 64, 128, 3)
        # A 1:1 source in a 2:1 canvas fills the height, so content is 64 wide.
        assert (out[0, :, :64, 0] == 255).all()

    def test_pad_falls_back_to_replicate_when_reflection_has_no_source(self):
        """A pad run wider than the resized extent has nothing left to mirror;
        reflect would raise, so the helper switches to edge replication."""
        frames = torch.full((1, 4, 4, 3), 128, dtype=torch.uint8)
        out = resize_fit_pad_uint8(frames, target_h=64, target_w=64)
        assert out.shape == (1, 64, 64, 3)
        assert (out[0, 4:, :4] == 128).all()


class TestResizeCenterCrop:
    """CPU-runnable checks of the shared Lanczos resize/crop front."""

    def _pil_reference(self, frames_u8: torch.Tensor, target_h: int, target_w: int):
        """The reference implementation's geometry: PIL LANCZOS cover-scale
        + center crop (vllm-omni ``_preprocess_condition_image``)."""
        out = []
        for frame in frames_u8.numpy():
            h, w = frame.shape[:2]
            scale = max(target_w / w, target_h / h)
            resize_w = int(math.ceil(scale * w))
            resize_h = int(math.ceil(scale * h))
            img = Image.fromarray(frame).resize((resize_w, resize_h), Image.Resampling.LANCZOS)
            left = (resize_w - target_w) // 2
            top = (resize_h - target_h) // 2
            out.append(np.asarray(img.crop((left, top, left + target_w, top + target_h))))
        return torch.from_numpy(np.stack(out))

    def test_parity_with_pil_lanczos(self):
        rng = np.random.default_rng(7)
        frames = torch.from_numpy(rng.integers(0, 256, (3, 64, 96, 3), dtype=np.uint8))
        for target_h, target_w in ((32, 48), (48, 32), (128, 96)):
            ours = resize_center_crop_uint8(frames, target_h, target_w)
            ref = self._pil_reference(frames, target_h, target_w)
            assert ours.shape == ref.shape == (3, target_h, target_w, 3)
            diff = (ours.int() - ref.int()).abs()
            # PIL quantizes filter coefficients to fixed point; we keep
            # float. Bounded, not bit-exact.
            assert diff.max().item() <= 2, f"{target_h}x{target_w}: max diff {diff.max()}"

    def test_native_resolution_is_identity(self):
        frames = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        assert resize_center_crop_uint8(frames, 32, 32) is frames

    def test_taps_are_local_support_and_cached(self):
        # The taps depend only on (in, out) sizes — one build per clip, not
        # one per frame — and their width is the filter's true support
        # (K = ceil(2 * a * scale) + 1), NOT the full input row: this is what
        # makes the resample O(out * K) instead of O(out * in).
        _lanczos_taps.cache_clear()
        weights, taps = _lanczos_taps(1920, 1280, "cpu")
        assert taps.shape == weights.shape
        assert taps.shape[1] <= 2 * math.ceil(3 * (1920 / 1280)) + 1  # K = 10 << 1920
        frames = torch.zeros(4, 40, 40, 3, dtype=torch.uint8)
        resize_center_crop_uint8(frames, 20, 20)
        info = _lanczos_taps.cache_info()
        resize_center_crop_uint8(frames, 20, 20)
        assert _lanczos_taps.cache_info().hits >= info.hits + 2


class TestImportIsolation:
    def test_import_tensorrt_llm_does_not_load_pynvvideocodec(self):
        """PyNvVideoCodec is driver-linked; ``import tensorrt_llm`` (and the
        decode module itself) must not load it — only an actual decode may."""
        code = (
            "import sys; import tensorrt_llm; "
            "import tensorrt_llm.media.decoding; "
            "assert 'PyNvVideoCodec' not in sys.modules, "
            "'driver-linked PyNvVideoCodec loaded at import time'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


def _status_protocol_rank(rank: int, world_size: int, init_file: str, results_dir: str):
    """Spawn target: run the convergence protocol with rank 1 failing."""
    import os

    # Containers often have hostnames that don't resolve to a usable
    # interface; pin gloo to loopback or its rendezvous hangs.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")

    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen.utils import synchronize_media_prepare_status

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        local_error = ValueError("rank-local decode failure") if rank == 1 else None
        try:
            synchronize_media_prepare_status(local_error)
            outcome = "no-error"
        except ValueError as exc:
            outcome = f"client:{exc}"
        except Exception as exc:  # pragma: no cover - diagnostic path
            outcome = f"unexpected:{type(exc).__name__}:{exc}"
    finally:
        dist.destroy_process_group()
    Path(results_dir, f"rank{rank}.txt").write_text(outcome)


class TestPrepareStatusProtocol:
    def test_local_failure_is_reraised_without_group(self):
        err = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            synchronize_media_prepare_status(err)

    def test_success_passes_through_without_group(self):
        synchronize_media_prepare_status(None)

    def test_two_rank_convergence_over_gloo(self, tmp_path):
        """Rank 1 fails decode, rank 0 is healthy: both must exit the
        protocol with the SAME client error — rank 1 re-raising its own,
        rank 0 raising the reconstructed equivalent — instead of rank 0
        proceeding into (and hanging in) model collectives."""
        import torch.multiprocessing as mp

        init_file = tmp_path / "gloo_init"
        ctx = mp.spawn(
            _status_protocol_rank,
            args=(2, str(init_file), str(tmp_path)),
            nprocs=2,
            join=False,
        )
        # ``ProcessContext.join`` returns False whenever ANY child has
        # exited but others still run — it is designed to be polled.
        deadline = time.monotonic() + 240
        converged = False
        while time.monotonic() < deadline:
            if ctx.join(timeout=deadline - time.monotonic()):
                converged = True
                break
        if not converged:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
        assert converged, "status protocol did not converge (hang)"
        healthy = (tmp_path / "rank0.txt").read_text()
        failing = (tmp_path / "rank1.txt").read_text()
        assert failing == "client:rank-local decode failure"
        assert healthy == "client:[rank 1] rank-local decode failure"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDecodeVideoReferenceWindow:
    _DEVICE = torch.device("cuda:0")

    def _decode(self, data: bytes, *, window=5, keep="first", **kwargs):
        """Express a (window, keep) request as the decoder's frame range."""
        span = (0, window - 1) if keep == "first" else (-window, -1)
        defaults = dict(target_h=64, target_w=64, device=self._DEVICE)
        defaults.update(kwargs)
        return decode_video_reference_window(
            data, first_frame=span[0], last_frame=span[1], **defaults
        )

    @pytest.mark.parametrize("fixture", [_MP4, _AVI], ids=["mp4", "avi"])
    def test_keep_first_display_order(self, fixture):
        # Display order despite forced B-frames, in both supported containers.
        window = self._decode(fixture.read_bytes())
        assert window.shape == (5, 64, 64, 3) and window.dtype == torch.uint8
        assert window.device.type == "cuda"
        assert _frame_indices(window) == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("fixture", [_MP4, _AVI], ids=["mp4", "avi"])
    def test_keep_last_ring_reorder(self, fixture):
        window = self._decode(fixture.read_bytes(), keep="last")
        assert _frame_indices(window) == [4, 5, 6, 7, 8]

    def test_frame_step_thins_the_window(self):
        window = decode_video_reference_window(
            _MP4.read_bytes(),
            first_frame=0,
            last_frame=8,
            target_h=64,
            target_w=64,
            device=self._DEVICE,
            frame_step=2,
        )
        assert window.shape[0] == 5
        assert _frame_indices(window) == [0, 2, 4, 6, 8]

    def test_frame_step_keeps_a_partial_tail(self):
        # The range end is not a multiple of the step: 0, 3, 6 and stop.
        window = decode_video_reference_window(
            _MP4.read_bytes(),
            first_frame=0,
            last_frame=7,
            target_h=64,
            target_w=64,
            device=self._DEVICE,
            frame_step=3,
        )
        assert _frame_indices(window) == [0, 3, 6]

    def test_frame_step_one_matches_the_default(self):
        span = dict(first_frame=0, last_frame=4, target_h=64, target_w=64, device=self._DEVICE)
        assert torch.equal(
            decode_video_reference_window(_MP4.read_bytes(), **span),
            decode_video_reference_window(_MP4.read_bytes(), frame_step=1, **span),
        )

    def test_frame_step_rejects_trailing_windows(self):
        # The trailing form wraps a ring whose length is unknown until EOS.
        with pytest.raises(ValueError, match="non-negative ranges"):
            self._decode(_MP4.read_bytes(), keep="last", frame_step=2)

    def test_frame_step_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            self._decode(_MP4.read_bytes(), frame_step=0)

    def test_rgb_channel_layout(self):
        # Frame i carries a green horizontal bar at rows [7i, 7i+7) and a
        # blue vertical bar at cols [7i, 7i+7): asserts the NVDEC output is
        # RGB (not BGR) and spatially unflipped. 4:2:0 chroma blurs edges,
        # so thresholds are generous.
        window = self._decode(_MP4.read_bytes(), window=9, keep="first")
        for i in (2, 6):
            frame = window[i].float()
            band = slice(7 * i, 7 * i + 7)
            assert frame[band, :, 1].mean() > 150  # green bar rows
            assert frame[:, band, 2].mean() > 150  # blue bar cols
            assert frame[:, :, 1].mean() < frame[band, :, 1].mean() - 60

    def test_surface_ownership_across_full_decode(self):
        # If the ring held DLPack views instead of owned copies, NVDEC's
        # surface recycling would overwrite earlier frames during the later
        # decodes; per-frame content proves each retained frame is intact
        # after the decoder finished the whole stream.
        window = self._decode(_MP4.read_bytes(), window=9, keep="first")
        assert _frame_indices(window) == list(range(9))

    def test_target_resolution_resize(self):
        window = self._decode(_MP4.read_bytes(), window=3, target_h=96, target_w=128)
        assert window.shape == (3, 96, 128, 3)

    def test_window_longer_than_clip_returns_all(self):
        window = self._decode(_MP4.read_bytes(), window=20)
        assert _frame_indices(window) == list(range(9))

    def test_corrupt_bytes_with_valid_magic_is_client_error(self):
        payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        with pytest.raises(ValueError):
            self._decode(payload)

    def test_decoder_imposes_no_frame_limit(self):
        # The decoder returns what it is asked for; bounding the request is
        # the caller's business (Cosmos3 derives it from the output length).
        window = self._decode(_MP4.read_bytes(), window=20, keep="last")
        assert window.shape[0] == 9

    def test_interior_range(self):
        window = decode_video_reference_window(
            _MP4.read_bytes(),
            first_frame=3,
            last_frame=5,
            target_h=64,
            target_w=64,
            device=self._DEVICE,
        )
        assert _frame_indices(window) == [3, 4, 5]

    def test_negative_range_excluding_final_frames(self):
        window = decode_video_reference_window(
            _MP4.read_bytes(),
            first_frame=-4,
            last_frame=-3,
            target_h=64,
            target_w=64,
            device=self._DEVICE,
        )
        assert _frame_indices(window) == [5, 6]

    @pytest.mark.parametrize(
        "span", [(0, -1), (-1, 0), (5, 2)], ids=["mixed", "mixed-rev", "reversed"]
    )
    def test_malformed_range_rejected(self, span):
        with pytest.raises(ValueError):
            decode_video_reference_window(
                _MP4.read_bytes(),
                first_frame=span[0],
                last_frame=span[1],
                target_h=64,
                target_w=64,
                device=self._DEVICE,
            )

    def test_resize_perf_representative(self):
        # Representative evidence for the local-tap resample: a 1080p frame
        # to 720p-cover must be in the low-millisecond range. The bound is
        # ~100x actual so it never flakes; a dense O(out*in) implementation
        # (~11 GMAC/frame) would still trip it under contention.
        frame = torch.randint(0, 256, (1, 1080, 1920, 3), dtype=torch.uint8, device=self._DEVICE)
        resize_center_crop_uint8(frame, 720, 1280)  # warmup + tap-cache build
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            resize_center_crop_uint8(frame, 720, 1280)
        torch.cuda.synchronize()
        per_frame = (time.perf_counter() - start) / 10
        assert per_frame < 0.25, f"resize took {per_frame * 1e3:.1f} ms/frame"


# ---------------------------------------------------------------------------
# PNG decode (host inflate + GPU unfilter): bitwise Pillow pixels or None
# ---------------------------------------------------------------------------


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))


def _encode_rgb8_png(pixels: np.ndarray, extra_chunks=(), idat_parts: int = 1) -> bytes:
    """RGB8 PNG whose rows cycle through all five filter types (spec section 9)."""
    h, w, _ = pixels.shape
    x = pixels.reshape(h, w * 3).astype(np.int32)
    rows = []
    for r in range(h):
        a = np.concatenate([np.zeros(3, np.int32), x[r, :-3]])
        b = x[r - 1] if r else np.zeros_like(x[r])
        c = np.concatenate([np.zeros(3, np.int32), b[:-3]])
        p = a + b - c
        pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
        paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        ft = r % 5
        pred = [np.zeros_like(a), a, b, (a + b) >> 1, paeth][ft]
        rows.append(bytes([ft]) + ((x[r] - pred) & 255).astype(np.uint8).tobytes())
    z = zlib.compress(b"".join(rows), 6)
    cuts = [len(z) * i // idat_parts for i in range(idat_parts + 1)]
    out = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    for chunk in extra_chunks:
        out += chunk
    for i in range(idat_parts):
        out += _png_chunk(b"IDAT", z[cuts[i] : cuts[i + 1]])
    return out + _png_chunk(b"IEND", b"")


def _pil_rgb(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as img:
        return np.array(img)


class TestPngDecode:
    _RNG = np.random.default_rng(7)
    _PIXELS = _RNG.integers(0, 256, (37, 53, 3), dtype=np.uint8)

    def test_scope_parse(self):
        """In-scope files parse; anything PIL may decode differently (other colour
        types, interlace, text/EXIF chunks, bad CRCs, tiny sizes) is left to PIL."""
        good = _encode_rgb8_png(self._PIXELS, [_png_chunk(b"gAMA", struct.pack(">I", 45455))], 3)
        width, height, idat = png_decode.parse_png_rgb8(good)
        assert (width, height, len(idat)) == (53, 37, 3)

        buffer = BytesIO()
        Image.fromarray(self._PIXELS).convert("RGBA").save(buffer, format="PNG")
        corrupt = bytearray(good)
        corrupt[29] ^= 1  # IHDR CRC
        out_of_scope = [
            buffer.getvalue(),
            _encode_rgb8_png(self._PIXELS, [_png_chunk(b"tEXt", b"k\0v")]),
            _encode_rgb8_png(self._PIXELS, [_png_chunk(b"gAMA", b"\0")]),
            _encode_rgb8_png(self._PIXELS[:8, :8]),
            bytes(corrupt),
            good[:-12],  # no IEND
        ]
        for data in out_of_scope:
            assert png_decode.parse_png_rgb8(data) is None

    def test_inflate_is_exact_size(self):
        raw = bytes(range(256)) * 9
        z = zlib.compress(raw)
        out = torch.empty(len(raw), dtype=torch.uint8)
        for use_libdeflate in (False, True):
            assert png_decode.inflate_scanlines_into(z, out, _use_libdeflate=use_libdeflate)
            assert out.numpy().tobytes() == raw
            short = torch.empty(len(raw) - 1, dtype=torch.uint8)
            assert not png_decode.inflate_scanlines_into(z, short, _use_libdeflate=use_libdeflate)
            assert not png_decode.inflate_scanlines_into(
                z + b"\0", out, _use_libdeflate=use_libdeflate
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("shape", [(37, 53), (16, 16), (300, 17), (2100, 24)])
    def test_device_pixels_are_pillow_pixels(self, shape):
        pixels = self._RNG.integers(0, 256, (*shape, 3), dtype=np.uint8)
        for data in (_encode_rgb8_png(pixels, idat_parts=2), self._pillow_png(pixels)):
            decoded = png_decode.decode_png_rgb8_to_device(data, "cuda")
            assert decoded is not None
            np.testing.assert_array_equal(decoded.cpu().numpy(), _pil_rgb(data))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_out_of_scope_and_bad_filters_return_none(self):
        data = bytearray(_encode_rgb8_png(self._PIXELS))
        assert png_decode.decode_png_rgb8_to_device(bytes(data), "cpu") is None
        rows = bytearray(zlib.decompress(bytes(png_decode.parse_png_rgb8(data)[2][0])))
        rows[0] = 5  # filter byte out of range
        z = zlib.compress(bytes(rows))
        bad = bytes(data[:33]) + _png_chunk(b"IDAT", z) + _png_chunk(b"IEND", b"")
        assert png_decode.decode_png_rgb8_to_device(bad, "cuda") is None

    @staticmethod
    def _pillow_png(pixels: np.ndarray) -> bytes:
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue()
