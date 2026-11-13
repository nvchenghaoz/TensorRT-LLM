"""Idea 152 unit check: the pageable fallback of media/encoding._host_numpy.

Saves one synthetic 124x768x1344x3 uint8 CUDA video (+ 32 kHz stereo audio) through
encode_video (_FfmpegCliEncoder) on:
  A  the pinned path (fresh process-wide pinned buffer),
  B  a forced pin failure (torch.empty(pin_memory=True) monkeypatched to raise), twice,
  C  the pinned path again after the patch is removed (retry grows the buffer again),
  D  main's path: _host_numpy replaced by tensor.cpu().numpy() (pageable, what main does).
Every MP4's sha256 must be equal; exactly one warning must be logged for the two failures;
no exception may reach the caller. Prints one JSON object as the last line.
"""

import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import torch


def log(*a):
    print("[pinfb]", *a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # The container has no ffmpeg: take the same binary the frozen adapters use (store-cached,
    # sha-reported) via ab_common.ensure_ffmpeg. argv[1] = experiments/ascent-benchmark/ab_common.py.
    if len(sys.argv) > 1:
        bench_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
        sys.path.insert(0, bench_dir)
        from ab_common import ensure_ffmpeg, load_json

        info = ensure_ffmpeg(load_json("reference_manifest.json")["store"])
        log("ffmpeg", info)
    from tensorrt_llm.media import encoding

    assert encoding._check_ffmpeg_available(), "ffmpeg missing"
    enc = encoding._FfmpegCliEncoder()

    T, H, W, fps, sr = 124, 768, 1344, 24.0, 32000
    g = torch.Generator(device="cuda").manual_seed(42)
    t = torch.arange(T, device="cuda").view(T, 1, 1, 1).float()
    y = torch.arange(H, device="cuda").view(1, H, 1, 1).float()
    x = torch.arange(W, device="cuda").view(1, 1, W, 1).float()
    c = torch.arange(3, device="cuda").view(1, 1, 1, 3).float()
    base = 127 + 100 * torch.sin((x + 3 * t) / 37 + c) * torch.cos((y - 2 * t) / 23 + c)
    noise = torch.randint(-12, 13, (T, H, W, 3), device="cuda", generator=g).float()
    # [B, T, H, W, C]-contiguous like the pipeline output; encode_video sees the [T,H,W,C] view.
    video = (base + noise).clamp(0, 255).to(torch.uint8).unsqueeze(0).contiguous()[0]
    n = math.ceil(T * sr / fps)
    ts = torch.arange(n, device="cuda").float() / sr
    audio = torch.stack([0.3 * torch.sin(2 * math.pi * 440 * ts), 0.2 * torch.sin(2 * math.pi * 660 * ts)], 0)
    torch.cuda.synchronize()
    log("video", tuple(video.shape), video.stride(), "audio", tuple(audio.shape))

    warnings = []
    orig_warning = encoding.logger.warning

    def counting_warning(msg, *a, **k):
        warnings.append(str(msg))
        return orig_warning(msg, *a, **k)

    encoding.logger.warning = counting_warning

    orig_empty = torch.empty
    pin_calls = {"raised": 0}

    def failing_empty(*a, **k):
        if k.get("pin_memory"):
            pin_calls["raised"] += 1
            raise RuntimeError("CUDA error: out of memory (simulated cudaHostAlloc failure)")
        return orig_empty(*a, **k)

    out = tempfile.mkdtemp(prefix="pinfb_")
    res = {}
    errors = []

    def save(tag):
        p = os.path.join(out, f"{tag}.mp4")
        t0 = time.perf_counter()
        try:
            enc.encode_video(video, p, fps, audio=audio, audio_sample_rate=sr)
        except Exception as e:  # must never happen
            errors.append(f"{tag}: {type(e).__name__}: {e}")
            log(tag, "RAISED", e)
            return
        ms = (time.perf_counter() - t0) * 1e3
        buf = encoding._PINNED_BUFFER
        res[tag] = {"sha256": sha(p), "bytes": os.path.getsize(p), "ms": round(ms, 1),
                    "pinned_buffer_bytes": None if buf is None else int(buf.numel())}
        log(tag, res[tag])

    # A: pinned path.
    encoding._PINNED_BUFFER = None
    save("A_pinned")
    # B: forced pin failure, twice (buffer dropped so the grow is attempted).
    encoding._PINNED_BUFFER = None
    torch.empty = failing_empty
    try:
        save("B_fallback_1")
        save("B_fallback_2")
    finally:
        torch.empty = orig_empty
    # C: pinned again after the pressure is gone.
    save("C_pinned_again")
    # D: main's pageable path.
    orig_host_numpy = encoding._host_numpy

    @contextlib.contextmanager
    def main_host_numpy(tensor):
        yield tensor.cpu().numpy()

    encoding._host_numpy = main_host_numpy
    try:
        save("D_main_pageable")
    finally:
        encoding._host_numpy = orig_host_numpy
    encoding.logger.warning = orig_warning

    shas = {k: v["sha256"] for k, v in res.items()}
    fb_warnings = [w for w in warnings if "Pinned video staging buffer" in w]
    checks = {
        "no_exception": not errors,
        "all_saved": len(res) == 5,
        "all_sha_equal": len(shas) == 5 and len(set(shas.values())) == 1,
        "pin_alloc_was_forced_to_fail": pin_calls["raised"] >= 2,
        "fallback_had_no_pinned_buffer": all(
            k in res and res[k]["pinned_buffer_bytes"] is None for k in ("B_fallback_1", "B_fallback_2")
        ),
        "one_warning_per_process": len(fb_warnings) == 1,
        "pinned_regrown_after": res.get("C_pinned_again", {}).get("pinned_buffer_bytes") == T * H * W * 3,
    }
    log("warnings", fb_warnings)
    log("checks", checks)
    print(json.dumps({"passed": all(checks.values()), "checks": checks, "saves": res,
                      "pin_raises": pin_calls["raised"], "warnings": fb_warnings, "errors": errors,
                      "compared": "fallback MP4 sha256 vs pinned-path and main-path (pageable) MP4"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
