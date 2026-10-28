"""Branch 68 (idea 114 Stage A): GPU Huffman-only deflate behind image_to_bytes.

Usage: python b68_gpu_png.py check|sweep|ab [block_rows ...]
  check : untimed correctness. Images: saved served FLUX.3 tensors (b47 dump) when
          found, the docs jpgs, synthetic edges. Every image and every block-row config:
          PIL verify, decoded pixels == source == main path (TRTLLM_PNG_GPU_DEFLATE=0,
          the unchanged zlib strip writer) decode, inflated IDAT == main's filtered
          stream, chunk CRCs, zlib Adler, b64 round trip, repeated bytes identical,
          input unchanged. Plus raw-scan round trips (random, constant, Fibonacci
          histogram that forces the 15-bit length limit).
  sweep : interleaved timing of main / main_aa / gpu at each block_rows (encode+b64).
  ab    : like sweep with the module default only (main, main_aa, new), >=60 rounds.
Last line: JSON.
"""
import base64
import glob
import io
import json
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b47_common import find_dump, load_images, log  # noqa: E402

os.environ.pop("TRTLLM_VISUALGEN_DUMP_DIR", None)
from tensorrt_llm.media import encoding as enc  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
ROWS = [int(x) for x in sys.argv[2:]] or [64, 256, 1024]
DEFAULT_ROWS = enc._PNG_GPU_BLOCK_ROWS


def set_cfg(cfg):
    if cfg in ("main", "main_aa"):
        os.environ["TRTLLM_PNG_GPU_DEFLATE"] = "0"
    else:
        os.environ.pop("TRTLLM_PNG_GPU_DEFLATE", None)
        enc._PNG_GPU_BLOCK_ROWS = cfg


def encode(g):
    return enc.image_to_bytes(g, format="PNG")


def chunks(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, out = 8, []
    while pos < len(png):
        n = int.from_bytes(png[pos:pos + 4], "big")
        tag = png[pos + 4:pos + 8]
        payload = png[pos + 8:pos + 8 + n]
        crc = int.from_bytes(png[pos + 8 + n:pos + 12 + n], "big")
        if zlib.crc32(tag + payload) != crc:
            raise ValueError(f"bad CRC in {tag}")
        out.append((tag, payload))
        pos += 12 + n
    return out


def inflate(png):
    return zlib.decompress(b"".join(p for t, p in chunks(png) if t == b"IDAT"))


def decode(png):
    im = Image.open(io.BytesIO(png))
    im.verify()
    im = Image.open(io.BytesIO(png))
    im.load()
    return np.asarray(im)


def images():
    out = []
    dump = find_dump()
    if dump:
        for seq, kind, src, g, stride in load_images(dump):
            out.append((f"served{seq:02d}_{kind}", g, src[0].numpy() if src.dim() == 4 else src.numpy(),
                        kind))
        log(f"dump {dump}: {len(out)} served tensors, stride {out[0][1].stride() if out else None}")
    else:
        log("no b47 dump found")
    for p in sorted(glob.glob("docs/source/blogs/media/*.jpg"))[:8]:
        arr = np.asarray(Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)).copy()
        out.append((os.path.basename(p)[:40], torch.from_numpy(arr).cuda(), arr, "docs"))
    log(f"{len(out)} real images")
    return out


def synthetic():
    g = torch.Generator().manual_seed(0)
    s = []

    def add(name, a):
        a = np.ascontiguousarray(a)
        s.append((name, torch.from_numpy(a).cuda(), a, "synthetic"))

    add("1x1x3", np.array([[[1, 2, 3]]], np.uint8))
    add("1x1x1", np.array([[[7]]], np.uint8))
    add("2x3x4", torch.randint(0, 256, (2, 3, 4), generator=g, dtype=torch.uint8).numpy())
    add("7x5x3", torch.randint(0, 256, (7, 5, 3), generator=g, dtype=torch.uint8).numpy())
    add("odd_1023x777x3", torch.randint(0, 256, (1023, 777, 3), generator=g, dtype=torch.uint8).numpy())
    add("noise_1024x1024x3", torch.randint(0, 256, (1024, 1024, 3), generator=g, dtype=torch.uint8).numpy())
    add("zeros_1024x1024x3", np.zeros((1024, 1024, 3), np.uint8))
    add("ones255_1024x1024x3", np.full((1024, 1024, 3), 255, np.uint8))
    y, x = np.mgrid[0:1024, 0:1024]
    add("gradient_rgb", np.stack([(x // 4) % 256, (y // 4) % 256, ((x + y) // 8) % 256], -1).astype(np.uint8))
    add("tall_300x1x3", torch.randint(0, 256, (300, 1, 3), generator=g, dtype=torch.uint8).numpy())
    add("wide_1x3000x4", torch.randint(0, 256, (1, 3000, 4), generator=g, dtype=torch.uint8).numpy())
    add("gray_64x64", torch.randint(0, 256, (64, 64), generator=g, dtype=torch.uint8).numpy())
    add("rgba_513x257", torch.randint(0, 4, (513, 257, 4), generator=g, dtype=torch.uint8).numpy())
    big = (np.arange(1536)[None, :] + np.arange(2048)[:, None] // 3) % 256
    add("big_2048x1536x3_smooth", np.stack([big] * 3, -1).astype(np.uint8))
    return s


def raw_scan_cases():
    g = torch.Generator().manual_seed(1)
    cases = {
        "rand_5x9": torch.randint(0, 256, (5, 9), generator=g, dtype=torch.uint8),
        "const_100x3073": torch.full((100, 3073), 9, dtype=torch.uint8),
        "two_vals_64x65": torch.randint(0, 2, (64, 65), generator=g, dtype=torch.uint8) * 200,
    }
    fib = [1, 1]
    while len(fib) < 26:
        fib.append(fib[-1] + fib[-2])
    vals = np.concatenate([np.full(f, i, np.uint8) for i, f in enumerate(fib)])
    np.random.default_rng(3).shuffle(vals)
    pad = (-len(vals)) % 1001
    vals = np.concatenate([vals, np.zeros(pad, np.uint8)])
    cases["fibonacci_hist"] = torch.from_numpy(vals.reshape(-1, 1001))
    return cases


def check():
    real = images()
    imgs = real + synthetic()
    pristine = [gg.clone() for _, gg, _, _ in imgs]
    fails = []
    sizes = {"main": [], **{r: [] for r in ROWS}}
    for i, (name, gg, src, kind) in enumerate(imgs):
        set_cfg("main")
        ref = encode(gg)
        ref_dec = decode(ref)
        ref_filt = inflate(ref)
        if kind != "synthetic":
            sizes["main"].append(len(ref))
        if not np.array_equal(ref_dec.reshape(src.shape), src):
            fails.append((name, "main_decode!=source"))
        for r in ROWS:
            set_cfg(r)
            pngs = [encode(gg) for _ in range(3)]
            png = pngs[0]
            try:
                dec = decode(png)
                filt = inflate(png)
            except Exception as e:  # noqa: BLE001
                fails.append((name, r, f"decode_error {e}"))
                continue
            if any(p != png for p in pngs[1:]):
                fails.append((name, r, "repeat_bytes_differ"))
            if not np.array_equal(dec.reshape(src.shape), src):
                fails.append((name, r, "decode!=source"))
            if dec.shape != ref_dec.shape or not np.array_equal(dec, ref_dec):
                fails.append((name, r, "decode!=main_decode"))
            if filt != ref_filt:
                fails.append((name, r, "filtered!=main_filtered"))
            b = base64.b64encode(png)
            if base64.b64decode(b) != png:
                fails.append((name, r, "b64"))
            if png == ref:
                fails.append((name, r, "gpu_path_not_used"))
            if kind != "synthetic":
                sizes[r].append(len(png))
        if not torch.equal(gg, pristine[i]):
            fails.append((name, "input_changed"))
        log(f"{name}: main {len(ref)} B; fails so far {len(fails)}")
    # raw scan round trips through the deflate stage itself
    for name, scan in raw_scan_cases().items():
        for r in ROWS + [1, 3]:
            enc._PNG_GPU_BLOCK_ROWS = r
            stream, adler = enc._png_gpu_deflate(scan.cuda())
            z = b"\x78\x01" + bytes(stream) + adler.to_bytes(4, "big")
            try:
                ok = zlib.decompress(z) == scan.numpy().tobytes()
            except Exception as e:  # noqa: BLE001
                ok = False
                log(f"raw {name} rows {r}: {e}")
            if adler != zlib.adler32(scan.numpy().tobytes()):
                ok = False
            if not ok:
                fails.append(("raw", name, r))
        log(f"raw {name}: fails so far {len(fails)}")
    enc._PNG_GPU_BLOCK_ROWS = DEFAULT_ROWS
    os.environ.pop("TRTLLM_PNG_GPU_DEFLATE", None)
    nreal = len(sizes["main"])
    mean_mb = {str(k): round(float(np.mean(v)) / 1e6, 5) for k, v in sizes.items() if v}
    out = {"passed": not fails and nreal > 0, "n_images": len(imgs), "n_real": nreal,
           "rows": ROWS, "fails": fails[:20], "mean_png_mb": mean_mb,
           "compared": "PIL verify+decode vs source uint8 and vs main path (TRTLLM_PNG_GPU_DEFLATE=0) "
                       "decode; inflated IDAT vs main filtered stream; CRC/Adler; 3 repeats identical; "
                       "input unchanged; raw deflate round trips incl. Fibonacci histogram"}
    print(json.dumps(out), flush=True)


def stats(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2], s[(9 * n) // 10], s[n // 10]


def timing(cfgs, rounds):
    real = images()
    timed = [x for x in real if x[3] == "timed"][:12] or real[:12]
    for _, gg, _, _ in timed[:2]:
        for c in cfgs:
            set_cfg(c)
            for _ in range(2):
                base64.b64encode(encode(gg))
    t = {str(c): [] for c in cfgs}
    mb = {str(c): [] for c in cfgs}
    rng = random.Random(42)
    for r in range(rounds):
        for _, gg, _, _ in timed:
            order = list(cfgs)
            rng.shuffle(order)
            for c in order:
                set_cfg(c)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                b = base64.b64encode(encode(gg))
                t[str(c)].append((time.perf_counter() - t0) * 1e3)
                if r == 0:
                    mb[str(c)].append(len(b) * 0.75 / 1e6)
        if r % 10 == 0:
            log(f"round {r}: " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in t.items()))
    set_cfg(DEFAULT_ROWS)
    res = {}
    for k in t:
        med, p90, p10 = stats(t[k])
        res[k] = {"median_ms": round(med, 3), "p90_ms": round(p90, 3), "p10_ms": round(p10, 3),
                  "png_mb": round(float(np.mean(mb[k])), 5)}
        log(f"{k}: {res[k]}")
    return res, len(timed)


def breakdown(rows):
    real = images()
    gg = real[0][1]
    set_cfg(rows)
    h, w, c = gg.shape[-3:]
    ts = {"filter": [], "deflate": [], "assemble": [], "b64": []}
    for i in range(40):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scan = enc._png_filter_scanlines(gg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        st, ad = enc._png_gpu_deflate(scan)
        t2 = time.perf_counter()
        png = enc._png_assemble(w, h, c, st, ad)
        t3 = time.perf_counter()
        base64.b64encode(png)
        t4 = time.perf_counter()
        if i >= 5:
            for k, a, b in (("filter", t0, t1), ("deflate", t1, t2), ("assemble", t2, t3), ("b64", t3, t4)):
                ts[k].append((b - a) * 1e3)
    return {k: round(float(np.median(v)), 3) for k, v in ts.items()}


def prof(rows):
    from torch.profiler import ProfilerActivity, profile
    real = images()
    gg = real[0][1]
    set_cfg(rows)
    h, w, c = gg.shape[-3:]
    for _ in range(5):
        encode(gg)
    scan = enc._png_filter_scanlines(gg)
    cnt = np.bincount(scan.view(-1).cpu().numpy(), minlength=257).tolist()
    t0 = time.perf_counter()
    for _ in range(50):
        enc._huffman_lengths(cnt, 15)
    log(f"huffman_lengths(257): {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
    ll = np.asarray(enc._huffman_lengths(cnt, 15), dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(50):
        enc._canonical_rev_codes(ll, 15)
    log(f"canonical(257): {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
    st, ad = enc._png_gpu_deflate(scan)
    for k in (1, 2, 4, 8, 16):
        enc._PNG_IDAT_PIECES = k
        t0 = time.perf_counter()
        for _ in range(50):
            enc._png_assemble(w, h, c, st, ad)
        log(f"assemble pieces {k}: {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
    enc._PNG_IDAT_PIECES = 8
    for r in (1024, 256, 64):
        set_cfg(r)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            enc._png_gpu_deflate(scan)
            ts.append(time.perf_counter() - t0)
        log(f"gpu_deflate rows {r}: {np.median(ts) * 1e3:.3f} ms")
    set_cfg(rows)
    t0 = time.perf_counter()
    for _ in range(50):
        zlib.crc32(st)
    log(f"crc32 whole stream 1 thread: {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
    t0 = time.perf_counter()
    for _ in range(50):
        b"".join([b"x", st, b"y"])
    log(f"join copy: {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            enc._png_gpu_deflate(scan)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=25), flush=True)


if MODE == "check":
    check()
elif MODE == "prof":
    prof(ROWS[0])
elif MODE == "sweep":
    for r in ROWS:
        log(f"breakdown rows {r}: {breakdown(r)}")
    res, n = timing(["main", "main_aa"] + ROWS, 20)
    print(json.dumps({"passed": True, "n_timed": n, "ab": res}), flush=True)
else:
    res, n = timing(["main", "main_aa", DEFAULT_ROWS], 60)
    aa = abs(res["main_aa"]["median_ms"] - res["main"]["median_ms"])
    aa90 = abs(res["main_aa"]["p90_ms"] - res["main"]["p90_ms"])
    new = res[str(DEFAULT_ROWS)]
    print(json.dumps({"passed": True, "n_timed": n, "rounds": 60, "ab": res,
                      "aa_noise_ms": {"median": round(aa, 3), "p90": round(aa90, 3)},
                      "gain_ms": {"median": round(res["main"]["median_ms"] - new["median_ms"], 3),
                                  "p90": round(res["main"]["p90_ms"] - new["p90_ms"], 3)}}), flush=True)
