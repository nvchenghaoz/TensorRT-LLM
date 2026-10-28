"""Branch 47 (idea 42 Part 1): strip-deflate config sweep on saved served tensors.

Loads the uint8 tensors saved by b47_capture.py (TRTLLM_VISUALGEN_DUMP_DIR), rebuilds
each on the GPU in its served layout (shape + strides), and times the server's
b64_json encode, base64.b64encode(image_to_bytes(img, "PNG")), for several deflate
configs of tensorrt_llm.media.encoding's strip writer, interleaved in one process
(config order shuffled per image), idle and under an ~80-process CPU load. "main" is
run twice (main, main_aa) as the A/A noise floor. Score = encode+b64 wall
+ (PNG MB - main PNG MB) x {5.3, 9.3} ms. Correctness: every saved tensor, every
config, all three layouts: Pillow-decoded pixels == source == main's decode,
b64 == b64encode(PNG), 3 repeats byte-identical.
Usage: python b47_sweep.py <dump_dir>
"""
import base64
import glob
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
import zlib

import numpy as np
from PIL import Image

CONFIGS = {
    "main": dict(rows=64, strategy=zlib.Z_FILTERED, mem=9, dict=32768, workers=16),
    "main_aa": dict(rows=64, strategy=zlib.Z_FILTERED, mem=9, dict=32768, workers=16),
    "filt_r32_w16": dict(rows=32, strategy=zlib.Z_FILTERED, mem=9, dict=32768, workers=16),
    "filt_r32_w32": dict(rows=32, strategy=zlib.Z_FILTERED, mem=9, dict=32768, workers=32),
    "filt_r16_w32": dict(rows=16, strategy=zlib.Z_FILTERED, mem=9, dict=32768, workers=32),
    "filt_r32_w32_d8k": dict(rows=32, strategy=zlib.Z_FILTERED, mem=9, dict=8192, workers=32),
    "rle_r64_w16": dict(rows=64, strategy=zlib.Z_RLE, mem=9, dict=0, workers=16),
    "rle_r32_w32": dict(rows=32, strategy=zlib.Z_RLE, mem=9, dict=0, workers=32),
    "rle_r16_w32": dict(rows=16, strategy=zlib.Z_RLE, mem=9, dict=0, workers=32),
    "huff_r64_w16": dict(rows=64, strategy=zlib.Z_HUFFMAN_ONLY, mem=9, dict=0, workers=16),
    "huff_r32_w32": dict(rows=32, strategy=zlib.Z_HUFFMAN_ONLY, mem=9, dict=0, workers=32),
}
WEIGHTS = (5.3, 9.3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_cfg(enc, c):
    enc._PNG_STRIP_ROWS = c["rows"]
    enc._PNG_STRATEGY = c["strategy"]
    enc._PNG_MEM_LEVEL = c["mem"]
    enc._PNG_DICT_BYTES = c["dict"]
    enc._PNG_MAX_WORKERS = c["workers"]


def busy(stop):
    x = 0
    while not stop.is_set():
        for _ in range(10000):
            x += 1


def stats(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2], s[min(n - 1, int(0.9 * n))], s[n // 10]


def load_images(dump):
    import torch
    out = []
    for p in sorted(glob.glob(os.path.join(dump, "img_*.pt"))):
        d = torch.load(p)
        src = d["data"]
        g = torch.empty_strided(d["shape"], d["stride"], dtype=torch.uint8, device="cuda")
        g.copy_(src.cuda())
        assert list(g.stride()) == d["stride"]
        out.append((d["seq"], src, g, d["stride"]))
    return out


def layouts(g):
    return {
        "real": g,
        "chfirst": g.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1),
        "nhwc": g.contiguous(),
    }


def decode(png):
    return np.asarray(Image.open(io.BytesIO(png)))


def sweep(enc, imgs, names, iters, tag):
    import torch
    t = {n: [] for n in names}
    sizes = {n: [] for n in names}
    rng = random.Random(1234)
    for it in range(iters):
        for g in imgs:
            order = list(names)
            rng.shuffle(order)
            for n in order:
                set_cfg(enc, CONFIGS[n])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                b = base64.b64encode(enc.image_to_bytes(g, format="PNG"))
                t[n].append((time.perf_counter() - t0) * 1e3)
                if it == 0:
                    sizes[n].append(len(b) * 3 / 4 / 1e6)
        if it % 10 == 0:
            log(f"{tag} iter {it}/{iters}")
    res = {}
    main_mb = float(np.mean(sizes["main"]))
    for n in names:
        med, p90, p10 = stats(t[n])
        mb = float(np.mean(sizes[n]))
        res[n] = {"median_ms": round(med, 3), "p90_ms": round(p90, 3), "p10_ms": round(p10, 3),
                  "png_mb": round(mb, 5), "d_mb": round(mb - main_mb, 5)}
        for w in WEIGHTS:
            res[n][f"score_med_x{w}"] = round(med + (mb - main_mb) * w, 3)
            res[n][f"score_p90_x{w}"] = round(p90 + (mb - main_mb) * w, 3)
        log(f"{tag} {n:18s} {json.dumps(res[n])}")
    return res


def main(dump, stamps_timed=None):
    os.environ.pop("TRTLLM_VISUALGEN_DUMP_DIR", None)
    import torch
    from tensorrt_llm.media import encoding as enc

    reqs = json.load(open(os.path.join(dump, "requests.json")))
    kinds = {r["seq"]: r["kind"] for r in reqs}
    allimgs = load_images(dump)
    log(f"{len(allimgs)} images, strides {sorted({str(s) for _, _, _, s in allimgs})}")

    # ---- correctness: every saved tensor, every config, all three layouts
    fails = []
    for seq, src, g, _ in allimgs:
        ref = src[0].numpy() if src.dim() == 4 else src.numpy()
        set_cfg(enc, CONFIGS["main"])
        main_png = enc.image_to_bytes(g, format="PNG")
        main_dec = decode(main_png)
        if not np.array_equal(main_dec, ref):
            fails.append((seq, "main", "main_decode!=source"))
        for n, c in CONFIGS.items():
            set_cfg(enc, c)
            outs = []
            for lname, lg in layouts(g).items():
                reps = [enc.image_to_bytes(lg, format="PNG") for _ in range(3)]
                if len(set(reps)) != 1:
                    fails.append((seq, n, lname, "repeat_bytes_differ"))
                outs.append(reps[0])
                b64_server = base64.b64encode(enc.image_to_bytes(lg, format="PNG"))
                if b64_server != base64.b64encode(reps[0]) or base64.b64decode(b64_server) != reps[0]:
                    fails.append((seq, n, lname, "b64"))
            if len(set(outs)) != 1:
                fails.append((seq, n, "layout_bytes_differ"))
            dec = decode(outs[0])
            if not (np.array_equal(dec, ref) and np.array_equal(dec, main_dec)):
                fails.append((seq, n, "decode_mismatch"))
        log(f"correct seq {seq} ({kinds.get(seq)}): fails so far {len(fails)}")
    correctness = {"passed": not fails, "n_images": len(allimgs), "configs": len(CONFIGS),
                   "fails": fails[:20]}
    log(f"correctness {json.dumps(correctness)}")

    # ---- timing set: the 12 distinct timed images (first pass) in their served layout
    timed = [g for seq, _, g, _ in allimgs if kinds.get(seq) == "timed"][:12]
    names = list(CONFIGS)
    for g in timed[:2]:  # warm every pool / path
        for n in names:
            set_cfg(enc, CONFIGS[n])
            enc.image_to_bytes(g, format="PNG")
    idle = sweep(enc, timed, names, 30, "idle")

    stop = mp.Event()
    procs = [mp.Process(target=busy, args=(stop,), daemon=True) for _ in range(80)]
    for p in procs:
        p.start()
    time.sleep(2)
    try:
        load = sweep(enc, timed, names, 15, "load80")
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def beats(n):
        ok = True
        for r in (idle, load):
            for w in WEIGHTS:
                ok &= r[n][f"score_med_x{w}"] < r["main"][f"score_med_x{w}"]
                ok &= r[n][f"score_p90_x{w}"] < r["main"][f"score_p90_x{w}"]
        return ok

    winners = [n for n in names if n not in ("main", "main_aa") and beats(n)]
    winners.sort(key=lambda n: idle[n]["score_med_x9.3"] + load[n]["score_med_x9.3"])
    log(f"configs beating main at both weights, median+p90, idle+load: {winners}")
    chosen = winners[0] if winners else None

    # ---- per-layout time for main and the chosen config
    lay = {}
    for n in ["main"] + ([chosen] if chosen else []):
        set_cfg(enc, CONFIGS[n])
        for lname in ("real", "chfirst", "nhwc"):
            ts = []
            for _ in range(5):
                for g in timed:
                    lg = layouts(g)[lname]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    base64.b64encode(enc.image_to_bytes(lg, format="PNG"))
                    ts.append((time.perf_counter() - t0) * 1e3)
            med, p90, _ = stats(ts)
            lay[f"{n}/{lname}"] = {"median_ms": round(med, 3), "p90_ms": round(p90, 3)}
    log(f"layouts {json.dumps(lay)}")
    print(json.dumps({"passed": correctness["passed"], "chosen": chosen, "winners": winners,
                      "served_stamps_timed": stamps_timed and {
                          k: {"median": round(stats(v)[0], 3), "p90": round(stats(v)[1], 3)}
                          for k, v in stamps_timed.items()},
                      "idle": idle, "load80": load, "layouts": lay,
                      "correctness": correctness}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
