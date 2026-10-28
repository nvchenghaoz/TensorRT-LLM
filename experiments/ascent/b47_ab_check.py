"""Branch 47 (idea 42 Part 1): untimed correctness + in-process interleaved A/B/A-A.

On the saved served FLUX.3 tensors (b47_capture.py dump: 12 timed prompts x2, 12 extra
check prompts, plus the client test request; served layout rebuilt from shape/strides):
 1. correctness (untimed, same 60-round sequence as the timing): with the module's
    DEFAULT deflate config (this tree's), every image, every round: Pillow-decoded pixels
    == source tensor == main-config decode; b64 == b64encode(PNG); PNG bytes == round 0.
    Also all three layouts (served, channels-first view, contiguous NHWC) give the same
    bytes. Saved inputs checked unchanged before each round.
 2. timing: configs main (main's constants), main_aa (same, A/A noise floor) and new
    (this tree's defaults) interleaved per image in shuffled order, 60 rounds x 12 timed
    images; median/p90 of encode+b64 wall, PNG MB, score at 5.3 / 9.3 ms per MB.
Last line: {"passed": ..., ...}.
"""
import base64
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

KEYS = ("_PNG_STRIP_ROWS", "_PNG_STRATEGY", "_PNG_MEM_LEVEL", "_PNG_DICT_BYTES",
        "_PNG_MAX_WORKERS")
NEW = {k: getattr(enc, k) for k in KEYS}
MAIN = {"_PNG_STRIP_ROWS": 64, "_PNG_STRATEGY": zlib.Z_FILTERED, "_PNG_MEM_LEVEL": 9,
        "_PNG_DICT_BYTES": 32768, "_PNG_MAX_WORKERS": 16}
CFGS = {"main": MAIN, "main_aa": MAIN, "new": NEW}
ROUNDS = 60


def setc(c):
    for k, v in c.items():
        setattr(enc, k, v)


def step(g):
    return base64.b64encode(enc.image_to_bytes(g, format="PNG"))


def layouts(g):
    if g.dim() == 3:
        chf = g.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    else:
        chf = g.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
    return {"real": g, "chfirst": chf, "nhwc": g.contiguous()}


def stats(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2], s[(9 * n) // 10], s[n // 10]


def main():
    dump = find_dump()
    log(f"dump {dump}; new config {NEW}")
    imgs = load_images(dump)
    timed = [x for x in imgs if x[1] == "timed"][:12]
    log(f"{len(imgs)} images ({len(timed)} timed), stride {imgs[0][4]}")
    pristine = [g.clone() for _, _, _, g, _ in imgs]

    # ---- 1. correctness
    fails = []
    setc(MAIN)
    main_dec = [np.asarray(Image.open(io.BytesIO(enc.image_to_bytes(g, format="PNG"))))
                for _, _, _, g, _ in imgs]
    setc(NEW)
    first = [None] * len(imgs)
    for r in range(ROUNDS):
        for i, (seq, kind, src, g, _) in enumerate(imgs):
            if not torch.equal(g, pristine[i]):
                fails.append((seq, r, "input_changed"))
            if r >= 3 and kind != "timed":
                continue  # all images 3 rounds; the 12 timed images every round
            b = step(g)
            png = base64.b64decode(b)
            if b != base64.b64encode(png):
                fails.append((seq, r, "b64"))
            if first[i] is None:
                first[i] = png
                ref = src[0].numpy() if src.dim() == 4 else src.numpy()
                dec = np.asarray(Image.open(io.BytesIO(png)))
                if not np.array_equal(dec, ref):
                    fails.append((seq, "decode!=source"))
                if not np.array_equal(dec, main_dec[i]):
                    fails.append((seq, "decode!=main_decode"))
                if not np.array_equal(main_dec[i], ref):
                    fails.append((seq, "main_decode!=source"))
                for ln, lg in layouts(g).items():
                    if enc.image_to_bytes(lg, format="PNG") != png:
                        fails.append((seq, ln, "layout_bytes_differ"))
            elif png != first[i]:
                fails.append((seq, r, "bytes!=round0"))
        if r % 10 == 0:
            log(f"correctness round {r}: fails {len(fails)}")
    log(f"correctness fails: {fails[:10]}")

    # ---- 2. interleaved A/B/A-A timing
    for _, _, _, g, _ in timed[:2]:
        for c in CFGS.values():
            setc(c)
            step(g)
    t = {n: [] for n in CFGS}
    mb = {n: [] for n in CFGS}
    rng = random.Random(42)
    for r in range(ROUNDS):
        for _, _, _, g, _ in timed:
            order = list(CFGS)
            rng.shuffle(order)
            for n in order:
                setc(CFGS[n])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                b = step(g)
                t[n].append((time.perf_counter() - t0) * 1e3)
                if r == 0:
                    mb[n].append(len(b) * 0.75 / 1e6)
        if r % 10 == 0:
            log(f"timing round {r}")
    setc(NEW)
    res = {}
    for n in CFGS:
        med, p90, p10 = stats(t[n])
        m = float(np.mean(mb[n]))
        d = m - float(np.mean(mb["main"]))
        res[n] = {"median_ms": round(med, 3), "p90_ms": round(p90, 3), "p10_ms": round(p10, 3),
                  "png_mb": round(m, 5), "d_mb": round(d, 5),
                  "score_med_x5.3": round(med + 5.3 * d, 3), "score_med_x9.3": round(med + 9.3 * d, 3),
                  "score_p90_x5.3": round(p90 + 5.3 * d, 3), "score_p90_x9.3": round(p90 + 9.3 * d, 3)}
        log(f"{n}: {res[n]}")
    aa_med = abs(res["main_aa"]["median_ms"] - res["main"]["median_ms"])
    aa_p90 = abs(res["main_aa"]["p90_ms"] - res["main"]["p90_ms"])
    gain_med = res["main"]["median_ms"] - res["new"]["median_ms"]
    gain_p90 = res["main"]["p90_ms"] - res["new"]["p90_ms"]
    out = {
        "passed": not fails,
        "n_images": len(imgs), "rounds": ROUNDS, "fails": fails[:20],
        "new_config": {k: v for k, v in NEW.items()},
        "ab": res,
        "aa_noise_ms": {"median": round(aa_med, 3), "p90": round(aa_p90, 3)},
        "gain_ms": {"median": round(gain_med, 3), "p90": round(gain_p90, 3)},
        "gain_beyond_aa": gain_med > aa_med and gain_p90 > aa_p90,
        "compared": "default-config PNG decoded pixels vs saved served tensor and vs main-config "
                    "decode; b64 round trip; bytes vs round 0 and across 3 layouts",
    }
    print(json.dumps(out), flush=True)


main()
