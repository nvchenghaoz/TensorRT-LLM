"""Fast PNG writer on real Flux3 Klein NVFP4 outputs (frozen workload shape).

Generates >=8 images in process (tensorrt_llm.VisualGen, variant nvfp4,
text-to-image 1024x1024, 16 steps, fixed seeds), keeps the pipeline's raw
uint8 output tensors with their native strides, and for each one encodes
with the fast path and with TRTLLM_FAST_PNG=0 (main's Pillow path), reporting
decoded pixels/mode/size equality, filtered-stream equality, the byte-size
ratio (<= 1.03) and byte-identical output over 3 repeated calls.
Usage: python png_real_check.py <checkpoint>
"""

import base64
import io
import json
import os
import sys
import time
import zlib

import numpy as np
from PIL import Image

CKPT = sys.argv[1]

FALLBACK_PROMPTS = [
    "A red fox standing in fresh snow at sunrise",
    "A bowl of ramen with a soft-boiled egg on a wooden table, studio lighting",
    "A futuristic city skyline at night with neon signs reflected in rain puddles",
    "An old lighthouse on a rocky coast during a storm, dramatic clouds",
    "A watercolor painting of a hummingbird drinking from a purple flower",
    "A vintage red bicycle leaning against a yellow brick wall",
    "A close-up portrait of an elderly fisherman with a weathered face",
    "A cozy library with tall bookshelves and a fireplace, warm light",
    "A sign that says 'OPEN 24 HOURS' above a diner door at dusk",
    "Three green apples and a glass of water on a white tablecloth",
]


def find_prompts():
    """Benchmark/check prompts when reachable, else the fixed fallback list."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "prompt" and isinstance(v, str):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for p in ("experiments/ascent-benchmark/reference.json",):
        try:
            with open(p) as f:
                walk(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"prompts: cannot read {p}: {e}", flush=True)
    uniq = list(dict.fromkeys(found))
    if len(uniq) >= 8:
        return uniq[:10], "experiments/ascent-benchmark/reference.json"
    return FALLBACK_PROMPTS, "fallback_fixed_prompts"


def chunks(png):
    pos, out = 8, []
    while pos < len(png):
        n = int.from_bytes(png[pos : pos + 4], "big")
        tag = png[pos + 4 : pos + 8]
        payload = png[pos + 8 : pos + 8 + n]
        crc = int.from_bytes(png[pos + 8 + n : pos + 12 + n], "big")
        assert zlib.crc32(tag + payload) == crc, f"bad CRC in {tag}"
        out.append((tag, payload))
        pos += 12 + n
    return out


def filtered(png):
    return zlib.decompress(b"".join(p for t, p in chunks(png) if t == b"IDAT"))


def decoded(png):
    im = Image.open(io.BytesIO(png))
    im.load()
    return im.mode, im.size, np.asarray(im)


def main():
    import torch

    from tensorrt_llm import VisualGen, VisualGenArgs
    from tensorrt_llm.media import encoding as enc

    prompts, prompt_src = find_prompts()
    print(f"{len(prompts)} prompts from {prompt_src}", flush=True)

    config = VisualGenArgs()
    config.pipeline_config["variant"] = "nvfp4"
    print("loading VisualGen", flush=True)
    t0 = time.time()
    model = VisualGen(model=CKPT, args=config)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    def fast(img):
        os.environ.pop("TRTLLM_FAST_PNG", None)
        return enc.image_to_bytes(img, format="PNG")

    def pillow(img):
        os.environ["TRTLLM_FAST_PNG"] = "0"
        try:
            return enc.image_to_bytes(img, format="PNG")
        finally:
            os.environ.pop("TRTLLM_FAST_PNG", None)

    results, failures, ratios = {}, [], []
    for i, prompt in enumerate(prompts):
        params = model.default_params
        params.height, params.width, params.seed = 1024, 1024, 1000 + i
        params.num_inference_steps = 16
        t0 = time.time()
        out = model.generate(inputs=prompt, params=params)
        img = out.image  # raw pipeline uint8 tensor, native strides
        info = {
            "gen_s": round(time.time() - t0, 2),
            "shape": list(img.shape),
            "stride": list(img.stride()),
            "dtype": str(img.dtype),
            "device": str(img.device),
            "contiguous": img.is_contiguous(),
        }
        # also the openai_server path: _normalize_image_output -> same tensor
        ref = pillow(img)
        new = fast(img)
        reps = [fast(img) for _ in range(3)]
        ref_reps = [pillow(img) for _ in range(2)]
        mr, sr, pr = decoded(ref)
        mn, sn, pn = decoded(new)
        ratio = len(new) / len(ref)
        ok = {
            "is_cuda_uint8": img.is_cuda and img.dtype == torch.uint8,
            "fast_path_used": new != ref,  # Pillow bytes would be identical
            "decoded_pixels_equal": pr.dtype == pn.dtype and np.array_equal(pr, pn),
            "mode_equal": mr == mn,
            "size_equal": sr == sn,
            "filtered_equal": filtered(new) == filtered(ref),
            "deterministic_3_repeats": all(r == new for r in reps),
            "b64_deterministic": len({base64.b64encode(r) for r in reps + [new]}) == 1,
            "pillow_reference_stable": all(r == ref for r in ref_reps),
            "ratio_le_1.03": ratio <= 1.03,
        }
        info.update(ok, mode=mn, size=list(sn), main_bytes=len(ref), fast_bytes=len(new),
                    ratio=round(ratio, 5))
        ratios.append(ratio)
        results[f"img{i}"] = info
        if not all(ok.values()):
            failures.append(f"img{i}")
        print(f"img{i}", json.dumps(info), flush=True)

    try:
        model.shutdown()
    except Exception as e:  # noqa: BLE001
        print(f"shutdown: {e}", flush=True)
    print(json.dumps({
        "passed": not failures and len(results) >= 8,
        "failures": failures,
        "n_images": len(results),
        "prompt_source": prompt_src,
        "compared_against": "TRTLLM_FAST_PNG=0 (main's Pillow compress_level=1 path) on the same raw pipeline tensor",
        "size_ratio_max": round(max(ratios), 5),
        "size_ratio_mean": round(sum(ratios) / len(ratios), 5),
    }))


if __name__ == "__main__":
    main()
