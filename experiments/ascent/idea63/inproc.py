"""Idea 63: FLUX.3 i2i reference PNG fast decode (png_fast), in-process.

python inproc.py <mode> [checkpoint]
  unit  : no model. Tree _load_image_pixels vs main's (copied verbatim in pngtest) on the
          two frozen refs and synthetic PNGs (filters 0-4, cycles, multi-IDAT, levels 0/6/9,
          crop-path sizes, 4096^2), values + strides bitwise, and that the fast path was
          taken; fallback inputs (RGBA, palette, gray, 16-bit, Adam7, eXIf/tEXt, JPEG,
          truncated, bad CRC, bad zlib, filter byte 5, split IDAT runs, oversize) return
          None from png_fast and give PIL's pixels or PIL's exception; kill switch = main.
  check : served pipeline (engine.json, compile, CUDA graphs, 2048 warmup). Untimed
          t2i -> i2i(A) -> t2i -> i2i(B) sequence, N_ROUNDS rounds; each i2i runs the fast
          arm and the TRTLLM_FLUX3_PNG_FAST=0 arm (order alternating), stopped at the first
          transformer call. Allocator poisoned before every call. Reference pixels (+strides),
          VAE latents and every first-transformer tensor kwarg bitwise vs round 0 and vs the
          kill-switch arm; t2i kwargs vs round 0. Two full i2i requests per ref (fast vs kill
          switch): output image bitwise. Triton kernel caches of png_fast must not grow after
          warmup (no compile inside a request).
  pbench: method 544. One process, one pipeline; arm A = main's _load_image_pixels (copied),
          arm B = this tree's. Per iteration both arms on the same input, AB on even / BA on
          odd iterations. Span = infer entry -> first transformer call after cuda sync.
"""

import hashlib
import json
import os
import statistics
import struct
import sys
import time
import types
import zlib
from io import BytesIO

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pngtest import chunk, find_refs, log, make_png, pil_rgb, ref_load_image_pixels  # noqa: E402

MODE = sys.argv[1]
CKPT = sys.argv[2] if len(sys.argv) > 2 else None
PROMPTS = {"A": ("Make the sky a vivid sunset orange, keep everything else unchanged", 1234),
           "B": ("Turn the scene into a snowy winter day", 4321)}
T2I = ("A red fox standing in fresh snow at sunrise, photograph", 77)


def same(a, b):
    return bool(a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
                and torch.equal(a, b))


def stats(v):
    v = list(v)
    q = statistics.quantiles(sorted(v), n=10)
    med = statistics.median(v)
    return {"n": len(v), "mean": statistics.fmean(v), "median": med, "p10": q[0], "p90": q[-1],
            "p10_p90_over_median": (q[-1] - q[0]) / med}


def loadavg1():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:  # noqa: BLE001
        return -1.0


# ----------------------------------------------------------------------------- unit
def adam7_png(px):
    h, w = px.shape[:2]
    passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
              (0, 1, 1, 2))
    raw = b""
    for x0, y0, dx, dy in passes:
        sub = px[y0::dy, x0::dx]
        if sub.size == 0:
            continue
        for row in sub:
            raw += b"\x00" + np.ascontiguousarray(row).tobytes()
    return make_png(px, raw_override=raw, interlace=1)


def rgb16_png(h, w, rng):
    raw = b"".join(b"\x00" + rng.integers(0, 256, 6 * w, dtype=np.uint8).tobytes()
                   for _ in range(h))
    return make_png(np.zeros((h, w, 3), np.uint8), raw_override=raw, depth=16)


def fallback_cases(base, rng):
    from PIL import Image

    out = {}
    small = Image.fromarray(np.ascontiguousarray(base[:776, :1000]))
    b = BytesIO()
    small.save(b, format="JPEG", quality=90)
    out["jpeg"] = b.getvalue()
    rgba = small.copy()
    rgba.putalpha(Image.linear_gradient("L").resize(small.size))
    b = BytesIO()
    rgba.save(b, format="PNG")
    out["png_rgba"] = b.getvalue()
    b = BytesIO()
    small.convert("P", palette=Image.Palette.ADAPTIVE).save(b, format="PNG")
    out["png_palette"] = b.getvalue()
    b = BytesIO()
    small.convert("L").save(b, format="PNG")
    out["png_gray"] = b.getvalue()
    out["png_rgb16"] = rgb16_png(100, 130, rng)
    out["png_adam7"] = adam7_png(np.ascontiguousarray(base[:203, :317]))
    for orient in (1, 6):
        exif = Image.Exif()
        exif[0x0112] = orient
        b = BytesIO()
        small.save(b, format="PNG", exif=exif.tobytes())
        out[f"png_exif_orient{orient}"] = b.getvalue()
    px = np.ascontiguousarray(base[:120, :160])
    out["png_text_chunk"] = make_png(px, pre_chunks=[(b"tEXt", b"Comment\x00hello")])
    out["png_text_after_idat"] = make_png(px, post_chunks=[(b"tEXt", b"Comment\x00hello")])
    good = make_png(px, level=6, idat_size=4096)
    out["png_truncated_mid_idat"] = good[: len(good) // 2]
    out["png_truncated_no_iend"] = good[:-12]
    # Bad IHDR CRC.
    bad = bytearray(good)
    bad[8 + 8 + 13] ^= 0xFF
    out["png_bad_ihdr_crc"] = bytes(bad)
    # Well-formed chunks, zlib stream short by 1000 bytes of output.
    raw = b"".join(b"\x00" + np.ascontiguousarray(r).tobytes() for r in px)
    out["png_zlib_short"] = make_png(px, raw_override=raw[:-1000])
    out["png_zlib_long"] = make_png(px, raw_override=raw + b"\x00" * 481)
    z = zlib.compress(raw, 6)
    ihdr = struct.pack(">IIBBBBB", 160, 120, 8, 2, 0, 0, 0)
    out["png_zlib_trailing_garbage"] = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                                        + chunk(b"IDAT", z + b"garbage") + chunk(b"IEND", b""))
    out["png_zlib_corrupt"] = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                               + chunk(b"IDAT", z[:200] + bytes(64) + z[264:])
                               + chunk(b"IEND", b""))
    bad_adler = bytearray(z)
    bad_adler[-1] ^= 0x5A
    out["png_zlib_bad_adler"] = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                                 + chunk(b"IDAT", bytes(bad_adler)) + chunk(b"IEND", b""))
    rawf = bytearray(raw)
    rawf[5 * (1 + 3 * 160)] = 5
    out["png_filter_byte5"] = make_png(px, raw_override=bytes(rawf))
    zz = zlib.compress(raw, 6)
    out["png_split_idat_runs"] = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                                  + chunk(b"IDAT", zz[:500]) + chunk(b"tIME", bytes(7))
                                  + chunk(b"IDAT", zz[500:]) + chunk(b"IEND", b""))
    out["png_tiny_15x40"] = make_png(np.ascontiguousarray(base[:40, :15]))
    out["png_wide_4100x16"] = make_png(rng.integers(0, 256, (16, 4100, 3), dtype=np.uint8),
                                       level=1)
    return out


def unit():
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3
    from tensorrt_llm._torch.visual_gen.models.flux3 import png_fast

    dev = torch.device("cuda")
    refs = find_refs()
    rng = np.random.default_rng(7)
    base = pil_rgb(refs["A"])
    ok = True
    res = {"fast": {}, "fallback": {}}

    def cmp_loader(name, data, expect_fast):
        nonlocal ok
        fast = png_fast.decode_png_rgb8_to_device(data, dev) is not None
        row = {"fast_path": fast}
        try:
            r = ref_load_image_pixels(data, device=dev)
            rexc = None
        except Exception as exc:  # noqa: BLE001
            r, rexc = None, (type(exc).__name__, str(exc))
        try:
            c = flux3._load_image_pixels(data, device=dev)
            c2 = flux3._load_image_pixels(data, device=dev)
            cexc = None
        except Exception as exc:  # noqa: BLE001
            c, cexc = None, (type(exc).__name__, str(exc))
        if rexc or cexc:
            row["exception_ref"], row["exception_cand"] = rexc, cexc
            good = rexc == cexc
        else:
            good = same(r, c) and same(c, c2)
            row["shape"], row["stride"] = list(c.shape), list(c.stride())
        # Kill switch arm == main.
        os.environ["TRTLLM_FLUX3_PNG_FAST"] = "0"
        try:
            k = flux3._load_image_pixels(data, device=dev)
            kexc = None
        except Exception as exc:  # noqa: BLE001
            k, kexc = None, (type(exc).__name__, str(exc))
        finally:
            os.environ.pop("TRTLLM_FLUX3_PNG_FAST")
        good &= (kexc == rexc) if (rexc or kexc) else same(k, r)
        good &= fast == expect_fast
        row["ok"] = bool(good)
        ok &= good
        return row

    for k, d in refs.items():
        res["fast"][f"ref_{k}"] = cmp_loader(f"ref_{k}", d, True)
        log(f"ref_{k}", json.dumps(res["fast"][f"ref_{k}"]))
    sizes = ((16, 16), (17, 16), (37, 23), (300, 211), (2040, 2033), (2048, 2048), (1999, 2047),
             (4096, 4096), (2047, 4096), (4096, 2050))
    for (w, h) in sizes:
        if w <= base.shape[1] and h <= base.shape[0]:
            photo = np.ascontiguousarray(base[:h, :w])
        else:
            photo = np.ascontiguousarray(np.tile(base, (2, 2, 1))[:h, :w])
        noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        big = w * h > 2048 * 2048
        filts = ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)) if not big else ((0, 1, 2, 3, 4),)
        levels = ((0, None), (6, 8192), (9, None)) if not big else ((6, 65536),)
        for src_name, src in (("photo", photo), ("noise", noise)):
            if big and src_name == "noise":
                continue
            for filt in filts:
                for level, isz in levels:
                    name = f"{src_name}_{w}x{h}_f{''.join(map(str, filt))}_l{level}_i{isz}"
                    d = make_png(src, filters=filt, level=level, idat_size=isz)
                    res["fast"][name] = cmp_loader(name, d, True)
                    if not res["fast"][name]["ok"]:
                        log("FAIL", name, json.dumps(res["fast"][name]))
        log(f"size {w}x{h} done, ok so far={ok}")
    # Ancillary chunks that stay on the fast path.
    px = np.ascontiguousarray(base[:64, :80])
    anc = [(b"gAMA", struct.pack(">I", 45455)), (b"sRGB", b"\x00"), (b"pHYs", bytes(9)),
           (b"cHRM", bytes(32))]
    res["fast"]["ancillary"] = cmp_loader("ancillary", make_png(px, pre_chunks=anc,
                                                                post_chunks=[(b"tIME", bytes(7))]),
                                          True)
    for name, d in fallback_cases(base, rng).items():
        res["fallback"][name] = cmp_loader(name, d, False)
        log(name, json.dumps(res["fallback"][name]))
    # CPU device: unchanged PIL path.
    rc = ref_load_image_pixels(refs["A"], device="cpu")
    cc = flux3._load_image_pixels(refs["A"], device="cpu")
    res["cpu_ref_A_bitwise"] = same(rc, cc)
    ok &= res["cpu_ref_A_bitwise"]
    # A, B, A on cuda with no sync between calls.
    outs = [flux3._load_image_pixels(refs[k], device=dev) for k in "ABA"]
    torch.cuda.synchronize()
    ra, rb = ref_load_image_pixels(refs["A"], device=dev), ref_load_image_pixels(refs["B"], device=dev)
    res["ABA_no_sync"] = same(outs[0], ra) and same(outs[1], rb) and same(outs[2], ra)
    ok &= res["ABA_no_sync"]
    n_fast = sum(1 for v in res["fast"].values() if v["ok"])
    n_fb = sum(1 for v in res["fallback"].values() if v["ok"])
    summary = {"passed": bool(ok), "fast_cases_ok": f"{n_fast}/{len(res['fast'])}",
               "fallback_cases_ok": f"{n_fb}/{len(res['fallback'])}",
               "failed": [k for k, v in {**res["fast"], **res["fallback"]}.items() if not v["ok"]],
               "libdeflate": png_fast._libdeflate() is not None,
               "compared": "tree _load_image_pixels vs main's (copied verbatim) and vs "
                           "TRTLLM_FLUX3_PNG_FAST=0: values+strides bitwise or identical "
                           "exception (type, message); fast path taken/not taken as expected; "
                           "repeated calls; A/B/A without sync; cpu device"}
    log(json.dumps({k: v for k, v in res.items() if k not in ("fast", "fallback")}))
    print(json.dumps({**summary, "fallback": res["fallback"],
                      "refs": {k: res["fast"][f"ref_{k}"] for k in "AB"},
                      "ABA_no_sync": res["ABA_no_sync"], "cpu": res["cpu_ref_A_bitwise"]}))


# ----------------------------------------------------------------------------- pipeline
class _Stop(Exception):
    pass


def load_pipe():
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    tcc = dict(engine.pop("torch_compile_config", {}))
    tcc.update({"enable": True, "enable_fullgraph": False, "enable_autotune": False})
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        torch_compile_config=tcc,
        compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1],
                            "modes": ["t2i", "i2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"pipeline loaded + warmed up in {time.time() - t0:.0f} s; engine={json.dumps(engine)}")
    return pipe


def make_req(prompt, seed, ref=None):
    params = types.SimpleNamespace(
        height=2048, width=2048, num_inference_steps=16, guidance_scale=None, seed=seed,
        negative_prompt=None, extra_params={},
        image_reference=[types.SimpleNamespace(content=ref)] if ref is not None else [])
    return types.SimpleNamespace(prompt=prompt, params=params)


class Probe:
    def __init__(self, pipe):
        self.orig = pipe.transformer.forward
        self.stop = False
        self.capture = False
        self.t_host = self.t_gpu = None
        self.kwargs = None
        pipe.transformer.forward = self

    def __call__(self, *args, **kwargs):
        if self.stop:
            self.t_host = time.perf_counter()
            torch.cuda.synchronize()
            self.t_gpu = time.perf_counter()
            if self.capture:
                self.kwargs = {k: (v.clone(), v.stride()) for k, v in kwargs.items()
                               if isinstance(v, torch.Tensor)}
            raise _Stop()
        return self.orig(*args, **kwargs)


def span(pipe, probe, req):
    probe.stop = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        pipe.infer(req)
        raise RuntimeError("infer finished without reaching the denoiser")
    except _Stop:
        pass
    finally:
        probe.stop = False
    return (probe.t_gpu - t0) * 1e3, (probe.t_host - t0) * 1e3


def triton_cache_sizes():
    from tensorrt_llm._torch.visual_gen.models.flux3 import png_fast

    out = {}
    for name in ("_png_skew_kernel", "_png_wavefront_kernel", "_png_unskew_kernel"):
        fn = getattr(png_fast, name)
        n = 0
        dc = getattr(fn, "device_caches", None)
        if dc is not None:
            for v in dc.values():
                n += len(v[0]) if isinstance(v, tuple) else len(v)
        else:
            n = sum(len(v) for v in getattr(fn, "cache", {}).values())
        out[name] = n
    return out


def poison():
    # Fill freshly freed allocator blocks with 0xFF so torch.empty outputs start as garbage.
    blocks = [torch.full((s,), 255, dtype=torch.uint8, device="cuda")
              for s in (64 << 20, 32 << 20, 32 << 20, 16 << 20, 16 << 20)]
    del blocks


def check():
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    n_rounds = int(os.environ.get("N_ROUNDS", "50"))
    pipe = load_pipe()
    caches0 = triton_cache_sizes()
    log(f"triton caches after warmup: {caches0}")
    probe = Probe(pipe)
    probe.capture = True
    refs = find_refs()
    sha0 = {k: hashlib.sha256(v).hexdigest() for k, v in refs.items()}
    cand_load = flux3._load_image_pixels
    rec = {}

    def rec_load(source, *, device):
        out = cand_load(source, device=device)
        rec["pixels"] = (out.clone(), out.stride())
        return out

    flux3._load_image_pixels = rec_load
    orig_encode = pipe.vae.encode

    def rec_encode(x):
        out = orig_encode(x)
        rec.setdefault("latents", []).append(out.clone())
        return out

    pipe.vae.encode = rec_encode

    def run(kind, arm):
        if arm == "kill":
            os.environ["TRTLLM_FLUX3_PNG_FAST"] = "0"
        else:
            os.environ.pop("TRTLLM_FLUX3_PNG_FAST", None)
        rec.clear()
        poison()
        req = make_req(*T2I) if kind == "t2i" else make_req(PROMPTS[kind][0], PROMPTS[kind][1],
                                                            refs[kind])
        span(pipe, probe, req)
        os.environ.pop("TRTLLM_FLUX3_PNG_FAST", None)
        snap = {"kw": probe.kwargs}
        if kind != "t2i":
            snap["pixels"] = rec["pixels"]
            snap["latents"] = rec["latents"]
        return snap

    def eq(a, b):
        good = a["kw"].keys() == b["kw"].keys() and all(
            same(a["kw"][n][0], b["kw"][n][0]) and a["kw"][n][1] == b["kw"][n][1] for n in a["kw"])
        if "pixels" in a:
            good &= same(a["pixels"][0], b["pixels"][0]) and a["pixels"][1] == b["pixels"][1]
            good &= len(a["latents"]) == len(b["latents"]) and all(
                same(x, y) for x, y in zip(a["latents"], b["latents"]))
        return bool(good)

    first = {}
    bad = []
    n_calls = 0
    for r in range(n_rounds):
        for kind in ("t2i", "A", "t2i", "B"):
            arms = ["fast"] if kind == "t2i" else (["fast", "kill"] if r % 2 == 0 else
                                                   ["kill", "fast"])
            snaps = {arm: run(kind, arm) for arm in arms}
            n_calls += len(arms)
            if kind not in first:
                first[kind] = snaps["fast"]
                if kind != "t2i" and not eq(snaps["fast"], snaps["kill"]):
                    bad.append((r, kind, "fast_vs_kill"))
                continue
            if not eq(snaps["fast"], first[kind]):
                bad.append((r, kind, "fast_vs_round0"))
            if kind != "t2i" and not eq(snaps["kill"], snaps["fast"]):
                bad.append((r, kind, "fast_vs_kill"))
        if r % 10 == 0:
            log(f"round {r}: calls={n_calls} bad={bad[:5]}")
    log(f"sequence: {n_rounds} rounds, {n_calls} calls, bad={bad[:10]}")
    inputs_compared = sorted(first["A"]["kw"].keys())
    # Full requests: output image fast vs kill switch.
    probe.capture = False
    full = {}
    for kind in ("A", "B"):
        imgs = {}
        for arm in ("fast", "kill"):
            if arm == "kill":
                os.environ["TRTLLM_FLUX3_PNG_FAST"] = "0"
            out = pipe.infer(make_req(PROMPTS[kind][0], PROMPTS[kind][1], refs[kind]))
            torch.cuda.synchronize()
            os.environ.pop("TRTLLM_FLUX3_PNG_FAST", None)
            imgs[arm] = out.image.clone()
        full[kind] = same(imgs["fast"], imgs["kill"])
        log(f"full request {kind}: image fast==kill {full[kind]}")
    caches1 = triton_cache_sizes()
    sha1 = {k: hashlib.sha256(v).hexdigest() for k, v in refs.items()}
    passed = (not bad) and all(full.values()) and caches0 == caches1 and sha0 == sha1 \
        and all(v > 0 for v in caches0.values())
    print(json.dumps({
        "passed": bool(passed), "repeated_execution_passed": not bad, "rounds": n_rounds,
        "calls": n_calls, "bad": bad[:20], "full_request_image_fast_eq_kill": full,
        "triton_cache_after_warmup": caches0, "triton_cache_at_end": caches1,
        "no_triton_compile_in_requests": caches0 == caches1,
        "refs_pristine": sha0 == sha1, "first_transformer_kwargs_compared": inputs_compared,
        "compared": "served pipeline, untimed t2i->i2i(A)->t2i->i2i(B) x rounds; each i2i in "
                    "fast and TRTLLM_FLUX3_PNG_FAST=0 arms (order alternating), allocator "
                    "poisoned with 0xFF before every call; reference pixels (+strides), VAE "
                    "latents, every first-transformer tensor kwarg bitwise vs round 0 and vs "
                    "the kill-switch arm; full i2i output image fast vs kill switch",
    }))


def pbench():
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3

    n_warm = int(os.environ.get("N_ARM_WARM", "6"))
    n_paired = int(os.environ.get("N_PAIRED", "120"))
    n_t2i = int(os.environ.get("N_PAIRED_T2I", "60"))
    pipe = load_pipe()
    probe = Probe(pipe)
    refs = find_refs()
    cand_load = flux3._load_image_pixels
    has_fast = hasattr(flux3, "png_fast")

    def use(arm):
        flux3._load_image_pixels = ref_load_image_pixels if arm == "A" else cand_load
        assert flux3._load_image_pixels is (ref_load_image_pixels if arm == "A" else cand_load)

    def req(kind):
        if kind == "t2i":
            return make_req(*T2I)
        return make_req(PROMPTS[kind][0], PROMPTS[kind][1], refs[kind])

    out = {"batch": {}, "output": {"class": "exact"}, "arms": {}, "paired": {}, "load1": {},
           "tree_has_png_fast": has_fast}
    for case, kinds, n in (("i2i-2048", ["A", "B"], n_paired), ("t2i-2048", ["t2i"], n_t2i)):
        for i in range(n_warm):
            for arm in "AB":
                use(arm)
                span(pipe, probe, req(kinds[i % len(kinds)]))
        out["load1"][f"{case}_start"] = loadavg1()
        t = {"A": [], "B": []}
        by_ref = {"A": {}, "B": {}}
        diffs = {}
        for i in range(n):
            kind = kinds[i % len(kinds)]
            g = {}
            for arm in ("AB" if i % 2 == 0 else "BA"):
                use(arm)
                g[arm], _ = span(pipe, probe, req(kind))
                t[arm].append(g[arm])
                by_ref[arm].setdefault(kind, []).append(g[arm])
            diffs.setdefault(kind, []).append(g["B"] - g["A"])
            if i % 20 == 0:
                log(f"{case} iter {i} ({kind}): A {g['A']:.2f} ms, B {g['B']:.2f} ms")
        out["load1"][f"{case}_end"] = loadavg1()
        use("B")
        pr = {}
        for kind, d in diffs.items():
            q = statistics.quantiles(sorted(d), n=4)
            ra = [b / a for a, b in zip(by_ref["A"][kind], by_ref["B"][kind])]
            pr[kind] = {"diff_B_minus_A_ms": {"median": statistics.median(d), "p25": q[0],
                                              "p75": q[2], "iqr": q[2] - q[0]},
                        "ratio_B_over_A_median": statistics.median(ra), "n": len(d),
                        "A": stats(by_ref["A"][kind]), "B": stats(by_ref["B"][kind])}
        out["paired"][case] = pr
        nf = max(stats(v)["p10_p90_over_median"] for v in by_ref["B"].values())
        out["batch"][case] = {"step_ms": statistics.median(t["B"]), "noise_floor": nf}
        log(f"{case}: {json.dumps(pr)}")
    out["measure"] = ("Flux3Pipeline.infer entry (ref PNG bytes in) -> first transformer call, "
                      "after cuda sync; served engine.json pipeline, 2048^2; arm A = main's "
                      "_load_image_pixels (copied), arm B = tree's, interleaved per iteration "
                      "(AB even, BA odd), per-reference paired difference")
    print(json.dumps(out))


if __name__ == "__main__":
    {"unit": unit, "check": check, "pbench": pbench}[MODE]()
