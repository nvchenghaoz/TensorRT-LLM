"""Idea 63 STEP 0: where PIL's 2048^2 PNG decode time goes, which inflaters load, and the
GPU wavefront unfilter (png_fast) speed + bitwise vs PIL. No model load.

python step0.py
"""

import ctypes
import ctypes.util
import glob
import json
import os
import statistics
import struct
import sys
import time
import zlib
from io import BytesIO

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pngtest import find_refs, log, make_png, pil_rgb  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import png_fast  # noqa: E402


def tm(fn, n=7, sync=False):
    v = []
    for _ in range(n):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        v.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(v), 2)


def chunks_of(data):
    pos, out = 8, []
    while pos + 12 <= len(data):
        length, cid = struct.unpack_from(">I4s", data, pos)
        out.append((cid.decode("latin1"), length))
        pos += 12 + length
        if cid == b"IEND":
            break
    return out


def find_libdeflate():
    cands = []
    p = ctypes.util.find_library("deflate")
    if p:
        cands.append(p)
    for root in sys.path + ["/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]:
        if os.path.isdir(root):
            cands += glob.glob(os.path.join(root, "**", "libdeflate*.so*"), recursive=False)
            cands += glob.glob(os.path.join(root, "*", "libdeflate*.so*"))
            cands += glob.glob(os.path.join(root, "*", "*", "libdeflate*.so*"))
    for c in cands:
        try:
            return ctypes.CDLL(c), c
        except OSError:
            continue
    return None, cands


def main():
    from PIL import Image, features

    dev = torch.device("cuda")
    refs = find_refs()
    res = {"pil": {}, "split": {}, "alt": {}, "gpu": {}, "env": {}}
    res["env"]["pil_version"] = Image.__version__ if hasattr(Image, "__version__") else None
    try:
        import PIL
        res["env"]["pil_version"] = PIL.__version__
        res["env"]["pil_zlib_ng"] = features.check("zlib_ng") if "zlib_ng" in features.features else "n/a"
        res["env"]["pil_zlib_version"] = features.version("zlib")
    except Exception as exc:  # noqa: BLE001
        res["env"]["pil_err"] = repr(exc)
    res["env"]["zlib_runtime"] = zlib.ZLIB_RUNTIME_VERSION
    res["env"]["cpu_count"] = os.cpu_count()
    try:
        res["env"]["load1"] = os.getloadavg()[0]
    except OSError:
        pass

    for name, data in refs.items():
        log(f"ref {name}: {len(data)} bytes")
        ch = chunks_of(data)
        kinds = {}
        for c, ln in ch:
            kinds.setdefault(c, [0, 0])
            kinds[c][0] += 1
            kinds[c][1] += ln
        def pil_load():
            im = Image.open(BytesIO(data))
            im.load()
            return im
        im = pil_load()
        px = np.asarray(im)
        w, h = im.size
        sp = {"size": [w, h], "mode": im.mode, "chunks": kinds, "info": sorted(map(str, im.info))}
        sp["pil_open_load_ms"] = tm(pil_load)
        parsed = png_fast.parse_png_rgb8(data)
        sp["fast_path_eligible"] = parsed is not None
        sp["parse_ms"] = tm(lambda: png_fast.parse_png_rgb8(data), n=11)
        _, _, idat = parsed
        joined = b"".join(idat)
        sp["join_ms"] = tm(lambda: b"".join(idat), n=11)
        expected = h * (1 + 3 * w)
        raw = zlib.decompress(joined)
        assert len(raw) == expected
        sp["zlib_decompress_ms"] = tm(lambda: zlib.decompress(joined))
        sp["zlib_decompress_bufsize_ms"] = tm(lambda: zlib.decompress(joined, 15, expected))
        sp["unfilter_remainder_ms"] = round(sp["pil_open_load_ms"] - sp["zlib_decompress_ms"], 2)
        ft = np.frombuffer(raw, np.uint8)[:: 1 + 3 * w]
        sp["filter_types"] = {int(k): int(v) for k, v in zip(*np.unique(ft, return_counts=True))}
        res["split"][name] = sp
        log(json.dumps(sp))

        alt = {}
        # libdeflate via ctypes
        lib, where = find_libdeflate()
        if lib is not None:
            try:
                lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
                lib.libdeflate_zlib_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                           ctypes.c_size_t, ctypes.c_void_p,
                                                           ctypes.c_size_t, ctypes.c_void_p]
                dec = lib.libdeflate_alloc_decompressor()
                outt = torch.empty(expected, dtype=torch.uint8, pin_memory=True)
                actual = ctypes.c_size_t(0)
                src = (ctypes.c_char * len(joined)).from_buffer_copy(joined)

                def ld():
                    rc = lib.libdeflate_zlib_decompress(dec, src, len(joined), outt.data_ptr(),
                                                        expected, ctypes.byref(actual))
                    assert rc == 0, rc
                ld()
                alt["libdeflate"] = {"path": str(where), "ms": tm(ld),
                                     "equal_zlib": bytes(outt.numpy()) == raw}
            except Exception as exc:  # noqa: BLE001
                alt["libdeflate"] = repr(exc)
        else:
            alt["libdeflate"] = f"not found ({len(where)} candidates)"
        for mod, fn in (("isal.isal_zlib", "decompress"), ("zlib_ng.zlib_ng", "decompress"),
                        ("deflate", "zlib_decompress")):
            try:
                m = __import__(mod, fromlist=[fn])
                f = getattr(m, fn)
                if mod == "deflate":
                    o = f(joined, expected)
                    alt[mod] = {"ms": tm(lambda: f(joined, expected)), "equal_zlib": bytes(o) == raw}
                else:
                    o = f(joined)
                    alt[mod] = {"ms": tm(lambda: f(joined)), "equal_zlib": bytes(o) == raw}
            except Exception as exc:  # noqa: BLE001
                alt[mod] = repr(exc)[:200]
        try:
            import imagecodecs
            o = imagecodecs.zlib_decode(joined)
            alt["imagecodecs.zlib_decode"] = {"ms": tm(lambda: imagecodecs.zlib_decode(joined)),
                                              "equal_zlib": bytes(o) == raw}
            d = imagecodecs.png_decode(data)
            alt["imagecodecs.png_decode"] = {"ms": tm(lambda: imagecodecs.png_decode(data)),
                                             "equal_pil": bool(np.array_equal(d, px))}
        except Exception as exc:  # noqa: BLE001
            alt["imagecodecs"] = repr(exc)[:200]
        try:
            from nvidia import nvimgcodec
            dec = nvimgcodec.Decoder()
            o = dec.decode(data)
            alt["nvimgcodec"] = {"ms": tm(lambda: dec.decode(data), sync=True),
                                 "backends": repr(getattr(dec, "backends", None))}
        except Exception as exc:  # noqa: BLE001
            alt["nvimgcodec"] = repr(exc)[:200]
        try:
            import torchvision.io as tvio
            buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            d = tvio.decode_png(buf)
            alt["torchvision_decode_png"] = {"ms": tm(lambda: tvio.decode_png(buf)),
                                             "equal_pil": bool(np.array_equal(
                                                 d.permute(1, 2, 0).numpy(), px))}
        except Exception as exc:  # noqa: BLE001
            alt["torchvision_decode_png"] = repr(exc)[:200]
        res["alt"][name] = alt
        log(json.dumps(alt))

    # GPU unfilter configs on ref A (+ bitwise on both refs).
    raws = {}
    for name, data in refs.items():
        w, h, idat = png_fast.parse_png_rgb8(data)
        raws[name] = (w, h, torch.frombuffer(bytearray(zlib.decompress(b"".join(idat))),
                                             dtype=torch.uint8).to(dev))
    pils = {k: torch.from_numpy(pil_rgb(v)).to(dev) for k, v in refs.items()}
    for br in (2048,):
        for nw in (4, 8, 16, 32):
            png_fast.CFG[br] = {"num_warps": nw}
            key = f"lanes{br}-w{nw}"
            try:
                t0 = time.perf_counter()
                w, h, rd = raws["A"]
                o = png_fast.unfilter_rgb8_on_device(rd, w, h)
                torch.cuda.synchronize()
                first = (time.perf_counter() - t0) * 1e3
                eq = {k: bool(torch.equal(png_fast.unfilter_rgb8_on_device(r, w2, h2), pils[k]))
                      for k, (w2, h2, r) in raws.items()}
                ev = []
                for _ in range(9):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    png_fast.unfilter_rgb8_on_device(rd, w, h)
                    e.record()
                    e.synchronize()
                    ev.append(s.elapsed_time(e))
                res["gpu"][key] = {"first_call_ms": round(first, 1), "kernel_ms": round(
                    statistics.median(ev), 3), "bitwise_vs_pil": eq}
            except Exception as exc:  # noqa: BLE001
                res["gpu"][key] = repr(exc)[:300]
            log(key, json.dumps(res["gpu"][key]))
    best = min((k for k, v in res["gpu"].items() if isinstance(v, dict)
                and all(v["bitwise_vs_pil"].values())), key=lambda k: res["gpu"][k]["kernel_ms"])
    br, nw = best.split("-")
    png_fast.CFG[int(br[5:])] = {"num_warps": int(nw[1:])}
    res["gpu"]["best"] = best
    # Per-kernel split (best config) with a profiler-free event breakdown.
    w, h, rd = raws["A"]
    steps = w + h - 1
    si = torch.empty((3, steps, h), dtype=torch.uint8, device=dev)
    so = torch.empty_like(si)
    oo = torch.empty((h, w, 3), dtype=torch.uint8, device=dev)
    n = h * w * 3
    grid = ((n + 1023) // 1024,)
    split = {}
    for kname, fn in (
        ("skew", lambda: png_fast._png_skew_kernel[grid](rd, si, h, w, n, BLOCK=1024, num_warps=4)),
        ("wavefront", lambda: png_fast._png_wavefront_kernel[(3,)](
            rd, si, so, h, w, BLOCK_R=2048, **png_fast.CFG[2048])),
        ("unskew", lambda: png_fast._png_unskew_kernel[grid](so, oo, h, w, n, BLOCK=1024,
                                                             num_warps=4)),
    ):
        fn()
        ev = []
        for _ in range(9):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            ev.append(s.elapsed_time(e))
        split[kname] = round(statistics.median(ev), 3)
    res["gpu"]["kernel_split_ms"] = split
    log("kernel split", json.dumps(split))
    # Host inflate: libdeflate into pinned vs zlib.
    for name, data in refs.items():
        w, h, idat = png_fast.parse_png_rgb8(data)
        z = b"".join(idat)
        st = torch.empty(h * (1 + 3 * w), dtype=torch.uint8, pin_memory=True)
        res["gpu"][f"inflate_{name}"] = {
            "libdeflate_ms": tm(lambda: png_fast.inflate_scanlines_into(z, st), n=9),
            "zlib_ms": tm(lambda: png_fast.inflate_scanlines_into(z, st, use_libdeflate=False), n=5),
            "libdeflate_available": png_fast._libdeflate() is not None,
        }
        log(name, json.dumps(res["gpu"][f"inflate_{name}"]))
    # Whole fast decode (host time to return, and synced) vs PIL.
    for name, data in refs.items():
        o = png_fast.decode_png_rgb8_to_device(data, dev)
        torch.cuda.synchronize()
        res["gpu"][f"decode_{name}"] = {
            "bitwise_vs_pil": bool(torch.equal(o, pils[name])),
            "host_ms": tm(lambda: png_fast.decode_png_rgb8_to_device(data, dev), n=9),
            "synced_ms": tm(lambda: png_fast.decode_png_rgb8_to_device(data, dev), n=9, sync=True),
        }
        log(name, json.dumps(res["gpu"][f"decode_{name}"]))
    # Synthetic quick bitwise: each filter, cycle, levels, multi-IDAT, odd sizes.
    rng = np.random.default_rng(1)
    syn = {}
    base = pil_rgb(refs["A"])
    for (w, h) in ((16, 16), (37, 23), (300, 211), (2040, 2033)):
        px = np.ascontiguousarray(base[:h, :w]) if w <= 2048 else None
        noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for src_name, src in (("photo", px), ("noise", noise)):
            for filt in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4), (4, 3, 2, 1, 0, 4, 4)):
                for level, isz in ((0, None), (6, 8192), (9, None)):
                    d = make_png(src, filters=filt, level=level, idat_size=isz)
                    o = png_fast.decode_png_rgb8_to_device(d, dev)
                    ok = o is not None and bool(torch.equal(o.cpu(), torch.from_numpy(pil_rgb(d))))
                    syn[f"{src_name}_{w}x{h}_f{''.join(map(str, filt))}_l{level}_i{isz}"] = ok
    res["synthetic_all_bitwise"] = all(syn.values())
    res["synthetic_failed"] = [k for k, v in syn.items() if not v]
    res["synthetic_n"] = len(syn)
    log(f"synthetic: {len(syn)} cases, all bitwise={res['synthetic_all_bitwise']}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
