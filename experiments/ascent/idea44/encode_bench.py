"""Idea 44: 2048^2 b64_json response encode (t2i + i2i), in-process, no model load.

Arms (swapped by module attribute, swap asserted):
  A = main's code, copied verbatim below: _b64_gpu (pinned D2H + tobytes) and
      _B64JsonSplicer.render/response (b"".join + Response(content=body)).
  B = this tree's code (encoding._b64_gpu, _B64JsonSplicer.response).
  C = tree b64 + main response, D = main b64 + tree response (diagnostics).
In the base tree B == A (A/A).

Sections:
  1. step-0 stage table on real 2048^2 references (sync brackets), unbracketed totals,
     PNG size findings (block rows 1024/256/128 GPU deflate; zlib 1/6/9 on the same
     filtered scanlines), client parse ms/MB.
  2. correctness (untimed): encoder bytes A vs B vs base64(image_to_bytes) on real + odd
     shapes/channels + both fallbacks (TRTLLM_PNG_GPU_DEFLATE=0, TRTLLM_FAST_PNG=0) + CPU;
     ASGI messages through PerfMetricsMiddleware (n=1, n=2, ambiguous placeholder);
     route-level (generations + edits handlers, stub generator) A vs B sequence A,B,A,t2i;
     two threads encoding concurrently.
  3. served timing: a uvicorn subprocess (trtllm-serve's Config defaults) serves
     /enc/<arm>/<key>: encode + response + send; an aiohttp client reads the body.
     Per case 10 warmups per arm, then N paired iterations (AB even, BA odd).
     Wire bytes A vs B (raw socket, Date header stripped) for every wire case.
  4. untimed replay of the timed sequence: every body == the reference sha.
Last stdout line: {"batch": {...step_ms...}, "output": {"class": "exact"}, "passed": ...}
"""
import asyncio
import base64
import glob
import hashlib
import io
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import torch

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {  # experiments/ascent-benchmark/inputs.json
    "check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
    "check01": "7928b9c476065f4c55503eefff5cc525d506354183ca588955108180127e8581",
    "check02": "38d5965fd136b9aa38e7618a49d3bae487f68cc9e47d3f9c957a844a98c594ab",
    "check10": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",
    "acc07": "8018670f8ea9054bcecbdcd4533337e7d5a1c8492d670d2330ed76af6c5e590d",
    "acc14": "2aeb66e1c4909cdb715aa21dcbbd596ab27cb3c3f2b0c32c18d63e355f8696ed",
    "held03": "1face6b49175ab785cbe5e1463b9f4a1655000141fcb3762cc3fc82a08b819b4",
    "check08": "8638edbd78cbbbde7fe10abf27c9c61168c43de02dfdefe9a293436e327ec4d5",
}
CASES = {"t2i-2048": ["check00", "check01", "check02", "check10"],
         "i2i-2048": ["acc07", "acc14", "held03", "check08"]}
N_PAIRS = int(os.environ.get("N_PAIRS", "160"))
N_DIAG = int(os.environ.get("N_DIAG", "60"))  # A/A pairs per case (noise reference)
N_STAGE = int(os.environ.get("N_STAGE", "40"))
PROMPT = "A red fox on a snowy hill, \"sharp\" text SIGN — café"
WIRE_HEADERS = {"Server-Timing": "server_e2e;dur=12.500000, server_gen;dur=4000.000000"}


def log(*a):
    print(f"[idea44 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def loadavg():
    try:
        return open("/proc/loadavg").read().split()[:3]
    except OSError:
        return None


def find_refs():
    want = {v: k for k, v in REFS.items()}
    found = {}
    files = [f for f in sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True))
             if os.path.isfile(f)]
    for f in files:
        if len(found) == len(want):
            break
        name = os.path.basename(f)
        if not any(s in name for s in want) and len(files) > 80:
            continue
        data = open(f, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        if h in want:
            found[want[h]] = data
    if len(found) != len(want):
        raise SystemExit(f"references not found: have {sorted(found)}")
    return found


def load_pixels(refs):
    from PIL import Image
    pix = {}
    for n, data in refs.items():
        im = Image.open(io.BytesIO(data)).convert("RGB")
        pix[n] = torch.from_numpy(np.asarray(im).copy()).cuda()
    g = torch.Generator().manual_seed(1234)
    for name, shape in (("odd37x53x3", (37, 53, 3)), ("one1x1x1", (1, 1, 1)),
                        ("odd17x9x4", (17, 9, 4)), ("odd5x2x1", (5, 2, 1)),
                        ("odd301x199x4", (301, 199, 4)), ("noise2048x3", (2048, 2048, 3))):
        pix[name] = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).cuda()
    # a 4-channel 2048^2 real image (alpha from the green channel)
    base = pix["check00"]
    pix["check00rgba"] = torch.cat((base, base[:, :, 1:2]), dim=2).contiguous()
    return pix


# --------------------------------------------------------------------------------------
# main's code, verbatim (e0d4536c tensorrt_llm/media/encoding.py _b64_gpu,
# tensorrt_llm/serve/openai_server.py _B64JsonSplicer.render/response)
# --------------------------------------------------------------------------------------
def make_main_fns(enc, osrv):
    _png_pinned = enc._png_pinned
    _B64_ALPHABET = enc._B64_ALPHABET
    Response = osrv.Response
    JSONResponse = osrv.JSONResponse

    def main_b64_gpu(segs, stream, gpu_stream: torch.Tensor) -> bytes:
        dev = gpu_stream.device
        small = b"".join(s for s in segs if isinstance(s, bytes))
        pin_s = _png_pinned("b64_small", len(small) + 64, torch.uint8)
        pin_s[: len(small)].numpy()[:] = np.frombuffer(small, dtype=np.uint8)
        pin_s[len(small) : len(small) + 64].numpy()[:] = np.frombuffer(_B64_ALPHABET, dtype=np.uint8)
        small_g = pin_s.to(dev, non_blocking=True)
        pieces, pos, n = [], 0, 0
        for s in segs:
            if isinstance(s, bytes):
                pieces.append(small_g[pos : pos + len(s)])
                pos += len(s)
                n += len(s)
            else:
                pieces.append(gpu_stream[s[0] : s[1]])
                n += s[1] - s[0]
        table = small_g[len(small) :]
        m = (n + 2) // 3
        pad = 3 * m - n
        if pad:
            pieces.append(torch.zeros(pad, dtype=torch.uint8, device=dev))
        t = torch.cat(pieces).view(m, 3).to(torch.int32)
        v = (t[:, 0] << 16) | (t[:, 1] << 8) | t[:, 2]
        idx = torch.stack((v >> 18, (v >> 12) & 63, (v >> 6) & 63, v & 63), dim=1)
        enc_ = table[idx]  # (m, 4) uint8
        pin_b = _png_pinned("b64_out", 4 * m, torch.uint8)
        pin_b.copy_(enc_.view(-1), non_blocking=True)
        torch.cuda.current_stream(dev).synchronize()
        arr = pin_b.numpy()
        if pad:
            arr[4 * m - pad :] = ord("=")
        return arr.tobytes()

    def main_render(self, content):
        body = self._dumps(content)
        parts = []
        pos = 0
        for i, payload in enumerate(self._payloads):
            marker = f'"{self._prefix}{i}"'.encode("ascii")
            idx = body.find(marker, pos)
            if idx < 0 or body.count(marker) != 1:
                return None
            parts.append(body[pos:idx + 1])
            parts.append(payload)
            pos = idx + len(marker) - 1
        parts.append(body[pos:])
        return b"".join(parts)

    def main_response(self, content, headers):
        body = main_render(self, content)
        if body is None:
            return JSONResponse(content=self.restore(content), headers=headers)
        return Response(content=body, media_type="application/json", headers=headers)

    return main_b64_gpu, main_render, main_response


class Arms:
    def __init__(self, enc, osrv):
        self.enc, self.osrv = enc, osrv
        self.tree_b64 = enc._b64_gpu
        self.tree_resp = osrv._B64JsonSplicer.response
        self.main_b64, self.main_render, self.main_resp = make_main_fns(enc, osrv)

    def set(self, arm):
        b64 = {"A": self.main_b64, "B": self.tree_b64, "C": self.tree_b64, "D": self.main_b64}[arm]
        resp = {"A": self.main_resp, "B": self.tree_resp, "C": self.main_resp, "D": self.tree_resp}[arm]
        self.enc._b64_gpu = b64
        self.osrv._B64JsonSplicer.response = resp
        assert self.enc._b64_gpu is b64 and self.osrv._B64JsonSplicer.response is resp


def build_content(osrv, images, prompt=PROMPT, ambiguous=False):
    from tensorrt_llm.serve.openai_protocol import ImageGenerationResponse, ImageObject
    sp = osrv._B64JsonSplicer()
    data = []
    for img in images:
        token = sp.placeholder(osrv.image_to_b64(img, format="PNG"))
        rp = prompt
        if ambiguous:
            rp = f'"{token}"'  # the quoted placeholder appears twice -> JSONResponse fallback
        data.append(ImageObject(b64_json=token, revised_prompt=rp))
    resp = ImageGenerationResponse(created=0, data=data, output_format="png",
                                   size=f"{images[0].shape[1]}x{images[0].shape[0]}")
    return sp, resp.model_dump()


async def collect(resp, wrap=True):
    from tensorrt_llm.serve.perf_metrics import PerfMetricsMiddleware
    msgs = []

    async def send(m):
        msgs.append(m)

    async def receive():
        return {"type": "http.disconnect"}

    async def app(scope, receive_, send_):
        await resp(scope, receive_, send_)

    scope = {"type": "http", "headers": [], "state": {}}
    if wrap:
        await PerfMetricsMiddleware(app, expose_headers=True)(scope, receive, send)
    else:
        await app(scope, receive, send)
    return msgs


def msgs_summary(msgs):
    start = msgs[0]
    bodies = msgs[1:]
    finals = [m for m in bodies if not m.get("more_body", False)]
    return {"status": start["status"], "headers": [tuple(h) for h in start["headers"]],
            "body": b"".join(bytes(m.get("body", b"")) for m in bodies),
            "n_body_msgs": len(bodies), "n_final": len(finals),
            "final_is_last": bool(bodies) and not bodies[-1].get("more_body", False)}


# --------------------------------------------------------------------------------------
# uvicorn subprocess
# --------------------------------------------------------------------------------------
def serve(port):
    import uvicorn
    from tensorrt_llm.media import encoding as enc
    from tensorrt_llm.serve import openai_server as osrv
    from tensorrt_llm.serve.perf_metrics import PerfMetricsMiddleware
    refs = find_refs()
    pix = load_pixels(refs)
    arms = Arms(enc, osrv)
    stats = []
    for k in list(pix)[:2]:  # CUDA/JIT warmup
        for a in "AB":
            arms.set(a)
            osrv.image_to_b64(pix[k], format="PNG")

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        parts = scope["path"].strip("/").split("/")
        if parts[0] == "stats":
            body = json.dumps(stats).encode()
            stats.clear()
            await osrv.Response(content=body, media_type="application/json")(scope, receive, send)
            return
        # /enc/<arm>/<key>  or  /wire/<arm>/<key>/<n>/<mode>
        kind, arm, key = parts[0], parts[1], parts[2]
        n = int(parts[3]) if len(parts) > 3 else 1
        mode = parts[4] if len(parts) > 4 else "gpu"
        env = {"zlib": ("TRTLLM_PNG_GPU_DEFLATE", "0"), "pil": ("TRTLLM_FAST_PNG", "0")}.get(mode)
        if env:
            os.environ[env[0]] = env[1]
        try:
            arms.set(arm)
            t0 = time.perf_counter()
            if key == "amb":
                sp, content = build_content(osrv, [pix["odd37x53x3"]], ambiguous=True)
            else:
                imgs = [pix[key]] if n == 1 else [pix[key], pix["check10" if key != "check10" else "check00"]]
                if mode == "cpu":
                    imgs = [i.cpu() for i in imgs]
                sp, content = build_content(osrv, imgs)
            t1 = time.perf_counter()
            resp = sp.response(content, WIRE_HEADERS if kind == "wire" else {})
            t2 = time.perf_counter()
        finally:
            if env:
                os.environ.pop(env[0], None)
        await resp(scope, receive, send)
        t3 = time.perf_counter()
        stats.append({"arm": arm, "key": key, "enc_ms": (t1 - t0) * 1e3,
                      "build_ms": (t2 - t1) * 1e3, "send_ms": (t3 - t2) * 1e3})

    cfg = uvicorn.Config(PerfMetricsMiddleware(app, expose_headers=False), host="127.0.0.1",
                         port=port, log_level="warning", timeout_keep_alive=5)
    server = uvicorn.Server(cfg)
    cfg.load()
    try:
        import uvloop  # noqa: F401
        has_uvloop = True
    except ImportError:
        has_uvloop = False
    import starlette
    print(f"READY http={getattr(cfg.http_protocol_class, '__module__', '?')} "
          f"loop={cfg.loop} uvloop_installed={has_uvloop} uvicorn={uvicorn.__version__} "
          f"starlette={starlette.__version__} py={sys.version.split()[0]}", flush=True)
    server.run()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def raw_get(port, path):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode())
    await w.drain()
    data = await r.read()
    w.close()
    head, _, body = data.partition(b"\r\n\r\n")
    lines = [ln for ln in head.split(b"\r\n") if not ln.lower().startswith(b"date:")]
    return b"\r\n".join(lines), body


# --------------------------------------------------------------------------------------
def stat(ts):
    q = statistics.quantiles(ts, n=10)
    med = statistics.median(ts)
    return {"step_ms": round(med, 3), "noise_floor": round((q[-1] - q[0]) / med, 4),
            "p10": round(q[0], 3), "p90": round(q[-1], 3), "mean_ms": round(statistics.mean(ts), 3),
            "rel_std": round(statistics.pstdev(ts) / statistics.mean(ts), 4), "n": len(ts)}


def main():
    from tensorrt_llm.media import encoding as enc
    from tensorrt_llm.serve import openai_server as osrv
    result = {"loadavg_start": loadavg()}
    passed = True
    refs = find_refs()
    pix = load_pixels(refs)
    arms = Arms(enc, osrv)
    real = [k for k in REFS]
    log(f"loaded {len(pix)} images; tree has _SplicedJSONResponse="
        f"{hasattr(osrv, '_SplicedJSONResponse')} owned_d2h={getattr(enc, '_B64_OWNED_D2H', None)}")
    for k in real[:2]:
        for a in "AB":
            arms.set(a)
            osrv.image_to_b64(pix[k], format="PNG")

    # ---------------- 1. step-0 stage table -----------------------------------------------
    sync = torch.cuda.synchronize
    pc = time.perf_counter
    SY = []
    orig_sync = torch.cuda.Stream.synchronize

    def stamped_sync(self):
        t = pc()
        orig_sync(self)
        SY.append((t, pc()))

    def staged_b64(variant):
        def fn(segs, stream, gpu_stream):
            st = {"t_start": pc()}
            dev = gpu_stream.device
            small = b"".join(s for s in segs if isinstance(s, bytes))
            pin_s = enc._png_pinned("b64_small", len(small) + 64, torch.uint8)
            pin_s[: len(small)].numpy()[:] = np.frombuffer(small, dtype=np.uint8)
            pin_s[len(small): len(small) + 64].numpy()[:] = np.frombuffer(enc._B64_ALPHABET, dtype=np.uint8)
            small_g = pin_s.to(dev, non_blocking=True)
            pieces, pos, n = [], 0, 0
            for s in segs:
                if isinstance(s, bytes):
                    pieces.append(small_g[pos: pos + len(s)])
                    pos += len(s)
                    n += len(s)
                else:
                    pieces.append(gpu_stream[s[0]: s[1]])
                    n += s[1] - s[0]
            table = small_g[len(small):]
            m = (n + 2) // 3
            pad = 3 * m - n
            if pad:
                pieces.append(torch.zeros(pad, dtype=torch.uint8, device=dev))
            t = torch.cat(pieces).view(m, 3).to(torch.int32)
            v = (t[:, 0] << 16) | (t[:, 1] << 8) | t[:, 2]
            idx = torch.stack((v >> 18, (v >> 12) & 63, (v >> 6) & 63, v & 63), dim=1)
            e = table[idx]
            st["t_enq"] = pc()
            if variant == "pinned":
                pin_b = enc._png_pinned("b64_out", 4 * m, torch.uint8)
                pin_b.copy_(e.view(-1), non_blocking=True)
                orig_sync(torch.cuda.current_stream(dev))
                st["t_d2h"] = pc()
                arr = pin_b.numpy()
                if pad:
                    arr[4 * m - pad:] = ord("=")
                out = arr.tobytes()
            else:
                sync()
                st["t_gpu"] = pc()
                out = bytearray(4 * m)
                torch.frombuffer(out, dtype=torch.uint8).copy_(e.view(-1))
                st["t_d2h"] = pc()
                if pad:
                    out[4 * m - pad:] = b"=" * pad
            st["t_end"] = pc()
            fn.st = st
            return out
        return fn

    stage_rows = {}
    totals_unbr = {"A": [], "B": []}
    for variant in ("pinned", "owned"):
        rows = []
        sb = staged_b64(variant)
        for it in range(5 + N_STAGE):
            k = real[it % 4]
            img = pix[k]
            h, w, c = img.shape
            torch.cuda.Stream.synchronize = stamped_sync
            enc._b64_gpu = sb
            try:
                sync()
                SY.clear()
                t0 = pc()
                scan = enc._png_filter_scanlines(img)
                sync()
                t1 = pc()
                SY.clear()
                stream_b, adler, sg = enc._png_gpu_deflate(scan)
                t2 = pc()
                s1, s2 = SY[0], SY[1]
                t3 = pc()
                b64 = enc._png_assemble(w, h, c, stream_b, adler, sg)
                t4 = pc()
                st = sb.st
            finally:
                torch.cuda.Stream.synchronize = orig_sync
                arms.set("A")
            spl = osrv._B64JsonSplicer()
            from tensorrt_llm.serve.openai_protocol import ImageGenerationResponse, ImageObject
            content = ImageGenerationResponse(created=0, data=[ImageObject(
                b64_json=spl.placeholder(b64), revised_prompt=PROMPT)], output_format="png",
                size="2048x2048").model_dump()
            t5 = pc()
            if variant == "pinned":
                body = arms.main_render(spl, content)
                t6 = pc()
                r = osrv.Response(content=body, media_type="application/json", headers=WIRE_HEADERS)
            elif hasattr(osrv, "_SplicedJSONResponse"):
                parts = spl.render_parts(content)
                t6 = pc()
                r = osrv._SplicedJSONResponse(parts, WIRE_HEADERS)
            else:  # base tree: the tree's render + Response
                body = spl.render(content)
                t6 = pc()
                r = osrv.Response(content=body, media_type="application/json", headers=WIRE_HEADERS)
            t7 = pc()
            assert len(SY) == 2, len(SY)
            row = {"filter": t1 - t0,
                   "hist_enqueue+sync1": s1[1] - t1,
                   "host_huffman+upload+pack_enqueue": s2[0] - s1[1],
                   "pack_gpu+D2H_wait_sync2": s2[1] - s2[0],
                   "deflate_return": t2 - s2[1],
                   "crc32_assemble": st["t_start"] - t3,
                   "b64_enqueue": st["t_enq"] - st["t_start"]}
            if variant == "pinned":
                row["b64_gpu+D2H_wait_sync3"] = st["t_d2h"] - st["t_enq"]
                row["tobytes"] = st["t_end"] - st["t_d2h"]
            else:
                row["b64_gpu_wait"] = st["t_gpu"] - st["t_enq"]
                row["owned_pageable_D2H"] = st["t_d2h"] - st["t_gpu"]
                row["pad"] = st["t_end"] - st["t_d2h"]
            row["b64_return"] = t4 - st["t_end"]
            row["model_dump"] = t5 - t4
            row["render" if variant == "pinned" else "render_parts"] = t6 - t5
            row["response_construct"] = t7 - t6
            row["bracketed_total"] = t7 - t0
            if it >= 5:
                rows.append(row)
        stage_rows[variant] = {k: round(statistics.median([r[k] * 1e3 for r in rows]), 3)
                               for k in rows[0]}
        stage_rows[variant]["sum_of_stage_medians"] = round(sum(
            v for kk, v in stage_rows[variant].items() if kk != "bracketed_total"), 3)
        log(f"stage table ({variant}): {stage_rows[variant]}")
    # unbracketed totals, interleaved A/B
    for it in range(5 + N_STAGE):
        k = real[it % 4]
        for a in (("A", "B") if it % 2 == 0 else ("B", "A")):
            arms.set(a)
            sync()
            t0 = pc()
            sp, content = build_content(osrv, [pix[k]])
            resp = sp.response(content, WIRE_HEADERS)
            t1 = pc()
            if it >= 5:
                totals_unbr[a].append((t1 - t0) * 1e3)
    result["stage_table_ms"] = stage_rows
    result["unbracketed_image_to_b64+response_ms"] = {a: stat(v) for a, v in totals_unbr.items()}
    log(f"unbracketed: {result['unbracketed_image_to_b64+response_ms']}")

    # PNG size findings (exact pixels, byte-changing: findings only)
    sizes = {}
    for k in real[:4]:
        img = pix[k]
        row = {}
        for br in (1024, 256, 128):
            old = enc._PNG_GPU_BLOCK_ROWS
            enc._PNG_GPU_BLOCK_ROWS = br
            try:
                sync()
                t0 = pc()
                png = enc.image_to_bytes(img, format="PNG")
                row[f"gpu_huff_rows{br}_bytes"] = len(png)
                row[f"gpu_huff_rows{br}_ms"] = round((pc() - t0) * 1e3, 2)
            finally:
                enc._PNG_GPU_BLOCK_ROWS = old
        scan = enc._png_filter_scanlines(img).cpu().numpy().tobytes()
        for lvl in (1, 6, 9):
            t0 = pc()
            row[f"zlib{lvl}_bytes"] = len(zlib.compress(scan, lvl))
            row[f"zlib{lvl}_ms"] = round((pc() - t0) * 1e3, 1)
        sizes[k] = row
        log(f"sizes {k}: {row}")
    result["png_size_findings"] = sizes

    # client parse ms/MB (the pinned client's response.json() + b64decode)
    sp, content = build_content(osrv, [pix["check10"]])
    body = sp.render(content)
    tj, tb = [], []
    for _ in range(15):
        t0 = pc()
        js = json.loads(body.decode("utf-8"))
        t1 = pc()
        base64.b64decode(js["data"][0]["b64_json"])
        t2 = pc()
        tj.append((t1 - t0) * 1e3)
        tb.append((t2 - t1) * 1e3)
    mb = len(body) / 1e6
    result["client_parse"] = {"body_mb": round(mb, 3), "json_ms": round(statistics.median(tj), 3),
                              "b64decode_ms": round(statistics.median(tb), 3),
                              "ms_per_mb": round((statistics.median(tj) + statistics.median(tb)) / mb, 3)}
    log(f"client parse {result['client_parse']}")

    # ---------------- 2. correctness (untimed) --------------------------------------------
    enc_fail = []
    keys = list(pix)
    for mode in ("gpu", "zlib", "pil", "cpu"):
        env = {"zlib": ("TRTLLM_PNG_GPU_DEFLATE", "0"), "pil": ("TRTLLM_FAST_PNG", "0")}.get(mode)
        if env:
            os.environ[env[0]] = env[1]
        try:
            # 1-channel (H, W, 1) tensors are not accepted by the PIL fallback on main either.
            sub = ["check00", "odd37x53x3", "odd17x9x4", "check00rgba"]
            if mode == "zlib":
                sub += ["odd5x2x1", "one1x1x1"]
            for k in (keys if mode == "gpu" else sub):
                img = pix[k] if mode != "cpu" else pix[k].cpu()
                ref = base64.b64encode(enc.image_to_bytes(img, format="PNG"))
                outs = {}
                for a in "ABAB":
                    arms.set(a)
                    outs.setdefault(a, []).append(bytes(osrv.image_to_b64(img, format="PNG")))
                if not all(o == ref for v in outs.values() for o in v):
                    enc_fail.append((mode, k))
        finally:
            if env:
                os.environ.pop(env[0], None)
    log(f"encoder A/B/ref equality failures: {enc_fail}")
    result["encoder_fail"] = enc_fail
    passed &= not enc_fail

    # ASGI messages through PerfMetricsMiddleware
    asgi = {}
    for tag, imgs, amb in (("n1_check01", [pix["check01"]], False),
                           ("n2_check02_acc07", [pix["check02"], pix["acc07"]], False),
                           ("odd_n2", [pix["odd17x9x4"], pix["one1x1x1"]], False),
                           ("ambiguous", [pix["odd37x53x3"]], True)):
        res = {}
        for a in "AB":
            arms.set(a)
            sp, content = build_content(osrv, imgs, ambiguous=amb)
            resp = sp.response(content, WIRE_HEADERS)
            res[a] = msgs_summary(asyncio.run(collect(resp)))
            res[a]["type"] = type(resp).__name__
            res[a]["attr_body_eq"] = bytes(resp.body) == res[a]["body"]
        ok = (res["A"]["headers"] == res["B"]["headers"] and res["A"]["body"] == res["B"]["body"]
              and res["A"]["status"] == res["B"]["status"] and res["B"]["n_final"] == 1
              and res["B"]["final_is_last"] and res["B"]["attr_body_eq"])
        cl = dict(res["B"]["headers"]).get(b"content-length")
        ok &= cl == str(len(res["B"]["body"])).encode()
        asgi[tag] = {"ok": ok, "A_type": res["A"]["type"], "B_type": res["B"]["type"],
                     "B_body_msgs": res["B"]["n_body_msgs"], "headers": [(k.decode(), v.decode()[:40]) for k, v in res["B"]["headers"]],
                     "body_len": len(res["B"]["body"]),
                     "sha": hashlib.sha256(res["B"]["body"]).hexdigest()[:16]}
        log(f"asgi {tag}: {asgi[tag]}")
        passed &= ok
    result["asgi"] = asgi

    # route-level: generations + edits handlers with a stub generator
    route = route_check(osrv, enc, arms, refs, pix)
    result["route"] = route
    passed &= route["ok"]

    # two threads encoding concurrently (tree arm)
    arms.set("B")
    ref_b64 = {k: base64.b64encode(enc.image_to_bytes(pix[k], format="PNG")) for k in real[:4]}
    thr_fail = []

    def worker(ks):
        for i in range(12):
            k = ks[i % len(ks)]
            if bytes(osrv.image_to_b64(pix[k], format="PNG")) != ref_b64[k]:
                thr_fail.append(k)

    ths = [threading.Thread(target=worker, args=(real[:2],)), threading.Thread(target=worker, args=(real[2:4],))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    log(f"two-thread failures: {thr_fail}")
    result["threads_fail"] = thr_fail
    passed &= not thr_fail

    # ---------------- 3. served timing -----------------------------------------------------
    port = free_port()
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = ""
    t_start = time.time()
    while time.time() - t_start < 600:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith("READY"):
            break
        print("  [server]", line.rstrip(), flush=True)
    log(f"server: {line.strip()}")
    result["server"] = line.strip()

    def drain():
        for ln in proc.stdout:
            print("  [server]", ln.rstrip(), flush=True)
    threading.Thread(target=drain, daemon=True).start()
    try:
        served = asyncio.run(client(port, osrv, enc, arms, pix, result))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    passed &= served["ok"]
    result["loadavg_end"] = loadavg()
    out = {"batch": {c: served["timing"][c]["B"] for c in CASES},
           "output": {"class": "exact"}, "passed": bool(passed),
           "compared": "image_to_b64 bytes vs base64(image_to_bytes) and main's _b64_gpu; ASGI "
                       "messages/headers vs main's Response(content=join); route bodies vs main; "
                       "uvicorn wire bytes vs main; replay of the timed sequence vs reference sha"}
    out.update(result)
    out["served"] = served
    print(json.dumps(out), flush=True)


def route_check(osrv, enc, arms, refs, pix):
    import types
    from starlette.requests import Request
    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest
    from tensorrt_llm.visual_gen.params import VisualGenParams
    import tempfile
    by_sha = {hashlib.sha256(refs[n]).hexdigest(): n for n in refs}

    class Handle:
        def __init__(self, out):
            self._out = out

        async def aresult(self, timeout=None):
            return self._out

        def result(self, timeout=None):
            return self._out

    class StubGen:
        model = None
        extra_param_specs = {}
        executor = types.SimpleNamespace(supports_image_edit=True, pipeline=None)

        @property
        def default_params(self):
            return VisualGenParams(width=2048, height=2048, num_inference_steps=16)

        def generate_async(self, inputs, params=None):
            params = params.model_copy(deep=True)
            refs_ = params.image_reference
            prompt = inputs if isinstance(inputs, str) else str(inputs)
            if refs_:
                img = pix[by_sha[hashlib.sha256(refs_[0].content).hexdigest()]].clone()
            else:
                img = torch.flip(pix["check00"], dims=[0]).contiguous()
            if "PAIR" in prompt:
                img = [img, pix["check10"].clone()]
            return Handle(types.SimpleNamespace(image=img, metrics=None))

        def generate(self, inputs, params=None):
            return self.generate_async(inputs, params).result()

    srv = object.__new__(osrv.OpenAIServer)
    srv._is_visual_gen = True
    srv.generator = StubGen()
    srv.media_storage_path = osrv.Path(tempfile.mkdtemp(prefix="idea44-"))
    srv._adjusted_steady_clock = AdjustedSteadyClock()

    def make_request(body, path):
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "scheme": "http", "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 5555),
                 "headers": [(b"content-type", b"application/json"), (b"host", b"127.0.0.1:8000"),
                             (b"content-length", str(len(body)).encode())],
                 "state": {"server_arrival_time": srv._adjusted_steady_clock.now()}}
        sent = {"done": False}

        async def receive():
            if not sent["done"]:
                sent["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}
        return Request(scope, receive)

    b64_refs = {n: base64.b64encode(refs[n]).decode() for n in refs}

    async def call(key, prompt):
        if key == "t2i":
            req = ImageGenerationRequest(prompt=prompt, size="2048x2048", seed=7, response_format="b64_json",
                                         num_inference_steps=16, model="flux-3-image-klein")
            resp = await srv.openai_image_generation(req, make_request(b"{}", "/v1/images/generations"))
        else:
            body = json.dumps({"prompt": prompt, "model": "flux-3-image-klein", "response_format": "b64_json",
                               "image": b64_refs[key], "seed": 42, "size": "2048x2048",
                               "num_inference_steps": 16}).encode()
            resp = await srv.openai_image_edit(make_request(body, "/v1/images/edits"))
        return msgs_summary(await collect(resp))

    CREATED = re.compile(rb'"created":\d+')

    def norm(s):
        hdr = [(k, re.sub(rb"dur=[0-9.]+", b"dur=X", v) if k == b"server-timing" else v) for k, v in s["headers"]]
        return s["status"], hdr, CREATED.sub(b'"created":0', s["body"], count=1)

    seq = [("check10", "edit A"), ("check00", "edit B"), ("check10", "edit A"), ("t2i", PROMPT),
           ("check01", "edit PAIR n2"), ("t2i", "PAIR " + PROMPT), ("acc07", "edit A"), ("check10", "edit A")]
    out = {"items": [], "ok": True}
    first = {}
    for key, prompt in seq:
        res = {}
        for a in "AB":
            arms.set(a)
            res[a] = norm(asyncio.run(call(key, prompt)))
        ok = res["A"] == res["B"] and res["B"][0] == 200
        sig = (key, prompt)
        if sig in first:
            ok &= first[sig] == res["B"]
        else:
            first[sig] = res["B"]
        out["items"].append({"key": key, "prompt": prompt[:16], "ok": ok, "len": len(res["B"][2]),
                             "sha": hashlib.sha256(res["B"][2]).hexdigest()[:16]})
        out["ok"] &= ok
        log(f"route {out['items'][-1]}")
    return out


async def client(port, osrv, enc, arms, pix, result):
    import aiohttp
    base = f"http://127.0.0.1:{port}"
    ok = True
    # reference shas from in-process main path (A)
    arms.set("A")
    ref_sha = {}
    for c, ks in CASES.items():
        for k in ks:
            sp, content = build_content(osrv, [pix[k]])
            ref_sha[k] = hashlib.sha256(arms.main_render(sp, content)).hexdigest()
    # wire bytes A vs B
    wire = {}
    cases = [(k, 1, "gpu") for k in list(REFS)[:5]] + [
        ("check01", 2, "gpu"), ("odd37x53x3", 1, "gpu"), ("odd17x9x4", 2, "gpu"), ("one1x1x1", 1, "gpu"),
        ("check00rgba", 1, "gpu"), ("check02", 1, "zlib"), ("odd5x2x1", 1, "zlib"), ("odd301x199x4", 1, "pil"), ("check00", 1, "pil"),
        ("odd301x199x4", 2, "cpu"), ("amb", 1, "gpu")]
    for k, n, mode in cases:
        got = {}
        for a in "AB":
            got[a] = await raw_get(port, f"/wire/{a}/{k}/{n}/{mode}")
        h = got["B"][0].lower()
        same = got["A"] == got["B"]
        framed = b"content-length:" in h and b"transfer-encoding" not in h
        wire[f"{k}/n{n}/{mode}"] = {"ok": same and framed, "body_len": len(got["B"][1]),
                                    "sha": hashlib.sha256(got["B"][1]).hexdigest()[:16]}
        ok &= same and framed
    log(f"wire: {wire}")
    log(f"wire head sample: {got['B'][0].decode(errors='replace')}")
    result["wire"] = wire

    timing = {}
    seqs = {}
    async with aiohttp.ClientSession() as session:
        async def get(path):
            t0 = time.perf_counter()
            async with session.get(base + path) as r:
                body = await r.read()
            return body, (time.perf_counter() - t0) * 1e3

        async def stats():
            async with session.get(base + "/stats") as r:
                return json.loads(await r.read())
        for c, ks in CASES.items():
            plan = [(("A", "B"), N_PAIRS)] + ([(("A", "A"), N_DIAG)] if N_DIAG > 0 else [])
            for arm_set, npairs in plan:
                for it in range(10):
                    for a in arm_set:
                        await get(f"/enc/{a}/{ks[it % 4]}")
                await stats()
                ts = [[], []]  # by position in arm_set (A/A safe)
                diffs = []
                seq = []
                bad = 0
                for it in range(npairs):
                    k = ks[it % 4]
                    order = (0, 1) if it % 2 == 0 else (1, 0)
                    got = [0.0, 0.0]
                    for p in order:
                        a = arm_set[p]
                        body, ms = await get(f"/enc/{a}/{k}")
                        got[p] = ms
                        if hashlib.sha256(body).hexdigest() != ref_sha[k]:
                            bad += 1
                        seq.append((a, k))
                    ts[0].append(got[0])
                    ts[1].append(got[1])
                    diffs.append(got[1] - got[0])
                st = await stats()
                srv = {}
                for a in set(arm_set):
                    rows = [s for s in st if s["arm"] == a]
                    srv[a] = {f: round(statistics.median([r[f] for r in rows]), 3)
                              for f in ("enc_ms", "build_ms", "send_ms")}
                q = statistics.quantiles(diffs, n=4)
                key = "".join(arm_set)
                timing.setdefault(c, {})
                if key == "AB":
                    timing[c]["A"] = stat(ts[0])
                    timing[c]["B"] = stat(ts[1])
                timing[c][f"pair_{key}"] = {"median_diff_ms": round(statistics.median(diffs), 3),
                                            "iqr_ms": [round(q[0], 3), round(q[2], 3)],
                                            "first_ms": stat(ts[0]),
                                            "second_ms": stat(ts[1]),
                                            "server_side": srv, "bad_bodies": bad}
                seqs[(c, key)] = seq
                ok &= bad == 0
                log(f"{c} {key}: {timing[c][f'pair_{key}']}")
        # ---------------- 4. untimed replay of the timed AB sequence ----------------------
        replay_bad = 0
        for c in CASES:
            for a, k in seqs[(c, "AB")]:
                async with session.get(base + f"/enc/{a}/{k}") as r:
                    body = await r.read()
                if hashlib.sha256(body).hexdigest() != ref_sha[k]:
                    replay_bad += 1
        log(f"untimed replay mismatches: {replay_bad}")
        ok &= replay_bad == 0
    return {"ok": ok, "timing": timing, "replay_bad": replay_bad}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--serve":
        serve(int(sys.argv[2]))
    else:
        main()
