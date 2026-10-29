"""Idea 22: /v1/images/edits handler, in-process (no model load).

Drives OpenAIServer.openai_image_edit / openai_image_generation of THIS tree with a stub
VisualGen generator that returns a CUDA uint8 HWC 2048x2048 image (the decoded pixels of the
reference it was given, so A and B give different outputs), on real 2048^2 reference PNGs from
the frozen inputs store. The tensorrt_llm logger is set to INFO (trtllm-serve's default) and
written to a file, as the server writes its log.

Correctness (untimed, same sequence as timing):
  * every b64_json body equals main's code path computed here:
    JSONResponse(content=ImageGenerationResponse(... b64_json=base64.b64encode(
    image_to_bytes(img)).decode() ...).model_dump()).body, with "created" normalized,
  * b64decode(b64_json) == image_to_bytes(img), key order identical (byte-identical body),
  * sequence A, B, A, t2i, A: each matches its own reference, A1 == A3 == A5 bytes,
  * malformed base64 / non-PNG payload / GIF -> HTTP 400 (status + message printed so both
    arms can be compared), response_format=url writes the file (bytes == image_to_bytes) and
    returns a /v1/images/<id>/content URL.
Timing: wall time of the whole handler call (JSON parse of the ~9 MB body, params, log line,
stub generate, encode, response render), alternating A/B, median over N iterations.

Last stdout line: {"batch": {"i2i-2048": {"step_ms":..,"noise_floor":..}, "t2i-2048": {...}},
"output": {"class": "exact"}, "passed": ..., "digests": {...}}
"""
import asyncio
import base64
import glob
import hashlib
import io
import json
import logging
import os
import re
import statistics
import sys
import tempfile
import time
import types

import numpy as np
import torch

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {  # from experiments/ascent-benchmark/inputs.json
    "check10": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",
    "check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
}
N_WARM = int(os.environ.get("N_WARM", "3"))
N_ITERS = int(os.environ.get("N_ITERS", "12"))


def log(*a):
    print(f"[idea22 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_refs():
    want = {v: k for k, v in REFS.items()}
    found = {}
    files = sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    log(f"store files: {len(files)}")
    for f in files:
        if len(found) == len(want):
            break
        name = os.path.basename(f)
        cand = [s for s in want if s in name]
        if not cand and len(files) > 80:
            continue
        data = open(f, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        if h in want:
            found[want[h]] = data
    if len(found) != len(want):
        raise SystemExit(f"references not found in {STORE}: have {list(found)}")
    return found


def main():
    from PIL import Image
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.logger import logger as trt_logger
    from tensorrt_llm.media.encoding import image_to_bytes
    from tensorrt_llm.serve import openai_server as osrv
    from tensorrt_llm.serve.openai_protocol import (ImageGenerationRequest,
                                                    ImageGenerationResponse, ImageObject)
    from tensorrt_llm.visual_gen.params import VisualGenParams

    tmpd = tempfile.mkdtemp(prefix="idea22-")
    log_path = os.path.join(tmpd, "server.log")
    trt_logger.set_level("info")
    lg = trt_logger._logger
    for h in list(lg.handlers):
        lg.removeHandler(h)
    fh = logging.FileHandler(log_path)
    lg.addHandler(fh)
    lg.propagate = False

    refs = find_refs()
    names = ["check10", "check00"]
    pix = {}
    for n in names:
        im = Image.open(io.BytesIO(refs[n])).convert("RGB")
        pix[n] = torch.from_numpy(np.asarray(im).copy()).cuda()
        log(f"ref {n}: {len(refs[n])} bytes, {tuple(pix[n].shape)}")
    by_sha = {hashlib.sha256(refs[n]).hexdigest(): n for n in names}
    t2i_img = torch.flip(pix["check00"], dims=[0]).contiguous()

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

        def _out(self, inputs, params):
            refs_ = params.image_reference
            if refs_:
                n = by_sha[hashlib.sha256(refs_[0].content).hexdigest()]
                img = pix[n].clone()
            else:
                img = t2i_img.clone()
            return types.SimpleNamespace(image=img, metrics=None)

        def generate_async(self, inputs, params=None):
            params = params.model_copy(deep=True)  # as VisualGen.generate_async
            return Handle(self._out(inputs, params))

        def generate(self, inputs, params=None):
            return self.generate_async(inputs, params).result()

    srv = object.__new__(osrv.OpenAIServer)
    srv._is_visual_gen = True
    srv.generator = StubGen()
    srv.media_storage_path = osrv.Path(tmpd)
    srv._adjusted_steady_clock = AdjustedSteadyClock()

    def make_request(body, path):
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "scheme": "http", "server": ("127.0.0.1", 8000),
                 "client": ("127.0.0.1", 5555),
                 "headers": [(b"content-type", b"application/json"),
                             (b"host", b"127.0.0.1:8000"),
                             (b"content-length", str(len(body)).encode())],
                 "state": {"server_arrival_time": srv._adjusted_steady_clock.now()}}
        sent = {"done": False}

        async def receive():
            if not sent["done"]:
                sent["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        return Request(scope, receive)

    b64_refs = {n: base64.b64encode(refs[n]).decode() for n in names}
    PROMPT = {"check10": "Make it look like a watercolor painting, keep \"the\" layout",
              "check00": "Turn the sky orange at sunset — café lights on"}

    def edit_body(n, response_format="b64_json", image=None):
        return json.dumps({"prompt": PROMPT[n] if n else "x",
                           "model": "flux-3-image-klein",
                           "response_format": response_format,
                           "image": image if image is not None else b64_refs[n],
                           "seed": 42, "size": "2048x2048",
                           "num_inference_steps": 16}).encode()

    async def call_edit(body):
        return await srv.openai_image_edit(make_request(body, "/v1/images/edits"))

    T2I_PROMPT = "A red fox on a snowy hill, \"sharp\" text SIGN"

    async def call_t2i():
        req = ImageGenerationRequest(prompt=T2I_PROMPT, size="2048x2048", seed=7,
                                     response_format="b64_json", num_inference_steps=16,
                                     model="flux-3-image-klein")
        return await srv.openai_image_generation(req, make_request(b"{}", "/v1/images/generations"))

    CREATED = re.compile(rb'"created":\d+')

    def norm(body):
        return CREATED.sub(b'"created":0', body, count=1)

    def ref_body(img, prompt, size):
        # main's edit (and pre-splicer generation) path.
        return norm(JSONResponse(content=ImageGenerationResponse(
            created=0,
            data=[ImageObject(b64_json=base64.b64encode(image_to_bytes(img, format="PNG")).decode("utf-8"),
                              revised_prompt=prompt)],
            output_format="png", size=size).model_dump()).body)

    ref_cache = {}

    def ref_for(key):
        if key not in ref_cache:
            if key == "t2i":
                ref_cache[key] = ref_body(t2i_img, T2I_PROMPT, "2048x2048")
            else:
                ref_cache[key] = ref_body(pix[key], PROMPT[key], "2048x2048")
        return ref_cache[key]

    def run_seq(seq, timed):
        out = []
        for key in seq:
            torch.cuda.synchronize()
            log_before = os.path.getsize(log_path)
            t0 = time.perf_counter()
            if key == "t2i":
                resp = asyncio.run(call_t2i())
            else:
                resp = asyncio.run(call_edit(edit_body(key)))
            ms = (time.perf_counter() - t0) * 1e3
            fh.flush()
            log_bytes = os.path.getsize(log_path) - log_before
            out.append((key, resp, ms, log_bytes))
        return out

    # ---------------- correctness, untimed, same sequence shape as timing --------------
    log("correctness: A, B, A, t2i, A + warmup/iteration sequence")
    seq_c = ["check10", "check00", "check10", "t2i", "check10"]
    res = run_seq(seq_c, timed=False)
    checks = []
    digests = {}
    bodies_by_key = {}
    for i, (key, resp, ms, lb) in enumerate(res):
        body = norm(resp.body)
        ref = ref_for(key)
        js = json.loads(resp.body)
        png = base64.b64decode(js["data"][0]["b64_json"])
        img = t2i_img if key == "t2i" else pix[key]
        ok = (resp.status_code == 200 and body == ref and png == image_to_bytes(img, format="PNG")
              and list(js.keys()) == list(json.loads(ref).keys())
              and resp.media_type == "application/json")
        bodies_by_key.setdefault(key, []).append(body)
        d = hashlib.sha256(body).hexdigest()
        digests[f"{i}:{key}"] = d
        checks.append({"i": i, "key": key, "status": resp.status_code, "body_eq_main_path": body == ref,
                       "png_eq": png == image_to_bytes(img, format="PNG"), "ok": ok,
                       "body_len": len(resp.body), "png_sha": hashlib.sha256(png).hexdigest()[:16],
                       "body_sha": d[:16], "log_bytes": lb,
                       "server_timing": resp.headers.get("server-timing", "")[:80]})
        log(checks[-1])
    aba_ok = len(set(bodies_by_key["check10"])) == 1 and bodies_by_key["check10"][0] != bodies_by_key["check00"][0]

    # Iteration-shaped repeat: same warmup/iteration counts as timing, compared to iter 0.
    seq_t = ["check10" if i % 2 == 0 else "check00" for i in range(N_WARM + N_ITERS)]
    rep = run_seq(seq_t, timed=False)
    rep_ok = all(r[1].status_code == 200 and norm(r[1].body) == ref_for(r[0]) for r in rep)
    log(f"iteration-shaped repeat ({len(seq_t)} calls) all == main path: {rep_ok}")

    # ---------------- error paths --------------------------------------------------------
    gif = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(gif, format="GIF")
    errs = {}
    for tag, image in (("not_base64", "this is !!! not base64 ***"),
                       ("not_image", base64.b64encode(b"hello, I am not an image" * 10).decode()),
                       ("gif", base64.b64encode(gif.getvalue()).decode())):
        r = asyncio.run(call_edit(edit_body("check10", image=image)))
        try:
            msg = json.loads(r.body)
        except Exception:
            msg = r.body[:200].decode(errors="replace")
        errs[tag] = {"status": r.status_code, "body": msg}
        log(f"error {tag}: {r.status_code} {msg}")
    err_ok = all(e["status"] == 400 for e in errs.values())

    # ---------------- url transport ------------------------------------------------------
    r = asyncio.run(call_edit(edit_body("check00", response_format="url")))
    js = json.loads(r.body)
    url = js["data"][0]["url"]
    m = re.search(r"/v1/images/(image_[0-9a-f]+)/content\?i=0$", url or "")
    url_ok = bool(r.status_code == 200 and m)
    if url_ok:
        f = os.path.join(tmpd, f"{m.group(1)}_0.png")
        url_ok = os.path.exists(f) and open(f, "rb").read() == image_to_bytes(pix["check00"], format="PNG")
    url_shape = {k: (v if k != "data" else [{kk: (vv if kk != "url" else "<url>") for kk, vv in d.items()} for d in v])
                 for k, v in js.items() if k != "created"}
    log(f"url transport ok={url_ok} {url_shape}")

    # ---------------- step 0 micro-timings ------------------------------------------------
    from tensorrt_llm.visual_gen.params import MediaRef
    p = VisualGenParams(width=2048, height=2048, num_inference_steps=16, seed=42,
                        image_reference=[MediaRef(content=refs["check10"], format="bytes")])
    helper = getattr(osrv, "_params_for_log", None)
    micro = {}

    def tmed(fn, n=7):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return round(statistics.median(ts), 2)

    micro["log_full_params_ms"] = tmed(lambda: trt_logger.info(f"Editing image: x with params: {p} and prompt: y"))
    micro["fstring_full_params_ms"] = tmed(lambda: f"Editing image: x with params: {p} and prompt: y")
    if helper is not None:
        micro["log_helper_params_ms"] = tmed(lambda: trt_logger.info(f"Editing image: x with params: {helper(p)} and prompt: y"))
        micro["helper_line_len"] = len(f"{helper(p)}")
    img = pix["check10"]

    def old_render():
        return JSONResponse(content=ImageGenerationResponse(
            created=0, data=[ImageObject(b64_json=base64.b64encode(image_to_bytes(img, format="PNG")).decode("utf-8"),
                                         revised_prompt="p")], output_format="png", size="2048x2048").model_dump()).body

    def new_render():
        sp = osrv._B64JsonSplicer()
        resp = ImageGenerationResponse(created=0, data=[ImageObject(
            b64_json=sp.placeholder(osrv.image_to_b64(img, format="PNG")), revised_prompt="p")],
            output_format="png", size="2048x2048")
        return sp.response(resp.model_dump(), {}).body

    micro["encode_b64_jsonresponse_ms"] = tmed(old_render)
    micro["image_to_b64_splicer_ms"] = tmed(new_render)
    micro["render_bytes_equal"] = old_render() == new_render()
    log(f"micro {micro}")

    # ---------------- timing --------------------------------------------------------------
    log(f"timing: warmup {N_WARM}, iters {N_ITERS}, alternating A/B edits; then t2i")
    tim = run_seq(seq_t, timed=True)
    t_edit = [r[2] for r in tim[N_WARM:]]
    log_bytes = [r[3] for r in tim[N_WARM:]]
    tim_t2i = run_seq(["t2i"] * (N_WARM + N_ITERS), timed=True)
    t_t2i = [r[2] for r in tim_t2i[N_WARM:]]
    timing_ok = all(r[1].status_code == 200 for r in tim + tim_t2i)

    def stat(ts):
        q = statistics.quantiles(ts, n=10)
        med = statistics.median(ts)
        return {"step_ms": round(med, 3), "noise_floor": round((q[-1] - q[0]) / med, 4),
                "mean_ms": round(statistics.mean(ts), 3), "min_ms": round(min(ts), 3),
                "samples": [round(x, 2) for x in ts]}

    s_edit, s_t2i = stat(t_edit), stat(t_t2i)
    log(f"edit handler {s_edit}")
    log(f"t2i handler {s_t2i}")
    passed = (all(c["ok"] for c in checks) and aba_ok and rep_ok and err_ok and url_ok and timing_ok)
    print(json.dumps({
        "batch": {"i2i-2048": s_edit, "t2i-2048": s_t2i},
        "output": {"class": "exact"},
        "passed": bool(passed),
        "compared": "handler body vs main's JSONResponse(model_dump()) + base64(image_to_bytes) path; "
                    "A,B,A,t2i,A; error statuses; url transport",
        "checks": checks, "aba_ok": aba_ok, "repeat_ok": rep_ok, "errors": errs, "errors_ok": err_ok,
        "url_ok": url_ok, "url_shape": url_shape, "micro": micro,
        "edit_log_bytes_per_request": statistics.median(log_bytes),
        "digests": digests,
        "tree_has_params_for_log": helper is not None,
    }), flush=True)


if __name__ == "__main__":
    main()
