"""Idea 40 step 0: per-stage host timing of the /v1/images/edits request side.

In-process, no model load. Real 2048^2 references from the frozen inputs store.
Each stage is timed separately (median over N alternating A/B iterations), plus
candidate trims (msgspec json decode) so the table shows what is removable.
Also times the whole edit handler and the t2i handler with the idea22 stub
generator (body prebuilt OUTSIDE timing, unlike idea22/handler.py which built
the 9.3 MB body with json.dumps inside its timed region).
"""
import asyncio
import base64
import glob
import hashlib
import io
import json
import logging
import os
import pickle
import statistics
import time
import types

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {
    "check10": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",
    "check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
}
N = int(os.environ.get("N_ITERS", "30"))


def log(*a):
    print(f"[idea40 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def loadavg():
    try:
        return open("/proc/loadavg").read().split()[:3]
    except Exception:
        return None


def find_refs():
    want = {v: k for k, v in REFS.items()}
    found = {}
    for f in sorted(glob.glob(os.path.join(STORE, "**", "*"), recursive=True)):
        if not os.path.isfile(f) or len(found) == len(want):
            continue
        data = open(f, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        if h in want:
            found[want[h]] = data
    assert len(found) == len(want), found.keys()
    return found


def main():
    import msgspec
    import numpy as np
    import torch
    from PIL import Image
    from starlette.requests import Request

    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.logger import logger as trt_logger
    from tensorrt_llm.serve import openai_server as osrv
    from tensorrt_llm.serve import visual_gen_utils as vgu
    from tensorrt_llm.serve.openai_protocol import ImageEditRequest, ImageGenerationRequest
    from tensorrt_llm.visual_gen import params as vparams
    from tensorrt_llm.visual_gen.params import VisualGenParams

    log("load", loadavg())
    trt_logger.set_level("info")
    lg = trt_logger._logger
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.addHandler(logging.FileHandler("/tmp/idea40_step0.log"))
    lg.propagate = False

    refs = find_refs()
    names = ["check10", "check00"]
    pix = {n: torch.from_numpy(np.asarray(Image.open(io.BytesIO(refs[n])).convert("RGB")).copy()).cuda()
           for n in names}
    by_sha = {hashlib.sha256(refs[n]).hexdigest(): n for n in names}
    t2i_img = torch.flip(pix["check00"], dims=[0]).contiguous()
    bodies = {n: json.dumps({"prompt": f"edit {n} keep layout", "model": "flux-3-image-klein",
                             "response_format": "b64_json",
                             "image": base64.b64encode(refs[n]).decode(),
                             "seed": 42, "size": "2048x2048", "num_inference_steps": 16}).encode()
              for n in names}
    for n in names:
        log(f"{n}: png {len(refs[n])} B, body {len(bodies[n])} B")

    stages = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        stages.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)
        return r

    default = VisualGenParams(width=2048, height=2048, num_inference_steps=16)

    class Gen:
        model = None
        extra_param_specs = {}
        executor = types.SimpleNamespace(supports_image_edit=True, pipeline=None)

        @property
        def default_params(self):
            return default.model_copy(deep=True)

    gen = Gen()
    CH = 65536

    async def body_of(body):
        it = iter([body[i:i + CH] for i in range(0, len(body), CH)])
        scope = {"type": "http", "method": "POST", "path": "/v1/images/edits", "headers": []}

        async def receive():
            try:
                return {"type": "http.request", "body": next(it), "more_body": True}
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}

        return await Request(scope, receive).body()

    log("stage loop")
    for i in range(N + 3):
        n = names[i % 2]
        body = bodies[n]
        timed("client_json_dumps(not server)", lambda: json.dumps(json.loads(body)))
        b = timed("starlette_body_join_64k", lambda: asyncio.run(body_of(body)))
        d = timed("json_loads", lambda: json.loads(b))
        d2 = timed("CAND_msgspec_json_decode", lambda: msgspec.json.decode(b))
        assert d2 == d and list(d2) == list(d)
        req = timed("pydantic_ImageEditRequest", lambda: ImageEditRequest(**d))
        timed("default_params", lambda: gen.default_params)
        payload = timed("b64decode_validate", lambda: vgu._decode_base64_media(req.image))
        timed("pil_verify", lambda: vgu._validate_png_jpeg_image(payload))
        p = timed("parse_visual_gen_params_total", lambda: vgu.parse_visual_gen_params(req, gen))
        timed("params_for_log+logger.info",
              lambda: trt_logger.info(
                  f"Editing image: x with params: {osrv._params_for_log(p)} and prompt: {req.prompt}"))
        rp = timed("model_copy_deep", lambda: p.model_copy(deep=True))
        timed("prepare_reference_slots", lambda: vparams.prepare_reference_slots(rp))
        dr = DiffusionRequest(request_id=i, prompt=[req.prompt], params=rp)
        timed("refs_to_shm", dr.refs_to_shm)
        timed("pickle_request", lambda: pickle.dumps(dr))
        timed("refs_from_shm(worker)", dr.refs_from_shm)
        assert dr.params.image_reference[0].content == refs[n]
        timed("sha256_payload(stub only)", lambda: hashlib.sha256(payload).hexdigest())
    log("load", loadavg())

    class Handle:
        def __init__(self, out):
            self._out = out

        async def aresult(self, timeout=None):
            return self._out

    class StubGen(Gen):
        def generate_async(self, inputs, params=None):
            params = params.model_copy(deep=True)
            r = params.image_reference
            img = pix[by_sha[hashlib.sha256(r[0].content).hexdigest()]].clone() if r else t2i_img.clone()
            return Handle(types.SimpleNamespace(image=img, metrics=None))

    srv = object.__new__(osrv.OpenAIServer)
    srv._is_visual_gen = True
    srv.generator = StubGen()
    srv.media_storage_path = osrv.Path("/tmp")
    srv._adjusted_steady_clock = AdjustedSteadyClock()

    def mk(body, path):
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "", "scheme": "http",
                 "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 5555),
                 "headers": [(b"content-type", b"application/json"), (b"host", b"127.0.0.1:8000"),
                             (b"content-length", str(len(body)).encode())],
                 "state": {"server_arrival_time": srv._adjusted_steady_clock.now()}}
        sent = {"d": False}

        async def receive():
            if not sent["d"]:
                sent["d"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        return osrv._MsgspecRequest(scope, receive)

    log("handler loop")
    for i in range(N + 3):
        n = names[i % 2]
        body = bodies[n]
        torch.cuda.synchronize()
        r = timed("HANDLER_edit", lambda: asyncio.run(srv.openai_image_edit(mk(body, "/v1/images/edits"))))
        assert r.status_code == 200, r.body[:300]
        torch.cuda.synchronize()
        req_t = ImageGenerationRequest(prompt="A red fox", size="2048x2048", seed=7,
                                       response_format="b64_json", num_inference_steps=16,
                                       model="flux-3-image-klein")
        r = timed("HANDLER_t2i", lambda: asyncio.run(
            srv.openai_image_generation(req_t, mk(b"{}", "/v1/images/generations"))))
        assert r.status_code == 200
    log("load", loadavg())

    table = {}
    for k, v in stages.items():
        v = v[3:]
        q = statistics.quantiles(v, n=10)
        table[k] = {"median_ms": round(statistics.median(v), 3), "p10": round(q[0], 3),
                    "p90": round(q[-1], 3)}
        log(f"{k:40s} {table[k]}")
    print(json.dumps({"table": table, "load_end": loadavg(), "n": N}), flush=True)


if __name__ == "__main__":
    main()
