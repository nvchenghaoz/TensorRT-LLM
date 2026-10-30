"""Idea 40: /v1/images/edits request side -- correctness + interleaved A/B timing.

One process, no model load, real 2048^2 references. Arm A = main's code (main's
``await raw_request.json()`` and main's ``_decode_base64_media`` copied verbatim,
swapped in by module attribute, asserted), arm B = this tree.

Checks (untimed):
  1. msgspec edge corpus: every body msgspec accepts decodes type-strictly equal
     to json.loads (key order too); rejected bodies fall back (B == A then).
  2. base64 corpus: GPU canonical decode == base64.b64decode(validate=True) on
     canonical inputs; returns None (fallback) on every non-canonical/invalid one;
     _decode_base64_media A == B on all (value or None / exception text).
  3. handler parity on both references: response body (created normalized),
     VisualGenParams.model_dump (content replaced by sha256) + reference sha256,
     A,B,A,t2i sequence per arm; malformed request corpus status + body A == B.
  3b. deep nesting (3000 levels) at the default recursion limit: tree raises
     exactly json.loads's RecursionError (also in the error-parity corpus).
  4. after timing: untimed interleaved A/A2/B sequence matched against arm-A
     iteration-0 bodies.
Timing: three arms in one process (A = main verbatim, A2 = second copy of main
= in-process A/A control, B = tree); 10 warmups per arm per case, then
2*ROUNDS*PER_ROUND iterations, each calling every arm on the same input in
rotating order ABA2/BA2A/A2AB, i2i (reference alternating) then t2i. Per case
(i2i pooled, per reference, t2i): per-arm median/p10/p90/rel-std, paired B-A and
A2-A median+IQR, median of per-round (PER_ROUND iterations) ratios; load1.

ARM env var: "both" (default) prints the full report; "A"/"B" prints the
evidence_format JSON for that arm only (still timing both interleaved).
"""
import asyncio
import base64
import binascii
import glob
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import statistics
import time
import types

STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
         "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
REFS = {
    "check10": "76bf07ca4e247699c100eaeff046d3335fea0019844ab3a2b60206e2d798a714",
    "check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
}
ROUNDS = int(os.environ.get("ROUNDS", "20"))
PER_ROUND = 10
ARM = os.environ.get("ARM", "both")
if len(__import__("sys").argv) > 1:
    ARM = __import__("sys").argv[1]


def log(*a):
    print(f"[idea40 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load1():
    try:
        return float(open("/proc/loadavg").read().split()[0])
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


def strict_eq(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return repr(a) == repr(b)
    return a == b


def main():
    import sys
    DEFAULT_RECURSION_LIMIT = sys.getrecursionlimit()  # the server's (nothing in tensorrt_llm sets it)
    sys.setrecursionlimit(50000)
    import msgspec
    import numpy as np
    import torch
    from PIL import Image

    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.logger import logger as trt_logger
    from tensorrt_llm.serve import openai_server as osrv
    from tensorrt_llm.serve import visual_gen_utils as vgu
    from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest
    from tensorrt_llm.visual_gen import params as vparams
    from tensorrt_llm.visual_gen.params import VisualGenParams

    trt_logger.set_level("info")
    lg = trt_logger._logger
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.addHandler(logging.FileHandler("/tmp/idea40_pair.log"))
    lg.propagate = False

    # ---------------- arms ----------------
    async def main_request_json(raw_request):  # main: body = await raw_request.json()
        return await raw_request.json()

    def main_decode_base64_media(value):  # verbatim from main e0d4536c
        payload = value
        if value.startswith("data:"):
            _, sep, payload = value.partition(",")
            if not sep:
                return None
        if len(payload) > ((vgu.IMAGE_EDIT_MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise ValueError(
                "Image edit input exceeds the per-image byte limit "
                f"before decoding ({len(payload)} encoded bytes)."
            )
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return None

    # In the base tree (run_gpu_pair baseline arm) the helpers do not exist, so
    # "tree" == main and the job is an A/A control.
    tree_has_change = hasattr(osrv, "_request_json_fast")
    if not tree_has_change:
        osrv._request_json_fast = main_request_json
    gpu_b64 = getattr(vgu, "_b64decode_canonical_gpu", lambda p: None)
    tree = {"json": osrv._request_json_fast, "b64": vgu._decode_base64_media}
    mainv = {"json": main_request_json, "b64": main_decode_base64_media}

    def fcopy(f):  # a second, distinct function object with main's verbatim code
        g = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        assert g is not f and g.__code__ is f.__code__
        return g

    mainv2 = {"json": fcopy(main_request_json), "b64": fcopy(main_decode_base64_media)}
    ARMS = {"A": mainv, "A2": mainv2, "B": tree}

    def use(arm):
        src = ARMS[arm]
        osrv._request_json_fast = src["json"]
        vgu._decode_base64_media = src["b64"]
        assert osrv._request_json_fast is src["json"] and vgu._decode_base64_media is src["b64"]

    passed = {}

    # ---------------- 1. msgspec edge corpus ----------------
    corpus = [
        b'{"a":1}', b'{"a":1,"a":2}', b'{"b":1,"a":2,"b":3}', b'{"x":-0}', b'{"x":-0.0}',
        b'{"x":1e400}', b'{"x":-1e400}', b'{"x":1E5}', b'{"x":1e+5}', b'{"x":1.0e-5}',
        b'{"x":0.1}', b'{"x":1.7976931348623157e308}', b'{"x":5e-324}', b'{"x":2.5e-324}',
        b'{"x":123456789012345678901234567890}', b'{"x":-123456789012345678901234567890}',
        b'{"x":18446744073709551615}', b'{"x":18446744073709551616}', b'{"x":-9223372036854775808}',
        b'{"x":-9223372036854775809}', b'{"x":9223372036854775807}', b'{"x":1.0}', b'{"x":10}',
        b'{"x":0.30000000000000004}', b'{"x":123456789012345678901234567890.5}',
        b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":true,"y":false,"z":null}',
        b'{"s":"\\ud83d\\ude00"}', b'{"s":"\\ud800"}', b'{"s":"\\udc00x"}', b'{"s":"\\u0000"}',
        b'{"s":"a\\/b\\"c\\\\d\\b\\f\\n\\r\\t"}', b'{"s":"\xc3\xa9\xe2\x80\x94"}',
        b'{"s":"\xed\xa0\x80"}', b'{"s":"\xff"}', b'\xef\xbb\xbf{"a":1}', b'{"s":"a\x01b"}',
        b'{"s":"a\tb"}', b' \t\r\n{"a":1} \t\r\n', b'{"a":1}\x0c', b'\x0c{"a":1}', b'{"a":1}x',
        b'{"a":1,}', b'[1,2]', b'"str"', b'123', b'null', b'', b'{', b'{"a":01}', b'{"a":1.}',
        b'{"a":.5}', b'{"a":+1}', b'{"a":0x10}', b"{'a':1}", b'{"a":[1,[2,[3,{"b":[]}]]]}',
        b'[' * 2000 + b']' * 2000, b'{"a":' * 500 + b'1' + b'}' * 500,
        '{"a":"é"}'.encode("utf-16"), '{"a":1}'.encode("utf-32"), b'{"a":1}\x00',
        b'{"\\u0061":1,"a":2}', b'{"a":"\\x41"}', b'{"a":"\\u00G1"}', b'{"a":1e}', b'{"a":-}',
        b'{"a":1.5e3}', b'{"a":12.30}', b'{"a":1e-400}', b'{"a":[]}', b'{"a":{}}', b'{}',
        b'{"prompt":"p","image":"QUJD","seed":42,"size":"2048x2048","n":1.0}',
    ]
    for _ in range(200):  # random float/int literals
        r = random.Random(_)
        lit = r.choice([str(r.uniform(-1e20, 1e20)), repr(r.random() * 10 ** r.randint(-300, 300)),
                        str(r.randint(-2 ** 80, 2 ** 80)), f"{r.random():.17e}", str(r.randint(-10, 10))])
        corpus.append(('{"v":%s}' % lit).encode())
    accepted = rejected = 0
    mism = []
    for body in corpus:
        try:
            m = msgspec.json.decode(body)
        except Exception:
            rejected += 1
            continue
        accepted += 1
        try:
            j = json.loads(body)
        except Exception as e:
            mism.append((body[:60].decode("latin1"), "json.loads raised", repr(e)[:80]))
            continue
        if not strict_eq(m, j):
            mism.append((body[:60].decode("latin1"), repr(m)[:80], repr(j)[:80]))
    log(f"msgspec corpus: {len(corpus)} bodies, accepted {accepted}, rejected {rejected}, mismatches {mism[:10]}")
    passed["msgspec_corpus"] = not mism

    # ---------------- 2. base64 corpus ----------------
    rnd = random.Random(0)
    b64_bad = []
    b64_cases = 0
    for size in [3 << 19, (3 << 19) + 1, (3 << 19) + 2, 7_000_001, 7_000_002, 7_000_003]:
        data = rnd.randbytes(size)
        s = base64.b64encode(data).decode()
        cases = {"canonical": s, "data_uri": "data:image/png;base64," + s}
        if s.endswith("="):
            # flip pad bits -> non-canonical but maybe strict-accepted
            pre = s.rstrip("=")
            pads = s[len(pre):]
            last = pre[-1]
            alt = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            idx = alt.index(last)
            cases["nonzero_pad_bits"] = pre[:-1] + alt[idx | 1] + pads
        cases["bad_char_mid"] = s[:1000] + "*" + s[1001:]
        cases["eq_mid"] = s[:1000] + "=" + s[1001:]
        cases["triple_pad"] = s[:-4] + "A==="
        cases["len_mod4"] = s[:-1] if not s.endswith("=") else s + "AA"
        cases["non_ascii"] = s[:500] + "é" + s[501:]
        cases["newline"] = s[:76] + "\n" + s[77:]
        cases["extra_after_pad"] = s + "=AAA" if s.endswith("=") else s[:-4] + "AA==AAAA"
        for tag, val in cases.items():
            b64_cases += 1
            payload = val.partition(",")[2] if val.startswith("data:") else val
            g = gpu_b64(payload) if tree_has_change else data if tag in ("canonical", "data_uri") else None
            try:
                ref = base64.b64decode(payload, validate=True)
            except Exception:
                ref = None
            canonical = tag in ("canonical", "data_uri")
            if canonical and g != data:
                b64_bad.append((size, tag, "gpu != data"))
            if g is not None and g != ref:
                b64_bad.append((size, tag, "gpu accepted but != strict"))
            if not canonical and g is not None and tag != "nonzero_pad_bits":
                b64_bad.append((size, tag, "gpu accepted non-canonical"))
            if tag == "nonzero_pad_bits" and g is not None:
                b64_bad.append((size, tag, "gpu accepted nonzero pad bits"))

            def run(fn):
                try:
                    return ("ok", fn(val))
                except Exception as e:
                    return ("exc", type(e).__name__, str(e))
            ra, rb = run(main_decode_base64_media), run(tree["b64"])
            if ra != rb:
                b64_bad.append((size, tag, "A != B"))
    log(f"b64 corpus: {b64_cases} cases, problems {b64_bad}")
    passed["b64_corpus"] = not b64_bad

    # ---------------- 3. handler parity ----------------
    refs = find_refs()
    names = ["check10", "check00"]
    pix = {n: torch.from_numpy(np.asarray(Image.open(io.BytesIO(refs[n])).convert("RGB")).copy()).cuda()
           for n in names}
    key_of = {(len(refs[n]), refs[n][-64:]): n for n in names}
    t2i_img = torch.flip(pix["check00"], dims=[0]).contiguous()
    b64s = {n: base64.b64encode(refs[n]).decode() for n in names}
    PROMPT = {"check10": "Make it look like a watercolor painting, keep \"the\" layout",
              "check00": "Turn the sky orange at sunset — café lights on"}

    def edit_body(n, **over):
        d = {"prompt": PROMPT[n], "model": "flux-3-image-klein", "response_format": "b64_json",
             "image": b64s[n], "seed": 42, "size": "2048x2048", "num_inference_steps": 16}
        d.update(over)
        return json.dumps(d).encode()

    default = VisualGenParams(width=2048, height=2048, num_inference_steps=16)
    captured = []

    class Handle:
        def __init__(self, out):
            self._out = out

        async def aresult(self, timeout=None):
            return self._out

    class StubGen:
        model = None
        extra_param_specs = {}
        executor = types.SimpleNamespace(supports_image_edit=True, pipeline=None)

        @property
        def default_params(self):
            return default.model_copy(deep=True)

        def generate_async(self, inputs, params=None):
            # VisualGen.generate_async's host work on the reference.
            rp = params.model_copy(deep=True)
            vparams.prepare_reference_slots(rp)
            dr = DiffusionRequest(request_id=0, prompt=[inputs], params=rp)
            dr.refs_to_shm()
            dr.refs_from_shm()
            if CAPTURE[0]:
                dump = rp.model_dump()
                for r in dump.get("image_reference") or []:
                    r["content"] = hashlib.sha256(r["content"]).hexdigest()
                captured.append(json.dumps(dump, sort_keys=True, default=repr))
            r = rp.image_reference
            img = pix[key_of[(len(r[0].content), r[0].content[-64:])]].clone() if r else t2i_img.clone()
            return Handle(types.SimpleNamespace(image=img, metrics=None))

    CAPTURE = [False]
    srv = object.__new__(osrv.OpenAIServer)
    srv._is_visual_gen = True
    srv.generator = StubGen()
    srv.media_storage_path = osrv.Path("/tmp")
    srv._adjusted_steady_clock = AdjustedSteadyClock()

    def mk(body, path, ctype=b"application/json"):
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "", "scheme": "http",
                 "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 5555),
                 "headers": [(b"content-type", ctype), (b"host", b"127.0.0.1:8000"),
                             (b"content-length", str(len(body)).encode())],
                 "state": {"server_arrival_time": srv._adjusted_steady_clock.now()}}
        chunks = [body[i:i + 65536] for i in range(0, len(body), 65536)] or [b""]
        it = iter(range(len(chunks)))

        async def receive():
            try:
                i = next(it)
                return {"type": "http.request", "body": chunks[i], "more_body": i < len(chunks) - 1}
            except StopIteration:
                return {"type": "http.disconnect"}

        return osrv._MsgspecRequest(scope, receive)

    CREATED = re.compile(rb'"created":\d+')

    def norm(b):
        return CREATED.sub(b'"created":0', b, count=1)

    def call(key, body=None):
        if key == "t2i":
            req = ImageGenerationRequest(prompt="A red fox on a snowy hill, \"sharp\" text SIGN",
                                         size="2048x2048", seed=7, response_format="b64_json",
                                         num_inference_steps=16, model="flux-3-image-klein")
            return asyncio.run(srv.openai_image_generation(req, mk(b"{}", "/v1/images/generations")))
        return asyncio.run(srv.openai_image_edit(mk(body if body is not None else bodies[key],
                                                    "/v1/images/edits")))

    bodies = {n: edit_body(n) for n in names}
    seq = ["check10", "check00", "check10", "t2i"]
    res = {}
    for arm in ("A", "B"):
        use(arm)
        captured.clear()
        CAPTURE[0] = True
        out = []
        for k in seq:
            r = call(k)
            out.append((r.status_code, hashlib.sha256(norm(r.body)).hexdigest(), r.media_type))
        CAPTURE[0] = False
        res[arm] = {"resp": out, "params": [hashlib.sha256(c.encode()).hexdigest() for c in captured]}
        log(f"arm {arm}: {[(o[0], o[1][:12]) for o in out]} params {[p[:12] for p in res[arm]['params']]}")
    par_ok = (res["A"] == res["B"] and all(o[0] == 200 for o in res["A"]["resp"])
              and res["A"]["resp"][0] == res["A"]["resp"][2] and res["A"]["resp"][0] != res["A"]["resp"][1]
              and len(res["A"]["params"]) == 4)
    passed["handler_parity_ABAt2i"] = par_ok
    ref_resp = {k: res["A"]["resp"][i] for i, k in enumerate(seq)}

    # malformed / edge requests
    gif = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(gif, format="GIF")
    big = b64s["check10"]
    bad = {
        "not_base64": edit_body("check10", image="this is !!! not base64 ***"),
        "not_image": edit_body("check10", image=base64.b64encode(b"hello, not an image" * 100000).decode()),
        "gif": edit_body("check10", image=base64.b64encode(gif.getvalue()).decode()),
        "b64_bad_char_big": edit_body("check10", image=big[:5000] + "*" + big[5001:]),
        "b64_eq_mid_big": edit_body("check10", image=big[:5000] + "=" + big[5001:]),
        "b64_len_mod4_big": edit_body("check10", image=big[:-1]),
        "b64_newline_big": edit_body("check10", image=big[:76] + "\n" + big[77:]),
        "data_uri_big": edit_body("check10", image="data:image/png;base64," + big),
        "truncated_png_big": edit_body("check10", image=base64.b64encode(refs["check10"][:-3000]).decode()),
        "invalid_json": b'{"prompt": "x", "image": ',
        "empty_body": b"",
        "top_level_array": b"[1,2,3]",
        "nan_seed": b'{"prompt":"x","image":"QUJD","seed":NaN}',
        "bom_body": b"\xef\xbb\xbf" + edit_body("check00"),
        "invalid_utf8": b'{"prompt":"\xff","image":"QUJD"}',
        "lone_surrogate_prompt": b'{"prompt":"\\ud800","image":"QUJD"}',
        "dup_keys_image": edit_body("check10")[:-1] + b', "image": "' + b64s["check00"].encode() + b'"}',
        "big_int_seed": edit_body("check10", seed=2 ** 70),
        "float_seed": edit_body("check10", seed=1.5),
        "missing_prompt": json.dumps({"image": big[:100]}).encode(),
        "extra_field": edit_body("check00", bogus=1),
        "mask_set": edit_body("check00", mask="QUJD"),
        "list_images": edit_body("check00", image=[b64s["check00"]]),
        "empty_list": edit_body("check00", image=[]),
        "deep_nesting_3000": b"[" * 3000 + b"]" * 3000,
        "deep_nesting_obj_3000": b'{"a":' * 3000 + b"1" + b"}" * 3000,
    }
    DEEP = ("deep_nesting_3000", "deep_nesting_obj_3000")
    err = {"A": {}, "B": {}}
    for arm in ("A", "B"):
        use(arm)
        for tag, body in bad.items():
            if tag in DEEP:  # run at the server's default recursion limit
                sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)
                try:
                    r = call("edit", body)
                finally:
                    sys.setrecursionlimit(50000)
            else:
                r = call("edit", body)
            err[arm][tag] = (r.status_code, hashlib.sha256(norm(r.body)).hexdigest()[:16],
                             r.body[:160].decode(errors="replace") if r.status_code != 200 else "")
    diffs = {t: (err["A"][t], err["B"][t]) for t in bad if err["A"][t] != err["B"][t]}
    for t in bad:
        log(f"edge {t}: A={err['A'][t][:2]} {err['A'][t][2][:100]!r}")
    log(f"edge diffs A vs B: {diffs}")
    passed["error_parity"] = not diffs

    # deep nesting at the server's default recursion limit: tree's decoder must
    # raise exactly json.loads's error (msgspec's RecursionError is swallowed).
    deep_ok = True
    sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)
    try:
        for tag in DEEP:
            body = bad[tag]

            def exc_of(fn):
                try:
                    fn()
                    return ("ok",)
                except BaseException as e:  # noqa
                    return (type(e).__name__, str(e))
            ms = exc_of(lambda: msgspec.json.Decoder().decode(body))
            jl = exc_of(lambda: json.loads(body))
            tr = exc_of(lambda: asyncio.run(tree["json"](mk(body, "/v1/images/edits"))))
            mn = exc_of(lambda: asyncio.run(main_request_json(mk(body, "/v1/images/edits"))))
            log(f"deep {tag}: msgspec={ms} json.loads={jl} tree={tr} main={mn}")
            deep_ok &= tr == jl == mn and tr[0] == "RecursionError"
    finally:
        sys.setrecursionlimit(50000)
    passed["deep_nesting_fallback"] = deep_ok
    # ascii-only bodies: also msgpack-header route untouched (not exercised by this route's client)

    # ---------------- timing ----------------
    def one(key):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = call(key)
        ms = (time.perf_counter() - t0) * 1e3
        assert r.status_code == 200
        return ms

    # Per iteration every arm runs the same input, order rotating ABA2 / BA2A / A2AB.
    # Each iteration does one i2i call per arm (reference alternating by
    # iteration) and then one t2i call per arm (same rotated order), so both
    # cases get ITERS paired samples per arm. A round = PER_ROUND iterations.
    ITERS = ROUNDS * PER_ROUND * 2  # default 400 -> 200 per i2i reference, 400 t2i
    ORDERS = [("A", "A2", "B"), ("A2", "B", "A"), ("B", "A", "A2")]
    AR = ("A", "A2", "B")
    l_poll0 = load1()
    t_poll = time.time()
    while (load1() or 0) >= 4.0 and time.time() - t_poll < 180:  # poll <= 3 min, then run anyway
        time.sleep(10)
    log(f"load poll: load1 {l_poll0} -> {load1()} after {time.time() - t_poll:.0f} s")
    l_start = load1()
    log(f"timing: load1 {l_start}, iters {ITERS} (rounds of {PER_ROUND}), arms {AR}")
    for arm in AR:  # 10 warmups per arm per case
        use(arm)
        for i in range(10):
            one(names[i % 2])
            one("t2i")
    cases = ["i2i", "i2i:check10", "i2i:check00", "t2i"]
    T = {c: {a: [] for a in AR} for c in cases}
    rnd_of = {c: [] for c in cases}  # round index per sample
    for it in range(ITERS):
        order = ORDERS[it % 3]
        ref = names[it % 2]
        rd = it // PER_ROUND
        for arm in order:
            use(arm)
            x = one(ref)
            T["i2i"][arm].append(x)
            T["i2i:" + ref][arm].append(x)
        for arm in order:
            use(arm)
            T["t2i"][arm].append(one("t2i"))
        rnd_of["i2i"].append(rd)
        rnd_of["i2i:" + ref].append(rd)
        rnd_of["t2i"].append(rd)
        if it % 50 == 49:
            log(f"iter {it}: i2i A {statistics.median(T['i2i']['A'][-50:]):.2f} "
                f"A2 {statistics.median(T['i2i']['A2'][-50:]):.2f} B {statistics.median(T['i2i']['B'][-50:]):.2f} "
                f"| t2i A {statistics.median(T['t2i']['A'][-50:]):.3f} A2 {statistics.median(T['t2i']['A2'][-50:]):.3f} "
                f"B {statistics.median(T['t2i']['B'][-50:]):.3f} load1 {load1()}")
    l_end = load1()

    def st(xs):
        q = statistics.quantiles(xs, n=10)
        med = statistics.median(xs)
        return {"step_ms": round(med, 3), "p10": round(q[0], 3), "p90": round(q[-1], 3),
                "noise_floor": round((q[-1] - q[0]) / med, 4),
                "rel_std": round(statistics.pstdev(xs) / statistics.mean(xs), 4), "n": len(xs)}

    def cmp(case, arm):
        a, b = T[case]["A"], T[case][arm]
        d = [y - x for x, y in zip(a, b)]
        dq = statistics.quantiles(d, n=4)
        rr = []
        for r in sorted(set(rnd_of[case])):
            idx = [i for i, k in enumerate(rnd_of[case]) if k == r]
            rr.append(statistics.median([b[i] for i in idx]) / statistics.median([a[i] for i in idx]))
        rq = statistics.quantiles(rr, n=10)
        # per-rotation ratio = per iteration (one step of the A/A2/B rotation,
        # same input for every arm); median with a bootstrap 95% CI (fixed seed).
        pr = [y / x for x, y in zip(a, b)]
        g = random.Random(1234)
        boots = sorted(statistics.median(g.choices(pr, k=len(pr))) for _ in range(2000))
        return {"ratio_per_rotation_median": round(statistics.median(pr), 4),
                "ratio_per_rotation_ci95": [round(boots[49], 4), round(boots[1949], 4)],
                "paired_diff_ms_median": round(statistics.median(d), 3),
                "paired_diff_iqr": [round(dq[0], 3), round(dq[-1], 3)],
                "ratio_median_of_rounds": round(statistics.median(rr), 4),
                "ratio_rounds_p10_p90": [round(rq[0], 4), round(rq[-1], 4)], "rounds": len(rr),
                "pairs": len(d)}

    summary = {"load1_start": l_start, "load1_end": l_end, "load1_before_poll": l_poll0}
    for c in cases:
        summary[c] = {"per_arm": {a: st(T[c][a]) for a in AR},
                      "B_vs_A": cmp(c, "B"), "A2_vs_A": cmp(c, "A2")}
        log(f"timing {c}: {json.dumps(summary[c])}")
    log(f"load1 start {l_start} end {l_end}")

    # Verdict rules fixed in advance (area lead 1283 / top lead 1277).
    def ci(c, k):
        return summary[c][k]["ratio_per_rotation_ci95"]
    verdict = {}
    void = any(not (ci(c, "A2_vs_A")[0] <= 1.0 <= ci(c, "A2_vs_A")[1]) for c in ("i2i", "t2i"))
    verdict["void_A2_ci_excludes_1"] = void
    verdict["t2i_pass"] = ci("t2i", "B_vs_A")[0] <= 1.0  # CI includes 1 or lies below it
    verdict["i2i_pass"] = ci("i2i", "B_vs_A")[1] < 1.0 and ci("i2i", "B_vs_A")[1] < ci("i2i", "A2_vs_A")[0]
    summary["verdict"] = verdict
    log(f"verdict {verdict}")

    # ---------------- 4. untimed interleaved sequence after timing ----------------
    # Same shape as timing (rotated arm order, alternating reference, i2i then
    # t2i per arm); every output must equal the arm-A pass-1 (iteration-0) body.
    post_ok = True
    for it in range(12):
        order = ORDERS[it % 3]
        ref = names[it % 2]
        for k in (ref, "t2i"):
            for arm in order:
                use(arm)
                r = call(k)
                got = (r.status_code, hashlib.sha256(norm(r.body)).hexdigest(), r.media_type)
                post_ok &= got == ref_resp[k]
    passed["post_timing_alternating"] = post_ok
    log(f"post-timing interleaved A/A2/B == arm-A iteration-0 refs: {post_ok}")
    use("B")
    ok = all(passed.values())
    if ARM in ("A", "B"):
        s = summary["i2i"]["per_arm"][ARM]
        s2 = summary["t2i"]["per_arm"][ARM]
        out = {"batch": {"i2i-2048": {"step_ms": s["step_ms"], "noise_floor": s["noise_floor"]},
                         "t2i-2048": {"step_ms": s2["step_ms"], "noise_floor": s2["noise_floor"]}},
               "output": {"class": "exact"}, "passed": ok, "checks": passed, "summary": summary,
               "tree_has_change": tree_has_change}
    else:
        batch = {}
        for c, lab in (("i2i", "i2i-2048"), ("t2i", "t2i-2048")):
            sb = summary[c]["per_arm"]["B"]
            batch[lab] = {"step_ms": sb["step_ms"], "noise_floor": sb["noise_floor"],
                          "base_step_ms": summary[c]["per_arm"]["A"]["step_ms"],
                          "B_over_A": summary[c]["B_vs_A"]["ratio_per_rotation_median"],
                          "B_over_A_ci95": summary[c]["B_vs_A"]["ratio_per_rotation_ci95"],
                          "A2_over_A": summary[c]["A2_vs_A"]["ratio_per_rotation_median"],
                          "A2_over_A_ci95": summary[c]["A2_vs_A"]["ratio_per_rotation_ci95"],
                          "B_minus_A_ms_median": summary[c]["B_vs_A"]["paired_diff_ms_median"],
                          "B_minus_A_ms_iqr": summary[c]["B_vs_A"]["paired_diff_iqr"]}
        out = {"batch": batch, "output": {"class": "exact"}, "verdict": summary["verdict"],
               "passed": ok, "checks": passed, "summary": summary,
               "compared": "arm A (main code swapped in) vs arm B (tree): msgspec corpus vs json.loads, "
                           "b64 corpus vs base64.b64decode(validate=True), handler bodies + VisualGenParams "
                           "dump + ref sha256 (A,B,A,t2i), malformed-request status+body (incl. 3000-deep "
                           "nesting at default recursion limit), post-timing interleaved A/A2/B"}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
