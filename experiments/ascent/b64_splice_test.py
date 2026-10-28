"""Correctness for idea 16 Part B (openai_server._B64JsonSplicer).

The spliced HTTP body and headers must be byte-identical to main's
JSONResponse(content=ImageGenerationResponse(...).model_dump(), headers=...)
for n=1 and n=2 images, prompts with quotes, backslashes, newlines,
control characters and non-ASCII, with None fields, and on the fallback path
(a prompt that collides with a placeholder). Last stdout line is JSON.
"""
import base64
import json
import random
import time

from starlette.responses import JSONResponse

from tensorrt_llm.serve import openai_server as osrv
from tensorrt_llm.serve.openai_protocol import ImageGenerationResponse, ImageObject

PROMPTS = [
    "a plain prompt",
    'He said "hello" and left',
    "back\\slash \\n literal and C:\\path\\to\\file",
    "line one\nline two\r\nline three\ttabbed",
    "control \x01\x1f chars and \u2028 separator",
    "caf\u00e9 na\u00efve \u6f22\u5b57 \U0001F600 emoji \u0444",
    "</script><!-- \"mixed\" \\ \u00e9\n -->",
    None,
    "",
]


def build(values, prompt, created, size="1024x1024", output_format="png"):
    return ImageGenerationResponse(
        created=created,
        data=[ImageObject(b64_json=v, revised_prompt=prompt) for v in values],
        output_format=output_format,
        size=size,
    )


def check_case(pngs, prompt, headers):
    created = int(time.time())
    b64s = [base64.b64encode(p) for p in pngs]
    ref = JSONResponse(
        content=build([b.decode("utf-8") for b in b64s], prompt, created).model_dump(),
        headers=headers,
    )
    sp = osrv._B64JsonSplicer()
    resp = build([sp.placeholder(b) for b in b64s], prompt, created)
    spliced_body = sp.render(resp.model_dump())
    out = sp.response(resp.model_dump(), headers)
    return {
        "n": len(pngs),
        "prompt": repr(prompt)[:40],
        "spliced_path": spliced_body is not None,
        "body_equal": out.body == ref.body,
        "headers_equal": out.raw_headers == ref.raw_headers,
        "media_type_equal": out.media_type == ref.media_type,
        "status_equal": out.status_code == ref.status_code,
        "json_roundtrip_equal": json.loads(out.body) == json.loads(ref.body),
    }


def main():
    rng = random.Random(7)
    headers = {"Server-Timing": "total;dur=1234.5, gen;dur=1100.0"}
    sizes = [(1_904_700,), (1_904_700, 1_520_003), (1,), (2, 3)]
    results = []
    for prompt in PROMPTS:
        for sz in sizes:
            pngs = [rng.randbytes(s) for s in sz]
            results.append(check_case(pngs, prompt, headers))
    # Fallback: prompt equal to the placeholder of item 0 (forces render()
    # to see the marker twice) must still be byte-identical via JSONResponse.
    sp_probe = osrv._B64JsonSplicer()
    created = int(time.time())
    pngs = [rng.randbytes(1000), rng.randbytes(2000)]
    b64s = [base64.b64encode(p) for p in pngs]
    tokens = [sp_probe.placeholder(b) for b in b64s]
    collide_prompt = tokens[0]
    ref = JSONResponse(
        content=build([b.decode() for b in b64s], collide_prompt, created).model_dump(),
        headers=headers,
    )
    fb_resp = build(tokens, collide_prompt, created)
    fb_render = sp_probe.render(fb_resp.model_dump())
    fb_out = sp_probe.response(fb_resp.model_dump(), headers)
    fallback = {
        "render_returned_none": fb_render is None,
        "body_equal": fb_out.body == ref.body,
        "headers_equal": fb_out.raw_headers == ref.raw_headers,
    }
    # Escaped collision (prompt with quotes around the token) stays on the
    # spliced path and is still identical.
    sp2 = osrv._B64JsonSplicer()
    tok = sp2.placeholder(b64s[0])
    esc_prompt = f'"{tok}"'
    ref2 = JSONResponse(
        content=build([b64s[0].decode()], esc_prompt, created).model_dump(), headers=headers
    )
    out2 = sp2.response(build([tok], esc_prompt, created).model_dump(), headers)
    escaped = {"body_equal": out2.body == ref2.body, "headers_equal": out2.raw_headers == ref2.raw_headers}

    ok_cases = all(
        r["body_equal"] and r["headers_equal"] and r["media_type_equal"] and r["status_equal"]
        and r["spliced_path"] for r in results
    )
    ok_fb = fallback["render_returned_none"] and fallback["body_equal"] and fallback["headers_equal"]
    ok_esc = escaped["body_equal"] and escaped["headers_equal"]
    for r in results:
        print(r, flush=True)
    print("fallback", fallback, flush=True)
    print("escaped", escaped, flush=True)
    print(json.dumps({
        "passed": bool(ok_cases and ok_fb and ok_esc),
        "compared": "spliced body/headers vs starlette JSONResponse(model_dump()) (main's path)",
        "cases": len(results),
        "n_values": [1, 2],
        "all_cases_identical": ok_cases,
        "fallback": fallback,
        "escaped_collision": escaped,
    }), flush=True)


main()
