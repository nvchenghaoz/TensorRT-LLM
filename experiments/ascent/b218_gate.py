"""Idea 218: gate-v2 replica on serving-pair outputs (the frozen check.py is not shipped to jobs).

usage: python experiments/ascent/b218_gate.py <label>=<proxy_artifact_dir> [<label>=<dir> ...]

Each dir is a run_serving_pair arm's artifacts/proxy (proxy_c1.json, media_proxy_c1/, proxy_server.log).
The proxy sends the 12 check prompts (prompts.json 'check', seeds 100-111) twice after the client's
untimed test request, i.e. check.py's run A. Per arm:
  * gate v2 per prompt (pass 1): LPIPS (AlexNet v0.1, as metrics.py) <= 0.265977 and PSNR (uint8 RGB)
    >= 17.658 dB vs the frozen reference image (store_dir/<sha256>.png, sha256 verified);
  * repeated execution: pass 2 vs pass 1 per prompt (bitwise count and the same tolerance);
  * compile audit of proxy_server.log: lines after the last "Warmup completed" that report a
    Dynamo/Inductor compile, a recompile or a cuDNN FE FP8 plan build; plus per-request server
    post_denoise ms (a compile inside a request would add seconds).
Between arms (first label = base): per-prompt PSNR/LPIPS candidate vs base.
Last stdout line: {"passed": ...} = every arm valid, all 12 within gate v2, pass 2 == pass 1 within tol.
"""

import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

STORE = "/home/scratch.chengzhang/trtllm-dev-flux3-klein-20261027-125409/ascent-benchmark-data/flux3-klein-nvfp4/v1/ref"
REF = {  # experiments/ascent-benchmark/reference.json images (check prompts)
    "check00": "6281cc18a39f950ac89b3513513b4957941f8b3760a6abf6c6010d4614e65910",
    "check01": "3e8c204c77fd5ff41e9636fc3471f64a6264be31cc1af24d84619ea741efd95d",
    "check02": "337723400385cfb0dd51111bf108cb9db4ce196f6e069048667aa4ee322f275a",
    "check03": "e0f8f049f3d7f2366268f74900ca926dd7df3dfd8032af1be780c47a6bca18fa",
    "check04": "bb2987bccc6847beb6ed218b9162ee61ce1adee8d84defab4c41cd8706ef0a4b",
    "check05": "3e9da040e7bdc551108abe2e51c3f227556f2be791877f0c72fcd99c85636c16",
    "check06": "829a1e0775f116aa8b1e2aacf88d2e9996dd070ceb6686492ae688c5d119ca96",
    "check07": "f7049fd916722d2554e3fc26e801deeb94d7bf61af2cde00d4aef28041c43943",
    "check08": "75c27e96ba0b60f1ed2d723bb58f72f200b33c3c0343da51c242a55b9221845b",
    "check09": "3b52010c7b0037be7a14465bcb4201e73c7f83486dca04aa616dfbc70bb8bc0f",
    "check10": "5c908d8736812d476f53a3292db4a2945223f930a2a075b6dc738aa20010123d",
    "check11": "29f21a2f3aa6f63c5df84335b536c79fc1df1be1293c4f4504cbf0e3a933e054",
}
PROMPTS = [
    "A red fox standing in fresh snow at sunrise",
    "A lighthouse on a cliff during a thunderstorm, oil painting",
    "A bowl of ramen with a soft boiled egg, studio photo",
    "A futuristic city skyline at night with flying cars",
    "Portrait of an elderly fisherman with a white beard, dramatic lighting",
    "A cozy reading nook with a window, plants and warm lamp light",
    "A watercolor painting of a hot air balloon over green hills",
    "A golden retriever puppy playing with a tennis ball on grass",
    "A vintage travel poster that says VISIT MARS in bold letters",
    "Macro photo of a dew drop on a green leaf",
    "A medieval castle surrounded by autumn forest, aerial view",
    "A robot chef cooking pancakes in a modern kitchen",
]
LPIPS_MAX, PSNR_MIN = 0.265977, 17.658
_LP = {}


def log(*a):
    print(*a, flush=True)


def lpips_model():
    if "m" in _LP:
        return _LP["m"]
    import lpips

    try:
        m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    except TypeError:
        import lpips.pretrained_networks as pn
        import torchvision.models as tvm

        original = tvm.alexnet

        def alexnet(pretrained=False, **kwargs):
            if pretrained:
                kwargs.setdefault("weights", tvm.AlexNet_Weights.IMAGENET1K_V1)
            return original(**kwargs)

        pn.tv.alexnet = alexnet
        m = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    _LP["m"] = m.to("cuda").eval()
    return _LP["m"]


def to_t(a):
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0).cuda()


def compare(a, b):
    diff = a.astype(np.int16) - b.astype(np.int16)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    bit = mse == 0.0
    psnr = 100.0 if bit else min(100.0, 10.0 * math.log10(255.0 ** 2 / mse))
    try:
        with torch.no_grad():
            lp = float(lpips_model()(to_t(a), to_t(b)).item())
    except Exception as e:  # noqa: BLE001
        lp = None
        log("lpips unavailable:", repr(e)[:300])
    return {"lpips": None if lp is None else round(lp, 6), "psnr_db": round(psnr, 4), "bitwise_equal": bit,
            "max_abs_diff": int(np.abs(diff).max())}


def within(m):
    return m["lpips"] is not None and m["lpips"] <= LPIPS_MAX and m["psnr_db"] >= PSNR_MIN


def load(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


refs, ref_err = {}, []
for k, sha in REF.items():
    p = os.path.join(STORE, f"{sha}.png")
    if not os.path.isfile(p):
        ref_err.append(f"{k} missing")
        continue
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != sha:
        ref_err.append(f"{k} sha mismatch")
        continue
    refs[k] = load(p)
log("references loaded:", len(refs), ref_err)

arms, out = [], {"reference_errors": ref_err, "tolerance": {"lpips_max": LPIPS_MAX, "psnr_min_db": PSNR_MIN}}
for spec in sys.argv[1:]:
    label, d = spec.split("=", 1)
    res = json.load(open(os.path.join(d, "proxy_c1.json")))
    recs = sorted([r for r in res.get("requests") or [] if r.get("success")], key=lambda r: int(r["index"]))
    imgs = {}
    for r in recs:
        i = int(r["index"])
        key = None
        if r.get("prompt") in PROMPTS:
            key = f"check{PROMPTS.index(r['prompt']):02d}"
        else:
            key = f"check{i % 12:02d}"
        imgs.setdefault(key, []).append((i, r["output_paths"][0], r.get("server_post_denoise")))
    rows, rep = [], []
    for key in sorted(imgs):
        seq = sorted(imgs[key])
        a = load(seq[0][1])
        m = compare(refs[key], a) if key in refs else {"lpips": None, "psnr_db": None}
        m.update({"key": key, "passed": within(m) if key in refs else False})
        rows.append(m)
        for i, p, _ in seq[1:]:
            mm = compare(a, load(p))
            mm.update({"key": key, "later_index": i, "passed": within(mm)})
            rep.append(mm)
    post = [r.get("server_post_denoise") for r in recs]
    post_ms = [round(v * 1000, 2) for v in post if v is not None]
    # compile audit
    lines = open(os.path.join(d, "proxy_server.log"), errors="replace").read().splitlines()
    warm = [i for i, ln in enumerate(lines) if "Warmup completed" in ln]
    w_end = warm[-1] if warm else None
    pat = re.compile(r"torchdynamo start tracing|Recompiling function|recompil|Compiling FX graph|"
                     r"fp8_conv3x3 built|fp8_conv3x3 autotuned|compile_fx", re.I)
    after = [ln[-240:] for i, ln in enumerate(lines) if w_end is not None and i > w_end and pat.search(ln)]
    before_fp8 = [ln[-200:] for i, ln in enumerate(lines)
                  if (w_end is None or i < w_end) and ("fp8_conv3x3 built" in ln or "prepare_fp8_convs" in ln
                                                       or "vae_decode_fp8_convs" in ln)]
    arm = {"label": label, "dir": d, "n_requests": len(recs), "c1_pass1": rows,
           "c1_within_gate": sum(r["passed"] for r in rows), "total": len(rows),
           "repeat_pairs": rep, "repeat_bitwise": sum(m["bitwise_equal"] for m in rep),
           "repeat_within_tol": all(m["passed"] for m in rep) and len(rep) == len(rows),
           "post_denoise_ms": {"min": min(post_ms) if post_ms else None, "max": max(post_ms) if post_ms else None,
                               "all": post_ms},
           "audit": {"warmup_completed_line": w_end, "compile_or_build_lines_after_warmup": after[:10],
                     "n_after_warmup": len(after), "fp8_lines_before_warmup_end": before_fp8[:16]},
           "_imgs": {k: sorted(v)[0][1] for k, v in imgs.items()}}
    arm["passed"] = arm["c1_within_gate"] == 12 and arm["repeat_within_tol"] and not ref_err
    arms.append(arm)
    log(json.dumps({k: v for k, v in arm.items() if k not in ("_imgs", "c1_pass1", "repeat_pairs")}))
    for r in rows:
        log(f"  [{label}] {r['key']} lpips={r['lpips']} psnr={r['psnr_db']} passed={r['passed']}")

if len(arms) >= 2:
    base = arms[0]
    for arm in arms[1:]:
        cross = []
        for k in sorted(arm["_imgs"]):
            if k in base["_imgs"]:
                m = compare(load(base["_imgs"][k]), load(arm["_imgs"][k]))
                m["key"] = k
                cross.append(m)
        arm["vs_" + base["label"]] = cross
        log(f"{arm['label']} vs {base['label']}: min psnr {min(c['psnr_db'] for c in cross)}, "
            f"max lpips {max((c['lpips'] or 0) for c in cross)}")
for a in arms:
    a.pop("_imgs", None)
out["arms"] = arms
out["passed"] = bool(arms) and all(a["passed"] for a in arms)
print(json.dumps(out), flush=True)
