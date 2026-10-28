"""Branch 62 (idea 112): ruling-500 exactness of strided FP8R q/k/v on the full pipeline.

One process, FLUX.3 Klein nvfp4 loaded with PipelineLoader on the benchmark defaults plus
experiments/ascent-config/engine.json (FLASHINFER fp8/fp8, cuda graphs on, torch_compile +
autotune, warmup [[1024,1024]] t2i), generating as VisualGen infer() does.
Module switch: flux3_fp8r_qkv._FP8R_QKV_PLANAR / _FP8R_QKV_CONTIGUOUS.
  ON   = strided (False, False), as loaded and warmed up (tree default):
         P1 (11 T=80 prompts), P2 (same), L1, L2 (T=160 prompt), P3 (11 T=80 again)
  OFF  = planar (main 700ddd59) after torch._dynamo.reset() + CUDA graph runner clear:
         W (prompt 0, recompile + capture, as the warmup was for ON), then the same sequence.
  ON2  = strided again after reset/clear: W, P1[0:3] (compile-to-compile variation of ON).
  NOG  = strided after reset, CUDA graphs disabled: W, P1[0:3] (graph-on == graph-off).
Compared bitwise (final latent + decoded image): ON vs OFF at every position (ruling 500),
P1==P2 and L1==L2 within ON and OFF, ON P1 vs ON2 P1, ON P1 vs NOG P1. P3 vs P1 is reported
(history effect, knowledge 847) but is not a pass criterion.
usage: python inproc_check.py <checkpoint>
"""
import copy
import json
import os
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "attend_fastpath"))
_argv = sys.argv
sys.argv = [sys.argv[0], CKPT, os.path.join(HERE, "..", "..", "..")]
import e2e_repeat_check as base_chk  # noqa: E402

sys.argv = _argv
PROMPTS, SEEDS = base_chk.PROMPTS, base_chk.SEEDS
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as fq  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

# Newest landing this check depends on: landing 34 (planar switch) must be present.
assert hasattr(fq, "_FP8R_QKV_PLANAR"), "landing 34 (_FP8R_QKV_PLANAR) missing"
try:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.join(HERE, "..", "..", "..")).stdout.strip() or "n/a"
except Exception:  # noqa: BLE001
    sha = "n/a"
log("tensorrt_llm", tensorrt_llm.__file__, "git", sha,
    "tree default planar/contiguous:", fq._FP8R_QKV_PLANAR, fq._FP8R_QKV_CONTIGUOUS)

args = {
    "attention_config": {"backend": "VANILLA"},
    "compilation_config": {"modes": ["t2i"], "num_frames": [1], "resolutions": [[1024, 1024]],
                           "skip_warmup": False},
    "cuda_graph_config": {"enable": False},
    "parallel_config": {"tp_size": 1},
    "pipeline_config": {"variant": "nvfp4"},
    "torch_compile_config": {"enable": True, "enable_autotune": True, "enable_fullgraph": False},
}
with open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")) as f:
    overlay = json.load(f)
base_chk.deep_merge(args, copy.deepcopy(overlay))
log("overlay as applied:", json.dumps(args))
pipe = PipelineLoader(VisualGenArgs(model=CKPT, **args), device="cuda:0").load()
torch.cuda.synchronize()
log("loaded + warmed up")
runners = getattr(pipe, "_cuda_graph_runners", {})
log("cuda graph runners:", list(runners))

STRIDED, PLANAR = (False, False), (True, True)
fails = []


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


def set_flags(flags):
    fq._FP8R_QKV_PLANAR, fq._FP8R_QKV_CONTIGUOUS = flags


def reset(graphs=True):
    torch._dynamo.reset()
    for r in runners.values():
        r.clear()
        r.enabled = graphs


def generate(i):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(PROMPTS[i], None)
        with autotune_context(pipe.device, pipe.rank, tune=False):
            latent = pipe._prepare_t2i(ctx=ctx, height=1024, width=1024,
                                       num_inference_steps=16, seed=SEEDS[i])
        img = pipe._package_output(pipe._decode_latents(latent)).image
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent[0].clone(), img.clone()


SHORT = list(range(len(PROMPTS) - 1))
LONG = len(PROMPTS) - 1


def phase(label, seq):
    out = []
    for tag, idxs in seq:
        for i in idxs:
            t = time.time()
            T, lat, img = generate(i)
            out.append((tag, i, lat, img))
            log(f"{label}/{tag} prompt {i} T={T} {time.time() - t:.2f}s")
    return out


def eq(a, b):
    return torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])


def pick(rows, tag):
    return {i: r for r in rows if r[0] == tag for i in [r[1]]}


SEQ = [("P1", SHORT), ("P2", SHORT), ("L1", [LONG]), ("L2", [LONG]), ("P3", SHORT)]
set_flags(STRIDED)
on = phase("ON", SEQ)
reset()
set_flags(PLANAR)
off = phase("OFF", [("W", [0])] + SEQ)
off = [r for r in off if r[0] != "W"]
reset()
set_flags(STRIDED)
on2 = phase("ON2", [("W", [0]), ("P1", SHORT[:3])])
reset(graphs=False)
nog = phase("NOG", [("W", [0]), ("P1", SHORT[:3])])
reset(graphs=True)

res = {"on_vs_off": [], "repeats": {}, "history_P3_vs_P1": {}}
for a, b in zip(on, off):
    assert a[:2] == b[:2]
    d = {"pos": a[0], "prompt": a[1], "bitwise": eq(a, b),
         "image_max_abs": int((a[3].int() - b[3].int()).abs().max())}
    res["on_vs_off"].append(d)
    note(d["bitwise"], f"ON vs OFF {a[0]} prompt {a[1]}")
for label, rows in (("ON", on), ("OFF", off)):
    p1, p2, p3 = pick(rows, "P1"), pick(rows, "P2"), pick(rows, "P3")
    l1, l2 = pick(rows, "L1"), pick(rows, "L2")
    rep = all(eq(p1[i], p2[i]) for i in SHORT) and eq(l1[LONG], l2[LONG])
    res["repeats"][label] = rep
    note(rep, f"{label} P1==P2 and L1==L2")
    res["history_P3_vs_P1"][label] = [eq(p1[i], p3[i]) for i in SHORT]
on_p1 = pick(on, "P1")
res["on_vs_on2_after_reset"] = [eq(on_p1[i], r) for i, r in pick(on2, "P1").items()]
res["graph_on_vs_graph_off"] = [eq(on_p1[i], r) for i, r in pick(nog, "P1").items()]
note(all(res["graph_on_vs_graph_off"]) or not all(res["on_vs_on2_after_reset"]),
     "graph-on == graph-off (strided)")
res["passed"] = not fails
res["failures"] = fails
res["compared_against"] = ("_FP8R_QKV_PLANAR=True (main 700ddd59 planar path), same process, "
                           "after torch._dynamo.reset() + graph clear, same prompts/seeds/positions")
print(json.dumps(res), flush=True)
