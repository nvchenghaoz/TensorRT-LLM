"""Branch 184 (idea 244): in-process pipeline check of stage D (qr8.L2DISCARD) and stage R
(flux3_fp8r_qkv.FP8R_QKV_FP8_DIRECT), served engine.json, main's routing otherwise.

Configs: base (D off, R off = main), D, R, both. At each position (A T=80, B T=160, A again)
every config runs one graph-on and one graph-off request; final latents and uint8 images are
compared against base (same position, same graph mode) and graph-on vs graph-off. Graph-off
requests census the routes: VEC mode-2 launches per discard variant, fp8_sm12x K1/K2 quantize
calls (main's image.0 path), and qknorm_rope_fp8 calls. Then two consecutive graph-on 'both'
requests of A. The graph runner is cleared after every flip; R is read at trace time (Dynamo
guards on it), D per call.
mode 'profile <base|D|R|both>': warm 2 graph-on requests, then one inside cudaProfilerStart/Stop.
Usage: python pipe_check.py <checkpoint> [check|profile <cfg>]
"""

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from te_common import prompt_for_bucket  # noqa: E402

CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "check"
fails = []


def log(*a):
    print(*a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as f8r  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

counts = {}
_fold_pass = qr8._fold_pass
_quant = fp8_sm12x.quantize_qkv_e4m3
_qr8 = qr8.qknorm_rope_fp8


def _count_fold(mode, key_base, args, rows, n_part, grid, sm_scale, vec=False, discard=0, hint=0):
    k = f"{'vec' if vec else 'fold'}{mode}_d{discard}_h{hint}_ld{key_base[2]}"
    counts[k] = counts.get(k, 0) + 1
    return _fold_pass(mode, key_base, args, rows, n_part, grid, sm_scale, vec, discard, hint)


def _count_quant(*a, **k):
    counts["K1K2_quantize_qkv_e4m3"] = counts.get("K1K2_quantize_qkv_e4m3", 0) + 1
    return _quant(*a, **k)


qr8._fold_pass = _count_fold
fp8_sm12x.quantize_qkv_e4m3 = _count_quant

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
    **engine,
)
CONFIGS = {"base": (False, False), "D": (True, False), "R": (False, True), "both": (True, True)}
_state = {"cfg": None}


def set_cfg(cfg, graphs, runner=None):
    if _state["cfg"] != cfg:
        qr8.L2DISCARD, f8r.FP8R_QKV_FP8_DIRECT = CONFIGS[cfg]
        if runner is not None:
            runner.clear()  # graphs captured under the other config must not replay
        _state["cfg"] = cfg
    if runner is not None:
        runner.enabled = graphs


# Warmup / capture under the served default (both levers on).
set_cfg("both", True)
log("engine overlay", engine, pipeline_config, "L2DISCARD", qr8.L2DISCARD, "R",
    f8r.FP8R_QKV_FP8_DIRECT, "VEC", qr8.VEC)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled}")
note(runner is not None and runner.enabled, "graph runner present and enabled")
te = pipe.text_encoder


def generate(prompt, seed):
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        latent, ids = pipe._prepare_t2i(
            ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
        )
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent.clone(), img.clone()


PROMPTS = {
    "A": ("A red fox standing in fresh snow at sunrise", 11),
    "B": (prompt_for_bucket(te, 2), 22),
}
results = {"requests": [], "comparisons": []}


def h(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:12]


def run(label, name, cfg, graphs):
    set_cfg(cfg, graphs, runner)
    counts.clear()
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {"label": label, "req": name, "cfg": cfg, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2), "latent_sha": h(lat.float()), "image_sha": h(img)}
    if not graphs:
        rec["census"] = dict(counts)
    log(label, json.dumps(rec))
    results["requests"].append(rec)
    return rec, lat, img


def cmp(a, b, what):
    d = {
        "what": what,
        "latent_bitwise": bool(torch.equal(a[1], b[1])),
        "image_bitwise": bool(torch.equal(a[2], b[2])),
        "latent_max_abs": float((a[1].float() - b[1].float()).abs().max()),
        "image_max_abs": int((a[2].int() - b[2].int()).abs().max()),
    }
    log("CMP", json.dumps(d))
    results["comparisons"].append(d)
    note(d["latent_bitwise"] and d["image_bitwise"], what)


def census_ok(tag, cfg, census):
    d_on, r_on = CONFIGS[cfg]
    vec2 = {k: v for k, v in census.items() if k.startswith("vec2")}
    total2 = sum(vec2.values())
    k1k2 = census.get("K1K2_quantize_qkv_e4m3", 0)
    want2 = 512 + (16 if r_on else 0)
    note(total2 == want2, f"{tag} {cfg}: {want2} VEC mode-2 launches (got {total2}: {vec2})")
    note(k1k2 == (0 if r_on else 16), f"{tag} {cfg}: K1/K2 calls {k1k2}")
    undiscarded = sum(v for k, v in vec2.items() if "_d0_" in k)
    note(undiscarded == (0 if d_on else total2),
         f"{tag} {cfg}: discard variant census {vec2}")


def position(tag, name):
    out = {}
    for cfg, graphs in (("both", True), ("base", True), ("D", True), ("R", True),
                        ("R", False), ("D", False), ("base", False), ("both", False)):
        out[(cfg, graphs)] = run(f"{tag}_{cfg}_{'graph' if graphs else 'eager'}", name, cfg,
                                 graphs)
    for cfg in ("D", "R", "both"):
        cmp(out[(cfg, True)], out[("base", True)], f"{tag}: graph-on {cfg} == base")
        cmp(out[(cfg, False)], out[("base", False)], f"{tag}: graph-off {cfg} == base")
    for cfg in CONFIGS:
        cmp(out[(cfg, True)], out[(cfg, False)], f"{tag}: {cfg} graph-on == graph-off")
        census_ok(tag, cfg, out[(cfg, False)][0]["census"])
    return out


if MODE == "profile":
    cfg = sys.argv[3] if len(sys.argv) > 3 else "both"
    set_cfg(cfg, True, runner)
    generate(*PROMPTS["A"])
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"passed": True, "profiled": f"one graph-on request, cfg {cfg}"}))
    sys.exit(0)

a1 = position("A_T80", "A")
note(results["requests"][0]["T"] == 80, "A is T=80")
position("B_T160", "B")
note(results["requests"][-1]["T"] == 160, "B is T=160")
a3 = position("A_T80_after_T160", "A")
leak = {"graph_on_both_A3_eq_A1": bool(torch.equal(a1[("both", True)][2], a3[("both", True)][2])),
        "graph_on_base_A3_eq_A1": bool(torch.equal(a1[("base", True)][2], a3[("base", True)][2]))}
results["A3_vs_A1_info_only"] = leak
r1 = run("REPEAT_1_both_graph", "A", "both", True)
r2 = run("REPEAT_2_both_graph", "A", "both", True)
cmp(r1, r2, "two consecutive graph-on 'both' requests bitwise")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "base config (L2DISCARD off, FP8R_QKV_FP8_DIRECT off = main's code), "
    "same process, same position; final latent + uint8 image bitwise",
    **{k: v for k, v in results.items() if k != "requests"},
    "requests": results["requests"],
}))
