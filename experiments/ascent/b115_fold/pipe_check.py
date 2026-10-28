"""Branch 115 (idea 182): in-process pipeline check of the folded FP8 scale reduce.

ONE process, pipeline as served (engine.json overlay; main's default routing, so the NVFP4 route
covers single_blocks.0-27 and image mode blocks 1-4 since landing 56: 32 routed calls per step).
Switch: flux3_sm120_qknorm_rope_fp8.FOLD (read per call inside the custom op; the graph runner is
cleared and re-captured after every flip so a graph-on request replays the flag's own path).
ON = fold (this branch's default); OFF = landing 54 (mode 1 + one-program reduce + mode 2).

At each position (A T=80, B T=160, A again) four requests run back to back:
graph-on ON, graph-on OFF, graph-off ON, graph-off OFF. Final latents and uint8 images compared
ON vs OFF (graph-on and graph-off) and graph-on vs graph-off. Graph-off requests count launches:
fold mode 1 / mode 2 (_fold_pass) and landing-54 passes (_pass, each paired with one reduce).
Then two consecutive graph-on ON requests of A compare equal.
mode 'profile': one graph-on ON request inside cudaProfilerStart/Stop (for profile_timeline).
Usage: python pipe_check.py <checkpoint> [check|profile]
"""

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


from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

counts = {"routed": 0, "routed_rows": {}, "fold1": 0, "fold2": 0, "l54_pass1": 0, "l54_pass2": 0}
_impl = fp.nvfp4_qkv_rope_fp8_impl
_fold_pass = qr8._fold_pass
_pass = qr8._pass


def _count_impl(*a, **k):
    counts["routed"] += 1
    rows = int(a[0].shape[0])
    counts["routed_rows"][rows] = counts["routed_rows"].get(rows, 0) + 1
    return _impl(*a, **k)


def _count_fold(mode, *a, **k):
    counts[f"fold{mode}"] += 1
    return _fold_pass(mode, *a, **k)


def _count_pass(mode, *a, **k):
    counts[f"l54_pass{mode}"] += 1
    return _pass(mode, *a, **k)


fp.nvfp4_qkv_rope_fp8_impl = _count_impl
qr8._fold_pass = _count_fold
qr8._pass = _count_pass

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
log("engine overlay", engine, pipeline_config, "direct", primsts.FLUX3_QKV_FP8_DIRECT,
    "fold", qr8.FOLD)
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
_state = {"flag": None}


def set_mode(flag, graphs):
    if _state["flag"] != flag:
        qr8.FOLD = flag
        runner.clear()  # graphs captured under the other flag must not replay
        _state["flag"] = flag
    runner.enabled = graphs


def run(label, name, flag, graphs):
    set_mode(flag, graphs)
    for k in counts:
        counts[k] = {} if k == "routed_rows" else 0
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {"label": label, "req": name, "fold": flag, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2)}
    if not graphs:
        rec.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in counts.items()})
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


def position(tag, name):
    g_on = run(f"{tag}_graph_ON", name, True, True)
    g_off = run(f"{tag}_graph_OFF", name, False, True)
    e_off = run(f"{tag}_eager_OFF", name, False, False)
    e_on = run(f"{tag}_eager_ON", name, True, False)
    cmp(g_on, g_off, f"{tag}: graph-on fold ON == OFF (landing 54)")
    cmp(e_on, e_off, f"{tag}: graph-off fold ON == OFF (landing 54)")
    cmp(g_on, e_on, f"{tag}: fold ON graph-on == graph-off")
    cmp(g_off, e_off, f"{tag}: OFF graph-on == graph-off")
    on, off = e_on[0], e_off[0]
    note(on["routed"] == 512, f"{tag}: 512 routed calls (got {on['routed']})")
    note(on["fold1"] == 512 and on["fold2"] == 512 and on["l54_pass1"] == 0
         and on["l54_pass2"] == 0, f"{tag}: ON = 512 fold mode1 + 512 fold mode2, 0 landing-54")
    note(off["fold1"] == 0 and off["fold2"] == 0 and off["l54_pass1"] == 512
         and off["l54_pass2"] == 512, f"{tag}: OFF = 512 landing-54 mode1/mode2 (+reduce)")


if MODE == "profile":
    set_mode(True, True)
    generate(*PROMPTS["A"])
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"passed": True, "profiled": "one graph-on request, fold ON"}))
    sys.exit(0)

position("A_T80", "A")
note(results["requests"][0]["T"] == 80, "A is T=80")
position("B_T160", "B")
note(results["requests"][-1]["T"] == 160, "B is T=160")
position("A_T80_after_T160", "A")
r1 = run("REPEAT_1_graph_ON", "A", True, True)
r2 = run("REPEAT_2_graph_ON", "A", True, True)
cmp(r1, r2, "two consecutive graph-on fold ON requests bitwise")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "flux3_sm120_qknorm_rope_fp8.FOLD=False (landing 54: mode 1 + one-program "
    "Triton reduce + mode 2), same process, same position; final latent + uint8 image bitwise",
    **results,
}))
