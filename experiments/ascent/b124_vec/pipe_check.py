"""Branch 124 (idea 192): in-process pipeline check of the vectorized fold (VEC) vs main's fold.

ONE process, pipeline as served (engine.json overlay, main's default routing: 32 routed NVFP4
QKV calls per step, 512 per request). Switch: flux3_sm120_qknorm_rope_fp8.VEC (read per call;
FOLD stays on). The graph runner is cleared and re-captured after every flip so a graph-on request
replays the flag's own path. ON = VEC kernel (this branch's default); OFF = main's landing-58 fold
kernel (TRTLLM_FLUX3_QKV_FP8_VEC=0 path, byte for byte main's code).

At each position (A T=80, B T=160, A again) four requests run back to back:
graph-on ON, graph-on OFF, graph-off OFF, graph-off ON. Final latents and uint8 images compared
ON vs OFF (graph-on and graph-off) and graph-on vs graph-off. Graph-off requests count launches
of _fold_pass by kernel (vec / fold) and mode. Then two consecutive graph-on ON requests of A.
All hashes come from this one process (HASH RULE).
mode 'profile <on|off>': warm 2 graph-on requests, then one graph-on request inside
cudaProfilerStart/Stop (for profile_timeline capture=region).
Usage: python pipe_check.py <checkpoint> [check|profile on|profile off]
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

counts = {"routed": 0, "vec1": 0, "vec2": 0, "fold1": 0, "fold2": 0}
_impl = fp.nvfp4_qkv_rope_fp8_impl
_fold_pass = qr8._fold_pass


def _count_impl(*a, **k):
    counts["routed"] += 1
    return _impl(*a, **k)


def _count_fold(mode, *a, **k):
    vec = a[6] if len(a) > 6 else k.get("vec", False)
    counts[f"{'vec' if vec else 'fold'}{mode}"] += 1
    return _fold_pass(mode, *a, **k)


fp.nvfp4_qkv_rope_fp8_impl = _count_impl
qr8._fold_pass = _count_fold

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
    "fold", qr8.FOLD, "vec", qr8.VEC, "pair_kernel_env",
    os.environ.get("TRTLLM_FLUX3_SM120_PAIR_KERNEL"))
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
        qr8.VEC = flag
        qr8.FOLD = True
        runner.clear()  # graphs captured under the other flag must not replay
        _state["flag"] = flag
    runner.enabled = graphs


def h(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:12]


def run(label, name, flag, graphs):
    set_mode(flag, graphs)
    for k in counts:
        counts[k] = 0
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {"label": label, "req": name, "vec": flag, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2), "latent_sha": h(lat.float()), "image_sha": h(img)}
    if not graphs:
        rec.update(dict(counts))
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
    cmp(g_on, g_off, f"{tag}: graph-on VEC ON == OFF (main's fold)")
    cmp(e_on, e_off, f"{tag}: graph-off VEC ON == OFF (main's fold)")
    cmp(g_on, e_on, f"{tag}: VEC ON graph-on == graph-off")
    cmp(g_off, e_off, f"{tag}: OFF graph-on == graph-off")
    on, off = e_on[0], e_off[0]
    note(on["routed"] == 512, f"{tag}: 512 routed calls (got {on['routed']})")
    note(on["vec1"] == 512 and on["vec2"] == 512 and on["fold1"] == 0 and on["fold2"] == 0,
         f"{tag}: ON = 512 vec mode1 + 512 vec mode2, 0 main fold")
    note(off["fold1"] == 512 and off["fold2"] == 512 and off["vec1"] == 0 and off["vec2"] == 0,
         f"{tag}: OFF = 512 main fold mode1 + mode2, 0 vec")
    return g_on, e_on


if MODE == "profile":
    flag = (sys.argv[3] if len(sys.argv) > 3 else "on") == "on"
    set_mode(flag, True)
    generate(*PROMPTS["A"])
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"passed": True, "profiled": f"one graph-on request, VEC {flag}"}))
    sys.exit(0)

a1 = position("A_T80", "A")
note(results["requests"][0]["T"] == 80, "A is T=80")
position("B_T160", "B")
note(results["requests"][-1]["T"] == 160, "B is T=160")
a3 = position("A_T80_after_T160", "A")
# Main-wide leak (job 205): A after B may differ from A's first pass in BOTH arms; reported only.
leak = {"graph_on_A3_eq_A1": bool(torch.equal(a1[0][2], a3[0][2])),
        "graph_off_A3_eq_A1": bool(torch.equal(a1[1][2], a3[1][2]))}
results["A3_vs_A1_info_only"] = leak
log("A3 vs A1 (info only, main-wide leak owned by job 205)", leak)
r1 = run("REPEAT_1_graph_ON", "A", True, True)
r2 = run("REPEAT_2_graph_ON", "A", True, True)
cmp(r1, r2, "two consecutive graph-on VEC ON requests bitwise")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "flux3_sm120_qknorm_rope_fp8.VEC=False (main's landing-58 fold kernel), "
    "same process, same position; final latent + uint8 image bitwise",
    **results,
}))
