"""Branch 100 (idea 159): in-process pipeline check of the NVFP4-QKV -> E4M3 direct route.

ONE process, pipeline as served (engine.json overlay: FLASHINFER FP8 attention, BF16 VAE decode,
cuda_graph_config on; torch_compile + autotune; warmup [[1024,1024]] t2i).
Switch: primsts.FLUX3_QKV_FP8_DIRECT (Dynamo guards on it; the graph runner is cleared and
re-captured after every flip, so a graph-on request always replays the flag's own path).

At each position (A T=80, B T=160, A again) four requests run back to back:
graph-on ON, graph-on OFF, graph-off ON, graph-off OFF. Final latents and uint8 images are compared
ON vs OFF (graph-on and graph-off) and graph-on vs graph-off (each flag). Graph-off requests count
calls: routed nvfp4_qkv_rope_fp8_impl calls (expect 28 x 16 = 448 per request with ON, 0 with OFF),
fp8_sm12x.quantize_qkv_e4m3 (main's K1/K2) calls, need_v (single-block) routed calls.
Then two consecutive graph-on ON requests of A after B compare equal (no state across requests).
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


from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import primsts  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

counts = {"routed": 0, "routed_rows": {}, "quantize_qkv_e4m3": 0}
_impl = fp.nvfp4_qkv_rope_fp8_impl
_quant = fp8_sm12x.quantize_qkv_e4m3


def _count_impl(*a, **k):
    counts["routed"] += 1
    rows = int(a[0].shape[0])
    counts["routed_rows"][rows] = counts["routed_rows"].get(rows, 0) + 1
    return _impl(*a, **k)


def _count_quant(*a, **k):
    counts["quantize_qkv_e4m3"] += 1
    return _quant(*a, **k)


fp.nvfp4_qkv_rope_fp8_impl = _count_impl
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
log("engine overlay", engine, pipeline_config, "flag", primsts.FLUX3_QKV_FP8_DIRECT)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
    f"captures={len(runner.graphs) if runner else 0}")
note(runner is not None and runner.enabled, "graph runner present and enabled")
te = pipe.text_encoder
tr = pipe.transformer
blocks = [(f"single_blocks.{i}", b) for i, b in enumerate(tr.single_blocks)]
for mode, mb in getattr(tr, "content_mode_blocks", {}).items():
    blocks += [(f"content_mode_blocks.{mode}.{i}", b) for i, b in enumerate(mb)]
eligible = []
for name, b in blocks:
    b = getattr(b, "_orig_mod", b)
    att = b.attn
    if (isinstance(getattr(att, "_flux3_primsts_nvfp4_qkv", None), primsts.PreparedNVFP4)
            and primsts._fp8_direct_backend(att) is not None):
        eligible.append(name)
log(f"eligible blocks ({len(eligible)}): {eligible}")


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
results = {"eligible_blocks": len(eligible), "requests": [], "comparisons": []}
_state = {"flag": None}


def set_mode(flag, graphs):
    if _state["flag"] != flag:
        primsts.FLUX3_QKV_FP8_DIRECT = flag
        runner.clear()  # graphs captured under the other flag must not replay
        _state["flag"] = flag
    runner.enabled = graphs


def run(label, name, flag, graphs):
    set_mode(flag, graphs)
    for k in ("routed", "quantize_qkv_e4m3"):
        counts[k] = 0
    counts["routed_rows"] = {}
    t1 = time.time()
    T, lat, img = generate(*PROMPTS[name])
    rec = {"label": label, "req": name, "flag": flag, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2)}
    if not graphs:
        rec.update(routed=counts["routed"], routed_rows=dict(counts["routed_rows"]),
                   quantize_qkv_e4m3=counts["quantize_qkv_e4m3"])
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
    cmp(g_on, g_off, f"{tag}: graph-on direct ON == OFF (main)")
    cmp(e_on, e_off, f"{tag}: graph-off direct ON == OFF (main)")
    cmp(g_on, e_on, f"{tag}: direct ON graph-on == graph-off")
    cmp(g_off, e_off, f"{tag}: OFF graph-on == graph-off")
    note(e_on[0]["routed"] == 448, f"{tag}: 448 routed calls with ON (got {e_on[0]['routed']})")
    note(e_off[0]["routed"] == 0, f"{tag}: 0 routed calls with OFF")
    note(e_off[0]["quantize_qkv_e4m3"] - e_on[0]["quantize_qkv_e4m3"] == 448,
         f"{tag}: ON removes exactly 448 K1/K2 quantize calls "
         f"({e_off[0]['quantize_qkv_e4m3']} -> {e_on[0]['quantize_qkv_e4m3']})")


if MODE == "profile":
    set_mode(True, True)
    generate(*PROMPTS["A"])
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    generate(*PROMPTS["A"])
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(json.dumps({"passed": True, "profiled": "one graph-on request, direct ON"}))
    sys.exit(0)

position("A_T80", "A")
note(results["requests"][0]["T"] == 80, "A is T=80")
position("B_T160", "B")
note(results["requests"][-1]["T"] == 160, "B is T=160")
position("A_T80_after_T160", "A")
r1 = run("REPEAT_1_graph_ON", "A", True, True)
r2 = run("REPEAT_2_graph_ON", "A", True, True)
cmp(r1, r2, "two consecutive graph-on direct ON requests bitwise")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "primsts.FLUX3_QKV_FP8_DIRECT=False (main's qkv_forward + _attend + "
    "fp8_sm12x quantize path), same process, same position; final latent + uint8 image bitwise",
    **results,
}))
