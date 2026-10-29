"""Idea 14 in-process pipeline check at the served 2048^2 shapes (engine.json overlay).

1. Real-input op check: during eager (graph-off) t2i and i2i requests every routed FP8-direct QKV
   call (flux3_primsts.nvfp4_qkv_rope_fp8_impl) runs main's chain (EPI off) and then the fused
   route 3x on the same block inputs; qkv8 bytes and scales[8] must be bitwise main's on every
   call and repeat (and the gemm v plane when keep_v). The request continues on the fused result.
2. End to end: graph-on requests with the route on (t2i, t2i again, i2i, t2i after i2i) vs
   graph-on requests with the route off (kill switch, graph runner cleared): final latent and
   uint8 image bitwise; graph-on vs graph-off (eager) with the route on; consecutive repeats.
3. Census per eager request: fused-route calls, main-chain qknorm_rope_fp8 calls, by M.
Usage: python pipe_check.py <checkpoint>
"""

import glob
import hashlib
import json
import os
import sys
import time

import torch

CKPT = sys.argv[1]
fails = []


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

state = {"compare": False, "calls": 0, "by_m": {}, "op_fail": 0, "epi": 0, "main": 0}
_impl = fp.nvfp4_qkv_rope_fp8_impl
_epi_run = epi.qkv_rope_fp8
_qr8 = qr8.qknorm_rope_fp8


def _count_epi(*a, **k):
    state["epi"] += 1
    return _epi_run(*a, **k)


def _count_qr8(*a, **k):
    state["main"] += 1
    return _qr8(*a, **k)


epi.qkv_rope_fp8 = _count_epi
qr8.qknorm_rope_fp8 = _count_qr8


def u8(t):
    return t.view(torch.uint8)


def _hook(activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin,
          positions, qkv_scale, sm_scale, keep_v=True):
    args = (activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin,
            positions, qkv_scale, sm_scale, keep_v)
    if not state["compare"]:
        return _impl(*args)
    m = activation.shape[0]
    state["calls"] += 1
    state["by_m"][m] = state["by_m"].get(m, 0) + 1
    epi.ENABLED = False
    ref = _impl(*args[:-1], True)  # main's chain, v kept so its v plane is comparable
    ref = tuple(t.clone() for t in ref)
    epi.ENABLED = True
    note(fp._qkv_epi_fp8_route(activation, weight, q_norm, k_norm, cos_sin),
         f"fused route covers served call M={m}")
    width = weight.shape[0] // 3
    out = None
    for i in range(3):
        got = _impl(*args)
        ok = torch.equal(u8(got[0]), u8(ref[0])) and torch.equal(
            got[1].view(torch.int32), ref[1].view(torch.int32))
        if keep_v:
            ok = ok and torch.equal(got[2].view(3, m, width)[2].view(torch.int16),
                                    ref[2].view(3, m, width)[2].view(torch.int16))
        if not ok:
            state["op_fail"] += 1
            if state["op_fail"] <= 5:
                d = [int((u8(got[0][p]) != u8(ref[0][p])).sum()) for p in range(3)]
                log(f"OP MISMATCH call {state['calls']} M={m} rep {i} bytes differ {d} "
                    f"scales {ref[1].tolist()} vs {got[1].tolist()}")
        out = got if out is None else out
    return out


fp.nvfp4_qkv_rope_fp8_impl = _hook

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
    f"EPI={epi.ENABLED} VEC={qr8.VEC} STATIC={qr8.STATIC_QK}")
note(runner is not None and runner.enabled, "graph runner present and enabled")

# i2i reference: the frozen check00 PNG (benchmark inputs, content-addressed by sha256).
store = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/" \
        "flux3-klein-nvfp4-2048/v1/inputs"
sha = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"
cands = [p for p in glob.glob(os.path.join(store, "**", "*"), recursive=True) if sha in p]
log("i2i reference", cands[:2])
ref_png = cands[0]
with open(ref_png, "rb") as f:
    raw = f.read()
note(hashlib.sha256(raw).hexdigest() == sha, "reference sha256")
ref_img = _load_image_pixels(raw, device=pipe.device)

PROMPT_T = ("A vintage travel poster that says VISIT MARS in bold letters, red planet, rockets", 108)
PROMPT_I = ("Make it night time with a full moon and glowing windows", 7)


def generate(mode):
    prompt, seed = PROMPT_T if mode == "t2i" else PROMPT_I
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if mode == "t2i":
            latent = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048, num_inference_steps=16,
                                       seed=seed)
        else:
            latent = pipe._prepare_i2i(ctx=ctx, image_cond=[ref_img], height=2048, width=2048,
                                       num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents(latent)).image
    torch.cuda.synchronize()
    return ctx[0].shape[1], latent[0].clone(), img.clone()


results = {"requests": [], "comparisons": []}
_cfg = {"epi": True}


def h(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:12]


def run(label, mode, epi_on, graphs, compare=False):
    if _cfg["epi"] != epi_on:
        epi.ENABLED = epi_on
        runner.clear()  # graphs captured under the other route must not replay
        _cfg["epi"] = epi_on
    epi.ENABLED = epi_on
    runner.enabled = graphs
    state.update(compare=compare, calls=0, by_m={}, op_fail=0, epi=0, main=0)
    t1 = time.time()
    T, lat, img = generate(mode)
    state["compare"] = False
    rec = {"label": label, "mode": mode, "epi": epi_on, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2), "latent_sha": h(lat.float()), "image_sha": h(img)}
    if not graphs:
        rec["census"] = {"fused_route_calls": state["epi"], "qknorm_rope_fp8_calls": state["main"]}
    if compare:
        rec["op_check"] = {"calls": state["calls"], "by_m": state["by_m"],
                           "mismatching_repeats": state["op_fail"]}
        note(state["op_fail"] == 0, f"{label}: real-input op check bitwise ({state['op_fail']} bad)")
        note(state["calls"] > 0, f"{label}: op check saw routed calls")
    log(label, json.dumps(rec))
    results["requests"].append(rec)
    return rec, lat, img


def cmp(a, b, what):
    d = {"what": what, "latent_bitwise": bool(torch.equal(a[1], b[1])),
         "image_bitwise": bool(torch.equal(a[2], b[2])),
         "latent_max_abs": float((a[1].float() - b[1].float()).abs().max()),
         "image_max_abs": int((a[2].int() - b[2].int()).abs().max())}
    log("CMP", json.dumps(d))
    results["comparisons"].append(d)
    note(d["latent_bitwise"] and d["image_bitwise"], what)


# 1. real-input op checks (eager, route on for the request's own outputs)
e_t = run("t2i_eager_epi_opcheck", "t2i", True, False, compare=True)
e_i = run("i2i_eager_epi_opcheck", "i2i", True, False, compare=True)
# 2. graph-on with the route, alternating t2i -> t2i -> i2i -> t2i
g_t1 = run("t2i_graph_epi_1", "t2i", True, True)
g_t2 = run("t2i_graph_epi_2", "t2i", True, True)
g_i1 = run("i2i_graph_epi_1", "i2i", True, True)
g_t3 = run("t2i_graph_epi_3_after_i2i", "t2i", True, True)
g_i2 = run("i2i_graph_epi_2", "i2i", True, True)
# kill switch (route off = main), graph runner cleared
k_t = run("t2i_graph_main", "t2i", False, True)
k_i = run("i2i_graph_main", "i2i", False, True)
k_te = run("t2i_eager_main", "t2i", False, False)

cmp(g_t1, k_t, "t2i graph-on: fused route == main (kill switch)")
cmp(g_i1, k_i, "i2i graph-on: fused route == main (kill switch)")
cmp(g_t1, g_t2, "t2i graph-on: consecutive repeat bitwise")
cmp(g_t1, g_t3, "t2i graph-on after i2i == first t2i")
cmp(g_i1, g_i2, "i2i graph-on: repeat after t2i bitwise")
cmp(e_t, g_t1, "t2i: route on, eager == graph-on")
cmp(e_i, g_i1, "i2i: route on, eager == graph-on")
cmp(e_t, k_te, "t2i eager: fused route == main")
ce = e_t[0].get("census", {})
cm = k_te[0].get("census", {})
note(ce.get("fused_route_calls", 0) > 0 and cm.get("fused_route_calls", 1) == 0,
     f"census: fused route used with EPI on ({ce}) and never with EPI off ({cm})")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "main's chain (route off = TRTLLM_FLUX3_QKV_EPI_FP8=0 semantics) in the "
                        "same process: per-call qkv8/scales/v bitwise on real block inputs (3 "
                        "repeats), final latent + uint8 image bitwise, graph-on and eager",
    "comparisons": results["comparisons"],
    "requests": results["requests"],
}))
