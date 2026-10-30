"""Branch 36 (idea 51) in-process check at the served 2048^2 shapes (engine.json overlay).

Arms (all in ONE process, so hashes compare bitwise; graph runner cleared on every arm switch):
  REV  : this branch as served (fused QKV route, v_quant reverse walk, static_qk cleanup)
  FWD  : TRTLLM_FLUX3_QKV_EPI_VQ_REVERSE=0 semantics (main's forward v_quant)
  OFF  : TRTLLM_FLUX3_QKV_EPI_FP8=0 semantics (pre-branch-7 GEMM + qknorm_rope_fp8 chain)
  MAIN : main's code for the cleanup: image_cond attn._flux3_static_qk=False and a route gate
         that ignores static_qk (main's _qkv_epi_fp8_route), forward v_quant  == main b6400bed
1. Real-input op check (eager t2i and i2i requests, alternating M 16384/16464/32848 in one
   process): every routed nvfp4_qkv_rope_fp8_impl call runs OFF and FWD (keep_v=True) as
   references, then REV with keep_v True and False, 2 eager repeats each: qkv8 bytes + scales[8]
   bitwise vs both refs, v plane bitwise when keep_v. The first GRAPH_CALLS calls per (request, M)
   also capture REV in a CUDA graph (keep_v True and False) and replay it 3x: bitwise vs refs,
   inputs unchanged. Census by (M, static_qk, fused route).
2. End to end (graph-on): REV t2i, i2i, t2i, i2i; FWD t2i, i2i; OFF t2i, i2i; MAIN i2i:
   final latent + uint8 image bitwise across arms and repeats; eager REV == graph REV.
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
GRAPH_CALLS = int(os.environ.get("B36_GRAPH_CALLS", "2"))
fails = []


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


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

log("module defaults: EPI", epi.ENABLED, "VQ_REVERSE", epi.VQ_REVERSE)
note(epi.ENABLED and epi.VQ_REVERSE, "branch defaults: fused route on, reverse v_quant on")

_impl = fp.nvfp4_qkv_rope_fp8_impl
_route = fp._qkv_epi_fp8_route
state = {"compare": False, "calls": 0, "census": {}, "op_fail": 0, "graph_calls": {},
         "graph_checked": 0, "epi_runs": 0}
_epi_run = epi.qkv_rope_fp8


def _count_epi(*a, **k):
    state["epi_runs"] += 1
    return _epi_run(*a, **k)


epi.qkv_rope_fp8 = _count_epi


def _main_route(activation, weight, q_norm, k_norm, cos_sin, static_qk=True):
    return _route(activation, weight, q_norm, k_norm, cos_sin)  # main: static_qk never read


def u8(t):
    return t.view(torch.uint8)


def same(got, ref, m, width, keep_v):
    ok = torch.equal(u8(got[0]), u8(ref[0])) and torch.equal(
        got[1].view(torch.int32), ref[1].view(torch.int32))
    if keep_v:
        ok = ok and torch.equal(got[2].view(3, m, width)[2].view(torch.int16),
                                ref[2].view(3, m, width)[2].view(torch.int16))
    return ok


def _hook(activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin,
          positions, qkv_scale, sm_scale, keep_v=True, static_qk=True):
    base = (activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin,
            positions, qkv_scale, sm_scale)
    if not state["compare"]:
        return _impl(*base, keep_v, static_qk)
    m = activation.shape[0]
    width = weight.shape[0] // 3
    fused = fp._qkv_epi_fp8_route(activation, weight, q_norm, k_norm, cos_sin, static_qk)
    key = f"M{m}/static_qk={bool(static_qk)}/fused={fused}"
    state["census"][key] = state["census"].get(key, 0) + 1
    state["calls"] += 1
    rev0 = epi.VQ_REVERSE
    epi.ENABLED = False
    ref_off = tuple(t.clone() for t in _impl(*base, True, static_qk))
    epi.ENABLED = True
    epi.VQ_REVERSE = False
    ref_fwd = tuple(t.clone() for t in _impl(*base, True, static_qk))
    epi.VQ_REVERSE = True
    out = None
    bad = []
    for kv in (True, False):
        for rep in range(2):
            got = _impl(*base, kv, static_qk)
            for name, ref in (("off", ref_off), ("fwd", ref_fwd)):
                if not same(got, ref, m, width, kv):
                    bad.append(f"eager keep_v={kv} rep{rep} vs {name}")
            if kv == keep_v and out is None:
                out = got
    gkey = m
    if state["graph_calls"].get(gkey, 0) < GRAPH_CALLS:
        state["graph_calls"][gkey] = state["graph_calls"].get(gkey, 0) + 1
        state["graph_checked"] += 1
        snap = [t.clone() for t in base if isinstance(t, torch.Tensor)]
        for kv in (True, False):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                _impl(*base, kv, static_qk)  # warm on the capture stream
                with torch.cuda.graph(g, stream=s):
                    gout = _impl(*base, kv, static_qk)
            torch.cuda.current_stream().wait_stream(s)
            for rep in range(3):
                g.replay()
                torch.cuda.synchronize()
                for name, ref in (("off", ref_off), ("fwd", ref_fwd)):
                    if not same(gout, ref, m, width, kv):
                        bad.append(f"graph keep_v={kv} replay{rep} vs {name}")
            del g, gout
        now = [t for t in base if isinstance(t, torch.Tensor)]
        if not all(torch.equal(a, b) for a, b in zip(snap, now)):
            bad.append("inputs changed")
    epi.VQ_REVERSE = rev0
    if bad:
        state["op_fail"] += 1
        if state["op_fail"] <= 5:
            log(f"OP MISMATCH call {state['calls']} M={m} {bad[:6]}")
    return out


fp.nvfp4_qkv_rope_fp8_impl = _hook

here = os.path.dirname(os.path.abspath(__file__))
engine = json.load(open(os.path.join(here, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config",
                  {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(
    model=CKPT,
    pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
log("engine", json.dumps(engine))
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled}")
note(runner is not None and runner.enabled, "graph runner present and enabled")

cond_attns = []
tr = pipe.transformer
tr = getattr(tr, "_orig_mod", tr)
cmb = getattr(tr, "content_mode_blocks", {})
note("image_cond" in cmb, "transformer has content_mode_blocks.image_cond")
if "image_cond" in cmb:
    for blk in cmb["image_cond"]:
        cond_attns.append(getattr(blk, "_orig_mod", blk).attn)
cond_flags = [getattr(a, "_flux3_static_qk", "<unset>") for a in cond_attns]
log("image_cond attn._flux3_static_qk:", cond_flags)
note(all(f in ("<unset>", True) for f in cond_flags),
     "cleanup: no image_cond block carries _flux3_static_qk=False")

store = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/" \
        "flux3-klein-nvfp4-2048/v1/inputs"
sha = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"
cands = [p for p in glob.glob(os.path.join(store, "**", "*"), recursive=True) if sha in p]
log("i2i reference", cands[:2])
with open(cands[0], "rb") as f:
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


ARMS = {
    "REV": dict(epi=True, rev=True, main=False),
    "FWD": dict(epi=True, rev=False, main=False),
    "OFF": dict(epi=False, rev=True, main=False),
    "MAIN": dict(epi=True, rev=False, main=True),
}
cur = ["REV"]


def set_arm(name):
    if cur[0] != name:
        runner.clear()  # graphs captured under another arm must not replay
        cur[0] = name
    a = ARMS[name]
    epi.ENABLED = a["epi"]
    epi.VQ_REVERSE = a["rev"]
    fp._qkv_epi_fp8_route = _main_route if a["main"] else _route
    for at in cond_attns:
        if a["main"]:
            at._flux3_static_qk = False  # main's transformer_flux3.py:2305
        elif hasattr(at, "_flux3_static_qk"):
            delattr(at, "_flux3_static_qk")


def h(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:12]


results = {"requests": [], "comparisons": []}


def run(label, mode, arm, graphs, compare=False):
    set_arm(arm)
    runner.enabled = graphs
    state.update(compare=compare, calls=0, census={}, op_fail=0, graph_calls={},
                 graph_checked=0, epi_runs=0)
    t1 = time.time()
    T, lat, img = generate(mode)
    state["compare"] = False
    rec = {"label": label, "mode": mode, "arm": arm, "graphs": graphs, "T": T,
           "s": round(time.time() - t1, 2), "latent_sha": h(lat.float()), "image_sha": h(img)}
    if compare:
        rec["op_check"] = {"calls": state["calls"], "census": state["census"],
                           "graph_checked_calls": state["graph_checked"],
                           "mismatching_calls": state["op_fail"]}
        note(state["op_fail"] == 0, f"{label}: real-input op check bitwise ({state['op_fail']} bad)")
        note(state["calls"] > 0, f"{label}: op check saw routed calls")
    elif not graphs:
        rec["fused_runs"] = state["epi_runs"]
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


e_t = run("t2i_eager_REV_opcheck", "t2i", "REV", False, compare=True)
e_i = run("i2i_eager_REV_opcheck", "i2i", "REV", False, compare=True)
ce_t = e_t[0]["op_check"]["census"]
ce_i = e_i[0]["op_check"]["census"]
k16 = "M16384/static_qk=True/fused=True"
note(not any("fused=False" in k or "static_qk=False" in k for k in list(ce_t) + list(ce_i)),
     f"census: every routed call takes the fused static-Q/K op (t2i {ce_t}, i2i {ce_i})")
note(ce_i.get(k16, 0) - ce_t.get(k16, 0) == 80,
     f"i2i census: image_cond 0-4 add 80 fused M16384 calls per request "
     f"({ce_i.get(k16, 0)} - {ce_t.get(k16, 0)})")
g_t1 = run("t2i_graph_REV_1", "t2i", "REV", True)
g_i1 = run("i2i_graph_REV_1", "i2i", "REV", True)
g_t2 = run("t2i_graph_REV_2_after_i2i", "t2i", "REV", True)
g_i2 = run("i2i_graph_REV_2", "i2i", "REV", True)
f_t = run("t2i_graph_FWD", "t2i", "FWD", True)
f_i = run("i2i_graph_FWD", "i2i", "FWD", True)
o_t = run("t2i_graph_OFF", "t2i", "OFF", True)
o_i = run("i2i_graph_OFF", "i2i", "OFF", True)
m_i = run("i2i_graph_MAIN", "i2i", "MAIN", True)
m_t = run("t2i_graph_MAIN", "t2i", "MAIN", True)
set_arm("REV")

cmp(g_t1, g_t2, "t2i graph REV: repeat after i2i bitwise")
cmp(g_i1, g_i2, "i2i graph REV: repeat after t2i bitwise")
cmp(g_t1, f_t, "t2i graph: REV == FWD (VQ_REVERSE=0)")
cmp(g_i1, f_i, "i2i graph: REV == FWD (VQ_REVERSE=0)")
cmp(g_t1, o_t, "t2i graph: REV == OFF (TRTLLM_FLUX3_QKV_EPI_FP8=0 chain)")
cmp(g_i1, o_i, "i2i graph: REV == OFF (TRTLLM_FLUX3_QKV_EPI_FP8=0 chain)")
cmp(g_i1, m_i, "i2i graph: branch == main's code (static_qk cleanup + forward v_quant)")
cmp(g_t1, m_t, "t2i graph: branch == main's code")
cmp(e_t, g_t1, "t2i REV: eager == graph")
cmp(e_i, g_i1, "i2i REV: eager == graph")
print(json.dumps({
    "passed": not fails,
    "failures": fails,
    "compared_against": "in one process: per-call qkv8/scales[8]/v plane bitwise on real block "
                        "inputs vs the TRTLLM_FLUX3_QKV_EPI_FP8=0 chain and main's forward "
                        "v_quant (eager repeats + graph replays, keep_v both ways, M 16384/16464/"
                        "32848 alternating); final latent + uint8 image bitwise vs FWD, OFF and "
                        "main's code (MAIN arm), repeats t2i->i2i->t2i->i2i",
    "census": {"t2i": ce_t, "i2i": ce_i},
    "comparisons": results["comparisons"],
    "requests": results["requests"],
}))
