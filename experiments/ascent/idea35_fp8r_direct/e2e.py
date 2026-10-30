"""Idea 35: in-process pipeline check of the FP8R image.0 QKV straight to E4M3 (served config).

One process, the pipeline as served (engine.json overlay, torch.compile + autotune, CUDA graphs,
2048^2 t2i + i2i warmup), loaded with the branch default (TRTLLM_FLUX3_FP8R_QKV_FP8_DIRECT on):
  ON  graph-on : t2i A, i2i X, t2i B, i2i X(repeat), t2i A(repeat)   -> no compile / capture
  ON  graph-off: t2i A, i2i X with the census hooks
  flip flux3_fp8r_qkv.FP8R_QKV_FP8_DIRECT = False (main's path; Dynamo guards on it), runner.clear()
  OFF graph-on : t2i A, i2i X ; OFF graph-off: t2i A, i2i X with the census hooks
Census hooks (graph-off requests, Python op bodies): fp8_sm12x.quantize_qkv_e4m3 (K1/K2), the
in-place qknorm_rope pass, qknorm_rope_fp8 by layout/static_qk, landing 5's
flux3_qkv_epi_fp8.qkv_rope_fp8, the overlay GEMM routes and the new twin. image.0's per-call
qkv8 bytes / scales[0..3] / FMHA output are sha256-hashed: OFF from K1/K2 + the following fmha_v2
call, ON from the twin's qknorm_rope_fp8 (packed column views, row stride 3W) + the following
fmha_v2 call. Checks: ON == OFF (latent + uint8 image) for every request in both graph modes, the
16 image.0 calls per request equal call by call, repeats equal, graph-on == graph-off, census.
Usage: python e2e.py <checkpoint>
"""

import hashlib
import json
import os
import sys
import time

import torch
from torch._dynamo.utils import counters

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
fails = []


def log(*a):
    print(*a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


def store_png(sha):
    store = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
             "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs")
    for root, _, files in os.walk(store):
        for f in files:
            if sha in f:
                return os.path.join(root, f)
    for root, _, files in os.walk(store):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return p
    raise FileNotFoundError(sha)


from tensorrt_llm._torch.visual_gen.attention.backends import fp8_sm12x  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as _fi  # noqa: E402
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope as qr,
)
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: E402
    flux3_sm120_qknorm_rope_fp8 as qr8,
)
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as g  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_qkv as f8r  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_qkv_epi_fp8 as epi  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import (  # noqa: E402
    autotune_context,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (  # noqa: E402
    _load_image_pixels,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

HOOK = {"on": False, "pending": False}
CENSUS = {}
IMG0 = []  # per image.0 call: dict of hashes


def bump(k):
    CENSUS[k] = CENSUS.get(k, 0) + 1


def sha(t):
    t = t.detach().contiguous()
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


_quant = fp8_sm12x.quantize_qkv_e4m3
_qknorm_rope = qr.qknorm_rope
_qr8 = qr8.qknorm_rope_fp8
_fmha = _fi.run_fmha_v2_fp8_sm12x
_epi = epi.qkv_rope_fp8


def h_quant(q, k, v, sm_scale):
    out = _quant(q, k, v, sm_scale)
    if HOOK["on"]:
        bump(f"K1K2_quantize_qkv_e4m3_rows{q.shape[0] * q.shape[1]}")
        q8, k8, v8, sc = out
        IMG0.append({"q8": sha(q8), "k8": sha(k8), "v8": sha(v8), "scales0_3": sha(sc[:4]),
                     "src": "K1K2"})
        HOOK["pending"] = True
    return out


def h_qknorm_rope(q, k, v, dst, *a, **kw):
    if HOOK["on"]:
        bump(f"qknorm_rope_rows{q.shape[0]}_stride{q.stride(0)}_planar{int(kw.get('planar', False))}"
             f"_inplace{int(dst.data_ptr() == q.data_ptr())}")
    return _qknorm_rope(q, k, v, dst, *a, **kw)


def h_qr8(q, k, v, *a, **kw):
    out = _qr8(q, k, v, *a, **kw)
    if HOOK["on"]:
        packed = q.stride(0) == 3 * q.shape[1]
        bump(f"qknorm_rope_fp8_rows{q.shape[0]}_packed{int(packed)}"
             f"_static{int(bool(kw.get('static_qk', True)))}")
        if packed and not kw.get("static_qk", True):
            qkv8, sc = out
            IMG0.append({"q8": sha(qkv8[0]), "k8": sha(qkv8[1]), "v8": sha(qkv8[2]),
                         "scales0_3": sha(sc[:4]), "scales_all": [float(x) for x in sc.cpu()],
                         "src": "twin"})
            HOOK["pending"] = True
    return out


def h_fmha(q8, k8, v8, out, *a, **kw):
    r = _fmha(q8, k8, v8, out, *a, **kw)
    if HOOK["on"]:
        bump("fmha_v2_fp8_sm12x")
        if HOOK["pending"]:
            IMG0[-1]["fmha"] = sha(out)
            HOOK["pending"] = False
    return r


def h_epi(*a, **kw):
    if HOOK["on"]:
        bump("landing5_flux3_qkv_epi_fp8.qkv_rope_fp8")
    return _epi(*a, **kw)


fp8_sm12x.quantize_qkv_e4m3 = h_quant
qr.qknorm_rope = h_qknorm_rope
qr8.qknorm_rope_fp8 = h_qr8
_fi.run_fmha_v2_fp8_sm12x = h_fmha
epi.qkv_rope_fp8 = h_epi

inputs = {"check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
          "check01": "7928b9c476065f4c55503eefff5cc525d506354183ca588955108180127e8581"}
engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
pipeline_config = dict(engine.pop("pipeline_config", {}))
pipeline_config["variant"] = "nvfp4"
engine.setdefault("torch_compile_config",
                  {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
engine.setdefault("parallel_config", {"tp_size": 1})
args = VisualGenArgs(
    model=CKPT, pipeline_config=pipeline_config,
    compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1], "modes": ["t2i", "i2i"]},
    **engine,
)
log(f"FP8R_QKV_FP8_DIRECT={f8r.FP8R_QKV_FP8_DIRECT} engine={engine}")
note(f8r.FP8R_QKV_FP8_DIRECT, "branch default is ON")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
    f"graphs={len(runner.graphs) if runner else None}")
note(f8r.fp8r_qkv_rope_fp8 is g.fp8r_qkv_rope_fp8_cg, "install() rebound fp8r_qkv_rope_fp8 to twin")
log("gemm routes after warmup", dict(g.ROUTES), "twin calls", dict(g.QKV_FP8_ROUTES))
dev = torch.device("cuda:0")
refs = {}


def ref_image(key):
    if key not in refs:
        data = open(store_png(inputs[key]), "rb").read()
        assert hashlib.sha256(data).hexdigest() == inputs[key]
        refs[key] = _load_image_pixels(data, device=dev)
    return refs[key]


def stats():
    return {"frames": dict(counters["frames"]),
            "unique_graphs": counters["stats"].get("unique_graphs", 0),
            "graphs": len(runner.graphs) if runner else 0}


REQ = {
    "t2i_A": ("t2i", "A red fox standing in fresh snow at sunrise, photograph", 11, None),
    "t2i_B": ("t2i", "A cozy reading nook with a green armchair, a lamp and a stack of books", 22,
              None),
    "i2i_X": ("i2i", "Turn the scene into a watercolor painting", 5, "check00"),
}
records = []
outs = {}


def run(tag, name, graphs, census=False):
    mode, prompt, seed, ref = REQ[name]
    if runner is not None:
        runner.enabled = graphs
    CENSUS.clear()
    IMG0.clear()
    HOOK["on"], HOOK["pending"] = census, False
    before = stats()
    t1 = time.time()
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if mode == "t2i":
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                            num_inference_steps=16, seed=seed)
        else:
            latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[ref_image(ref)], height=2048,
                                            width=2048, num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    HOOK["on"] = False
    after = stats()
    rec = {"tag": tag, "req": name, "graphs": graphs, "s": round(time.time() - t1, 2),
           "latent": sha(latent.float()), "image": sha(img),
           "new_frames": {k: after["frames"].get(k, 0) - before["frames"].get(k, 0)
                          for k in after["frames"]},
           "new_unique_graphs": after["unique_graphs"] - before["unique_graphs"],
           "new_cuda_graphs": after["graphs"] - before["graphs"]}
    if census:
        rec["census"] = dict(CENSUS)
        rec["image0_calls"] = list(IMG0)
    log(tag, json.dumps({k: v for k, v in rec.items() if k != "image0_calls"}))
    records.append(rec)
    outs[tag] = rec
    return rec


def compiled_nothing(rec):
    return (not rec["new_unique_graphs"] and not rec["new_cuda_graphs"]
            and not any(rec["new_frames"].values()))


# ---- ON (branch default) ----
for tag, name in (("ON_t2i_A", "t2i_A"), ("ON_i2i_X", "i2i_X"), ("ON_t2i_B", "t2i_B"),
                  ("ON_i2i_X_rep", "i2i_X"), ("ON_t2i_A_rep", "t2i_A")):
    r = run(tag, name, True)
    note(compiled_nothing(r), f"{tag}: compiled or captured after warmup")
run("ON_t2i_A_eager", "t2i_A", False, census=True)
run("ON_i2i_X_eager", "i2i_X", False, census=True)
# ---- OFF (main's path) ----
f8r.FP8R_QKV_FP8_DIRECT = False
if runner is not None:
    runner.clear()
run("OFF_t2i_A", "t2i_A", True)
run("OFF_i2i_X", "i2i_X", True)
run("OFF_t2i_A_eager", "t2i_A", False, census=True)
run("OFF_i2i_X_eager", "i2i_X", False, census=True)


def same(a, b, what):
    ok = outs[a]["latent"] == outs[b]["latent"] and outs[a]["image"] == outs[b]["image"]
    note(ok, what)
    return ok


comparisons = {
    "ON==OFF t2i graph": same("ON_t2i_A", "OFF_t2i_A", "ON==OFF t2i graph"),
    "ON==OFF i2i graph": same("ON_i2i_X", "OFF_i2i_X", "ON==OFF i2i graph"),
    "ON==OFF t2i eager": same("ON_t2i_A_eager", "OFF_t2i_A_eager", "ON==OFF t2i eager"),
    "ON==OFF i2i eager": same("ON_i2i_X_eager", "OFF_i2i_X_eager", "ON==OFF i2i eager"),
    "ON t2i repeat": same("ON_t2i_A", "ON_t2i_A_rep", "ON t2i repeat"),
    "ON i2i repeat": same("ON_i2i_X", "ON_i2i_X_rep", "ON i2i repeat"),
    "ON t2i graph==eager": same("ON_t2i_A", "ON_t2i_A_eager", "ON t2i graph==eager"),
    "ON i2i graph==eager": same("ON_i2i_X", "ON_i2i_X_eager", "ON i2i graph==eager"),
}
image0 = {}
for case in ("t2i_A", "i2i_X"):
    on, off = outs[f"ON_{case}_eager"]["image0_calls"], outs[f"OFF_{case}_eager"]["image0_calls"]
    note(len(on) == 16 and len(off) == 16, f"{case}: 16 image.0 calls each (on {len(on)}, "
         f"off {len(off)})")
    eq = [all(a.get(k) == b.get(k) for k in ("q8", "k8", "v8", "scales0_3", "fmha"))
          and a.get("fmha") is not None for a, b in zip(on, off)]
    note(all(eq) and len(eq) == 16, f"{case}: image.0 qkv8/scales/FMHA ON==OFF per call {eq}")
    image0[case] = {"calls": len(on), "equal_calls": sum(eq), "on_src": on[0]["src"] if on else None,
                    "off_src": off[0]["src"] if off else None,
                    "scales_all_first_call_on": on[0].get("scales_all") if on else None}
    con, coff = outs[f"ON_{case}_eager"]["census"], outs[f"OFF_{case}_eager"]["census"]
    k1on = sum(v for k, v in con.items() if k.startswith("K1K2"))
    k1off = sum(v for k, v in coff.items() if k.startswith("K1K2"))
    inpl_on = sum(v for k, v in con.items() if k.startswith("qknorm_rope_rows") and "_inplace1" in k)
    note(k1on == 0, f"{case}: ON K1/K2 calls {k1on}")
    note(k1off == 16, f"{case}: OFF K1/K2 calls {k1off}")
    note(inpl_on == 0, f"{case}: ON in-place qknorm_rope passes {inpl_on}")
    e_on = con.get("landing5_flux3_qkv_epi_fp8.qkv_rope_fp8", 0)
    e_off = coff.get("landing5_flux3_qkv_epi_fp8.qkv_rope_fp8", 0)
    note(e_on == e_off and e_on > 0, f"{case}: landing-5 epi route calls ON {e_on} OFF {e_off}")
    other_on = {k: v for k, v in con.items() if not k.startswith(("K1K2", "qknorm_rope_fp8_rows16384_packed1"))}
    other_off = {k: v for k, v in coff.items() if not k.startswith(("K1K2", "qknorm_rope_rows16384_stride9216"))}
    image0[case]["non_image0_routes_on"] = other_on
    image0[case]["non_image0_routes_off"] = other_off
    note({k: v for k, v in other_on.items() if k != "fmha_v2_fp8_sm12x"}
         == {k: v for k, v in other_off.items() if k != "fmha_v2_fp8_sm12x"},
         f"{case}: non-image.0 routes identical ON vs OFF")
log("gemm routes", dict(g.ROUTES), "twin calls", dict(g.QKV_FP8_ROUTES))
print(json.dumps({
    "passed": not fails, "failures": fails, "comparisons": comparisons, "image0": image0,
    "twin_calls_total": dict(g.QKV_FP8_ROUTES), "gemm_routes": dict(g.ROUTES),
    "compared_against": "TRTLLM_FLUX3_FP8R_QKV_FP8_DIRECT off (main's image.0 path: GEMM -> "
    "in-place qknorm_rope -> fp8_sm12x K1/K2 -> fmha_v2) in the same process, same requests; "
    "latent + uint8 image sha256; image.0 qkv8/scales[0..3]/FMHA sha256 per call",
    "requests": [{k: v for k, v in r.items() if k != "image0_calls"} for r in records],
}))
