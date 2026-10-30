"""Branch 51 untimed correctness check for the ROW_REV default (ROW_REV=1 on bool / None-select
warp adaLN+NVFP4 head builds).

(A) synthetic, eager + CUDA graph: x_mod + fp4 + SF torch.equal ROW_REV 0 vs 1 at 16464 bool select
    (80-row partial tile, 48 SF padding rows, outputs pre-filled with sentinels) and 16384
    no-select, affine and LN_ID; u8 SEL_INDEX at 32848 never builds ROW_REV; a captured graph of
    the default launch replayed 3x equals eager; ROW_REV=0 cubins equal main's source byte for
    byte; inputs unchanged.
(B) served pipeline (engine.json, compile, autotune, CUDA graphs from warmup at the default):
    t2i -> i2i -> t2i -> i2i, every repeat bitwise to its first; then ROW_REV=0 with its own graphs
    (test-only extra key): t2i, i2i bitwise to the default's outputs; 0 Dynamo graphs / captures /
    NVRTC modules after capture; capture record of the defines.

  check.py <ckpt>
"""

import collections
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def log(*a):
    print(*a, flush=True)


def note(ok, what):
    if not ok:
        fails.append(what)
        log("FAIL", what)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


WKP = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_sm120_adaln_nvfp4.py"
INFO = {"log": sh("git log --oneline -4"),
        "anc_2e350b72": sh("git merge-base --is-ancestor 2e350b72 HEAD; echo rc=$?")}
log("INFO", json.dumps(INFO))

from torch._dynamo.utils import counters as _dc  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

DEV = torch.device("cuda", 0)
DEFAULT = wk._ROW_REV
INFO["default_row_rev"] = DEFAULT
note(DEFAULT == 1, "module default _ROW_REV == 1")


def cubin_of(src, defines):
    from cuda.bindings import nvrtc
    major, minor = torch.cuda.get_device_capability(DEV)
    prog = wk._check(nvrtc.nvrtcCreateProgram(src.encode(), b"flux3_adaln_nvfp4.cu", 0, [], []))
    opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"--fmad=false",
            b"--prec-div=true", b"--prec-sqrt=true", b"--ftz=false", b"-std=c++17",
            f"-DWARPS={wk._DEFAULT_WARPS}".encode(), b"-DUSE_PDL=0"] + \
        [f"-D{d}".encode() for d in defines]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert int(err) == 0, "nvrtc failed"
    size = wk._check(nvrtc.nvrtcGetCUBINSize(prog))
    buf = b" " * size
    wk._check(nvrtc.nvrtcGetCUBIN(prog, buf))
    nvrtc.nvrtcDestroyProgram(prog)
    return bytes(buf)


def synthetic():
    res = {}
    intro = sh(f"git log --format=%H -S'ROW_REV' -- {WKP} | tail -1")
    main_text = sh(f"git show {intro}^:{WKP}") if intro else ""
    try:
        start = main_text.index('_SRC = r"""') + len('_SRC = r"""')
        main_src = main_text[start:main_text.index('"""', start)]
    except ValueError:
        main_src = None
    note(main_src is not None, "extract main _SRC")
    res["main_src_from"] = intro[:12] + "^"
    if main_src is not None:
        for ln in ((), ("LN_ID=1",), ("SEL_INDEX=1",), ("LN_ID=1", "SEL_INDEX=1")):
            d = tuple(wk._DEFAULT_DEFINES) + ln
            eq = cubin_of(main_src, d) == cubin_of(wk._SRC, d)
            res["cubin_eq_main:" + "+".join(d)] = eq
            note(eq, f"ROW_REV=0 cubin == main cubin {d}")
    g = torch.Generator(device=DEV).manual_seed(4321)
    gs = torch.tensor([448.0 * 6.0 / 8.0], device=DEV, dtype=torch.float32)
    for form, m, kind in (("bool16464", 16464, "bool"), ("row16384", 16384, None),
                          ("u8_32848", 32848, "u8")):
        x = (torch.randn((1, m, 3072), generator=g, device=DEV) * 2).to(torch.bfloat16)
        if kind == "bool":
            sel = torch.zeros((1, m), dtype=torch.bool, device=DEV)
            sel[:, 80:] = True
            k = 2
        elif kind == "u8":
            sel = (torch.arange(m, device=DEV) % 3).to(torch.uint8).view(1, m)
            k = 3
        else:
            sel, k = None, 1
        scale = (torch.randn((1, k, 3072), generator=g, device=DEV) * .3).to(torch.bfloat16)
        shift = (torch.randn((1, k, 3072), generator=g, device=DEV) * .3).to(torch.bfloat16)
        for aff in ("affine", "identity"):
            if aff == "affine":
                w = (1 + torch.randn(3072, generator=g, device=DEV) * .1).to(torch.bfloat16)
                b = (torch.randn(3072, generator=g, device=DEV) * .1).to(torch.bfloat16)
            else:
                w = torch.ones(3072, dtype=torch.bfloat16, device=DEV)
                b = torch.zeros(3072, dtype=torch.bfloat16, device=DEV)
                wk.register_identity_affine(w, b)
            note(wk.supported(x, sel, scale, shift, w, b), f"{form} {aff} supported")
            padded = (m + 127) // 128 * 128
            x0 = x.clone()

            def fresh():
                return (torch.full_like(x, 7.0),
                        torch.full((m, 1536), 0x5A, dtype=torch.uint8, device=DEV),
                        torch.full((padded * 192,), 0xA5, dtype=torch.uint8, device=DEV))

            outs = {}
            for rr in (0, 1, 1, 0):
                wk._ROW_REV = rr
                o = fresh()
                r = wk.launch(x, sel, scale, shift, w, b, gs, 1e-6, outs=o)
                torch.cuda.synchronize()
                outs.setdefault(rr, []).append(tuple(t.clone() for t in r))
            # graph replay of the default launch vs eager
            wk._ROW_REV = DEFAULT
            o = fresh()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                wk.launch(x, sel, scale, shift, w, b, gs, 1e-6, outs=o)  # build outside capture
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            o = fresh()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                wk.launch(x, sel, scale, shift, w, b, gs, 1e-6, outs=o)
            g_ok = True
            for _ in range(3):
                for t, fill in zip(o, (7.0, 0x5A, 0xA5)):
                    t.fill_(fill)
                graph.replay()
                torch.cuda.synchronize()
                g_ok &= all(torch.equal(p, q) for p, q in zip(o, outs[DEFAULT][0]))
            del graph
            wk._ROW_REV = DEFAULT
            ref = outs[0][0]
            eq01 = all(all(torch.equal(p, q) for p, q in zip(ref, t))
                       for t in outs[0][1:] + outs[1])
            res[f"{form}_{aff}"] = {
                "R0_eq_R1_eager_x2": eq01, "graph_replay_x3_eq_eager": g_ok,
                "sf_sentinel_left": int((outs[1][0][2] == 0xA5).sum().item()),
                "fp4_unwritten_row": bool((outs[1][0][1] == 0x5A).all(1).any()),
                "pad_rows": padded - m}
            note(eq01, f"{form} {aff}: x_mod+fp4+SF R0 == R1 (eager, repeats)")
            note(g_ok, f"{form} {aff}: graph replay x3 == eager")
            note(res[f"{form}_{aff}"]["sf_sentinel_left"] == 0
                 and not res[f"{form}_{aff}"]["fp4_unwritten_row"], f"{form} {aff}: all written")
            note(torch.equal(x, x0), f"{form} {aff}: input unchanged")
            if aff == "identity":
                wk.unregister_identity_affine(w, b)
    bad = [k for k in wk._MODULES if "ROW_REV=1" in k[3] and "SEL_INDEX=1" in k[3]]
    note(not bad, f"no ROW_REV on SEL_INDEX build: {bad}")
    log("SYNTH", json.dumps(res))
    return res


SYN = synthetic()

ARM = ["default"]
ARMS = {"default": DEFAULT, "R0": 0}
CAPREC = collections.defaultdict(collections.Counter)
_orig_compile = wk._compile


def _rec_compile(device, warps, use_pdl, extra_defines=()):
    CAPREC[ARM[0]][(bool(torch.cuda.is_current_stream_capturing()), tuple(extra_defines))] += 1
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _rec_compile

engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
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
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
note(runner is not None, "transformer CUDA graph runner present")
log(f"loaded+warmup {time.time() - t0:.0f}s graphs={len(runner.graphs) if runner else None}")
warm_rec = {f"capturing={k[0]} defines={list(k[1])}": n for k, n in CAPREC["default"].items()}
log("WARMUP_CAPREC", json.dumps(warm_rec))
cap_defs = {k[1] for k in CAPREC["default"] if k[0]}
note(any("ROW_REV=1" in d for d in cap_defs), "warmup graphs captured ROW_REV=1 launches")
note(all(("ROW_REV=1" in d) != ("SEL_INDEX=1" in d) for d in cap_defs),
     f"every captured head build is exactly one of ROW_REV / SEL_INDEX: {cap_defs}")
# from here, the test-only extra key separates the R0 arm's graphs (default keeps "default")
runner.clear()
runner.register_extra_key_fn("b51_arm", lambda *a, **k: ARM[0])


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


SHA00 = "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"
_data = open(store_png(SHA00), "rb").read()
assert hashlib.sha256(_data).hexdigest() == SHA00
REF_IMG = _load_image_pixels(_data, device=DEV)
CASES = {"t2i": ("A red fox standing in fresh snow at sunrise, photograph", 11),
         "i2i": ("Turn the scene into a watercolor painting", 5)}


def generate(case):
    prompt, seed = CASES[case]
    with torch.inference_mode():
        ctx = pipe._encode_prompt(prompt, None)
    with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
        if case == "t2i":
            latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                            num_inference_steps=16, seed=seed)
        else:
            latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[REF_IMG], height=2048,
                                            width=2048, num_inference_steps=16, seed=seed)
        img = pipe._package_output(pipe._decode_latents((latent, ids))).image
    torch.cuda.synchronize()
    return latent.clone(), img.clone()


REFS = {}
reqs = []


def run(arm, case, tag, expect_static):
    ARM[0] = arm
    wk._ROW_REV = ARMS[arm]
    g0 = _dc["stats"].get("unique_graphs", 0)
    c0 = len(runner.graphs)
    n0 = len(wk._MODULES)
    lat, img = generate(case)
    rec = {"arm": arm, "case": case, "tag": tag,
           "new_graphs": _dc["stats"].get("unique_graphs", 0) - g0,
           "new_captures": len(runner.graphs) - c0, "new_nvrtc": len(wk._MODULES) - n0}
    if case not in REFS:
        REFS[case] = (lat, img)
        rec["bitwise"] = "ref"
    else:
        ok = torch.equal(REFS[case][0], lat) and torch.equal(REFS[case][1], img)
        rec["bitwise"] = ok
        note(ok, f"{tag} {arm} {case}: latent+image bitwise vs first default request")
    if expect_static:
        note(rec["new_graphs"] == 0 and rec["new_captures"] == 0 and rec["new_nvrtc"] == 0,
             f"{tag} {arm} {case}: 0 compiles/captures/NVRTC")
    log("REQ", json.dumps(rec))
    reqs.append(rec)


# default arm: first request per case re-captures under the new extra key
for i, case in enumerate(("t2i", "i2i", "t2i", "i2i", "t2i", "i2i")):
    run("default", case, f"default{i}", expect_static=i >= 2)
for i, case in enumerate(("t2i", "i2i", "t2i", "i2i")):
    run("R0", case, f"R0_{i}", expect_static=i >= 2)
for case in ("t2i", "i2i"):
    run("default", case, "default_after_R0", expect_static=True)
r0_caps = {k[1] for k in CAPREC["R0"] if k[0]}
note(bool(r0_caps) and not any("ROW_REV=1" in d for d in r0_caps),
     f"R0 arm captured no ROW_REV launches: {r0_caps}")
print(json.dumps({"passed": not fails, "failures": fails, "info": INFO,
                  "compared": "x_mod/fp4/SF R0 vs R1 synthetic (eager + graph); served "
                              "latent+uint8 image default (ROW_REV=1) repeats and ROW_REV=0 arm "
                              "vs first default request; ROW_REV=0 cubin vs main source",
                  "synthetic": SYN, "requests": reqs,
                  "r0_capture_defines": sorted(str(d) for d in r0_caps)}, default=str))
