"""Idea 85 (re-file of idea 69, branch 51): in-process ROW_REV census of the warp adaLN+NVFP4 head.

Template: experiments/ascent/b20_xpol/census.py (branch 20 X_POLICY census), X_POLICY -> ROW_REV.
Arms switched in-process through flux3_sm120_adaln_nvfp4._ROW_REV (launch() resolves it at call
time, bool / no-select builds only, never u8 SEL_INDEX); a test-only CUDA-graph extra key gives
every arm its own graphs:
  R0a : main default (ROW_REV off)     R1 : ROW_REV=1 on bool + None-select builds
  R0b : R0a again with fresh graphs (A/A twin)
Cases: t2i-2048 and i2i-2048 (frozen check00 reference image), 16 steps, served engine.json.

  census.py <ckpt> <census_rounds> <aba_rounds>

Pre-checks (synthetic, eager): fp4 + SF + x_mod torch.equal R0 vs R1 at 16464 bool select (80-row
partial tile, SF padding rows, outputs pre-filled with a sentinel) and 16384 no-select, affine and
LN_ID forms; u8 select at 32848 under _ROW_REV=1 builds no ROW_REV module; ROW_REV=0 cubin equals
main's source cubin byte for byte.
census rounds: torch.profiler around _run_denoise; per kernel keyed name+grid (calls, us), plus
the kernel right before (writer) and right after (consumer) every head launch, per head grid.
aba rounds: untraced rotated rounds, per request CUDA-event denoise span (decides).
Checks: every request's latent + uint8 image bitwise equal to the case's first R0a request;
0 Dynamo graphs / captures / NVRTC modules after capture.
"""

import collections
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

CKPT = sys.argv[1]
CENSUS_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ABA_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
NAME_LEN = 120
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
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
INFO = {"log": sh("git log --oneline -6"),
        "anc_7ab1b650": sh("git merge-base --is-ancestor 7ab1b650 HEAD; echo rc=$?"),
        "anc_2e350b72": sh("git merge-base --is-ancestor 2e350b72 HEAD; echo rc=$?"),
        "x_policy_default_line": sh(f"grep -n 'TRTLLM_FLUX3_ADALN_WARP_X_POLICY\", \"' {WKP}"),
        "L15_flux3_tres": sh("grep -c flux3_tres experiments/ascent-flashinfer/csrc/"
                             "flux3_fp4_pair_residual_sm120.cu"),
        "L18_last_block_rows": sh("grep -c FLUX3_LAST_BLOCK_ROWS tensorrt_llm/_torch/visual_gen/"
                                  "models/flux3/transformer_flux3.py"),
        "L22_vq_reverse": sh("grep -c VQ_REVERSE tensorrt_llm/_torch/visual_gen/models/flux3/"
                             "flux3_qkv_epi_fp8.py"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("TRTLLM_FLUX3")},
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
log("INFO", json.dumps(INFO))

from torch._dynamo.utils import counters as _dc  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

DEV = torch.device("cuda", 0)
UUID = str(torch.cuda.get_device_properties(0).uuid)
if not UUID.startswith("GPU-"):
    UUID = "GPU-" + UUID
INFO["gpu"] = sh(f"nvidia-smi -i {UUID} --query-gpu=uuid,name,clocks.sm,clocks.mem "
                 "--format=csv,noheader")
INFO["raster_for_16464"] = lptr.raster_for(16464)
INFO["raster_for_16384"] = lptr.raster_for(16384)
INFO["raster_for_32848"] = lptr.raster_for(32848)
INFO["module_X_POLICY"] = wk._X_POLICY
INFO["module_ROW_REV"] = wk._ROW_REV
log("INFO2", json.dumps({k: INFO[k] for k in ("gpu", "raster_for_16464", "raster_for_16384",
                                               "raster_for_32848", "module_X_POLICY",
                                               "module_ROW_REV")}))
note(INFO["module_X_POLICY"] == 0 and INFO["module_ROW_REV"] == 0, "module defaults X0/R0")

CLOCKS = []
_STOP = [False]


def _poll():
    while not _STOP[0]:
        v = sh(f"nvidia-smi -i {UUID} --query-gpu=clocks.sm --format=csv,noheader,nounits")
        try:
            CLOCKS.append(int(v.split()[0]))
        except Exception:  # noqa
            pass
        time.sleep(3)


# ---------------------------------------------------------------- synthetic pre-checks
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


def precheck():
    res = {}
    intro = sh(f"git log --format=%H -S'ROW_REV' -- {WKP} | tail -1")
    main_text = sh(f"git show {intro}^:{WKP}") if intro else ""
    ns = {}
    try:
        start = main_text.index('_SRC = r"""') + len('_SRC = r"""')
        main_src = main_text[start:main_text.index('"""', start)]
        ns["ok"] = True
    except ValueError:
        main_src = None
    res["main_src_from"] = intro[:12] + "^"
    if main_src is not None:
        same = {}
        for ln in ((), ("LN_ID=1",), ("SEL_INDEX=1",), ("LN_ID=1", "SEL_INDEX=1")):
            d = tuple(wk._DEFAULT_DEFINES) + ln
            a, b = cubin_of(main_src, d), cubin_of(wk._SRC, d)
            same["+".join(d)] = a == b
            note(a == b, f"ROW_REV=0 cubin == main cubin for {d}")
        res["cubin_equal_main"] = same
        r1 = cubin_of(wk._SRC, tuple(wk._DEFAULT_DEFINES) + ("ROW_REV=1",))
        res["cubin_rowrev_differs"] = r1 != cubin_of(wk._SRC, tuple(wk._DEFAULT_DEFINES))
    else:
        note(False, "could not extract main _SRC")
    # default key equals main's
    note(not wk._ROW_REV, "default _ROW_REV is 0")
    g = torch.Generator(device=DEV).manual_seed(1234)
    gs = torch.tensor([448.0 * 6.0 / 8.0], device=DEV, dtype=torch.float32)
    keys0 = set(wk._MODULES)
    for form, m, sel_kind in (("bool16464", 16464, "bool"), ("row16384", 16384, None),
                              ("u8_32848", 32848, "u8")):
        x = (torch.randn((1, m, 3072), generator=g, device=DEV) * 2).to(torch.bfloat16)
        if sel_kind == "bool":
            sel = torch.zeros((1, m), dtype=torch.bool, device=DEV)
            sel[:, 80:] = True
            k = 2
        elif sel_kind == "u8":
            sel = (torch.arange(m, device=DEV) % 3).to(torch.uint8).view(1, m)
            k = 3
        else:
            sel, k = None, 1
        if sel is None:
            scale = (torch.randn((1, 1, 3072), generator=g, device=DEV) * .3).to(torch.bfloat16)
            shift = (torch.randn((1, 1, 3072), generator=g, device=DEV) * .3).to(torch.bfloat16)
        else:
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
            if not wk.supported(x, sel, scale, shift, w, b):
                note(False, f"{form} {aff} unsupported synthetic")
                continue
            padded = (m + 127) // 128 * 128
            outs = {}
            for rr in (0, 1, 0):
                wk._ROW_REV = rr
                o = (torch.full_like(x, 7.0), torch.full((m, 1536), 0x5A, dtype=torch.uint8,
                                                         device=DEV),
                     torch.full((padded * 192,), 0xA5, dtype=torch.uint8, device=DEV))
                xin = x.clone()
                r = wk.launch(xin, sel, scale, shift, w, b, gs, 1e-6, outs=o)
                torch.cuda.synchronize()
                note(torch.equal(xin, x), f"{form} {aff} R{rr}: input unchanged")
                outs.setdefault(rr, []).append(tuple(t.clone() for t in r))
            wk._ROW_REV = 0
            a, b1, a2 = outs[0][0], outs[1][0], outs[0][1]
            eq = all(torch.equal(p, q) for p, q in zip(a, b1))
            eq0 = all(torch.equal(p, q) for p, q in zip(a, a2))
            sent_sf = int((b1[2] == 0xA5).sum().item())
            sent_fp4 = bool((b1[1] == 0x5A).all(1).any())
            res[f"{form}_{aff}"] = {"R1_eq_R0": eq, "R0_repeat_eq": eq0,
                                    "R1_sf_sentinel_left": sent_sf,
                                    "R1_fp4_unwritten_row": sent_fp4,
                                    "pad_rows": padded - m}
            note(sent_sf == 0 and not sent_fp4, f"{form} {aff}: R1 wrote every fp4 row + SF byte")
            note(eq, f"{form} {aff}: fp4+SF+x_mod torch.equal R0 vs R1")
            note(eq0, f"{form} {aff}: R0 repeat bitwise")
            if aff == "identity":
                wk.unregister_identity_affine(w, b)
    new_keys = set(wk._MODULES) - keys0
    res["new_module_defines"] = sorted(str(k[3]) for k in new_keys)
    note(not any("ROW_REV=1" in k[3] and any("SEL_INDEX=1" == d for d in k[3])
                 for k in wk._MODULES), "no ROW_REV on SEL_INDEX build")
    log("PRECHECK", json.dumps(res))
    return res


PRE = precheck()

# ---------------------------------------------------------------- served census
ARMS = {"R0a": 0, "R1": 1, "R0b": 0}
ARM = ["R0a"]
CAPREC = collections.defaultdict(collections.Counter)
_orig_compile = wk._compile


def _rec_compile(device, warps, use_pdl, extra_defines=()):
    CAPREC[ARM[0]][(bool(torch.cuda.is_current_stream_capturing()), tuple(extra_defines))] += 1
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _rec_compile


def set_arm(a):
    ARM[0] = a
    wk._ROW_REV = ARMS[a]


def counters():
    return {"unique_graphs": _dc["stats"].get("unique_graphs", 0)}


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
log("engine", json.dumps(engine), "pipeline_config", json.dumps(pipeline_config))
set_arm("R0a")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None} "
    f"graphs={len(runner.graphs) if runner else None}")
note(runner is not None, "transformer CUDA graph runner present")
for rr in (0, 1):
    for ln in ((), ("LN_ID=1",)):
        _orig_compile(DEV, wk._DEFAULT_WARPS, False,
                      tuple(wk._DEFAULT_DEFINES) + ln + (("ROW_REV=1",) if rr else ()))
log("modules", {str(k): v for k, v in wk.NUM_REGS.items()})
runner.clear()
runner.register_extra_key_fn("idea69r_arm", lambda *a, **k: ARM[0])
CAPREC.clear()


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

PROF = {"on": False, "events": None}
_DEN = {}
_orig_run_denoise = pipe._run_denoise


def _timed_run_denoise(*a, **k):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    if PROF["on"]:
        torch.cuda.synchronize()
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        prof.__enter__()
    s.record()
    out = _orig_run_denoise(*a, **k)
    e.record()
    if PROF["on"]:
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        PROF["events"] = prof
    _DEN["ev"] = (s, e)
    return out


pipe._run_denoise = _timed_run_denoise


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
    return ctx[0].shape[1], latent.clone(), img.clone()


HEAD = "flux3_adaln_nvfp4_warp"


def summarize_prof(prof):
    path = os.path.join(ART, f"_tmp_trace_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    os.remove(path)
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"
           and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    evs.sort(key=lambda e: e["ts"])
    per = collections.defaultdict(lambda: [0, 0.0])
    keys = []
    for e in evs:
        grid = e.get("args", {}).get("grid")
        g = "x".join(str(v) for v in grid) if grid else "-"
        k = f"{e['cat']}|{e['name'][:NAME_LEN]}|g{g}"
        keys.append(k)
        per[k][0] += 1
        per[k][1] += float(e["dur"])
    nb = collections.defaultdict(lambda: [0, 0.0])
    for i, k in enumerate(keys):
        if HEAD in k:
            hg = k.rsplit("|", 1)[1]
            if i > 0:
                r = nb[f"prev@{hg}|{keys[i - 1]}"]
                r[0] += 1
                r[1] += float(evs[i - 1]["dur"])
            if i + 1 < len(keys):
                r = nb[f"next@{hg}|{keys[i + 1]}"]
                r[0] += 1
                r[1] += float(evs[i + 1]["dur"])
    busy, cur_s, cur_e = 0.0, None, None
    for e in evs:
        s, t = e["ts"], e["ts"] + e["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (evs[-1]["ts"] + evs[-1]["dur"] - evs[0]["ts"]) if evs else 0.0
    return per, nb, busy, span


REFS = {}


def run(arm, case, profile=False, tag=""):
    set_arm(arm)
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs)
    n_mod = len(wk._MODULES)
    PROF["on"] = profile
    t1 = time.perf_counter()
    T, lat, img = generate(case)
    wall = (time.perf_counter() - t1) * 1000
    PROF["on"] = False
    s, e = _DEN["ev"]
    rec = {"arm": arm, "case": case, "tag": tag, "T": T, "wall_ms": round(wall, 1),
           "denoise_ms": round(s.elapsed_time(e), 3),
           "new_graphs": counters()["unique_graphs"] - prev,
           "new_captures": len(runner.graphs) - prev_caps,
           "new_nvrtc_modules": len(wk._MODULES) - n_mod}
    if profile:
        per, nb, busy, span = summarize_prof(PROF["events"])
        PROF["events"] = None
        rec["busy_us"] = round(busy, 1)
        rec["span_us"] = round(span, 1)
        rec["_per"] = {k: [v[0], round(v[1], 2)] for k, v in per.items()}
        rec["_nb"] = {k: [v[0], round(v[1], 2)] for k, v in nb.items()}
    if case not in REFS:
        REFS[case] = (lat, img)
        rec["bitwise_vs_ref"] = "ref"
    else:
        ok = torch.equal(REFS[case][0], lat) and torch.equal(REFS[case][1], img)
        rec["bitwise_vs_ref"] = ok
        note(ok, f"{tag} {arm} {case}: latent+image bitwise vs first R0a request")
    log("REQ", json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}))
    return rec


results = {"info": INFO, "precheck": PRE}
caprecs = []
for arm in ARMS:
    for case in ("t2i", "i2i"):
        caprecs.append(run(arm, case, tag="capture"))
results["capture_record"] = {a: {f"capturing={k[0]} defines={list(k[1])}": n
                                 for k, n in c.items()} for a, c in CAPREC.items()}
log("CAPREC", json.dumps(results["capture_record"]))
for arm, rr in ARMS.items():
    caps = {k[1] for k in CAPREC[arm] if k[0]}
    want = set()
    for ln in ((), ("LN_ID=1",)):
        base = tuple(wk._DEFAULT_DEFINES) + ln
        want.add(base + (("ROW_REV=1",) if rr else ()))
        want.add(base + ("SEL_INDEX=1",))
    note(bool(caps) and caps <= want, f"arm {arm} captured only ROW_REV={rr} launches: {caps}")
    if rr:
        note(any("ROW_REV=1" in c for c in caps), f"arm {arm} captured a ROW_REV=1 launch")
results["capture_requests"] = caprecs

for arm in ARMS:
    for case in ("t2i", "i2i"):
        r = run(arm, case, tag="warm")
        note(r["new_graphs"] == 0 and r["new_captures"] == 0 and r["new_nvrtc_modules"] == 0,
             f"warm {arm} {case}: 0 compiles/captures/NVRTC")

census = {c: {a: [] for a in ARMS} for c in ("t2i", "i2i")}
for rd in range(CENSUS_ROUNDS):
    order = list(ARMS) if rd % 2 == 0 else list(reversed(list(ARMS)))
    for arm in order:
        for case in ("t2i", "i2i"):
            r = run(arm, case, profile=True, tag=f"census{rd}")
            census[case][arm].append(r)
            note(r["new_graphs"] == 0 and r["new_captures"] == 0 and r["new_nvrtc_modules"] == 0,
                 f"census{rd} {arm} {case}: 0 compiles/captures/NVRTC")

threading.Thread(target=_poll, daemon=True).start()
ORDERS = [["R0a", "R1", "R0b"], ["R1", "R0b", "R0a"], ["R0b", "R0a", "R1"]]
aba = {c: {a: [] for a in ARMS} for c in ("t2i", "i2i")}
for rd in range(ABA_ROUNDS):
    order = ORDERS[rd % 3]
    for arm in order:
        for case in ("t2i", "i2i"):
            r = run(arm, case, tag=f"aba{rd}")
            aba[case][arm].append(r["denoise_ms"])
            note(r["new_graphs"] == 0 and r["new_captures"] == 0 and r["new_nvrtc_modules"] == 0,
                 f"aba{rd} {arm} {case}: 0 compiles/captures/NVRTC")
    log("ROUND", rd, json.dumps({c: {a: aba[c][a][-1] for a in ARMS} for c in aba}))
_STOP[0] = True
INFO["sm_clock_median_mhz"] = statistics.median(CLOCKS) if CLOCKS else None
INFO["sm_clock_min_max"] = [min(CLOCKS), max(CLOCKS)] if CLOCKS else None


def med(v):
    return statistics.median(v) if v else 0.0


out_census = {}
for case in census:
    out = {}
    for field in ("_per", "_nb"):
        names = set()
        for a in ARMS:
            for r in census[case][a]:
                names |= set(r[field])
        rows = {}
        for nm in names:
            row = {}
            for a in ARMS:
                us = [r[field].get(nm, [0, 0.0])[1] for r in census[case][a]]
                calls = [r[field].get(nm, [0, 0.0])[0] for r in census[case][a]]
                c = int(med(calls))
                row[a] = {"ms": round(med(us) / 1000, 3), "calls": c,
                          "us_call": round(med(us) / c, 2) if c else None}
            rows[nm] = row
        if field == "_per":
            sel = {k: v for k, v in rows.items()
                   if HEAD in k or "qkv_epi" in k.lower() or "pair_residual" in k.lower()
                   or "v_quant" in k.lower()}
            top = sorted(rows.items(), key=lambda kv: -kv[1]["R0a"]["ms"])[:25]
            out["focus"] = sel
            out["top"] = dict(top)
        else:
            out["neighbors"] = {k: v for k, v in sorted(rows.items(),
                                                        key=lambda kv: -kv[1]["R0a"]["ms"])
                                if v["R0a"]["calls"] >= 8 or v["R1"]["calls"] >= 8}
    out["busy_ms"] = {a: round(med([r["busy_us"] for r in census[case][a]]) / 1000, 3)
                      for a in ARMS}
    out["denoise_ms_profiled"] = {a: med([r["denoise_ms"] for r in census[case][a]])
                                  for a in ARMS}
    out_census[case] = out
    log("CENSUS", case, json.dumps(out))
results["census"] = out_census


def decide(case):
    d = aba[case]
    m = {a: round(med(d[a]), 3) for a in ARMS}
    mean = {a: round(statistics.mean(d[a]), 3) for a in ARMS}
    base = [(x + y) / 2 for x, y in zip(d["R0a"], d["R0b"])]
    per = [x - b for x, b in zip(d["R1"], base)]
    return {"median_ms": m, "mean_ms": mean, "AA_abs_median_ms": round(abs(m["R0a"] - m["R0b"]), 3),
            "AA_abs_mean_ms": round(abs(mean["R0a"] - mean["R0b"]), 3),
            "R1_minus_R0pair_per_round": {"median": round(med(per), 3), "min": round(min(per), 3),
                                          "max": round(max(per), 3),
                                          "neg": sum(p < 0 for p in per), "n": len(per)},
            "spread_rel": {a: round((max(d[a]) - min(d[a])) / m[a], 5) for a in ARMS},
            "raw": d}


results["aba"] = {c: decide(c) for c in aba}
results["info"] = INFO
log("ABA", json.dumps(results["aba"]))
json.dump(results, open(os.path.join(ART, "idea69r_census.json"), "w"), default=str)
print(json.dumps({"passed": not fails, "failures": fails,
                  "sm_clock_median_mhz": INFO["sm_clock_median_mhz"],
                  "aba": {c: {k: v for k, v in results["aba"][c].items() if k != "raw"}
                          for c in results["aba"]},
                  "head_focus": {c: {k: v for k, v in out_census[c]["focus"].items()
                                     if HEAD in k} for c in out_census}},
                 default=str))
