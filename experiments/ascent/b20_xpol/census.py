"""Branch 20 (idea 31): in-process X_POLICY 0/1/2 re-tune of the warp adaLN head's x loads at 2048^2.

Arms switched in-process through flux3_sm120_adaln_nvfp4._X_POLICY (launch() resolves it at call
time; a test-only CUDA-graph extra key gives every arm its own graphs; the custom op is opaque to
Dynamo, so every arm shares compiled code):
  X0 : plain loads           X1 : evict_last fraction 1.0
  X2 : evict_last frac 0.5 (main's default)     X2b: X2 again, separate graphs (A/A twin)
Cases: t2i-2048 and i2i-2048 (frozen check00 reference image), 16 steps, served engine.json.

  census.py <ckpt> <census_rounds> <aba_rounds>
census rounds: torch.profiler (CUDA activity) only around _run_denoise; per kernel name calls/us,
  GPU busy (union of kernel intervals) and idle per step vs the profiled denoise span.
aba rounds: UNTRACED rotation, per request CUDA-event denoise span (decides).
Checks: every request's latent + uint8 image bitwise equal to the case's first X2 request
(main's default), repeats included, t2i->i2i alternation; 0 Dynamo graphs / 0 captures / 0 NVRTC
modules after each arm's capture request; capture record of the defines each arm's graph used.
"""

import collections
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

import torch

CKPT = sys.argv[1]
CENSUS_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ABA_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
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
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa
        return f"ERR {e}"


QKV = "tensorrt_llm/_torch/visual_gen/models/flux3/flux3_qkv_epi_fp8.py"
INFO = {"log": sh("git log --oneline -8"),
        "anc_e0d4536c": sh("git merge-base --is-ancestor e0d4536c HEAD; echo rc=$?"),
        "qkv_epi_l2policy_grep": sh(f"grep -c QKV_EPI_L2POLICY {QKV}"),
        "x_policy_default_line": sh("grep -n 'TRTLLM_FLUX3_ADALN_WARP_X_POLICY\", \"' "
                                    "tensorrt_llm/_torch/visual_gen/models/flux3/"
                                    "flux3_sm120_adaln_nvfp4.py"),
        "env_x_policy": os.environ.get("TRTLLM_FLUX3_ADALN_WARP_X_POLICY"),
        "env_qkv_l2": os.environ.get("TRTLLM_FLUX3_QKV_EPI_L2POLICY"),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sh("nvidia-smi --query-gpu=uuid,name,clocks.sm,clocks.mem --format=csv,noheader")}
log("INFO", json.dumps(INFO))

from torch._dynamo.utils import counters as _dc  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as wk  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels  # noqa
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402

ARMS = {"X0": 0, "X1": 1, "X2": 2, "X2b": 2}
ARM = ["X2"]
CAPREC = collections.defaultdict(collections.Counter)
_orig_compile = wk._compile


def _rec_compile(device, warps, use_pdl, extra_defines=()):
    CAPREC[ARM[0]][(bool(torch.cuda.is_current_stream_capturing()), tuple(extra_defines))] += 1
    return _orig_compile(device, warps, use_pdl, extra_defines)


wk._compile = _rec_compile


def set_arm(a):
    ARM[0] = a
    wk._X_POLICY = ARMS[a]


def counters():
    return {"frames": dict(_dc["frames"]), "unique_graphs": _dc["stats"].get("unique_graphs", 0)}


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
log("module default _X_POLICY", wk._X_POLICY)
note(wk._X_POLICY == 2, "module default X_POLICY is 2 (main's default)")
set_arm("X2")
t0 = time.time()
pipe = PipelineLoader(args, device="cuda:0").load()
torch.cuda.synchronize()
runner = pipe._cuda_graph_runners.get("transformer")
log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None} "
    f"graphs={len(runner.graphs) if runner else None}")
note(runner is not None, "transformer CUDA graph runner present")
for xp in (0, 1, 2):
    for ln in ((), ("LN_ID=1",)):
        _orig_compile(torch.device("cuda", 0), wk._DEFAULT_WARPS, False,
                      tuple(wk._DEFAULT_DEFINES) + ((f"X_POLICY={xp}",) if xp else ()) + ln)
log("modules", {str(k): v for k, v in wk.NUM_REGS.items()})
runner.clear()
runner.register_extra_key_fn("b20_arm", lambda *a, **k: ARM[0])
CAPREC.clear()
blocks = [m for m in pipe.transformer.modules() if hasattr(m, "_sm120_pair_residual")]
log("pair_residual routed blocks", sum(bool(m._sm120_pair_residual) for m in blocks),
    "of", len(blocks))
ws_holder = lptr._WORKSPACES


def epoch():
    ws = ws_holder.get(0)
    if ws is None:
        return 0
    return int(ws[:256].view(torch.int32)[32].item())


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
REF_IMG = _load_image_pixels(_data, device=torch.device("cuda:0"))
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


def summarize_prof(prof):
    per = collections.defaultdict(lambda: [0, 0.0])
    iv = []
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if "Memcpy" in ev.name or "Memset" in ev.name:
            kind = "memop"
        else:
            kind = "kernel"
        d = ev.device_time_total
        r = per[(kind, ev.name[:160])]
        r[0] += 1
        r[1] += d
        iv.append((ev.time_range.start, ev.time_range.end))
    iv.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (iv[-1][1] - iv[0][0]) if iv else 0.0
    return per, busy, span


REFS = {}


def run(arm, case, profile=False, tag=""):
    set_arm(arm)
    prev = counters()["unique_graphs"]
    prev_caps = len(runner.graphs)
    n_mod = len(wk._MODULES)
    e0 = epoch()
    PROF["on"] = profile
    t1 = time.perf_counter()
    T, lat, img = generate(case)
    wall = (time.perf_counter() - t1) * 1000
    PROF["on"] = False
    s, e = _DEN["ev"]
    rec = {"arm": arm, "case": case, "tag": tag, "T": T, "wall_ms": round(wall, 1),
           "denoise_ms": round(s.elapsed_time(e), 3), "fused_pair_launches": epoch() - e0,
           "new_graphs": counters()["unique_graphs"] - prev,
           "new_captures": len(runner.graphs) - prev_caps,
           "new_nvrtc_modules": len(wk._MODULES) - n_mod}
    if profile:
        per, busy, span = summarize_prof(PROF["events"])
        PROF["events"] = None
        rec["busy_us"] = round(busy, 1)
        rec["span_us"] = round(span, 1)
        rec["idle_per_step_us"] = round((span - busy) / 16, 1)
        rec["kernel_us"] = round(sum(v[1] for k, v in per.items() if k[0] == "kernel"), 1)
        rec["_per"] = {f"{k[0]}|{k[1]}": [v[0], round(v[1], 2)] for k, v in per.items()}
    if case not in REFS:
        REFS[case] = (lat, img)
        rec["bitwise_vs_ref"] = "ref"
    else:
        ok = torch.equal(REFS[case][0], lat) and torch.equal(REFS[case][1], img)
        rec["bitwise_vs_ref"] = ok
        note(ok, f"{tag} {arm} {case}: latent+image bitwise vs first X2 request")
    log("REQ", json.dumps({k: v for k, v in rec.items() if k != "_per"}))
    return rec


results = {"info": INFO}
# capture requests: X2 first (the reference = main default), alternate t2i/i2i
caprecs = []
for arm in ("X2", "X0", "X1", "X2b"):
    for case in ("t2i", "i2i"):
        caprecs.append(run(arm, case, tag="capture"))
results["capture_record"] = {a: {f"capturing={k[0]} defines={list(k[1])}": n
                                 for k, n in c.items()} for a, c in CAPREC.items()}
log("CAPREC", json.dumps(results["capture_record"]))
for arm, xp in ARMS.items():
    caps = {k[1] for k in CAPREC[arm] if k[0]}
    want = {tuple(wk._DEFAULT_DEFINES) + ((f"X_POLICY={xp}",) if xp else ()) + ln
            for ln in ((), ("LN_ID=1",))}
    note(bool(caps) and caps <= want, f"arm {arm} graph captured only X_POLICY={xp} launches: {caps}")
results["capture_requests"] = caprecs

# warm, untimed: one request per arm and case after capture, checking no new compiles/captures
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

aba = {c: {a: [] for a in ARMS} for c in ("t2i", "i2i")}
for rd in range(ABA_ROUNDS):
    order = list(ARMS) if rd % 2 == 0 else list(reversed(list(ARMS)))
    for arm in order:
        for case in ("t2i", "i2i"):
            r = run(arm, case, tag=f"aba{rd}")
            aba[case][arm].append(r["denoise_ms"])
            note(r["new_graphs"] == 0 and r["new_captures"] == 0 and r["new_nvrtc_modules"] == 0,
                 f"aba{rd} {arm} {case}: 0 compiles/captures/NVRTC")
    log("ROUND", rd, json.dumps({c: {a: aba[c][a][-1] for a in ARMS} for c in aba}))


def classify(name):
    n = name.split("|", 1)[1]
    low = n.lower()
    if "flux3_adaln_nvfp4_warp" in n:
        return "warp_head"
    if "lpt_pair_residual" in low or "pairresidual" in low or "pair_residual" in low:
        return "pair_residual_fused"
    if "add_mul" in low and "triton" in low:
        return "inductor_residual"
    if "qkv_epi" in low or "qkvepi" in low:
        return "qkv_epi_fp8"
    if "v_quant" in low or "vquant" in low:
        return "v_quant"
    if "fmha" in low or "flash" in low or "attention" in low:
        return "fmha"
    if "gate_up" in low or "swiglu" in low or "gateup" in low:
        return "gate_up"
    if re.search(r"gemm|cutlass|nvjet|cublas|xmma|_mma|matmul|sm120", low):
        return "gemm_other"
    if name.startswith("memop"):
        return "memop"
    return "other"


def med(v):
    return statistics.median(v) if v else 0.0


out_census = {}
for case in census:
    names = set()
    for a in ARMS:
        for r in census[case][a]:
            names |= set(r["_per"])
    rows = {}
    for nm in names:
        row = {"class": classify(nm)}
        for a in ARMS:
            us = [r["_per"].get(nm, [0, 0.0])[1] for r in census[case][a]]
            calls = [r["_per"].get(nm, [0, 0.0])[0] for r in census[case][a]]
            row[a] = {"us": round(med(us), 1), "calls": int(med(calls))}
        rows[nm] = row
    cls = collections.defaultdict(lambda: {a: [0, 0.0] for a in ARMS})
    for nm, row in rows.items():
        for a in ARMS:
            cls[row["class"]][a][0] += row[a]["calls"]
            cls[row["class"]][a][1] += row[a]["us"]
    top = sorted(rows.items(), key=lambda kv: -kv[1]["X2"]["us"])[:60]
    out_census[case] = {
        "classes": {k: {a: {"calls": v[a][0], "ms": round(v[a][1] / 1000, 3)} for a in ARMS}
                    for k, v in cls.items()},
        "busy_ms": {a: round(med([r["busy_us"] for r in census[case][a]]) / 1000, 3) for a in ARMS},
        "span_ms": {a: round(med([r["span_us"] for r in census[case][a]]) / 1000, 3) for a in ARMS},
        "kernel_ms": {a: round(med([r["kernel_us"] for r in census[case][a]]) / 1000, 3)
                      for a in ARMS},
        "idle_per_step_us": {a: med([r["idle_per_step_us"] for r in census[case][a]])
                             for a in ARMS},
        "denoise_ms_profiled": {a: med([r["denoise_ms"] for r in census[case][a]]) for a in ARMS},
        "top_kernels": [{"name": nm, **row} for nm, row in top],
    }
    log("CENSUS", case, json.dumps({k: v for k, v in out_census[case].items()
                                    if k != "top_kernels"}))
results["census"] = out_census


def decide(case):
    d = aba[case]
    m = {a: med(d[a]) for a in ARMS}
    aa = abs(m["X2"] - m["X2b"])
    base = [(x + y) / 2 for x, y in zip(d["X2"], d["X2b"])]
    out = {"median_ms": m, "AA_abs_ms": round(aa, 3),
           "noise_rel": {a: round((max(d[a]) - min(d[a])) / m[a], 5) if d[a] else None
                         for a in ARMS}}
    for a in ("X0", "X1"):
        per = [x - b for x, b in zip(d[a], base)]
        out[f"{a}_minus_X2pair_per_round"] = {"median": round(med(per), 3),
                                              "min": round(min(per), 3), "max": round(max(per), 3),
                                              "neg": sum(p < 0 for p in per), "n": len(per)}
    out["raw"] = d
    return out


results["aba"] = {c: decide(c) for c in aba}
log("ABA", json.dumps(results["aba"]))
json.dump(results, open(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"),
                                     "b20_census.json"), "w"), default=str)
print(json.dumps({"passed": not fails, "failures": fails,
                  "aba": {c: {k: v for k, v in results["aba"][c].items() if k != "raw"}
                          for c in results["aba"]},
                  "census_summary": {c: {k: v for k, v in out_census[c].items()
                                         if k != "top_kernels"} for c in out_census}},
                 default=str))
