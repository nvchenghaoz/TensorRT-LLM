"""Branch 15 (idea 24) in-process served-pipeline checks (2048^2, 16 steps, served engine settings:
benchmark BASE_SERVER visual_gen_args + experiments/ascent-config/engine.json; torch.compile with
autotune, CUDA graphs on, warmup of t2i+i2i at 2048^2 inside load).

parent:  python e2e.py <checkpoint> [joint_table values, default "0,1"]
  For each value v, a fresh child under nsys (--cuda-graph-trace=node, cudaProfilerApi range)
  with TRTLLM_FLUX3_JOINT_TABLE=v (v=0 is main's i2i route: per-token joint() modulation):
   1. ROUTE: one i2i + one t2i request with CUDA-graph capture/replay bypassed
      (disallow_cuda_graph_capture), every single / mode block wrapped to log the route the
      fused adaLN+NVFP4 head takes and the first guard that fails (same checks as
      flux3_adaln_quant.adaln_nvfp4_input), FP8R single-block count, attention arm env.
   2. SEQUENCE: t2i A, i2i B, t2i A, i2i B (graphs on), each repeat compared bitwise to its first
      (latents and decoded pixels); Dynamo unique_graphs / recompile counters before and after.
   3. CENSUS: one replayed i2i request B inside cudaProfilerStart/Stop; the parent reduces the
      nsys cuda_gpu_trace to kernel counts/time and an ordered kernel window of one single block.
Last stdout line: {"passed": ..., "children": {v: {...}}}.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CKPT = sys.argv[1]
NSYS = "/usr/local/cuda/bin/nsys"
STORE = (
    "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/"
    "ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs"
)
PROMPT_T2I = "A red fox standing in fresh snow at sunrise, photorealistic"
PROMPT_I2I = "Make it a watercolor painting with soft pastel colors"


def log(*a):
    print(f"[b15-e2e {time.strftime('%H:%M:%S')}]", *a, flush=True)


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ----------------------------------------------------------------------------- child
def guard_reason(block, x, mod):
    """First failing condition of adaln_nvfp4_input (branch code), or 'fused'."""
    import torch

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_adaln_quant as aq

    t = getattr(block, "_orig_mod", block)
    if not aq.FLUX3_ADALN_NVFP4:
        return "FLUX3_ADALN_NVFP4 off"
    if not getattr(t, "_flux3_adaln_nvfp4", False):
        why = []
        if getattr(t, "_fuse_proj_out", True):
            why.append("_fuse_proj_out")
        ff = getattr(t, "ff", None)
        ad = getattr(getattr(ff, "gate_up_proj", None), "_flux3_primsts_nvfp4", None)
        if not isinstance(ad, aq.PreparedNVFP4):
            why.append(f"gate_up adapter={type(ad).__name__}")
        if getattr(getattr(t, "attn", None), "_flux3_primsts_nvfp4_qkv", None) is None:
            why.append("no _flux3_primsts_nvfp4_qkv")
        return "static _flux3_adaln_nvfp4=False (" + ",".join(why or ["?"]) + ")"
    d = x.shape[-1]
    if not (x.dim() == 3 and x.shape[0] == 1 and x.dtype == torch.bfloat16 and x.is_contiguous()):
        return "x layout"
    if len(mod) == 4:
        shift, scale, _, select = mod
        if select.dtype == torch.uint8:
            rows = shift.shape[1] if shift.dim() == 3 else 0
            if not (2 <= rows <= 255 and aq._row_ok(shift, rows, d) and aq._row_ok(scale, rows, d)):
                return f"u8 index: table rows {tuple(shift.shape)}"
            return f"fused (K={rows} table + u8 index)"
        if not (aq._row_ok(shift, 2, d) and aq._row_ok(scale, 2, d)):
            return f"bool select: _row_ok(shift,2) shift={tuple(shift.shape)}"
        return "fused (K=2 table + bool select)"
    shift, scale, _ = mod
    if not (aq._row_ok(shift, 1, d) and aq._row_ok(scale, 1, d)):
        return f"3-tuple: _row_ok(shift,1) failed, shift={tuple(shift.shape)} (per-token mod)"
    return "fused (row form)"


def child(jt):
    import torch

    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    log("child JOINT_TABLE env", os.environ.get("TRTLLM_FLUX3_JOINT_TABLE"),
        "module flag", getattr(tf, "FLUX3_JOINT_TABLE", None))
    args = {
        "pipeline_config": {"variant": "nvfp4"},
        "parallel_config": {"tp_size": 1},
        "attention_config": {"backend": "VANILLA"},
        "cuda_graph_config": {"enable": False},
        "torch_compile_config": {"enable": True, "enable_fullgraph": False,
                                 "enable_autotune": True},
        "compilation_config": {"resolutions": [[2048, 2048]], "num_frames": [1],
                               "modes": ["t2i", "i2i"]},
    }
    overlay = json.load(open(os.path.join(WT, "experiments/ascent-config/engine.json")))
    overlay.pop("env", None)
    overlay = overlay.get("visual_gen_args", overlay)
    deep_merge(args, overlay)
    args["pipeline_config"]["variant"] = "nvfp4"
    log("visual_gen_args", json.dumps(args))
    t0 = time.time()
    pipe = PipelineLoader(VisualGenArgs(model=CKPT, **args), device="cuda:0").load()
    log(f"loaded + warmed in {time.time() - t0:.0f} s")
    from torch._dynamo.utils import counters

    def dyn():
        return {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                "calls_captured": int(counters["stats"].get("calls_captured", 0)),
                "recompiles": int(sum(counters["recompiles"].values()))
                if "recompiles" in counters else 0}

    dyn_after_load = dyn()
    files = sorted(f for f in os.listdir(STORE) if f.endswith(".png"))
    ref_img = _load_image_pixels(os.path.join(STORE, files[0]), device=pipe.device)
    log("reference", files[0], tuple(ref_img.shape))
    ctx_t = pipe._encode_prompt(PROMPT_T2I, None)
    ctx_i = pipe._encode_prompt(PROMPT_I2I, None)

    def run(mode, seed):
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            if mode == "t2i":
                lat = pipe._prepare_t2i(ctx=ctx_t, height=2048, width=2048,
                                        num_inference_steps=16, seed=seed)
            else:
                lat = pipe._prepare_i2i(ctx=ctx_i, image_cond=[ref_img], height=2048,
                                        width=2048, num_inference_steps=16, seed=seed)
            img = pipe._decode_latents(lat)
        torch.cuda.synchronize()
        return lat[0].clone(), img.clone() if isinstance(img, torch.Tensor) else img

    # ---- 1. ROUTE (graphs bypassed, wrappers log the route per block) ----
    tr = pipe.transformer
    routes, wrapped = {}, []

    class Spy(torch.nn.Module):
        def __init__(self, inner, name):
            super().__init__()
            self.inner = inner
            self.name = name

        def forward(self, x, freqs, mod, *a, **k):
            t = getattr(self.inner, "_orig_mod", self.inner)
            key = (self.name, tuple(x.shape))
            if key not in routes:
                routes[key] = {
                    "block": self.name, "type": type(t).__name__, "M": int(x.shape[1]),
                    "route": guard_reason(self.inner, x, mod),
                    "fp8r_head": getattr(t, "_flux3_fp8r_adaln_head", None) is not None,
                    "mod": [f"{tuple(m.shape)}:{str(m.dtype).replace('torch.', '')}" for m in mod],
                    "tail_pair": bool(getattr(t, "_sm120_tail_pair", False)),
                    "side_gate_up": int(getattr(t, "_sm120_side_gate_up", 0) or 0),
                }
            return self.inner(x, freqs, mod, *a, **k)

    def wrap_list(lst, prefix):
        for i in range(len(lst)):
            wrapped.append((lst, i, lst[i]))
            lst[i] = Spy(lst[i], f"{prefix}.{i}")

    wrap_list(tr.single_blocks, "single_blocks")
    wrap_list(tr.txt_mode_blocks, "txt_mode_blocks")
    for key, lst in tr.content_mode_blocks.items():
        wrap_list(lst, f"content_mode_blocks.{key}")
    fp8r_single = sum(
        isinstance(getattr(b, "_orig_mod", b), tf._Flux3FP8RSingleStreamBlock)
        for _, _, b in wrapped if True
    )
    runners = list(getattr(pipe, "_cuda_graph_runners", {}).values())
    prev_enabled = [r.enabled for r in runners]
    for r in runners:
        r.enabled = False  # eager forward (no replay), so the block wrappers run
    try:
        run("i2i", 7)
        i2i_routes = dict(routes)
        routes.clear()
        run("t2i", 7)
        t2i_routes = dict(routes)
    finally:
        for r, e in zip(runners, prev_enabled):
            r.enabled = e
    for lst, i, orig in wrapped:
        lst[i] = orig
    dyn_after_route = dyn()

    def compact(rs):
        out = {}
        for r in rs.values():
            k = f"{r['route']} | {r['type']} M={r['M']} tail_pair={r['tail_pair']} side={r['side_gate_up']}"
            out.setdefault(k, []).append(r["block"])
        return {k: (len(v), v[0] + ".." + v[-1] if len(v) > 1 else v[0]) for k, v in out.items()}

    route_info = {
        "i2i": compact(i2i_routes), "t2i": compact(t2i_routes),
        "i2i_single_mod": next((r["mod"] for r in i2i_routes.values()
                                if r["block"].startswith("single")), None),
        "fp8r_single_blocks": int(fp8r_single),
        "attention_env": {k: os.environ.get(k) for k in (
            "TRTLLM_FLUX3_ATTN_NVFP4_OUT", "TRTLLM_FLUX3_SIDE_GATE_UP")},
    }
    log("ROUTES", json.dumps(route_info, indent=1))

    # ---- 2. SEQUENCE t2i A, i2i B, t2i A, i2i B (graphs on) ----
    dyn_before_seq = dyn()
    seq = []
    for mode in ("t2i", "i2i", "t2i", "i2i"):
        t = time.time()
        lat, img = run(mode, 11 if mode == "t2i" else 23)
        seq.append((mode, lat, img))
        log(f"seq {mode} {(time.time() - t) * 1000:.0f} ms lat_sha "
            f"{hashlib.sha256(lat.float().cpu().numpy().tobytes()).hexdigest()[:16]}")
    rep = {}
    for mode in ("t2i", "i2i"):
        a, b = [s for s in seq if s[0] == mode]
        rep[mode] = {"latent_bitwise": bool(torch.equal(a[1], b[1])),
                     "image_bitwise": bool(torch.equal(a[2], b[2]))
                     if isinstance(a[2], torch.Tensor) else None}
    dyn_after_seq = dyn()
    log("REPEAT", json.dumps(rep), "dynamo", dyn_after_load, dyn_after_route, dyn_before_seq,
        dyn_after_seq)

    # ---- 3. CENSUS: one replayed i2i request ----
    run("i2i", 23)
    torch.cuda.profiler.start()
    t = time.time()
    lat, img = run("i2i", 23)
    torch.cuda.profiler.stop()
    census_ms = (time.time() - t) * 1000
    rep["i2i_census_vs_first"] = bool(torch.equal(lat, seq[1][1]))
    out = {"routes": route_info, "repeat": rep, "dynamo": {
        "after_load": dyn_after_load, "after_route": dyn_after_route,
        "before_seq": dyn_before_seq, "after_seq": dyn_after_seq},
        "census_request_ms_under_nsys": census_ms}
    print("CHILD_JSON " + json.dumps(out), flush=True)


# ----------------------------------------------------------------------------- parent
def short(name):
    for pre in ("void ", "cutlass::"):
        if name.startswith(pre):
            name = name[len(pre):]
    return name[:90]


def reduce(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Name", "")
            if not name or r.get("Duration (ns)") in (None, ""):
                continue
            if r.get("Grid X") in (None, "") and "memcpy" in name.lower():
                continue
            rows.append((int(float(r["Start (ns)"])), int(float(r["Duration (ns)"])), short(name),
                         r.get("Grid X"), r.get("Grid Y"), r.get("Grid Z")))
    rows.sort()
    agg = {}
    for _, dur, name, *_ in rows:
        a = agg.setdefault(name, [0, 0])
        a[0] += 1
        a[1] += dur
    table = sorted(([n, c, round(t / 1e6, 3)] for n, (c, t) in agg.items()), key=lambda x: -x[2])
    # ordered window: kernels of one mid-request single block (between consecutive FMHA calls
    # at the single-block count position), as names + grids
    fmha_idx = [i for i, r in enumerate(rows) if "fmha" in r[2].lower() or "flash" in r[2].lower()
                and "attn" in r[2].lower()]
    window = []
    if len(fmha_idx) > 40:
        mid = len(fmha_idx) // 2
        lo, hi = fmha_idx[mid] - 25, fmha_idx[mid + 1] + 5
        window = [f"{r[2][:70]} grid=({r[3]},{r[4]},{r[5]}) {r[1] / 1e3:.1f}us"
                  for r in rows[max(0, lo):hi]]
    span = (rows[-1][0] + rows[-1][1] - rows[0][0]) / 1e6 if rows else None
    return {"kernels": len(rows), "gpu_span_ms": span,
            "gpu_busy_ms": round(sum(r[1] for r in rows) / 1e6, 3),
            "table": table[:60], "window": window}


def parent(values):
    tmp = os.environ.get("AUTO_DEV_ARTIFACTS") or tempfile.mkdtemp()
    res = {"children": {}}
    for v in values:
        rep = os.path.join(tmp, f"b15_jt{v}")
        env = dict(os.environ, TRTLLM_FLUX3_JOINT_TABLE=v)
        cmd = [NSYS, "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, __file__, CKPT, "--child", v]
        log(f"=== JOINT_TABLE={v}: " + " ".join(cmd))
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        child_json = None
        tail = []
        for line in p.stdout:
            if line.startswith("CHILD_JSON "):
                child_json = json.loads(line[len("CHILD_JSON "):])
                continue
            tail.append(line)
            tail = tail[-200:]
            if line.startswith("[b15-e2e") or "Error" in line or "error" in line[:80]:
                print(line, end="", flush=True)
        p.wait()
        if child_json is None:
            log("child failed rc", p.returncode, "".join(tail[-60:]))
        s = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_trace", "-f", "csv", "-o", rep,
                            "--force-export=true", rep + ".nsys-rep"],
                           capture_output=True, text=True)
        csv_path = rep + "_cuda_gpu_trace.csv"
        cen = reduce(csv_path) if os.path.exists(csv_path) else {"error": s.stderr[-800:]}
        with open(os.path.join(tmp, f"b15_jt{v}_census.json"), "w") as f:
            json.dump(cen, f, indent=1)
        res["children"][v] = {"child": child_json, "census": {
            k: cen.get(k) for k in ("kernels", "gpu_span_ms", "gpu_busy_ms")},
            "top": (cen.get("table") or [])[:40], "window": cen.get("window")}
        log(f"JOINT_TABLE={v}", json.dumps(res["children"][v], indent=1))
    ok = True
    for v, c in res["children"].items():
        ch = c.get("child")
        if not ch:
            ok = False
            continue
        r = ch["repeat"]
        ok &= all(r[m]["latent_bitwise"] and r[m]["image_bitwise"] is not False
                  for m in ("t2i", "i2i")) and r.get("i2i_census_vs_first", False)
        d = ch["dynamo"]
        ok &= d["after_seq"]["unique_graphs"] == d["before_seq"]["unique_graphs"]
    res["passed"] = bool(ok)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent((sys.argv[2] if len(sys.argv) > 2 else "0,1").split(","))
