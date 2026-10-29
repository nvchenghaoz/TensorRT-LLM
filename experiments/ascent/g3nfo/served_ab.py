"""Idea 6 served-pipeline A/B/A: G3 nvfp4_out vs main's 4-warp nvfp4_out inside ONE pipeline.

Cross-server image comparisons are confounded by per-process AutoTuner/inductor picks, so this
loads the FLUX.3 pipeline once in-process (PipelineLoader, the benchmark's BASE_SERVER args +
experiments/ascent-config/engine.json, 2048^2 t2i+i2i warmup with autotune) with CUDA graphs OFF
(so the module swap below takes effect every step; graphs only replay the same launches), and
runs at 2048x2048, 16 steps, fixed seeds:
  t2i  A1 (G3 nfo, this branch) -> B (4-warp nfo = main's served kernel) -> A2 (G3)
  i2i  (reference = t2i A1 PNG)  A1 -> B -> A2
  t2i  A3 (after i2i)
B swaps flux3_primsts.attn_nvfp4_out_module for the same lever set minus g3 (main's kernel: on
main the G3 guard sent every nvfp4_out call to that 4-warp kernel). Every request runs under
torch.profiler; the fmha kernel census per request shows which kernel each arm ran.
Pass: A1 == B == A2 (pixels) per mode, t2i A3 == A1, census as expected.
usage: python served_ab.py <checkpoint>
"""

import copy
import hashlib
import io
import json
import os
import socket
import sys
import time

import torch

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

BASE_VGA = {
    "pipeline_config": {"variant": "nvfp4"},
    "parallel_config": {"tp_size": 1},
    "attention_config": {"backend": "VANILLA"},
    "cuda_graph_config": {"enable": False},
    "torch_compile_config": {"enable": True, "enable_fullgraph": False, "enable_autotune": True},
    "compilation_config": {"resolutions": [[2048, 2048]], "num_frames": [1],
                           "modes": ["t2i", "i2i"]},
}
T2I = [("t2i_a", 100, "A red fox standing in fresh snow at sunrise"),
       ("t2i_b", 108, "A vintage travel poster that says VISIT MARS in bold letters")]
I2I = [("i2i_a", 200, "Make it a snowy winter night with falling snow")]


def log(*a):
    print(f"[served_ab {time.strftime('%H:%M:%S')}]", *a, flush=True)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    os.environ.update(TLLM_DISABLE_MPI="1", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port.getsockname()[1]), RANK="0", WORLD_SIZE="1",
                      LOCAL_RANK="0")
    port.close()
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", init_method="env://", world_size=1,
                            rank=0, device_id=torch.device("cuda:0"))
    from PIL import Image

    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as fx
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs
    from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams

    engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
    vga = deep_merge(BASE_VGA, engine.get("visual_gen_args", {}))
    vga = deep_merge(vga, {k: v for k, v in engine.items() if k not in ("visual_gen_args", "env")})
    vga["pipeline_config"]["variant"] = "nvfp4"
    vga["cuda_graph_config"] = {"enable": False}
    for k, v in (engine.get("env") or {}).items():
        os.environ.setdefault(str(k), str(v))
    vga["model"] = CKPT
    log("VGA", json.dumps(vga))
    t0 = time.time()
    pipe = PipelineLoader(VisualGenArgs(**vga), device="cuda:0").load(skip_warmup=False)
    log(f"loaded + warmup {time.time() - t0:.0f}s")

    orig_mod = fp.attn_nvfp4_out_module

    def mod_4w(key=None, value=None):
        levers = fx.ROUTED_LEVERS
        if key is not None and value is not None:
            levers = fx.load_trim_levers(levers, key, value)
        return fx.get_flux3_d128_module((*[x for x in levers if x != "g3"], "nvfp4_out"))

    def run(key, seed, prompt, ref_png=None):
        params = VisualGenParams(height=2048, width=2048, num_inference_steps=16, seed=seed,
                                 image_reference=([MediaRef(content=ref_png, format="bytes")]
                                                  if ref_png else None))
        for name, dv in pipe.default_generation_params.items():
            if hasattr(params, name) and getattr(params, name) is None:
                setattr(params, name, dv)
        specs = pipe.extra_param_specs or {}
        params.extra_params = {k: s.default for k, s in specs.items()}
        req = DiffusionRequest(request_id=0, prompt=[prompt], params=params)
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        t = time.time()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            pipe.prepare_request(req)
            out = pipe.run_inference(req)
            torch.cuda.synchronize()
        wall = time.time() - t
        cen = {}
        for e in prof.key_averages():
            if "fmha_v2_flux3" in e.key:
                tag = ("g3_" if e.key.endswith("_g3") else "4w_") + ("nfo" if "_nfo_" in e.key
                                                                    else "bf16")
                cen[tag] = cen.get(tag, 0) + e.count
        img = out.image
        img = img[0] if img.dim() == 4 else img
        arr = img.detach().cpu().numpy()
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        log(f"{key} wall {wall:.1f}s sha {sha[:16]} census {cen}")
        return arr, sha, cen

    res, failures = {"requests": []}, []

    def arm(tag, fn, *a):
        fp.attn_nvfp4_out_module = orig_mod if tag.startswith("A") else mod_4w
        arr, sha, cen = fn(*a)
        res["requests"].append({"arm": tag, "key": a[0], "sha256": sha, "census": cen})
        return arr, sha, cen

    import numpy as np

    out = {}
    for key, seed, prompt in T2I:
        out[key] = {t: arm(t, run, key, seed, prompt) for t in ("A1", "B", "A2")}
    buf = io.BytesIO()
    Image.fromarray(out["t2i_a"]["A1"][0]).save(buf, format="PNG")
    ref_png = buf.getvalue()
    for key, seed, prompt in I2I:
        out[key] = {t: arm(t, run, key, seed, prompt, ref_png) for t in ("A1", "B", "A2")}
    key, seed, prompt = T2I[0]
    out[key]["A3"] = arm("A3", run, key, seed, prompt)
    fp.attn_nvfp4_out_module = orig_mod

    cmp = {}
    for key, arms in out.items():
        a1 = arms["A1"]
        for t, v in arms.items():
            if t == "A1":
                continue
            eq = a1[0].shape == v[0].shape and bool((a1[0] == v[0]).all())
            cmp[f"{key}:{t}_vs_A1"] = {"bitwise": eq, "max_abs_px": int(
                np.abs(a1[0].astype(int) - v[0].astype(int)).max())}
            if not eq:
                failures.append(f"{key}:{t}")
        steps = 16
        a_cen, b_cen = arms["A1"][2], arms["B"][2]
        exp_nfo = 32 * steps
        if a_cen.get("g3_nfo", 0) != exp_nfo or a_cen.get("4w_nfo", 0):
            failures.append(f"{key}:census_A {a_cen}")
        if b_cen.get("4w_nfo", 0) != exp_nfo or b_cen.get("g3_nfo", 0):
            failures.append(f"{key}:census_B {b_cen}")
    res.update(passed=not failures, failures=failures, compare=cmp,
               note=("one in-process pipeline, CUDA graphs off; A = G3 nvfp4_out (branch), B = "
                     "4-warp nvfp4_out (main's served kernel); census = fmha_v2_flux3 kernels per "
                     "request from torch.profiler"))
    print(json.dumps(res), flush=True)
    try:
        pipe.cleanup()
    except Exception:  # noqa: BLE001
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
