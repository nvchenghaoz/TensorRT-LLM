"""In-process FLUX.3 sync census + bitwise e2e check (idea 54, branch 42).

Loads the pipeline as trtllm-serve does (benchmark engine settings + the tree's
experiments/ascent-config/engine.json overlay), warmup included. Then, in ONE process:
  * arm "tree": the tree's pipeline_flux3 (sample_image_latents/_as_image_grid);
  * arm "main": module switch to verbatim main copies (from flux3_boundary_bench.py).
Runs requests along infer()'s path (CudaPhaseTimer pre/denoise/post), order:
  tree(P0,s100), tree(P0,s100), tree(P1,s101), main(P0,s100), main(P1,s101),
  then timed alternation tree/main x4 (census off) for pre_denoise/post_denoise.
Sync census: torch.cuda.set_sync_debug_mode("warn") in two windows, each warning logged
with its Python stack: (a) te_to_denoise = encode() return -> first denoise step;
(b) denoise_to_vae = denoise loop end -> vae.decode entry. vae.decode is a third window.
Checks: images bitwise tree==main per prompt/seed and request 1 == request 2, decoder
input z (values, strides) tree==main, x/x_ids/orig_ids tree==main, no dynamo
recompile after warmup.
Usage: python flux3_sync_census.py <checkpoint_root>
"""

import json
import os
import statistics
import sys
import time
import traceback
import warnings

import torch
import torch._dynamo.utils as dyn_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux3_boundary_bench as ref  # noqa: E402  (verbatim main copies; runs no bench on import? see guard)

import tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 as pf  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context  # noqa: E402
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer  # noqa: E402
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader  # noqa: E402
from tensorrt_llm.visual_gen.args import VisualGenArgs  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


root = sys.argv[1]
cfg = dict(
    model=root,
    pipeline_config={"variant": "nvfp4"},
    attention_config={"backend": "VANILLA"},
    compilation_config={"modes": ["t2i"], "num_frames": [1], "resolutions": [[1024, 1024]]},
    cuda_graph_config={"enable": False},
    torch_compile_config={"enable": True, "enable_autotune": True, "enable_fullgraph": False},
)
overlay = os.path.join(os.getcwd(), "experiments/ascent-config/engine.json")
if os.path.exists(overlay):
    ov = json.load(open(overlay))

    def merge(d, o):
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v

    merge(cfg, ov)
    log(f"engine.json overlay: {ov}")
t0 = time.time()
pipe = PipelineLoader(VisualGenArgs(**cfg)).load()
log(f"loaded+warmup in {time.time() - t0:.1f}s")
log(f"compute dtype {pipe.transformer.model_config.torch_dtype}, "
    f"vae param dtype {next(pipe.vae.parameters()).dtype}")


# ---- sync census -----------------------------------------------------------------
class Census:
    def __init__(self):
        self.enabled = False
        self.window = None
        self.events = []
        self._cw = None

    def _show(self, message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        stack = "".join(traceback.format_stack(limit=14)[:-2])
        self.events.append({"window": self.window, "msg": msg[:200], "stack": stack})
        print(f"[sync] window={self.window} {msg[:160]}\n{stack}", flush=True)

    def start(self, name):
        if not self.enabled:
            return
        self.stop()
        self.window = name
        self._cw = warnings.catch_warnings()
        self._cw.__enter__()
        warnings.simplefilter("always")
        warnings.showwarning = self._show
        torch.cuda.set_sync_debug_mode("warn")

    def stop(self):
        if self.window is None:
            return
        torch.cuda.set_sync_debug_mode(0)
        self._cw.__exit__(None, None, None)
        self.window = None


census = Census()
_orig_encode = pipe._encode_prompt
_orig_steps = pipe._profile_denoise_steps
_orig_vae_decode = pipe.vae.decode
captured = {}


def encode(prompt, neg):
    out = _orig_encode(prompt, neg)
    census.start("te_to_denoise")
    return out


def steps(timesteps):
    first = True
    for item in _orig_steps(timesteps):
        if first:
            census.stop()
            first = False
        yield item
    census.start("denoise_to_vae")


def vae_decode(z):
    census.stop()
    captured["z"] = z.detach().clone()
    captured["z_stride"] = tuple(z.stride())
    census.start("vae_decode")
    out = _orig_vae_decode(z)
    census.stop()
    return out


pipe._encode_prompt = encode
pipe._profile_denoise_steps = steps
pipe.vae.decode = vae_decode

TREE = (pf.sample_image_latents, pf._as_image_grid)
MAIN = (ref.ref_sample_image_latents, ref.ref_as_image_grid)


def use(arm):
    pf.sample_image_latents, pf._as_image_grid = TREE if arm == "tree" else MAIN


width, height = pipe.pipeline_config.extra_attrs["flux3_checkpoint"].config["sampler"]["size"]
PROMPTS = ["A red fox standing in fresh snow at sunrise",
           "A lighthouse on a cliff during a thunderstorm, oil painting"]


def request(arm, prompt, seed, census_on):
    use(arm)
    census.enabled = census_on
    n0 = len(census.events)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode():
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        ctx = pipe._encode_prompt(prompt, None)
        with autotune_context(pipe.device, pipe.rank, tune=False):
            latent = pipe._prepare_t2i(ctx=ctx, height=height, width=width, seed=seed,
                                       timer=timer)
        timer.mark_post_start()
        result = pipe._package_output(pipe._decode_latents(latent))
        timer.mark_end()
        timer.fill(result)
    census.stop()
    census.enabled = False
    wall = (time.perf_counter() - t) * 1000
    ev = census.events[n0:]
    rec = {"arm": arm, "seed": seed, "wall_ms": wall,
           "pre_denoise_ms": result.pre_denoise * 1000, "denoise_ms": result.denoise * 1000,
           "post_denoise_ms": result.post_denoise * 1000,
           "syncs": {w: sum(1 for e in ev if e["window"] == w)
                     for w in ("te_to_denoise", "denoise_to_vae", "vae_decode")},
           "orig_ids_device": str(latent[1].device)}
    log(f"request {json.dumps(rec)}")
    return result.image.clone(), captured["z"], captured["z_stride"], rec


before = {k: dict(v) for k, v in dyn_utils.counters.items() if k in ("frames", "stats", "recompiles")}
recs = []
img_t0, z_t0, zs_t0, r = request("tree", PROMPTS[0], 100, True); recs.append(r)
img_t1, z_t1, zs_t1, r = request("tree", PROMPTS[0], 100, True); recs.append(r)
img_t2, z_t2, zs_t2, r = request("tree", PROMPTS[1], 101, True); recs.append(r)
img_m0, z_m0, zs_m0, r = request("main", PROMPTS[0], 100, True); recs.append(r)
img_m2, z_m2, zs_m2, r = request("main", PROMPTS[1], 101, True); recs.append(r)
timed = {"tree": [], "main": []}
for i in range(8):
    arm = ("tree", "main")[(i + i // 2) % 2]
    _, _, _, r = request(arm, PROMPTS[i % 2], 100 + i % 2, False)
    timed[arm].append(r)
after = {k: dict(v) for k, v in dyn_utils.counters.items() if k in ("frames", "stats", "recompiles")}

# x / x_ids / orig_ids, tree vs main, on the real device
dev = next(pipe.transformer.parameters()).device
scale = pipe.pipeline_config.extra_attrs["flux3_checkpoint"].config["dit"]["image_rope_scale"]
inp_eq = {}
for seed in (100, 101):
    a = TREE[0](height=height, width=width, seed=seed, device=dev, image_rope_scale=scale,
                copies=2)
    b = MAIN[0](height=height, width=width, seed=seed, device=dev, image_rope_scale=scale,
                copies=2)
    inp_eq[seed] = {
        "x": a[0].stride() == b[0].stride() and torch.equal(a[0], b[0]),
        "x_ids": a[1].stride() == b[1].stride() and torch.equal(a[1], b[1]),
        "orig_ids": torch.equal(a[2].cpu(), b[2].cpu()) and a[2].dtype == b[2].dtype,
    }


def summ(arm, key):
    v = [r[key] for r in timed[arm]]
    return {"median": statistics.median(v), "values": v}


census_rows = {}
for r in recs:
    census_rows.setdefault(r["arm"], []).append(r["syncs"])
tree_clean = all(s["te_to_denoise"] == 0 and s["denoise_to_vae"] == 0
                 for s in census_rows["tree"])
res = {
    "image_tree_eq_main_seed100": bool(torch.equal(img_t0, img_m0)),
    "image_tree_eq_main_seed101": bool(torch.equal(img_t2, img_m2)),
    "image_repeat_req1_eq_req2": bool(torch.equal(img_t0, img_t1)),
    "z_tree_eq_main": bool(torch.equal(z_t0, z_m0) and torch.equal(z_t2, z_m2)),
    "z_strides_tree": zs_t0, "z_strides_main": zs_m0,
    "z_strides_equal": zs_t0 == zs_m0 and zs_t2 == zs_m2,
    "inputs_eq": inp_eq,
    "no_recompile_after_warmup": after == before,
    "counters_before": before, "counters_after": after,
}
passed = (res["image_tree_eq_main_seed100"] and res["image_tree_eq_main_seed101"]
          and res["image_repeat_req1_eq_req2"] and res["z_tree_eq_main"]
          and res["z_strides_equal"] and res["no_recompile_after_warmup"]
          and all(all(v.values()) for v in inp_eq.values()) and tree_clean)
sync_sites = {}
for e in census.events:
    key = (e["window"], e["stack"].strip().splitlines()[-2:][0].strip() if e["stack"] else "")
    sync_sites[str(key)] = sync_sites.get(str(key), 0) + 1
print(json.dumps({
    "passed": bool(passed),
    "compared": "one process, module switch: tree pipeline_flux3 vs verbatim main "
                "sample_image_latents/_as_image_grid; images bitwise (2 seeds, 2 consecutive "
                "requests), decoder input z values+strides, x/x_ids/orig_ids, dynamo counters; "
                "sync census windows te_to_denoise and denoise_to_vae",
    "sync_census_per_request": census_rows,
    "sync_sites": sync_sites,
    "tree_sync_free_both_windows": tree_clean,
    **res,
    "timed_pre_denoise_ms": {a: summ(a, "pre_denoise_ms") for a in timed},
    "timed_post_denoise_ms": {a: summ(a, "post_denoise_ms") for a in timed},
    "timed_wall_ms": {a: summ(a, "wall_ms") for a in timed},
}, default=str))
