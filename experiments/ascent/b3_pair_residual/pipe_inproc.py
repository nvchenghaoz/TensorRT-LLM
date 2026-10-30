"""Branch 3 (idea 12) end-to-end check at 2048x2048 in ONE process (served engine.json,
torch.compile + autotune, CUDA graphs, warmup [[2048, 2048]] t2i + i2i).

Why one process: main's own i2i output is not reproducible across fresh processes (job 314 vs
352, kill switch off: i2i latent hashes 5e7bef90.. vs 1c46569a.. on identical source; t2i stable),
so a two-process A/B cannot show bitwise equality for i2i. Here both routes run in the same
process, after the same compile / autotune:
  1. graph runs: t2i check00, i2i edit00, t2i check08, i2i edit04, t2i check00 (repeat),
     i2i edit00 (repeat), each compared with its first run;
  2. eager runs (CUDA-graph runner off), route ON: t2i check00, i2i edit00, i2i edit04, compared
     with the graph runs (graph == eager), with route counters;
  3. route OFF in place (block._sm120_tail_one_launch = False on every block, which is exactly
     what TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=0 sets at prepare; Dynamo recompiles the blocks to
     main's path), eager: the same three requests, bitwise equal to step 2, counters show 0
     one-launch calls;
  4. route ON again, eager i2i edit00: equal to step 2 (no state leak).
Last stdout line: JSON.
usage: python pipe_inproc.py <ckpt>
"""

import hashlib
import json
import os
import sys
import threading
import time

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STORE = ("/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/"
         "flux3-klein-nvfp4-2048/v1/inputs")
REFS = {
    "check00": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e",
    "check04": "c958aa887a532e00750102d72e79250a5018c078d825fef3e1bbbce30eebb3a6",
}


def log(*a):
    print(*a, flush=True)


def ref_bytes(key):
    sha = REFS[key]
    for name in sorted(os.listdir(STORE)):
        path = os.path.join(STORE, name)
        if os.path.isfile(path) and (sha in name):
            data = open(path, "rb").read()
            if hashlib.sha256(data).hexdigest() == sha:
                return data
    for name in sorted(os.listdir(STORE)):
        path = os.path.join(STORE, name)
        if os.path.isfile(path):
            data = open(path, "rb").read()
            if hashlib.sha256(data).hexdigest() == sha:
                return data
    raise FileNotFoundError(key)


def main():
    stop = threading.Event()

    def beat():
        t0 = time.time()
        while not stop.wait(60):
            log(f"[heartbeat] {time.time() - t0:.0f}s")

    threading.Thread(target=beat, daemon=True).start()

    import torch

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as tol
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    counts = {"one_launch": 0, "lpt_run": 0, "gate_modes": {}}
    o_tol, o_lpt = tol.run, lpt.run

    def c_tol(*a, **k):
        counts["one_launch"] += 1
        gm = str(k.get("gate_mode"))
        counts["gate_modes"][gm] = counts["gate_modes"].get(gm, 0) + 1
        return o_tol(*a, **k)

    def c_lpt(*a, **k):
        counts["lpt_run"] += 1
        return o_lpt(*a, **k)

    tol.run = c_tol
    lpt.run = c_lpt

    engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1],
                            "modes": ["t2i", "i2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    runner = pipe._cuda_graph_runners.get("transformer")
    flagged = [m for n, m in pipe.transformer.named_modules()
               if getattr(m, "_sm120_tail_one_launch", False)]
    names = [n for n, m in pipe.transformer.named_modules()
             if getattr(m, "_sm120_tail_one_launch", False) and not n.endswith("_orig_mod")]
    log(f"[load] {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
        f"one_launch_blocks={len(names)}")
    refs = {k: _load_image_pixels(ref_bytes(k), device=pipe.device) for k in REFS}

    def generate(mode, prompt, seed, ref=None):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            if mode == "t2i":
                latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                                num_inference_steps=16, seed=seed)
            else:
                latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[refs[ref]], height=2048,
                                                width=2048, num_inference_steps=16, seed=seed)
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        lat = hashlib.sha256(latent.float().cpu().contiguous().numpy().tobytes()).hexdigest()[:20]
        im = hashlib.sha256(img.cpu().contiguous().numpy().tobytes()).hexdigest()[:20]
        return lat + ":" + im

    R = {
        "t2i_check00": ("t2i", "A red fox standing in fresh snow at sunrise", 100, None),
        "i2i_edit00": ("i2i", "Make it a winter night with falling snow", 400, "check00"),
        "t2i_check08": ("t2i", "A vintage travel poster that says VISIT MARS in bold letters",
                        108, None),
        "i2i_edit04": ("i2i", "Give the man a red knitted hat", 404, "check04"),
    }
    fails = []
    res = {}

    def run(tag, key):
        h = generate(*R[key])
        res[f"{tag}:{key}"] = h
        log(f"[req] {tag} {key} {h}")
        return h

    # 1. graph runs + repeats
    for key in ("t2i_check00", "i2i_edit00", "t2i_check08", "i2i_edit04"):
        run("graph", key)
    for key in ("t2i_check00", "i2i_edit00"):
        if run("graph_rep", key) != res[f"graph:{key}"]:
            fails.append(f"graph repeat {key} differs")
    # 2. eager, route on
    if runner is not None:
        runner.enabled = False
    eager_keys = ("t2i_check00", "i2i_edit00", "i2i_edit04")
    route_counts = {}
    for key in eager_keys:
        counts.update(one_launch=0, lpt_run=0, gate_modes={})
        if run("eager_on", key) != res[f"graph:{key}"]:
            fails.append(f"eager (route on) {key} != graph run")
        route_counts[f"on:{key}"] = json.loads(json.dumps(counts))
    # 3. route off in place (what the kill switch sets at prepare)
    for m in flagged:
        m._sm120_tail_one_launch = False
    for key in eager_keys:
        counts.update(one_launch=0, lpt_run=0, gate_modes={})
        if run("eager_off", key) != res[f"eager_on:{key}"]:
            fails.append(f"route off vs on {key} differs")
        route_counts[f"off:{key}"] = json.loads(json.dumps(counts))
    # 4. route on again
    for m in flagged:
        m._sm120_tail_one_launch = True
    counts.update(one_launch=0, lpt_run=0, gate_modes={})
    if run("eager_on2", "i2i_edit00") != res["eager_on:i2i_edit00"]:
        fails.append("route on again: i2i_edit00 differs")
    route_counts["on2:i2i_edit00"] = json.loads(json.dumps(counts))

    for key in ("t2i_check00",):
        rc = route_counts[f"on:{key}"]
        if rc["one_launch"] != 0:
            fails.append(f"t2i took the one-launch route: {rc}")
    for key in ("i2i_edit00", "i2i_edit04"):
        rc = route_counts[f"on:{key}"]
        if rc["one_launch"] != 16 * 28 or rc["gate_modes"] != {"2": 16 * 28}:
            fails.append(f"i2i route counts {key}: {rc}")
        if route_counts[f"off:{key}"]["one_launch"] != 0:
            fails.append(f"route off still took one-launch calls: {route_counts[f'off:{key}']}")
    stop.set()
    print(json.dumps({
        "passed": not fails, "failures": fails, "hashes": res, "route_counts": route_counts,
        "one_launch_blocks": names,
        "compared": "one process: graph runs + repeats; eager vs graph; route on vs route off "
                    "(block._sm120_tail_one_launch, the kill switch's static flag) on the same "
                    "compiled / autotuned model, final latent fp32 + uint8 image sha256, t2i and "
                    "i2i at 2048x2048; route counters"}))


if __name__ == "__main__":
    main()
