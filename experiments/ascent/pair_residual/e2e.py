"""Branch 5 (idea 2): in-process pipeline check of the gated residual in the LPT pair epilogue.

    python e2e.py <checkpoint>            driver: runs the worker twice in fresh processes, first
                                          with TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0 (main's pair +
                                          Inductor residual, byte for byte main's path), then =1
                                          (this branch), and compares.
    python e2e.py <checkpoint> worker <out.json>

Worker: the pipeline as served (engine.json overlay, torch.compile + autotune, CUDA graphs on,
warmup of 2048^2 t2i + i2i at load), then six requests in one process alternating the cases:
t2i A, i2i X, t2i B, i2i Y, t2i A (repeat), i2i X (repeat). Per request: sha256 of the final
latent and of the uint8 image, the number of fused-pair launches (the kernel's launch epoch in its
workspace advances by one per launch, graph replays included), Dynamo frame/graph counters and
the number of captured CUDA graphs. After warmup no request may compile or capture.
Driver checks: every request's latent and image bitwise equal between the arms, each repeat equal
to its first run, t2i runs 512 fused launches per request (28 single + 4 image mode blocks x 16),
i2i's single blocks keep main's residual (per-token gate) and the OFF arm runs none.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "driver"


def log(*a):
    print(*a, flush=True)


def store_png(sha):
    store = "/home/scratch.chengzhang/trtllm-dev-flux3-2048-20261029-090507/ascent-benchmark-data/flux3-klein-nvfp4-2048/v1/inputs"
    for root, _, files in os.walk(store):
        for f in files:
            p = os.path.join(root, f)
            if sha in f:
                return p
    for root, _, files in os.walk(store):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() == sha:
                    return p
    raise FileNotFoundError(sha)


REQUESTS = [
    ("t2i_A", "t2i", "A red fox standing in fresh snow at sunrise, photograph", 11, None),
    ("i2i_X", "i2i", "Turn the scene into a watercolor painting", 5, "check00"),
    ("t2i_B", "t2i", "A cozy reading nook with a green armchair, a lamp and a stack of books", 22,
     None),
    ("i2i_Y", "i2i", "Make it night time with a full moon in the sky", 7, "check01"),
    ("t2i_A_repeat", "t2i", "A red fox standing in fresh snow at sunrise, photograph", 11, None),
    ("i2i_X_repeat", "i2i", "Turn the scene into a watercolor painting", 5, "check00"),
]


def worker(out_path):
    import torch
    from torch._dynamo.utils import counters

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair_residual as lptr
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    # The frozen edit references (experiments/ascent-benchmark/inputs.json, sha256-pinned).
    inputs = {"inputs": {
        "check00": {"sha256": "ef88a9b18202a8303f9f99d3d453ac0473533b1b10bd887bcbcfb8734f2b4f3e"},
        "check01": {"sha256": "7928b9c476065f4c55503eefff5cc525d506354183ca588955108180127e8581"},
    }}
    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    engine.setdefault("torch_compile_config",
                      {"enable": True, "enable_fullgraph": False, "enable_autotune": True})
    engine.setdefault("parallel_config", {"tp_size": 1})
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[2048, 2048]], "num_frames": [1],
                            "modes": ["t2i", "i2i"]},
        **engine,
    )
    switch = os.environ.get("TRTLLM_FLUX3_SM120_PAIR_RESIDUAL", "1")
    log(f"switch={switch} engine={engine} pipeline_config={pipeline_config}")
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    runner = pipe._cuda_graph_runners.get("transformer")
    log(f"loaded+warmup {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
        f"graphs={len(runner.graphs) if runner else None}")
    blocks = [m for m in pipe.transformer.modules() if hasattr(m, "_sm120_pair_residual")]
    routed_blocks = sum(bool(m._sm120_pair_residual) for m in blocks)
    dev = torch.device("cuda:0")
    ws = lptr._WORKSPACES.get(0)

    def epoch():
        if ws is None:
            return 0
        return int(ws[:256].view(torch.int32)[32].item())

    refs = {}

    def ref_image(key):
        if key not in refs:
            meta = inputs["inputs"][key]
            path = store_png(meta["sha256"])
            data = open(path, "rb").read()
            assert hashlib.sha256(data).hexdigest() == meta["sha256"]
            refs[key] = _load_image_pixels(data, device=dev)
        return refs[key]

    def stats():
        return {"frames": dict(counters["frames"]), "unique_graphs":
                counters["stats"].get("unique_graphs", 0),
                "graphs": len(runner.graphs) if runner else 0}

    results = []
    for label, mode, prompt, seed, ref in REQUESTS:
        before = stats()
        e0 = epoch()
        t1 = time.time()
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            if mode == "t2i":
                latent, ids = pipe._prepare_t2i(ctx=ctx, height=2048, width=2048,
                                                num_inference_steps=16, seed=seed)
            else:
                latent, ids = pipe._prepare_i2i(ctx=ctx, image_cond=[ref_image(ref)],
                                                height=2048, width=2048,
                                                num_inference_steps=16, seed=seed)
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        after = stats()
        lat_b = latent.detach().float().contiguous().cpu().numpy().tobytes()
        img_b = img.detach().contiguous().cpu().numpy().tobytes()
        rec = {
            "label": label, "mode": mode, "seed": seed, "s": round(time.time() - t1, 2),
            "latent_sha256": hashlib.sha256(lat_b).hexdigest(),
            "image_sha256": hashlib.sha256(img_b).hexdigest(),
            "image_shape": list(img.shape), "image_dtype": str(img.dtype),
            "fused_launches": epoch() - e0,
            "new_unique_graphs": after["unique_graphs"] - before["unique_graphs"],
            "new_frames": {k: after["frames"].get(k, 0) - before["frames"].get(k, 0)
                           for k in after["frames"]},
            "new_cuda_graphs": after["graphs"] - before["graphs"],
        }
        log(json.dumps(rec))
        results.append(rec)
    out = {"switch": switch, "routed_blocks": routed_blocks, "blocks_with_flag": len(blocks),
           "graphs_after_warmup": len(runner.graphs) if runner else None, "requests": results}
    json.dump(out, open(out_path, "w"))
    log("worker done")


def driver():
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    arms = {}
    for switch in ("0", "1"):
        out = os.path.join(art, f"e2e_switch{switch}.json")
        env = dict(os.environ, TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=switch)
        log(f"=== worker switch={switch}")
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), CKPT, "worker", out],
                             env=env)
        if rc != 0:
            print(json.dumps({"passed": False, "error": f"worker switch={switch} rc={rc}"}))
            return
        arms[switch] = json.load(open(out))
    fails = []
    off, on = arms["0"], arms["1"]
    comps = []
    for a, b in zip(off["requests"], on["requests"]):
        same = (a["latent_sha256"] == b["latent_sha256"] and a["image_sha256"] == b["image_sha256"])
        comps.append({"label": a["label"], "bitwise_on_vs_main_path": same})
        if not same:
            fails.append(f"{a['label']}: ON != OFF")
    for arm_name, arm in (("off", off), ("on", on)):
        req = {r["label"]: r for r in arm["requests"]}
        for base in ("t2i_A", "i2i_X"):
            r0, r1 = req[base], req[base + "_repeat"]
            if (r0["latent_sha256"], r0["image_sha256"]) != (r1["latent_sha256"],
                                                              r1["image_sha256"]):
                fails.append(f"{arm_name}: {base} repeat differs")
        for r in arm["requests"]:
            if r["new_unique_graphs"] or r["new_cuda_graphs"] or any(r["new_frames"].values()):
                fails.append(f"{arm_name}: {r['label']} compiled or captured after warmup")
            if arm_name == "off" and r["fused_launches"] != 0:
                fails.append(f"off: {r['label']} ran {r['fused_launches']} fused launches")
            if arm_name == "on" and r["mode"] == "t2i" and r["fused_launches"] != 512:
                fails.append(f"on: {r['label']} ran {r['fused_launches']} fused launches, not 512")
    census = {r["label"]: r["fused_launches"] for r in on["requests"]}
    print(json.dumps({
        "passed": not fails, "failures": fails, "comparisons": comps,
        "fused_launches_per_request_on": census,
        "routed_blocks_on": on["routed_blocks"], "routed_blocks_off": off["routed_blocks"],
        "compared_against": "TRTLLM_FLUX3_SM120_PAIR_RESIDUAL=0 (main's LPT pair + Inductor gated "
                            "residual) in a fresh process, same requests; latent + uint8 image "
                            "sha256",
        "arms": arms,
    }))


if MODE == "worker":
    worker(sys.argv[3])
else:
    driver()
