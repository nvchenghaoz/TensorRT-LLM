"""Branch 3 (idea 12) end-to-end check at 2048x2048: TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=1 (branch)
vs 0 (main's path), one fresh process per arm, served engine.json, torch.compile + autotune, CUDA
graphs on, warmup [[2048, 2048]] t2i + i2i outside the compared requests.

Each child runs the SAME request sequence, alternating modes:
  t2i check00, i2i edit00 (ref check00), t2i check08, i2i edit04 (ref check04),
  t2i T2 (a longer prompt, second text bucket), t2i check00 (repeat), i2i edit00 (repeat),
  then with the CUDA-graph runner disabled (eager replay of the compiled blocks, route counters on):
  t2i check00, i2i edit00.
Per request: sha256 of the final latent (fp32) and of the uint8 image, Dynamo unique graphs and
CUDA-graph captures added. Parent: every request bitwise equal across arms; in each arm the
repeats equal the first run and the eager runs equal the graph runs; no compile/capture in
steady requests; routes: arm 1 runs pair_residual and no LPT pair on the NVFP4 tail blocks, arm 0
the reverse.
usage: python pipe_check.py <ckpt>        (parent)
       python pipe_check.py <ckpt> --child
"""

import hashlib
import json
import os
import subprocess
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
        if sha in name:
            data = open(os.path.join(STORE, name), "rb").read()
            assert hashlib.sha256(data).hexdigest() == sha, name
            return data
    for name in sorted(os.listdir(STORE)):
        path = os.path.join(STORE, name)
        if os.path.isfile(path):
            data = open(path, "rb").read()
            if hashlib.sha256(data).hexdigest() == sha:
                return data
    raise FileNotFoundError(key)


def child():
    import torch

    sys.path.insert(0, os.path.join(HERE, ".."))
    from te_common import prompt_for_bucket

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_tail_one_launch as pr
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import _load_image_pixels
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    counts = {"pair_residual": 0, "lpt_run": 0, "gate_modes": {}}
    o_pr, o_lpt = pr.run, lpt.run

    def c_pr(*a, **k):
        counts["pair_residual"] += 1
        gm = str(k.get("gate_mode"))
        counts["gate_modes"][gm] = counts["gate_modes"].get(gm, 0) + 1
        return o_pr(*a, **k)

    def c_lpt(*a, **k):
        counts["lpt_run"] += 1
        return o_lpt(*a, **k)

    pr.run = c_pr
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
    blocks = [n for n, m in pipe.transformer.named_modules()
              if getattr(m, "_sm120_tail_one_launch", False) and not n.endswith("_orig_mod")]
    log(f"[child] env={os.environ.get('TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH')} enabled={pr.enabled()} "
        f"load {time.time() - t0:.0f}s runner={runner is not None and runner.enabled} "
        f"fused_blocks={len(blocks)} {blocks[:2]}..{blocks[-2:]}")
    te = pipe.text_encoder
    refs = {k: _load_image_pixels(ref_bytes(k), device=pipe.device) for k in REFS}

    def ugraphs():
        from torch._dynamo.utils import counters as c

        return c["stats"].get("unique_graphs", 0)

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
        return ctx[0].shape[1], lat, im

    t2 = prompt_for_bucket(te, 2)
    seq = [
        ("t2i_check00", "t2i", "A red fox standing in fresh snow at sunrise", 100, None),
        ("i2i_edit00", "i2i", "Make it a winter night with falling snow", 400, "check00"),
        ("t2i_check08", "t2i", "A vintage travel poster that says VISIT MARS in bold letters", 108,
         None),
        ("i2i_edit04", "i2i", "Give the man a red knitted hat", 404, "check04"),
        ("t2i_T2", "t2i", t2, 22, None),
        ("t2i_check00_rep", "t2i", "A red fox standing in fresh snow at sunrise", 100, None),
        ("i2i_edit00_rep", "i2i", "Make it a winter night with falling snow", 400, "check00"),
    ]
    recs = []
    for label, mode, prompt, seed, ref in seq:
        g0 = ugraphs()
        c0 = len(runner.graphs) if runner is not None else 0
        t1 = time.time()
        T, lat, im = generate(mode, prompt, seed, ref)
        rec = {"label": label, "T": T, "lat": lat, "img": im, "new_graphs": ugraphs() - g0,
               "new_captures": (len(runner.graphs) if runner is not None else 0) - c0,
               "s": round(time.time() - t1, 1)}
        recs.append(rec)
        log("[req] " + json.dumps(rec))
    if runner is not None:
        runner.enabled = False
    eager_counts = {}
    for label, mode, prompt, seed, ref in (seq[0], seq[1]):
        counts["pair_residual"], counts["lpt_run"], counts["gate_modes"] = 0, 0, {}
        T, lat, im = generate(mode, prompt, seed, ref)
        rec = {"label": "eager_" + label, "T": T, "lat": lat, "img": im}
        recs.append(rec)
        eager_counts[mode] = json.loads(json.dumps(counts))
        log("[req] " + json.dumps(rec) + " counts " + json.dumps(counts))
    print("CHILD_JSON " + json.dumps({"recs": recs, "counts": eager_counts,
                                      "fused_blocks": blocks}), flush=True)


def parent():
    out = {}
    stop = threading.Event()

    def beat():
        t0 = time.time()
        while not stop.wait(60):
            log(f"[heartbeat] {time.time() - t0:.0f}s")

    threading.Thread(target=beat, daemon=True).start()
    for env in ("1", "0"):
        cmd = [sys.executable, __file__, CKPT, "--child"]
        e = dict(os.environ, TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=env)
        t0 = time.time()
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        cj = None
        tail = []
        for line in p.stdout:
            tail = (tail + [line.rstrip()[:300]])[-30:]
            if line.startswith("CHILD_JSON "):
                cj = json.loads(line[len("CHILD_JSON "):])
            elif line.startswith(("[child", "[req", "Traceback")) or "Error" in line \
                    or "one-launch block tail" in line:
                log(f"  env{env}| {line.rstrip()[:600]}")
        p.wait()
        log(f"[parent] env {env} rc={p.returncode} {time.time() - t0:.0f}s")
        if cj is None:
            log("\n".join(tail))
        out[env] = cj
    stop.set()
    fails = []
    on, off = out.get("1"), out.get("0")
    if not on or not off:
        print(json.dumps({"passed": False, "error": "child failed"}))
        return
    cmp = []
    for a, b in zip(on["recs"], off["recs"]):
        same = a["lat"] == b["lat"] and a["img"] == b["img"]
        cmp.append({"label": a["label"], "T": a["T"], "bitwise_on_vs_off": same})
        if not same:
            fails.append(f"{a['label']} differs between arms")
    for arm, r in (("1", on), ("0", off)):
        by = {x["label"]: x for x in r["recs"]}
        for x, y in (("t2i_check00", "t2i_check00_rep"), ("i2i_edit00", "i2i_edit00_rep"),
                     ("t2i_check00", "eager_t2i_check00"), ("i2i_edit00", "eager_i2i_edit00")):
            if (by[x]["lat"], by[x]["img"]) != (by[y]["lat"], by[y]["img"]):
                fails.append(f"env{arm}: {y} != {x}")
    # Compiles / captures: a request in a new text bucket (t2i_T2, T=160) recompiles on main too.
    # The branch must add none beyond main's on any request, and none at all in the warmed bucket.
    compile_info = []
    for a, b in zip(on["recs"], off["recs"]):
        if "new_graphs" not in a:
            continue
        ca = (a["new_graphs"], a["new_captures"])
        cb = (b["new_graphs"], b["new_captures"])
        compile_info.append({"label": a["label"], "T": a["T"], "env1": ca, "env0": cb})
        if ca[0] > cb[0] or ca[1] > cb[1]:
            fails.append(f"{a['label']}: branch compiles/captures {ca} > main {cb}")
        if a["T"] == on["recs"][0]["T"] and (ca[0] or ca[1]):
            fails.append(f"{a['label']}: compile/capture in the warmed text bucket")
    c1, c0 = on["counts"], off["counts"]
    n1 = len(on["fused_blocks"])
    if n1 == 0 or off["fused_blocks"]:
        fails.append(f"fused blocks env1={n1} env0={len(off['fused_blocks'])}")
    # Default bucket M >= 24576: only the i2i single blocks (28 per step at M=32848, per-token
    # gate, gate_mode 2) take the one-launch tail; t2i (16464/16384) and the i2i mode blocks
    # (16384) stay on main's route. lpt_run counts are informational (main's route may change).
    want1 = {"t2i": (0, {}), "i2i": (16 * 28, {"2": 16 * 28})}
    for mode in ("t2i", "i2i"):
        n_f, gm = want1[mode]
        if c1[mode]["pair_residual"] != n_f or c1[mode]["gate_modes"] != gm:
            fails.append(f"env1 {mode} routes {c1[mode]}")
        if c0[mode]["pair_residual"] != 0:
            fails.append(f"env0 {mode} routes {c0[mode]}")
    res = {"passed": not fails, "failures": fails,
           "compared": "TRTLLM_FLUX3_SM120_TAIL_ONE_LAUNCH=1 vs 0, fresh processes, identical "
                       "alternating t2i/i2i request sequences at 2048x2048; final latent (fp32) + "
                       "uint8 image sha256; repeats and eager (CUDA graphs off) vs graph runs",
           "requests": cmp, "routes_env1": c1, "routes_env0": c0, "fused_blocks": on["fused_blocks"],
           "compiles": compile_info,
           "recs_env1": on["recs"], "recs_env0": off["recs"]}
    print(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child()
    else:
        parent()
