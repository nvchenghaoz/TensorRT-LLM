"""Idea 184 STEP 2 pipeline check: TRTLLM_FLUX3_ATTN_NVFP4_OUT=1 (branch) vs '0' (main's path),
one fresh process per arm, served engine.json, torch.compile + autotune, CUDA graphs on.

Each child runs the SAME request sequence (so main's known T=80 -> T=160 -> T=80 history effect,
if any, is identical in both arms):
  probe : A (check00, T=80), B (T=160 prompt), A again        -- interleave probe
  check : the 12 check prompts at their fixed seeds              -- benchmark prompts
  repeat: check00 again (graph-on)
  eager : check00 with the CUDA-graph runner disabled, with route counters (python bodies of the
          custom ops run on every call): FMHA nvfp4_out calls, the LPT prequant pair calls,
          nvfp4_quantize_prepared calls made from nvfp4_out_pair, lpt.run calls.
Per request: sha256 of the final latent (fp32 bytes) and of the uint8 image, Dynamo unique graphs
and CUDA-graph captures added (must be 0 after warmup). Parent: every request bitwise equal
across arms; routes = 512 nvfp4_out FMHA + 512 prequant LPT + 0 quantize in arm 1, 0/0/512 in arm 0.
usage: python pipe_check.py <ckpt>        (parent)
       python pipe_check.py <ckpt> --child
"""

import hashlib
import json
import os
import subprocess
import sys
import time

CKPT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def log(*a):
    print(*a, flush=True)


# The benchmark's 12 check prompts and seeds (experiments/ascent-benchmark/prompts.json "check").
CHECK_PROMPTS = [
    {"key": "check00", "seed": 100, "prompt": "A red fox standing in fresh snow at sunrise"},
    {"key": "check01", "seed": 101,
     "prompt": "A lighthouse on a cliff during a thunderstorm, oil painting"},
    {"key": "check02", "seed": 102, "prompt": "A bowl of ramen with a soft boiled egg, studio photo"},
    {"key": "check03", "seed": 103, "prompt": "A futuristic city skyline at night with flying cars"},
    {"key": "check04", "seed": 104,
     "prompt": "Portrait of an elderly fisherman with a white beard, dramatic lighting"},
    {"key": "check05", "seed": 105,
     "prompt": "A cozy reading nook with a window, plants and warm lamp light"},
    {"key": "check06", "seed": 106,
     "prompt": "A watercolor painting of a hot air balloon over green hills"},
    {"key": "check07", "seed": 107,
     "prompt": "A golden retriever puppy playing with a tennis ball on grass"},
    {"key": "check08", "seed": 108,
     "prompt": "A vintage travel poster that says VISIT MARS in bold letters"},
    {"key": "check09", "seed": 109, "prompt": "Macro photo of a dew drop on a green leaf"},
    {"key": "check10", "seed": 110,
     "prompt": "A medieval castle surrounded by autumn forest, aerial view"},
    {"key": "check11", "seed": 111, "prompt": "A robot chef cooking pancakes in a modern kitchen"},
]


def child():
    import torch

    sys.path.insert(0, os.path.join(HERE, ".."))
    from te_common import prompt_for_bucket

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    counts = {"nfo_fmha": 0, "prequant_pair": 0, "pair_quant": 0, "lpt_run": 0, "pair": 0}
    o_nfo, o_quant, o_lpt = fp._prequant_attention_nvfp4_out, fp.nvfp4_quantize_prepared, lpt.run

    def c_nfo(*a, **k):
        counts["nfo_fmha"] += 1
        return o_nfo(*a, **k)

    def c_quant(*a, **k):
        if sys._getframe(1).f_code.co_name == "nvfp4_out_pair":
            counts["pair_quant"] += 1
        return o_quant(*a, **k)

    def c_lpt(*a, **k):
        counts["lpt_run"] += 1
        caller = sys._getframe(1).f_code.co_name
        if caller == "nvfp4_out_pair_prequant":
            counts["prequant_pair"] += 1
        elif caller == "nvfp4_out_pair":
            counts["pair"] += 1
        return o_lpt(*a, **k)

    fp._prequant_attention_nvfp4_out = c_nfo
    fp.nvfp4_quantize_prepared = c_quant
    lpt.run = c_lpt

    engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    runner = pipe._cuda_graph_runners.get("transformer")
    log(f"[child] env={os.environ.get('TRTLLM_FLUX3_ATTN_NVFP4_OUT')} enabled="
        f"{fp.attn_nvfp4_out_enabled()} load {time.time() - t0:.0f}s runner="
        f"{runner is not None and runner.enabled}")
    blocks = [n for n, m in pipe.transformer.named_modules()
              if getattr(m, "_sm120_attn_nvfp4_out", False) and not n.endswith("_orig_mod")]
    te = pipe.text_encoder

    def ugraphs():
        from torch._dynamo.utils import counters as c

        return c["stats"].get("unique_graphs", 0)

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        lat = hashlib.sha256(latent.float().cpu().contiguous().numpy().tobytes()).hexdigest()[:20]
        im = hashlib.sha256(img.cpu().contiguous().numpy().tobytes()).hexdigest()[:20]
        return ctx[0].shape[1], lat, im

    check = CHECK_PROMPTS
    A = (check[0]["prompt"], check[0]["seed"])
    B = (prompt_for_bucket(te, 2), 22)
    seq = [("probe_A", *A), ("probe_B", *B), ("probe_A2", *A)]
    seq += [(c["key"], c["prompt"], c["seed"]) for c in check]
    seq += [("repeat_check00", *A)]
    recs = []
    for label, prompt, seed in seq:
        g0 = ugraphs()
        c0 = len(runner.graphs) if runner is not None else 0
        T, lat, im = generate(prompt, seed)
        rec = {"label": label, "T": T, "lat": lat, "img": im, "new_graphs": ugraphs() - g0,
               "new_captures": (len(runner.graphs) if runner is not None else 0) - c0}
        recs.append(rec)
        log("[req] " + json.dumps(rec))
    if runner is not None:
        runner.enabled = False
    for k in counts:
        counts[k] = 0
    T, lat, im = generate(*A)
    recs.append({"label": "eager_check00", "T": T, "lat": lat, "img": im})
    log("[req] " + json.dumps(recs[-1]) + " counts " + json.dumps(counts))
    print("CHILD_JSON " + json.dumps({"recs": recs, "counts": counts, "nfo_blocks": len(blocks),
                                      "nfo_block_names": blocks[:3] + blocks[-3:]}), flush=True)


def parent():
    out = {}
    head = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    rc = subprocess.run(["git", "-C", WT, "merge-base", "--is-ancestor", "0b4693c4", "HEAD"]).returncode
    log(f"[head] {head} ancestry(0b4693c4) rc={rc}")
    for env in ("1", "0"):
        cmd = [sys.executable, __file__, CKPT, "--child"]
        e = dict(os.environ, TRTLLM_FLUX3_ATTN_NVFP4_OUT=env)
        t0 = time.time()
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        cj = None
        for line in p.stdout:
            if line.startswith("CHILD_JSON "):
                cj = json.loads(line[len("CHILD_JSON "):])
            elif line.startswith(("[child", "[req", "Traceback")) or "Error" in line:
                log(f"  env{env}| {line.rstrip()[:400]}")
        p.wait()
        log(f"[parent] env {env} rc={p.returncode} {time.time() - t0:.0f}s")
        out[env] = cj
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
            fails.append(f"{a['label']} differs")
    for arm, r in (("1", on), ("0", off)):
        for rec in r["recs"][3:-1]:
            if rec.get("new_graphs") or rec.get("new_captures"):
                fails.append(f"env{arm} {rec['label']}: compile/capture in steady request")
    c1, c0 = on["counts"], off["counts"]
    if not (c1["nfo_fmha"] == 512 and c1["prequant_pair"] == 512 and c1["pair_quant"] == 0
            and c1["pair"] == 0):
        fails.append(f"env1 routes {c1}")
    if not (c0["nfo_fmha"] == 0 and c0["prequant_pair"] == 0 and c0["pair_quant"] == 512
            and c0["pair"] == 512):
        fails.append(f"env0 routes {c0}")
    rep_on = [r for r in on["recs"] if r["label"] in ("check00", "repeat_check00", "probe_A")]
    info = {
        "env1_probeA_vs_probeA2": (on["recs"][0]["lat"], on["recs"][2]["lat"]),
        "env0_probeA_vs_probeA2": (off["recs"][0]["lat"], off["recs"][2]["lat"]),
        "env1_check00_vs_repeat": [r["lat"] for r in rep_on],
    }
    res = {"passed": not fails, "failures": fails, "head": head, "ancestry_rc": rc,
           "compared": "env TRTLLM_FLUX3_ATTN_NVFP4_OUT=1 vs 0, fresh processes, identical "
                       "request sequences; final latent + uint8 image sha256",
           "requests": cmp, "routes_env1": c1, "routes_env0": c0,
           "nfo_blocks_env1": on["nfo_blocks"], "nfo_blocks_env0": off["nfo_blocks"],
           "nfo_block_names": on["nfo_block_names"], "history_info": info}
    print(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child()
    else:
        parent()
