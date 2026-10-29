"""Idea 184 STEP 4: fresh-process full-model A/B/A (method 729). Arms differ ONLY by env
TRTLLM_FLUX3_ATTN_NVFP4_OUT fixed at process start: A = '0' (main's path: BF16 O + attn_out quantize
+ LPT pair), B = '1' (FMHA nvfp4_out epilogue + LPT pair). Served engine.json, torch.compile +
autotune, CUDA graphs on. Each child: WARM requests, then TIMED requests of one prompt/seed;
CUDA-event denoise ms (VisualGen output.denoise), NVML SM clock, sha256 of image and final latent.
GO only if every B beats both A neighbours beyond the A/A spread.
usage: python aba.py <ckpt> [order=ABABA] [warm=3] [timed=8]
       python aba.py <ckpt> --child <arm> <warm> <timed> <out.json>
"""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import types

PROMPT = "A red fox standing in fresh snow at sunrise"
SEED = 100
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def log(m):
    print(m, flush=True)


def child(ckpt, arm, warm, timed, outpath):
    import torch

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs
    from tensorrt_llm.visual_gen.params import VisualGenParams

    engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=ckpt, pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"[arm {arm}] env={os.environ.get('TRTLLM_FLUX3_ATTN_NVFP4_OUT')} "
        f"enabled={fp.attn_nvfp4_out_enabled()}")
    last = {}
    orig = pipe._decode_latents

    def decode(lr):
        last["lat"] = lr[0].detach().clone()
        return orig(lr)

    pipe._decode_latents = decode

    def gen():
        params = VisualGenParams(height=1024, width=1024, seed=SEED, num_inference_steps=16)
        out = pipe.infer(types.SimpleNamespace(prompt=PROMPT, params=params))
        torch.cuda.synchronize()
        img = out.image.detach().cpu().contiguous()
        lat = last["lat"].float().cpu().contiguous()
        return (out.denoise * 1e3, hashlib.sha256(img.numpy().tobytes()).hexdigest()[:16],
                hashlib.sha256(lat.numpy().tobytes()).hexdigest()[:16])

    for i in range(warm):
        d, _, _ = gen()
        log(f"[arm {arm}] warm {i}: {d:.2f} ms")
    clocks, stop = [], threading.Event()
    try:
        import pynvml

        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        h = (pynvml.nvmlDeviceGetHandleByUUID(vis) if vis.startswith("GPU-")
             else pynvml.nvmlDeviceGetHandleByIndex(int(vis or 0)))

        def sampler():
            while not stop.is_set():
                try:
                    clocks.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.02)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
    except Exception:  # noqa: BLE001
        th = None
    ds, shas = [], set()
    for i in range(timed):
        d, si, sl = gen()
        ds.append(d)
        shas.add((si, sl))
        log(f"[arm {arm}] timed {i}: {d:.2f} ms")
    stop.set()
    if th is not None:
        th.join()
    cs = sorted(clocks)
    res = {"arm": arm, "denoise_ms": ds, "median": statistics.median(ds),
           "mean": statistics.mean(ds), "sha": sorted(shas), "repeat_bitwise": len(shas) == 1,
           "sm_mhz_median": cs[len(cs) // 2] if cs else None}
    json.dump(res, open(outpath, "w"))
    log("RESULT " + json.dumps(res))


def parent(ckpt, order, warm, timed):
    outdir = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "idea184_aba")
    os.makedirs(outdir, exist_ok=True)
    head = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    rc = subprocess.run(["git", "-C", WT, "merge-base", "--is-ancestor", "0b4693c4", "HEAD"]).returncode
    log(f"[head] {head} ancestry(0b4693c4) rc={rc}")
    results = []
    for i, arm in enumerate(order):
        out = os.path.join(outdir, f"arm{i}_{arm}.json")
        env = dict(os.environ, TRTLLM_FLUX3_ATTN_NVFP4_OUT="1" if arm == "B" else "0")
        t0 = time.time()
        p = subprocess.Popen([sys.executable, __file__, ckpt, "--child", arm, str(warm), str(timed),
                              out], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        for line in p.stdout:
            if line.startswith(("[arm", "RESULT", "Traceback")) or "Error" in line:
                log(f"  {i}{arm}| {line.rstrip()[:300]}")
        p.wait()
        log(f"[parent] arm {i}{arm} rc={p.returncode} {time.time() - t0:.0f}s")
        if p.returncode != 0:
            print(json.dumps({"passed": False, "failed_arm": i}))
            return
        results.append(json.load(open(out)))
    meds = [r["median"] for r in results]
    a_meds = [r["median"] for r in results if r["arm"] == "A"]
    aa = max(a_meds) - min(a_meds) if len(a_meds) > 1 else None
    verdict = []
    for i, r in enumerate(results):
        if r["arm"] != "B":
            continue
        nb = [results[j]["median"] for j in (i - 1, i + 1)
              if 0 <= j < len(results) and results[j]["arm"] == "A"]
        verdict.append({"pos": i, "B": r["median"], "A_neighbours": nb,
                        "delta_vs_mean_neighbours": r["median"] - statistics.mean(nb),
                        "beats_both": all(r["median"] < a for a in nb)})
    shas = {tuple(map(tuple, r["sha"])) for r in results}
    go = all(v["beats_both"] for v in verdict) and aa is not None and all(
        -v["delta_vs_mean_neighbours"] > aa for v in verdict)
    print(json.dumps({
        "passed": len(shas) == 1 and all(r["repeat_bitwise"] for r in results),
        "GO": bool(go), "order": order, "medians_ms": meds, "aa_spread_ms": aa,
        "verdict": verdict, "outputs_bitwise_across_arms": len(shas) == 1,
        "repeat_bitwise_each_arm": [r["repeat_bitwise"] for r in results],
        "sm_mhz": [r["sm_mhz_median"] for r in results], "head": head, "ancestry_rc": rc,
        "per_arm_denoise_ms": [r["denoise_ms"] for r in results],
    }))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[1], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    else:
        parent(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ABABA",
               int(sys.argv[3]) if len(sys.argv) > 3 else 3,
               int(sys.argv[4]) if len(sys.argv) > 4 else 8)
