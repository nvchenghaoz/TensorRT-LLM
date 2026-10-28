"""Idea 162 S3: full-model fresh-process A/B/A and in-model nsys of the NVFP4 tail pair.

Arms differ ONLY by env TRTLLM_FLUX3_SM120_PAIR_KERNEL fixed at process start:
  A = 0 : main's two-stream pair (down t18 on current, attn quant + attn_out t18 on the side stream)
  B = 1 : the one-launch LPT pair (attn quant, args kernel, one persistent 2-group CUTLASS GEMM)
Pipeline loaded as served: benchmark defaults + experiments/ascent-config/engine.json (FLASHINFER
fp8 attention, vae bf16, CUDA graphs on), torch_compile + autotune, warmup [[1024,1024]] t2i.

  aba <order e.g. ABABA> <warm> <timed> <ckpt> <outdir> : one fresh child per letter. Each child:
        warm requests, then timed requests of the same prompt/seed; CUDA-event denoise ms
        (VisualGen output.denoise), NVML power / SM clock sampled every 20 ms during the timed
        requests, sha256 of the image and final latent. GO only if every B beats both A
        neighbours beyond the A/A spread.
  nsys <A|B> <ckpt> <outdir> : child under nsys (--cuda-graph-trace=node, cudaProfilerApi range
        around one warm request), sqlite export, per-pair table: for each attention kernel in
        the 28 tail-pair blocks, the wall from the attention kernel's end to the start of the
        first kernel after it that is not part of the pair (quant / memset / args / GEMMs).
"""

import hashlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import types

PROMPT = "A red fox standing in fresh snow at sunrise, photorealistic"
SEED = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def log(msg):
    print(msg, flush=True)


def load_pipe(ckpt):
    import torch

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine_path = os.path.join(WT, "experiments", "ascent-config", "engine.json")
    engine = json.load(open(engine_path)) if os.path.exists(engine_path) else {}
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=ckpt,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"[load] {time.time() - t0:.0f}s engine={engine} pipeline_config={pipeline_config}")
    return pipe


class Gen:
    def __init__(self, pipe):
        self.pipe = pipe
        self.last_latent = None
        orig = pipe._decode_latents

        def decode(latent_result):
            self.last_latent = latent_result[0].detach().clone()
            return orig(latent_result)

        pipe._decode_latents = decode

    def __call__(self):
        import torch

        from tensorrt_llm.visual_gen.params import VisualGenParams

        params = VisualGenParams(height=1024, width=1024, seed=SEED, num_inference_steps=16)
        out = self.pipe.infer(types.SimpleNamespace(prompt=PROMPT, params=params))
        torch.cuda.synchronize()
        img = out.image.detach().cpu().contiguous()
        lat = self.last_latent.float().cpu().contiguous()
        return out.denoise * 1e3, hashlib.sha256(img.numpy().tobytes()).hexdigest()[:16], \
            hashlib.sha256(lat.numpy().tobytes()).hexdigest()[:16]


class Nvml:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # noqa: BLE001
            log(f"[nvml] unavailable: {exc}")
            self.nv = None

    def _run(self):
        while not self.stop.is_set():
            try:
                p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                c = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                self.samples.append((p, c))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)

    def __enter__(self):
        if self.nv is not None:
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        if self.nv is not None:
            self.t.join()

    def summary(self):
        if not self.samples:
            return None
        ps = sorted(s[0] for s in self.samples)
        cs = sorted(s[1] for s in self.samples)
        n = len(ps)
        return {"n": n, "power_w_median": ps[n // 2], "power_w_p10": ps[n // 10],
                "sm_mhz_median": cs[n // 2], "sm_mhz_p10": cs[n // 10], "sm_mhz_p90": cs[(9 * n) // 10]}


def child(arm, warm, timed, ckpt, outpath):
    pipe = load_pipe(ckpt)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    log(f"[arm {arm}] TRTLLM_FLUX3_SM120_PAIR_KERNEL={os.environ.get('TRTLLM_FLUX3_SM120_PAIR_KERNEL')} "
        f"lpt.enabled={lpt.enabled()} sched={lpt.sched_mode()}")
    gen = Gen(pipe)
    for i in range(warm):
        d, si, sl = gen()
        log(f"[arm {arm}] warm {i}: denoise {d:.2f} ms img {si} lat {sl}")
    ds, shas = [], set()
    with Nvml() as nv:
        for i in range(timed):
            d, si, sl = gen()
            ds.append(d)
            shas.add((si, sl))
            log(f"[arm {arm}] timed {i}: denoise {d:.2f} ms")
    res = {"arm": arm, "denoise_ms": ds, "median": statistics.median(ds),
           "mean": statistics.mean(ds), "image_latent_sha": sorted(shas),
           "repeat_bitwise": len(shas) == 1, "nvml": nv.summary()}
    with open(outpath, "w") as f:
        json.dump(res, f)
    log("RESULT " + json.dumps(res))


def run_child(arm, extra, outdir, tag, prefix=()):
    env = dict(os.environ, TRTLLM_FLUX3_SM120_PAIR_KERNEL="1" if arm == "B" else "0")
    logpath = os.path.join(outdir, f"log_{tag}.txt")
    cmd = [*prefix, sys.executable, __file__, *extra]
    log(f"[parent] {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with open(logpath, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True)
        for line in proc.stdout:
            f.write(line)
            if line.startswith(("[arm", "[load", "RESULT", "Traceback")) or "Error" in line:
                print(f"  {tag}| " + line.rstrip()[:300], flush=True)
        proc.wait()
    log(f"[parent] {tag}: rc={proc.returncode} in {time.time() - t0:.0f}s")
    return proc.returncode


def aba(order, warm, timed, ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []
    for i, arm in enumerate(order):
        out = os.path.join(outdir, f"arm{i}_{arm}.json")
        rc = run_child(arm, ["child", arm, str(warm), str(timed), ckpt, out], outdir, f"{i}_{arm}")
        if rc != 0:
            print(json.dumps({"passed": False, "failed_arm": i}))
            sys.exit(1)
        results.append(json.load(open(out)))
    meds = [r["median"] for r in results]
    a_meds = [m for r, m in zip(results, meds) if r["arm"] == "A"]
    verdict = []
    for i, r in enumerate(results):
        if r["arm"] != "B":
            continue
        nb = [results[j]["median"] for j in (i - 1, i + 1)
              if 0 <= j < len(results) and results[j]["arm"] == "A"]
        verdict.append({"pos": i, "B": r["median"], "A_neighbours": nb,
                        "delta_vs_mean_neighbours": r["median"] - statistics.mean(nb),
                        "beats_both": all(r["median"] < a for a in nb)})
    aa_spread = max(a_meds) - min(a_meds) if len(a_meds) > 1 else None
    shas = {tuple(map(tuple, r["image_latent_sha"])) for r in results}
    go = all(v["beats_both"] for v in verdict) and aa_spread is not None and all(
        -v["delta_vs_mean_neighbours"] > aa_spread for v in verdict)
    summary = {
        "order": order, "medians_ms": meds, "aa_spread_ms": aa_spread, "verdict": verdict,
        "GO": bool(go), "outputs_bitwise_across_arms": len(shas) == 1,
        "repeat_bitwise_each_arm": [r["repeat_bitwise"] for r in results],
        "nvml": [r["nvml"] for r in results],
    }
    log("[summary] " + json.dumps(summary))
    summary["passed"] = bool(summary["outputs_bitwise_across_arms"]
                             and all(summary["repeat_bitwise_each_arm"]))
    print(json.dumps(summary))


def nsys_child(arm, ckpt):
    import torch

    pipe = load_pipe(ckpt)
    gen = Gen(pipe)
    for i in range(2):
        d, _, _ = gen()
        log(f"[arm {arm}] warm {i}: denoise {d:.2f} ms")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    d, _, _ = gen()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log(f"[arm {arm}] profiled request denoise {d:.2f} ms")


PAIR_PARTS = ("quantize", "Quantize", "quant", "memset", "flux3_lpt_pair_args", "GemmUniversal",
              "cutlass", "Gemm", "gemm")


def is_attn(name):
    n = name.lower()
    return ("fmha" in n or "flash" in n or "attention" in n) and "gemm" not in n


def is_pair_part(name):
    return any(p in name for p in PAIR_PARTS)


def analyze(sqlite_path, outdir, arm):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    rows = []
    for start, end, name_id, stream in cur.execute(
            "SELECT start, end, demangledName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        rows.append((start, end, strings.get(name_id, "?"), stream, "k"))
    try:
        for start, end, stream in cur.execute(
                "SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            rows.append((start, end, "memset", stream, "m"))
    except sqlite3.Error:
        pass
    rows.sort()
    log(f"[nsys {arm}] {len(rows)} kernel/memset records")
    attn_idx = [i for i, r in enumerate(rows) if is_attn(r[2])]
    log(f"[nsys {arm}] attention kernels: {len(attn_idx)}")
    pairs = []
    for ai in attn_idx:
        a_end = rows[ai][1]
        parts = []
        succ = None
        gemm_seen = 0
        for r in rows[ai + 1:ai + 40]:
            if r[0] < a_end:
                continue  # overlapping other-stream work (e.g. gate_up) started before attn end
            if is_attn(r[2]):
                break
            if is_pair_part(r[2]) and succ is None:
                parts.append(r)
                if "Gemm" in r[2] or "gemm" in r[2] or "cutlass" in r[2]:
                    gemm_seen += 1
                continue
            succ = r
            break
        if gemm_seen == 0 or succ is None:
            continue
        pairs.append({
            "wall_us": (succ[0] - a_end) / 1e3,
            "gemm_span_us": (max(p[1] for p in parts) - a_end) / 1e3,
            "first_part": parts[0][2][:60],
            "first_part_start_us": (parts[0][0] - a_end) / 1e3,
            "parts": [(p[2][:50], round((p[0] - a_end) / 1e3, 1), round((p[1] - p[0]) / 1e3, 1))
                      for p in parts],
            "successor": succ[2][:60],
        })
    with open(os.path.join(outdir, f"nsys_pairs_{arm}.json"), "w") as f:
        json.dump(pairs, f, indent=1)
    walls = [p["wall_us"] for p in pairs]
    summ = {"arm": arm, "pairs_found": len(pairs)}
    if walls:
        summ.update({"wall_us_median": statistics.median(walls), "wall_us_mean": statistics.mean(walls),
                     "gemm_span_us_median": statistics.median(p["gemm_span_us"] for p in pairs)})
        for p in pairs[:3]:
            log(f"[nsys {arm}] example pair: {json.dumps(p)}")
    log("[nsys summary] " + json.dumps(summ))
    return summ, pairs


def analyze_fork(sqlite_path, arm, succ_prefix="triton_poi_fused_add_mul_view"):
    """Per tail pair: wall from the image/single FMHA kernel's end (the fork's only predecessor,
    fact 1205) to the start of the residual add after the join, and every kernel/memset in
    between (start offset and duration). Only windows that end in that residual add and contain a
    CUTLASS GEMM are kept (the 28 NVFP4 tail pairs per step)."""
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    rows = []
    for start, end, name_id in cur.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        rows.append((start, end, strings.get(name_id, "?")))
    try:
        for start, end in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            rows.append((start, end, "memset"))
    except sqlite3.Error:
        pass
    rows.sort()
    fmha = [i for i, r in enumerate(rows) if "fmha" in r[2].lower()]
    out = []
    for fi in fmha:
        f_end = rows[fi][1]
        inside = []
        succ = None
        for r in rows[fi + 1:fi + 60]:
            if r[1] <= f_end:
                continue
            if r[2].startswith(succ_prefix):
                succ = r
                break
            if "fmha" in r[2].lower():
                break
            inside.append(r)
        if succ is None or not any("cutlass" in r[2] for r in inside):
            continue
        gemms = [r for r in inside if "cutlass" in r[2] and r[0] >= f_end]
        out.append({
            "wall_us": (succ[0] - f_end) / 1e3,
            "first_gemm_start_us": (min(r[0] for r in gemms) - f_end) / 1e3 if gemms else None,
            "last_gemm_end_us": (max(r[1] for r in gemms) - f_end) / 1e3 if gemms else None,
            "parts": [(r[2][:48], round((r[0] - f_end) / 1e3, 1), round((r[1] - r[0]) / 1e3, 1))
                      for r in inside],
        })
    walls = sorted(p["wall_us"] for p in out)
    n = len(walls)
    summ = {"arm": arm, "pairs": n}
    if n:
        fg = sorted(p["first_gemm_start_us"] for p in out)
        lg = sorted(p["last_gemm_end_us"] for p in out)
        summ.update({"wall_us_median": walls[n // 2], "wall_us_mean": sum(walls) / n,
                     "wall_us_p10": walls[n // 10], "wall_us_p90": walls[(9 * n) // 10],
                     "first_gemm_start_us_median": fg[n // 2],
                     "last_gemm_end_us_median": lg[n // 2],
                     "examples": [out[0], out[n // 2]]})
        # wall split by the GEMM grid's M (single 4176 vs image 4096) is not in the sqlite rows
        # read here; the two populations show up as the bimodal p10/p90.
    log(f"[fork {arm}] " + json.dumps(summ))
    return summ


def nsys_parent(arm, ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    rep = os.path.join(outdir, f"pair_{arm}")
    prefix = ["nsys", "profile", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
              "--capture-range-end=stop", "-t", "cuda", "--force-overwrite=true", "-o", rep]
    rc = run_child(arm, ["nsys_child", arm, ckpt], outdir, f"nsys_{arm}", prefix=prefix)
    if rc != 0:
        print(json.dumps({"passed": False, "rc": rc}))
        sys.exit(1)
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o",
                    rep + ".sqlite", rep + ".nsys-rep"], check=True)
    summ, pairs = analyze(rep + ".sqlite", outdir, arm)
    fork = analyze_fork(rep + ".sqlite", arm)
    print(json.dumps({"passed": True, "summary": summ, "fork": fork}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "aba":
        aba(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif mode == "child":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif mode == "nsys":
        nsys_parent(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "nsys_child":
        nsys_child(sys.argv[2], sys.argv[3])
    elif mode == "analyze":
        analyze(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "fork":
        res = {arm: analyze_fork(path, arm) for arm, path in zip(sys.argv[2::2], sys.argv[3::2])}
        print(json.dumps({"passed": True, "fork": res}))
