"""Flux3Attention._attend single-segment fast path: STEP 0 census, bitwise A/B, recompiles, timing.

All comparisons run in ONE process (main has two start-up modes; never compare across restarts).
The switch is transformer_flux3._ATTEND_SINGLE_SEGMENT_FAST_PATH (False = today's general path).

usage:
  python ab_check.py ab <checkpoint> <outdir>          # one process: off/on bitwise, profile, timing
  python ab_check.py fixed <on|off> <checkpoint> <outdir>   # fresh process, flag fixed from load:
                                                            # recompiles in A,B,A,A + A==A bitwise
The wrapper re-runs itself in a child with TORCH_LOGS=recompiles and counts Dynamo
'Recompiling function' lines per phase.
"""
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time
import types

BASE = "A red fox standing in fresh snow at sunrise, photorealistic"
FILLER = (" The scene is rendered with careful attention to texture, light and atmosphere,"
          " with soft shadows, crisp details on fur and snow crystals, a pale pink sky,"
          " distant pine trees fading into mist, and a calm, quiet winter mood.")
PROMPT_A = BASE
PROMPT_B = BASE + FILLER * 2
PHASE = "=== PHASE "
ATTN_RE = re.compile(r"fmha|flash|sdpa|attention|attn_fwd|fort_native", re.I)


def log(msg):
    print(msg, flush=True)


def make_args():
    from tensorrt_llm import VisualGenArgs
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

    cfg = VisualGenArgs()
    att = AttentionConfig(
        backend="FLASHINFER",
        quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
    )
    cfg = cfg.model_copy(update={"attention_config": att})
    cfg.pipeline_config["variant"] = "nvfp4"
    return cfg


class Runner:
    def __init__(self, ckpt):
        import torch
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        t0 = time.time()
        args = make_args()
        self.pipe = PipelineLoader(args, device="cuda:0").load(checkpoint_dir=ckpt)
        log(f"[load] pipeline loaded in {time.time() - t0:.0f}s")
        self.torch = torch
        self.last_latent = None
        self.text_shapes = []
        orig_decode = self.pipe._decode_latents

        def decode(latent_result):
            self.last_latent = latent_result[0].detach().clone()
            return orig_decode(latent_result)

        self.pipe._decode_latents = decode
        tr = getattr(self.pipe, "transformer", None)
        if tr is not None:
            orig_fwd = tr.forward

            def fwd(*a, **kw):
                if self.text_shapes and self.text_shapes[-1] is None:
                    self.text_shapes[-1] = {k: tuple(v.shape) for k, v in kw.items()
                                            if "text" in k and hasattr(v, "shape")}
                return orig_fwd(*a, **kw)

            tr.forward = fwd

    def gen(self, prompt, seed=1000):
        from tensorrt_llm.visual_gen.params import VisualGenParams

        params = VisualGenParams(height=1024, width=1024, seed=seed, num_inference_steps=16)
        req = types.SimpleNamespace(prompt=prompt, params=params)
        self.text_shapes.append(None)
        t0 = time.time()
        out = self.pipe.infer(req)
        self.torch.cuda.synchronize()
        wall = time.time() - t0
        img = out.image.detach().cpu().clone()
        lat = self.last_latent.cpu().clone()
        return img, lat, out.denoise * 1e3, wall * 1e3, self.text_shapes[-1]


def set_flag(on):
    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

    tf._ATTEND_SINGLE_SEGMENT_FAST_PATH = bool(on)


def kernel_trace(runner, outpath, prompt):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        runner.gen(prompt)
    prof.export_chrome_trace(outpath)
    with open(outpath) as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    ks.sort(key=lambda e: e["ts"])
    return [(e["name"], float(e.get("dur", 0.0)), e.get("cat"),
             str(e.get("args", {}).get("grid", ""))) for e in ks]


def short(name, n=90):
    return name if len(name) <= n else name[:n] + "..."


def summarize_trace(ks):
    by = collections.OrderedDict()
    for name, dur, cat, grid in ks:
        key = short(name, 120) + " grid=" + grid
        c, t = by.get(key, (0, 0.0))
        by[key] = (c + 1, t + dur)
    return by


def neighborhoods(ks, max_calls=40, before=6, after=3):
    """Kernels around attention kernels (first max_calls), us/call."""
    idx = [i for i, k in enumerate(ks) if ATTN_RE.search(k[0])]
    out = []
    for i in idx[:max_calls]:
        lo, hi = max(0, i - before), min(len(ks), i + after + 1)
        out.append([{"rel": j - i, "name": short(ks[j][0]), "us": round(ks[j][1], 1),
                     "cat": ks[j][2], "grid": ks[j][3]} for j in range(lo, hi)])
    return idx, out


def run_ab(ckpt, outdir):
    import torch

    os.makedirs(outdir, exist_ok=True)
    set_flag(False)
    r = Runner(ckpt)
    res = {}
    log(PHASE + "warm_off")
    for _ in range(2):
        _, _, d, w, ts = r.gen(PROMPT_A)
        log(f"[warm_off] denoise {d:.1f} ms wall {w:.0f} ms text {ts}")

    seqs = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        set_flag(flag)
        if flag:
            log(PHASE + "warm_on")
            _, _, d, w, ts = r.gen(PROMPT_A)
            log(f"[warm_on] denoise {d:.1f} ms wall {w:.0f} ms text {ts}")
        log(PHASE + f"aba_{tag}")
        seq = []
        for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
            img, lat, d, w, ts = r.gen(p)
            seq.append((name, img, lat))
            log(f"[aba_{tag}] {name} denoise {d:.1f} ms text {ts} img {tuple(img.shape)} "
                f"lat {tuple(lat.shape)}")
        seqs[tag] = seq
    cmp = []
    for i, ((n0, i0, l0), (n1, i1, l1)) in enumerate(zip(seqs["off"], seqs["on"])):
        cmp.append({"pos": i, "prompt": n0, "image_bitwise": bool(torch.equal(i0, i1)),
                    "latent_bitwise": bool(torch.equal(l0, l1)),
                    "latent_max_abs": float((l0.float() - l1.float()).abs().max())})
    within = {tag: {"A_first_eq_A_last_image": bool(torch.equal(s[0][1], s[2][1])),
                    "A_first_eq_A_last_latent": bool(torch.equal(s[0][2], s[2][2])),
                    "A_ne_B_image": (not bool(torch.equal(s[0][1], s[1][1])))
                    if s[0][1].shape == s[1][1].shape else True}
              for tag, s in seqs.items()}
    res["bitwise_off_vs_on"] = cmp
    res["within_process_repeat"] = within
    log("[bitwise] " + json.dumps(cmp))
    log("[within] " + json.dumps(within))

    # STEP 0 / after: kernel census of one A request per flag (both already compiled).
    traces = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        set_flag(flag)
        log(PHASE + f"profile_{tag}")
        ks = kernel_trace(r, os.path.join(outdir, f"trace_{tag}.json"), PROMPT_A)
        traces[tag] = ks
        idx, nb = neighborhoods(ks)
        with open(os.path.join(outdir, f"neighborhoods_{tag}.json"), "w") as f:
            json.dump({"attn_kernel_count": len(idx), "first_calls": nb}, f, indent=1)
        with open(os.path.join(outdir, f"kernels_{tag}.tsv"), "w") as f:
            for k in ks:
                f.write(f"{k[2]}\t{k[1]:.2f}\t{k[3]}\t{k[0]}\n")
        log(f"[profile_{tag}] kernels/request {len(ks)} attention kernels {len(idx)} "
            f"total kernel us {sum(k[1] for k in ks):.0f}")
        for c in nb[:12]:
            log(f"[nb_{tag}] " + " | ".join(f"{e['rel']}:{e['name'][:60]}:{e['us']}us:{e['grid']}"
                                            for e in c))
    s_off, s_on = summarize_trace(traces["off"]), summarize_trace(traces["on"])
    delta = []
    for key in set(s_off) | set(s_on):
        c0, t0 = s_off.get(key, (0, 0.0))
        c1, t1 = s_on.get(key, (0, 0.0))
        if c0 != c1 or abs(t0 - t1) > 300:
            delta.append({"name": key, "calls_off": c0, "calls_on": c1,
                          "us_off": round(t0), "us_on": round(t1),
                          "us_per_call_off": round(t0 / c0, 2) if c0 else None})
    delta.sort(key=lambda d: -(abs(d["calls_off"] - d["calls_on"]) * 1000
                               + abs(d["us_off"] - d["us_on"])))
    res["census"] = {"kernels_per_request_off": len(traces["off"]),
                     "kernels_per_request_on": len(traces["on"]),
                     "kernel_us_off": round(sum(k[1] for k in traces["off"])),
                     "kernel_us_on": round(sum(k[1] for k in traces["on"])),
                     "delta_by_name": delta[:40]}
    log("[census] " + json.dumps(res["census"]))

    # Interleaved timing (CUDA-event denoise per request), A/A from same-arm positions.
    log(PHASE + "timing")
    pos = {0: [], 1: [], 2: [], 3: []}
    order = [False, True, True, False]
    for rnd in range(6):
        for p, flag in enumerate(order):
            set_flag(flag)
            _, _, d, w, _ = r.gen(PROMPT_A)
            pos[p].append(d)
        log(f"[timing] round {rnd} off {pos[0][-1]:.2f}/{pos[3][-1]:.2f} on "
            f"{pos[1][-1]:.2f}/{pos[2][-1]:.2f}")
    off = pos[0] + pos[3]
    on = pos[1] + pos[2]

    def stats(x):
        xs = sorted(x)
        med = statistics.median(xs)
        p10 = xs[max(0, int(0.1 * (len(xs) - 1)))]
        p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
        return {"median": round(med, 3), "mean": round(statistics.mean(xs), 3),
                "noise": round((p90 - p10) / med, 5), "n": len(xs)}

    res["timing_denoise_ms"] = {
        "off": stats(off), "on": stats(on),
        "aa_off_pos0_vs_pos3_ms": round(statistics.median(pos[0]) - statistics.median(pos[3]), 3),
        "aa_on_pos1_vs_pos2_ms": round(statistics.median(pos[1]) - statistics.median(pos[2]), 3),
        "paired_off_minus_on_ms": [round((a + b - c - e) / 2, 3)
                                   for a, b, c, e in zip(pos[0], pos[3], pos[1], pos[2])],
    }
    log("[timing] " + json.dumps(res["timing_denoise_ms"]))
    with open(os.path.join(outdir, "ab_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("RESULT " + json.dumps(res))


def run_fixed(flag, ckpt, outdir):
    import torch

    os.makedirs(outdir, exist_ok=True)
    set_flag(flag == "on")
    r = Runner(ckpt)
    log(PHASE + "first_request")
    _, _, d, w, ts = r.gen(PROMPT_A)
    log(f"[first] denoise {d:.1f} ms wall {w:.0f} text {ts}")
    log(PHASE + "timed_A_A")
    a1 = r.gen(PROMPT_A)
    a2 = r.gen(PROMPT_A)
    log(f"[timed] A denoise {a1[2]:.1f} / {a2[2]:.1f}")
    log(PHASE + "timed_B")
    b = r.gen(PROMPT_B)
    log(f"[timed] B denoise {b[2]:.1f} text {b[4]}")
    log(PHASE + "timed_A_after_B")
    a3 = r.gen(PROMPT_A)
    ok = bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[0], a3[0]) and torch.equal(a1[1], a3[1]))
    log("RESULT " + json.dumps({"flag": flag, "A_repeat_bitwise": ok,
                                 "denoise_ms": [a1[2], a2[2], b[2], a3[2]]}))


def wrapper(argv):
    outdir = argv[-1]
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, TORCH_LOGS="recompiles")
    logpath = os.path.join(outdir, f"log_{'_'.join(argv[:-2])}.txt")
    phase = "load"
    counts = collections.OrderedDict(load=0)
    result = None
    with open(logpath, "w") as f:
        proc = subprocess.Popen([sys.executable, __file__, "--inner", *argv], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True)
        for line in proc.stdout:
            f.write(line)
            if line.startswith(PHASE):
                phase = line[len(PHASE):].strip()
                counts.setdefault(phase, 0)
            if "Recompiling function" in line:
                counts[phase] = counts.get(phase, 0) + 1
                print(f"[recompile in {phase}] " + line.rstrip()[:240], flush=True)
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
            elif line.startswith(("[", "===", "Traceback")) or "Error" in line:
                print(line.rstrip()[:400], flush=True)
        proc.wait()
    if proc.returncode != 0 or result is None:
        print(json.dumps({"passed": False, "rc": proc.returncode, "recompiles_by_phase": counts}))
        sys.exit(1)
    result["recompiles_by_phase"] = counts
    if argv[0] == "ab":
        bit = all(c["image_bitwise"] and c["latent_bitwise"] for c in result["bitwise_off_vs_on"])
        rep = all(v["A_first_eq_A_last_image"] and v["A_first_eq_A_last_latent"]
                  for v in result["within_process_repeat"].values())
        result["passed"] = bool(bit and rep)
        result["compared_against"] = ("switch-off path (= today's general _attend path) in the same "
                                      "process, requests A(T=80), B(longer text), A; images + final latent")
        t = result["timing_denoise_ms"]
        result["batch"] = {"1": {"step_ms": t["on"]["median"], "noise_floor": t["on"]["noise"],
                                 "base_step_ms": t["off"]["median"], "base_noise": t["off"]["noise"]}}
        result["output"] = {"class": "exact"}
    else:
        result["passed"] = bool(result.get("A_repeat_bitwise"))
    print(json.dumps(result))


def main():
    if sys.argv[1] == "--inner":
        argv = sys.argv[2:]
        if argv[0] == "ab":
            run_ab(argv[1], argv[2])
        else:
            run_fixed(argv[1], argv[2], argv[3])
        return
    wrapper(sys.argv[1:])


if __name__ == "__main__":
    main()
