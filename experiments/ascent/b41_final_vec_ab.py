"""Branch 41 / idea 91: final-layer adaLN on the compact vec row. One-process A/B.

The switch: Flux3Model._final_layer_vec (on = branch: [1,1,D] row for a single compact
segment; off = identity = main's expanded [1,S,D] vec). Both arms share Dynamo history in
ONE process: warm, then interleaved per request A(T=80) off, A on, B(T=160) off, B on,
A off, A on. Images + final latent compared off vs on at the same position, and A==A
within each arm. Then a CUPTI kernel census of one A request per arm (torch.profiler),
final_layer ms/request (CUDA events around final_layer.image), and interleaved denoise
timing with A/A. Engine settings: benchmark defaults + experiments/ascent-config/engine.json.

usage: python b41_final_vec_ab.py <checkpoint> <outdir>
"""
import collections
import json
import math
import os
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
    cfg.pipeline_config["vae_decode_dtype"] = "bfloat16"
    return cfg


_ORIG = None


def set_flag(on):
    global _ORIG
    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as tf

    if _ORIG is None:
        _ORIG = tf.Flux3Model._final_layer_vec
    tf.Flux3Model._final_layer_vec = _ORIG if on else (lambda self, vec, inputs, lengths: vec)


class Runner:
    def __init__(self, ckpt):
        import torch
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        t0 = time.time()
        self.pipe = PipelineLoader(make_args(), device="cuda:0").load(checkpoint_dir=ckpt)
        log(f"[load] pipeline loaded in {time.time() - t0:.0f}s")
        self.torch = torch
        self.last_latent = None
        orig_decode = self.pipe._decode_latents

        def decode(latent_result):
            self.last_latent = latent_result[0].detach().clone()
            return orig_decode(latent_result)

        self.pipe._decode_latents = decode
        tr = self.pipe.transformer
        fl = tr.final_layer["image"]
        self.fl_events = []
        self.vec_shapes = []

        def pre(mod, args):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.fl_events.append([e, None])
            self.vec_shapes.append(tuple(args[1].shape))

        def post(mod, args, out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.fl_events[-1][1] = e

        fl.register_forward_pre_hook(pre)
        fl.register_forward_hook(post)

    def gen(self, prompt, seed=1000):
        from tensorrt_llm.visual_gen.params import VisualGenParams

        params = VisualGenParams(height=1024, width=1024, seed=seed, num_inference_steps=16)
        req = types.SimpleNamespace(prompt=prompt, params=params)
        self.fl_events, self.vec_shapes = [], []
        t0 = time.time()
        out = self.pipe.infer(req)
        self.torch.cuda.synchronize()
        wall = time.time() - t0
        fl_ms = sum(a.elapsed_time(b) for a, b in self.fl_events)
        img = out.image.detach().cpu().clone()
        lat = self.last_latent.cpu().clone()
        return {"img": img, "lat": lat, "denoise": out.denoise * 1e3, "wall": wall * 1e3,
                "fl_ms": fl_ms, "fl_calls": len(self.fl_events),
                "vec_shape": self.vec_shapes[0] if self.vec_shapes else None}


def kernel_trace(runner, prompt):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        runner.gen(prompt)
    rows = []
    for e in prof.events():
        if e.device_type.name == "CUDA":
            us = getattr(e, "device_time", None)
            if us is None:
                us = e.cuda_time
            rows.append((e.name, float(us)))
    return rows


def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def run(ckpt, outdir):
    import torch

    os.makedirs(outdir, exist_ok=True)
    set_flag(False)
    r = Runner(ckpt)
    res = {}
    log(PHASE + "warm")
    for flag in (False, True, False, True):
        set_flag(flag)
        g = r.gen(PROMPT_A)
        log(f"[warm] flag={flag} denoise {g['denoise']:.1f} ms vec {g['vec_shape']}")

    log(PHASE + "aba")
    seqs = {"off": [], "on": []}
    for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
        for flag in (False, True):
            set_flag(flag)
            g = r.gen(p)
            tag = "on" if flag else "off"
            seqs[tag].append((name, g))
            log(f"[aba] {name} {tag} denoise {g['denoise']:.1f} ms final_layer {g['fl_ms']:.3f} ms "
                f"({g['fl_calls']} calls) vec {g['vec_shape']} img {tuple(g['img'].shape)}")
    cmp = []
    for i, ((n0, a), (_, b)) in enumerate(zip(seqs["off"], seqs["on"])):
        ia, ib = a["img"], b["img"]
        la, lb = a["lat"].float(), b["lat"].float()
        cmp.append({"pos": i, "prompt": n0,
                    "image_bitwise": bool(torch.equal(ia, ib)),
                    "latent_bitwise": bool(torch.equal(a["lat"], b["lat"])),
                    "latent_max_abs": float((la - lb).abs().max()),
                    "latent_max_rel": float(((la - lb).abs() / la.abs().clamp_min(1e-3)).max()),
                    "image_max_abs_px": float((ia.float() - ib.float()).abs().max()),
                    "image_psnr_db": round(psnr(ia, ib), 3)})
    within = {tag: {"A_first_eq_A_last_image": bool(torch.equal(s[0][1]["img"], s[2][1]["img"])),
                    "A_first_eq_A_last_latent": bool(torch.equal(s[0][1]["lat"], s[2][1]["lat"]))}
              for tag, s in seqs.items()}
    res["off_vs_on"] = cmp
    res["within_arm_repeat"] = within
    log("[cmp] " + json.dumps(cmp))
    log("[within] " + json.dumps(within))

    log(PHASE + "profile")
    census = {}
    for flag in (False, True):
        set_flag(flag)
        rows = kernel_trace(r, PROMPT_A)
        by = collections.OrderedDict()
        for name, us in rows:
            c, t = by.get(name, (0, 0.0))
            by[name] = (c + 1, t + us)
        census["on" if flag else "off"] = by
    delta = []
    for key in set(census["off"]) | set(census["on"]):
        c0, t0 = census["off"].get(key, (0, 0.0))
        c1, t1 = census["on"].get(key, (0, 0.0))
        if c0 != c1 or abs(t0 - t1) > 300:
            delta.append({"name": key[:160], "calls_off": c0, "calls_on": c1,
                          "us_off": round(t0), "us_on": round(t1),
                          "us_per_call_off": round(t0 / c0, 2) if c0 else None,
                          "us_per_call_on": round(t1 / c1, 2) if c1 else None})
    delta.sort(key=lambda d: -abs(d["us_off"] - d["us_on"]))
    res["census"] = {
        "kernels_per_request_off": sum(c for c, _ in census["off"].values()),
        "kernels_per_request_on": sum(c for c, _ in census["on"].values()),
        "kernel_us_off": round(sum(t for _, t in census["off"].values())),
        "kernel_us_on": round(sum(t for _, t in census["on"].values())),
        "delta_by_name": delta[:30],
    }
    log("[census] " + json.dumps(res["census"]))

    log(PHASE + "timing")
    pos = {0: [], 1: [], 2: [], 3: []}
    fl = {0: [], 1: [], 2: [], 3: []}
    order = [False, True, True, False]
    for rnd in range(6):
        for p, flag in enumerate(order):
            set_flag(flag)
            g = r.gen(PROMPT_A)
            pos[p].append(g["denoise"])
            fl[p].append(g["fl_ms"])
        log(f"[timing] round {rnd} off {pos[0][-1]:.2f}/{pos[3][-1]:.2f} on "
            f"{pos[1][-1]:.2f}/{pos[2][-1]:.2f} final_layer off {fl[0][-1]:.3f} on {fl[1][-1]:.3f}")

    def stats(x):
        xs = sorted(x)
        med = statistics.median(xs)
        p10 = xs[max(0, int(0.1 * (len(xs) - 1)))]
        p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
        return {"median": round(med, 3), "mean": round(statistics.mean(xs), 3),
                "noise": round((p90 - p10) / med, 5), "n": len(xs)}

    res["timing_denoise_ms"] = {
        "off": stats(pos[0] + pos[3]), "on": stats(pos[1] + pos[2]),
        "aa_off_pos0_minus_pos3_ms": round(statistics.median(pos[0]) - statistics.median(pos[3]), 3),
        "aa_on_pos1_minus_pos2_ms": round(statistics.median(pos[1]) - statistics.median(pos[2]), 3),
        "paired_off_minus_on_ms": [round((a + b - c - e) / 2, 3)
                                   for a, b, c, e in zip(pos[0], pos[3], pos[1], pos[2])],
    }
    res["final_layer_ms_per_request"] = {"off": stats(fl[0] + fl[3]), "on": stats(fl[1] + fl[2])}
    log("[timing] " + json.dumps(res["timing_denoise_ms"]))
    log("[final_layer] " + json.dumps(res["final_layer_ms_per_request"]))
    log("RESULT " + json.dumps(res))


def wrapper(ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, TORCH_LOGS="recompiles")
    phase = "load"
    counts = collections.OrderedDict(load=0)
    result = None
    with open(os.path.join(outdir, "log.txt"), "w") as f:
        proc = subprocess.Popen([sys.executable, __file__, "--inner", ckpt, outdir],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
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
                print(line.rstrip()[:600], flush=True)
        proc.wait()
    if proc.returncode != 0 or result is None:
        print(json.dumps({"passed": False, "rc": proc.returncode, "recompiles_by_phase": counts}))
        sys.exit(1)
    result["recompiles_by_phase"] = counts
    bit = all(c["image_bitwise"] and c["latent_bitwise"] for c in result["off_vs_on"])
    rep = all(v["A_first_eq_A_last_image"] and v["A_first_eq_A_last_latent"]
              for v in result["within_arm_repeat"].values())
    result["passed"] = bool(rep)
    result["repeated_execution_passed"] = bool(rep)
    result["output"] = {"class": "exact" if bit else "reordered",
                        "min_psnr_db_on_vs_off": min(c["image_psnr_db"] for c in result["off_vs_on"])}
    result["compared_against"] = ("switch-off (main's expanded vec) in the same process, interleaved "
                                  "A(T=80),B(T=160),A per request; images + final latent")
    print(json.dumps(result))


if __name__ == "__main__":
    if sys.argv[1] == "--inner":
        run(sys.argv[2], sys.argv[3])
    else:
        wrapper(sys.argv[1], sys.argv[2])
