"""Branch 34 (idea 78) end-to-end: planar SM120 QKV switch off/on in ONE process.

  ab <ckpt> <outdir>   : for each arm (off = main's packed layout, then on = planar):
                         torch._dynamo.reset(), set the switch (module flag + the prepare-time
                         per-attention flag), warm with A, then A(T=80), B(T=160), A. Compare
                         images + final latent of each request with the switch-off request at
                         the SAME position (never A2 vs A1). Then one torch.profiler kernel
                         census of a steady A request per arm (direct_copy count, kernels per
                         request, qknorm_rope us/call), and which _sm120_cutlass_qkv branch runs.
  fixed <on|off> <ckpt> <outdir> : fresh process, switch fixed from load: first request, A, A, B,
                         A; Dynamo recompile count per phase (TORCH_LOGS=recompiles) and A==A.
The wrapper re-runs itself in a child with TORCH_LOGS=recompiles and counts 'Recompiling function'
lines per phase.
"""
import collections
import json
import os
import re
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
    return cfg


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

    def attns(self):
        return [m for m in self.pipe.transformer.modules()
                if getattr(m, "_flux3_primsts_nvfp4_qkv", None) is not None]

    def gen(self, prompt, seed=1000):
        from tensorrt_llm.visual_gen.params import VisualGenParams

        params = VisualGenParams(height=1024, width=1024, seed=seed, num_inference_steps=16)
        req = types.SimpleNamespace(prompt=prompt, params=params)
        t0 = time.time()
        out = self.pipe.infer(req)
        self.torch.cuda.synchronize()
        wall = time.time() - t0
        return out.image.detach().cpu().clone(), self.last_latent.cpu().clone(), \
            out.denoise * 1e3, wall * 1e3


def set_flag(runner, on, reset=True):
    import torch
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    fp.SM120_QKV_PLANAR = bool(on)
    n = 0
    if runner is not None:
        for m in runner.attns():
            ad = m._flux3_primsts_nvfp4_qkv
            m._flux3_primsts_nvfp4_qkv_planar = fp.sm120_qkv_planar(
                ad.weight.device, ad.weight.shape[0], m.norm_q.weight.numel(),
                m.norm_k.weight.numel())
            n += int(m._flux3_primsts_nvfp4_qkv_planar)
    if reset:
        torch._dynamo.reset()
    return n


def kernel_trace(runner, outpath):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        runner.gen(PROMPT_A)
    prof.export_chrome_trace(outpath)
    with open(outpath) as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    return [(e["name"], float(e.get("dur", 0.0)), e.get("cat"),
             str(e.get("args", {}).get("grid", ""))) for e in ks]


def census(ks):
    def agg(pred):
        sel = [k for k in ks if pred(k)]
        return {"calls": len(sel), "us": round(sum(k[1] for k in sel), 1),
                "us_per_call": round(sum(k[1] for k in sel) / len(sel), 2) if sel else None}

    by = collections.Counter()
    t = collections.Counter()
    for name, dur, _, grid in ks:
        key = name[:100] + " grid=" + grid
        by[key] += 1
        t[key] += dur
    return {
        "kernels_per_request": len(ks),
        "kernel_us": round(sum(k[1] for k in ks)),
        "direct_copy": agg(lambda k: "direct_copy" in k[0]),
        "elementwise_kernel": agg(lambda k: "elementwise_kernel" in k[0]),
        "qknorm_rope": agg(lambda k: re.search("qknorm", k[0], re.I) is not None),
        "top": [(k, by[k], round(t[k])) for k, _ in t.most_common(25)],
    }


def run_ab(ckpt, outdir):
    import torch
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    os.makedirs(outdir, exist_ok=True)
    fp.SM120_QKV_PLANAR = False
    r = Runner(ckpt)
    attns = r.attns()
    per_plane = sum(int(m._flux3_primsts_nvfp4_qkv.qkv_scale is not None) for m in attns)
    info = {"nvfp4_qkv_attentions": len(attns), "per_plane_alpha": per_plane,
            "one_alpha": len(attns) - per_plane, "backend": fp._sm120_qkv_backend()}
    log("[info] " + json.dumps(info))
    res = {"info": info}
    seqs, cens = {}, {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        n = set_flag(r, flag)
        log(PHASE + f"warm_{tag}")
        log(f"[flag] {tag}: planar attentions {n}")
        for _ in range(2):
            _, _, d, w = r.gen(PROMPT_A)
            log(f"[warm_{tag}] denoise {d:.1f} ms wall {w:.0f} ms")
        log(PHASE + f"aba_{tag}")
        seq = []
        for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
            img, lat, d, w = r.gen(p)
            seq.append((name, img, lat))
            log(f"[aba_{tag}] {name} denoise {d:.1f} ms wall {w:.0f} ms")
        seqs[tag] = seq
        log(PHASE + f"profile_{tag}")
        ks = kernel_trace(r, os.path.join(outdir, f"trace_{tag}.json"))
        cens[tag] = census(ks)
        log(f"[census_{tag}] " + json.dumps({k: v for k, v in cens[tag].items() if k != "top"}))
        for row in cens[tag]["top"][:12]:
            log(f"[top_{tag}] {row}")
    cmp = []
    for i, ((n0, i0, l0), (_, i1, l1)) in enumerate(zip(seqs["off"], seqs["on"])):
        cmp.append({"pos": i, "prompt": n0, "image_bitwise": bool(torch.equal(i0, i1)),
                    "latent_bitwise": bool(torch.equal(l0, l1)),
                    "latent_max_abs": float((l0.float() - l1.float()).abs().max())})
    res["bitwise_off_vs_on_same_position"] = cmp
    res["census"] = cens
    log("[bitwise] " + json.dumps(cmp))
    with open(os.path.join(outdir, "ab_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("RESULT " + json.dumps(res))


def run_fixed(flag, ckpt, outdir):
    import torch
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    os.makedirs(outdir, exist_ok=True)
    fp.SM120_QKV_PLANAR = flag == "on"
    r = Runner(ckpt)
    n = sum(int(m._flux3_primsts_nvfp4_qkv_planar) for m in r.attns())
    log(f"[flag] {flag}: planar attentions {n}")
    log(PHASE + "first_request")
    _, _, d, w = r.gen(PROMPT_A)
    log(f"[first] denoise {d:.1f} ms wall {w:.0f}")
    log(PHASE + "second_request")
    _, _, d, w = r.gen(PROMPT_A)
    log(f"[second] denoise {d:.1f} ms wall {w:.0f}")
    log(PHASE + "timed_A_A")
    a1 = r.gen(PROMPT_A)
    a2 = r.gen(PROMPT_A)
    log(f"[timed] A denoise {a1[2]:.1f} / {a2[2]:.1f}")
    log(PHASE + "B")
    b = r.gen(PROMPT_B)
    log(f"[B] denoise {b[2]:.1f}")
    log(PHASE + "A_after_B")
    a3 = r.gen(PROMPT_A)
    log(f"[A_after_B] denoise {a3[2]:.1f}")
    ok = bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]))
    log("RESULT " + json.dumps({"flag": flag, "planar_attentions": n, "A_repeat_bitwise": ok,
                                 "A_after_B_eq_A": bool(torch.equal(a1[0], a3[0])),
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
        bit = all(c["image_bitwise"] and c["latent_bitwise"]
                  for c in result["bitwise_off_vs_on_same_position"])
        result["passed"] = bool(bit)
        result["compared_against"] = ("switch-off (SM120_QKV_PLANAR=False = main's packed layout) in "
                                      "the same process, A(T=80), B(T=160), A at the same positions; "
                                      "images + final latent")
        result.pop("census_top", None)
        for c in result["census"].values():
            c.pop("top", None)
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
