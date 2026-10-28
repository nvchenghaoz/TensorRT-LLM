"""Branch 40 end-to-end: fused FP8R SwiGLU+quant vs today's silu_and_mul -> warp quant.

ONE process, switch primsts._FP8R_FUSED_SWIGLU_QUANT (False = main's path, byte-identical
code to the base). Requests A (T=80), B (T=160), A per switch value; images + final latent
compared bitwise. Then a CUDA kernel trace of one A request per switch: the per-block-step
chain gate_up GEMM -> silu_and_mul -> warp quant -> down GEMM (off) and
gate_up GEMM -> fused -> down GEMM (on) is listed with us/call for the first
image.0 (M=4096) and single_blocks.24 (M=4176) block-steps (STEP 0 rows).
Then `fixed` mode (fresh process, switch fixed on from load): A, A, B, A with Dynamo
recompiles and Triton JIT compiles counted per phase.

usage: e2e_ab.py ab <checkpoint> <outdir> | e2e_ab.py fixed <on|off> <checkpoint> <outdir>
"""
import collections
import json
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


def install_triton_hook():
    try:
        import triton

        def post(*a, **kw):
            name = kw.get("repr") or kw.get("fn") or (a[0] if a else "?")
            log(f"[triton-compile] {str(name)[:160]}")
            return None

        triton.knobs.runtime.jit_post_compile_hook = post
        log("[hook] triton jit_post_compile_hook installed")
    except Exception as exc:  # noqa: BLE001
        log(f"[hook] triton hook unavailable: {exc!r}")


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

    def gen(self, prompt, seed=1000):
        from tensorrt_llm.visual_gen.params import VisualGenParams

        params = VisualGenParams(height=1024, width=1024, seed=seed, num_inference_steps=16)
        req = types.SimpleNamespace(prompt=prompt, params=params)
        t0 = time.time()
        out = self.pipe.infer(req)
        self.torch.cuda.synchronize()
        wall = time.time() - t0
        return (out.image.detach().cpu().clone(), self.last_latent.cpu().clone(),
                out.denoise * 1e3, wall * 1e3)


def set_flag(on):
    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

    primsts._FP8R_FUSED_SWIGLU_QUANT = bool(on)


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
    return [(e["name"], float(e.get("dur", 0.0)), str(e.get("args", {}).get("grid", ""))) for e in ks]


def chains(ks):
    """MLP middle chains: (gemm_before, middle kernels..., gemm_after) with us."""
    out = []
    for i, (name, dur, grid) in enumerate(ks):
        if "silu_and_mul_kernel" in name or "_swiglu_fp8r" in name:
            j = i + 1
            mids = [(name, dur, grid)]
            if "silu_and_mul_kernel" in name and j < len(ks):
                mids.append(ks[j])
                j += 1
            out.append({"gate_up": (ks[i - 1][0][:80], round(ks[i - 1][1], 1), ks[i - 1][2]),
                        "middle": [(n[:80], round(d, 1), g) for n, d, g in mids],
                        "down": (ks[j][0][:80], round(ks[j][1], 1), ks[j][2]) if j < len(ks) else None,
                        "middle_us": round(sum(d for _, d, _ in mids), 1)})
    return out


def windows(ks, before=2, after=4):
    """Kernel windows around the FP8R MLP middle (image tokens: grid M >= 4096)."""
    out = []
    for i, (name, _, grid) in enumerate(ks):
        big = "silu_and_mul_kernel" in name and grid.startswith("[") and int(grid[1:].split(",")[0]) >= 4096
        if big or "_swiglu_fp8r" in name:
            m = int(grid[1:].split(",")[0])
            out.append({"M": m, "rows": [{"rel": j - i, "name": ks[j][0][:110], "us": round(ks[j][1], 1),
                                          "grid": ks[j][2]}
                                         for j in range(max(0, i - before), min(len(ks), i + after + 1))]})
    return out


def run_profile(ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    r = Runner(ckpt)
    res = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        set_flag(flag)
        log(PHASE + f"warm_{tag}")
        r.gen(PROMPT_A)
        r.gen(PROMPT_A)
        log(PHASE + f"profile_{tag}")
        ks = kernel_trace(r, os.path.join(outdir, f"trace_{tag}.json"), PROMPT_A)
        with open(os.path.join(outdir, f"kernels_{tag}.tsv"), "w") as f:
            for k in ks:
                f.write(f"{k[1]:.2f}\t{k[2]}\t{k[0]}\n")
        w = windows(ks)
        first = {}
        for c in w:
            first.setdefault(c["M"], c)
        res[tag] = {"fp8r_mlp_calls_per_request": len(w),
                    "by_M": {str(m): sum(1 for c in w if c["M"] == m) for m in sorted({c["M"] for c in w})},
                    "first_image0_M4096": first.get(4096), "first_single24_M4176": first.get(4176)}
        for key in ("first_image0_M4096", "first_single24_M4176"):
            c = res[tag][key]
            if c:
                for row in c["rows"]:
                    log(f"[step0_{tag}] {key} {row['rel']:+d} {row['us']:8.1f} us {row['grid']:>14s} {row['name']}")
    log("RESULT " + json.dumps(res))


def run_ab(ckpt, outdir):
    import torch

    os.makedirs(outdir, exist_ok=True)
    set_flag(False)
    r = Runner(ckpt)
    res = {}
    log(PHASE + "warm_off")
    for _ in range(2):
        _, _, d, w = r.gen(PROMPT_A)
        log(f"[warm_off] denoise {d:.1f} ms wall {w:.0f} ms")
    _, _, d, w = r.gen(PROMPT_B)
    log(f"[warm_off] B denoise {d:.1f} ms wall {w:.0f} ms")
    seqs = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        set_flag(flag)
        if flag:
            log(PHASE + "warm_on")
            for p in (PROMPT_A, PROMPT_B):
                _, _, d, w = r.gen(p)
                log(f"[warm_on] denoise {d:.1f} ms wall {w:.0f} ms")
        log(PHASE + f"aba_{tag}")
        seq = []
        for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
            img, lat, d, w = r.gen(p)
            seq.append((name, img, lat))
            log(f"[aba_{tag}] {name} denoise {d:.1f} ms img {tuple(img.shape)} lat {tuple(lat.shape)}")
        seqs[tag] = seq
    cmp = []
    for i, ((n0, i0, l0), (_, i1, l1)) in enumerate(zip(seqs["off"], seqs["on"])):
        cmp.append({"pos": i, "prompt": n0, "image_bitwise": bool(torch.equal(i0, i1)),
                    "latent_bitwise": bool(torch.equal(l0, l1)),
                    "latent_max_abs": float((l0.float() - l1.float()).abs().max())})
    within = {tag: {"A_first_eq_A_last_image": bool(torch.equal(s[0][1], s[2][1])),
                    "A_first_eq_A_last_latent": bool(torch.equal(s[0][2], s[2][2]))}
              for tag, s in seqs.items()}
    res["bitwise_off_vs_on"] = cmp
    res["within_process_repeat"] = within
    log("[bitwise] " + json.dumps(cmp))
    log("[within] " + json.dumps(within))

    res["chains"] = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        set_flag(flag)
        log(PHASE + f"profile_{tag}")
        ks = kernel_trace(r, os.path.join(outdir, f"trace_{tag}.json"), PROMPT_A)
        ch = chains(ks)
        with open(os.path.join(outdir, f"kernels_{tag}.tsv"), "w") as f:
            for k in ks:
                f.write(f"{k[1]:.2f}\t{k[2]}\t{k[0]}\n")
        # Per request: 16 steps x 5 FP8R blocks = 80 chains. Block order per step:
        # image.0 (M=4096) first, then single_blocks.24-27 (M=4176).
        mids = [c["middle_us"] for c in ch]
        res["chains"][tag] = {"count": len(ch), "first_image0": ch[0] if ch else None,
                              "first_single24": ch[1] if len(ch) > 1 else None,
                              "middle_us_total": round(sum(mids), 1),
                              "middle_us_median": round(statistics.median(mids), 1) if mids else None,
                              "kernel_us_total": round(sum(k[1] for k in ks))}
        log(f"[chains_{tag}] " + json.dumps(res["chains"][tag]))

    log(PHASE + "timing")
    pos = {0: [], 1: [], 2: [], 3: []}
    order = [False, True, True, False]
    for rnd in range(5):
        for p, flag in enumerate(order):
            set_flag(flag)
            pos[p].append(r.gen(PROMPT_A)[2])
        log(f"[timing] round {rnd} off {pos[0][-1]:.2f}/{pos[3][-1]:.2f} on {pos[1][-1]:.2f}/{pos[2][-1]:.2f}")
    off, on = pos[0] + pos[3], pos[1] + pos[2]
    res["timing_denoise_ms"] = {"off_median": round(statistics.median(off), 2),
                                "on_median": round(statistics.median(on), 2),
                                "aa_off_ms": round(statistics.median(pos[0]) - statistics.median(pos[3]), 2),
                                "aa_on_ms": round(statistics.median(pos[1]) - statistics.median(pos[2]), 2)}
    log("[timing] " + json.dumps(res["timing_denoise_ms"]))
    log("RESULT " + json.dumps(res))


def run_fixed(flag, ckpt, outdir):
    import torch

    os.makedirs(outdir, exist_ok=True)
    set_flag(flag == "on")
    r = Runner(ckpt)
    log(PHASE + "first_request")
    r.gen(PROMPT_A)
    log(PHASE + "warm_B")
    r.gen(PROMPT_B)
    log(PHASE + "timed_A_A")
    a1 = r.gen(PROMPT_A)
    a2 = r.gen(PROMPT_A)
    log(PHASE + "timed_B")
    b = r.gen(PROMPT_B)
    log(PHASE + "timed_A_after_B")
    a3 = r.gen(PROMPT_A)
    ok = bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[0], a3[0]) and torch.equal(a1[1], a3[1]))
    log("RESULT " + json.dumps({"flag": flag, "A_repeat_bitwise": ok,
                                 "denoise_ms": [a1[2], a2[2], b[2], a3[2]]}))


def wrapper(argv):
    outdir = argv[-1]
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, TORCH_LOGS="recompiles")
    phase = "load"
    counts = collections.OrderedDict(load=0)
    tcounts = collections.OrderedDict(load=0)
    result = None
    with open(os.path.join(outdir, f"log_{'_'.join(argv[:-2])}.txt"), "w") as f:
        proc = subprocess.Popen([sys.executable, __file__, "--inner", *argv], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True)
        for line in proc.stdout:
            f.write(line)
            if line.startswith(PHASE):
                phase = line[len(PHASE):].strip()
                counts.setdefault(phase, 0)
                tcounts.setdefault(phase, 0)
            if "Recompiling function" in line:
                counts[phase] = counts.get(phase, 0) + 1
                print(f"[recompile in {phase}] " + line.rstrip()[:240], flush=True)
            elif line.startswith("[triton-compile]"):
                tcounts[phase] = tcounts.get(phase, 0) + 1
                if "timed" in phase or "aba" in phase:
                    print(f"[triton in {phase}] " + line.rstrip()[:200], flush=True)
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
            elif line.startswith(("[", "===", "Traceback")) or "Error" in line:
                print(line.rstrip()[:600], flush=True)
        proc.wait()
    if proc.returncode != 0 or result is None:
        print(json.dumps({"passed": False, "rc": proc.returncode, "recompiles_by_phase": counts}))
        sys.exit(1)
    result["recompiles_by_phase"] = counts
    result["triton_jit_compiles_by_phase"] = tcounts
    if argv[0] == "ab":
        bit = all(c["image_bitwise"] and c["latent_bitwise"] for c in result["bitwise_off_vs_on"])
        rep = all(v["A_first_eq_A_last_image"] and v["A_first_eq_A_last_latent"]
                  for v in result["within_process_repeat"].values())
        result["passed"] = bool(bit and rep)
        result["compared_against"] = ("switch-off path (main's silu_and_mul -> warp_fp8_per_token_quant) "
                                      "in the same process, requests A(T=80), B(T=160), A; images + final latent")
        result["output"] = {"class": "exact"}
    elif argv[0] == "profile":
        result["passed"] = True
    else:
        timed = [p for p in counts if p.startswith("timed")]
        result["recompiles_in_timed"] = sum(counts[p] for p in timed)
        result["triton_compiles_in_timed"] = sum(tcounts.get(p, 0) for p in timed)
        result["passed"] = bool(result.get("A_repeat_bitwise"))
    print(json.dumps(result))


def main():
    if sys.argv[1] == "--inner":
        install_triton_hook()
        argv = sys.argv[2:]
        if argv[0] == "ab":
            run_ab(argv[1], argv[2])
        elif argv[0] == "profile":
            run_profile(argv[1], argv[2])
        else:
            run_fixed(argv[1], argv[2], argv[3])
        return
    wrapper(sys.argv[1:])


if __name__ == "__main__":
    main()
