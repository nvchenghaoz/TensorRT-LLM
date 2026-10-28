"""Idea 75 end-to-end check: image + final latent bitwise in ONE process, plus Dynamo counts.

Switch: every eligible block's _sm120_tail_pair flag (False = main's sequential down, attn_out).

  ab    : one process (TORCH_LOGS=recompiles), engine.json settings (FLASHINFER FP8 attention,
          BF16 VAE decode, per-block torch.compile, no CUDA graphs). Flags off: warm A, then
          A(T~80) B(longer text) A; flags on: warm A, then A B A. Images and final latents compared
          position by position (off vs on), plus A==A within each arm. Then interleaved
          off/on/on/off denoise timing (information only).
  fixed : fresh process with TRTLLM_FLUX3_SM120_TAIL_PAIR=<0|1> fixed from load, under
          TORCH_LOGS=graph_breaks,recompiles: counts during load+warmup and the first request,
          then A, B, A.
  all   : fixed 0, fixed 1, ab; one JSON summary.

usage: python e2e_check.py all <checkpoint> <outdir>
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
        tr = self.pipe.transformer
        blocks = [(f"single_blocks.{i}", b) for i, b in enumerate(tr.single_blocks)]
        if "image" in tr.content_mode_blocks:
            blocks += [(f"content_mode_blocks.image.{i}", b)
                       for i, b in enumerate(tr.content_mode_blocks["image"])]
        self.blocks = [(n, getattr(b, "_orig_mod", b)) for n, b in blocks]
        self.eligible = [n for n, b in self.blocks if getattr(b, "_sm120_tail_pair", False)]
        log(f"[load] tail pair eligible blocks: {len(self.eligible)} {self.eligible}")

    def set_flag(self, on):
        for name, b in self.blocks:
            if name in self.eligible:
                b._sm120_tail_pair = bool(on)

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


def run_ab(ckpt):
    import torch

    r = Runner(ckpt)
    res = {"eligible_blocks": r.eligible}
    seqs = {}
    for flag in (False, True):
        tag = "on" if flag else "off"
        r.set_flag(flag)
        log(PHASE + f"warm_{tag}")
        _, _, d, w = r.gen(PROMPT_A)
        log(f"[warm_{tag}] denoise {d:.1f} ms wall {w:.0f} ms")
        log(PHASE + f"aba_{tag}")
        seq = []
        for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
            img, lat, d, w = r.gen(p)
            seq.append((name, img, lat))
            log(f"[aba_{tag}] {name} denoise {d:.1f} ms lat {tuple(lat.shape)}")
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
    log(PHASE + "timing")
    pos = {0: [], 1: [], 2: [], 3: []}
    order = [False, True, True, False]
    for rnd in range(5):
        for p, flag in enumerate(order):
            r.set_flag(flag)
            pos[p].append(r.gen(PROMPT_A)[2])
        log(f"[timing] round {rnd} off {pos[0][-1]:.2f}/{pos[3][-1]:.2f} "
            f"on {pos[1][-1]:.2f}/{pos[2][-1]:.2f}")
    off, on = pos[0] + pos[3], pos[1] + pos[2]
    res["timing_denoise_ms"] = {
        "off_median": round(statistics.median(off), 3),
        "on_median": round(statistics.median(on), 3),
        "paired_off_minus_on_ms": [round((a + b - c - e) / 2, 3)
                                   for a, b, c, e in zip(pos[0], pos[3], pos[1], pos[2])],
    }
    log("[timing] " + json.dumps(res["timing_denoise_ms"]))
    log("RESULT " + json.dumps(res))


def run_fixed(flag, ckpt):
    import torch

    r = Runner(ckpt)
    log(PHASE + "first_request")
    a0 = r.gen(PROMPT_A)
    log(f"[first] denoise {a0[2]:.1f} ms")
    log(PHASE + "later_A_B_A")
    a1, b, a2 = r.gen(PROMPT_A), r.gen(PROMPT_B), r.gen(PROMPT_A)
    ok = bool(torch.equal(a0[0], a1[0]) and torch.equal(a1[0], a2[0])
              and torch.equal(a1[1], a2[1]))
    log("RESULT " + json.dumps({"flag": flag, "eligible_blocks": len(r.eligible),
                                "A_repeat_bitwise": ok,
                                "denoise_ms": [a0[2], a1[2], b[2], a2[2]]}))


def child(argv, logs, extra_env):
    env = dict(os.environ, TORCH_LOGS=logs, **extra_env)
    phase = "load"
    counts = collections.OrderedDict(load={"recompiles": 0, "graph_breaks": 0})
    result = None
    proc = subprocess.Popen([sys.executable, __file__, "--inner", *argv], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True)
    for line in proc.stdout:
        if line.startswith(PHASE):
            phase = line[len(PHASE):].strip()
            counts.setdefault(phase, {"recompiles": 0, "graph_breaks": 0})
        if "Recompiling function" in line:
            counts[phase]["recompiles"] += 1
            print(f"[recompile in {phase}] " + line.rstrip()[:240], flush=True)
        elif "Graph break" in line:
            counts[phase]["graph_breaks"] += 1
            print(f"[graph break in {phase}] " + line.rstrip()[:240], flush=True)
        elif line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        elif line.startswith(("[", "===", "Traceback")) or "Error" in line:
            print(line.rstrip()[:400], flush=True)
    proc.wait()
    return proc.returncode, result, counts


def main():
    if sys.argv[1] == "--inner":
        argv = sys.argv[2:]
        if argv[0] == "ab":
            run_ab(argv[1])
        else:
            run_fixed(argv[1], argv[2])
        return
    ckpt = sys.argv[2]
    summary = {}
    for flag in ("0", "1"):
        log(f"=== fixed TRTLLM_FLUX3_SM120_TAIL_PAIR={flag}")
        rc, res, counts = child(["fixed", flag, ckpt], "graph_breaks,recompiles",
                                {"TRTLLM_FLUX3_SM120_TAIL_PAIR": flag})
        summary[f"fixed_{flag}"] = {"rc": rc, "result": res, "counts_by_phase": counts}
        log(json.dumps(summary[f"fixed_{flag}"]))
    log("=== ab (one process)")
    rc, res, counts = child(["ab", ckpt], "recompiles", {})
    summary["ab"] = {"rc": rc, "result": res, "recompiles_by_phase": counts}
    ok_ab = rc == 0 and res is not None and all(
        c["image_bitwise"] and c["latent_bitwise"] for c in res["bitwise_off_vs_on"]) and all(
        v["A_first_eq_A_last_image"] and v["A_first_eq_A_last_latent"]
        for v in res["within_process_repeat"].values())
    f0, f1 = summary["fixed_0"], summary["fixed_1"]
    ok_fixed = (f0["rc"] == 0 and f1["rc"] == 0 and f0["result"]["A_repeat_bitwise"]
                and f1["result"]["A_repeat_bitwise"]
                and f0["counts_by_phase"] == f1["counts_by_phase"])
    print(json.dumps({
        "passed": bool(ok_ab and ok_fixed),
        "compared": "ab: switch off (main's sequential path) vs on in ONE process, A/B/A at the "
                    "same positions, image + final latent bitwise; fixed: graph-break/recompile "
                    "counts per phase (load+warmup, first request, later A/B/A) with the pair "
                    "off vs on from load",
        "summary": summary,
    }))


if __name__ == "__main__":
    main()
