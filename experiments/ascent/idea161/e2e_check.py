"""Idea 161 acceptance 3: two fresh graph-on processes, TRTLLM_FLUX3_SIDE_GATE_UP=0 vs the chosen
mode, each generating A, A, B (T=160 text bucket), A in one process (pipeline as served: engine.json
overlay, torch.compile + autotune, CUDA graphs on). Requires every image and final latent bitwise
equal between the two processes, A repeats bitwise within each process, and peak reserved memory
ON not above OFF by more than 0.1 GiB.

usage: python e2e_check.py <checkpoint> [mode]          (parent: spawns both children)
       python e2e_check.py <checkpoint> --child <out.pt> (child: env decides the mode)
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]


def log(*a):
    print(*a, flush=True)


def child(out_path):
    import torch

    sys.path.insert(0, os.path.dirname(HERE))
    from te_common import prompt_for_bucket

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
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
    blocks = [m for m in pipe.transformer.modules() if hasattr(m, "_sm120_side_gate_up")]
    modes = sorted({int(m._sm120_side_gate_up) for m in blocks})
    n_on = sum(1 for m in blocks if m._sm120_side_gate_up)
    log(f"loaded {time.time() - t0:.0f}s env={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP')} "
        f"blocks={len(blocks)} side_gate_up_blocks={n_on} modes={modes}")
    te = pipe.text_encoder

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        return latent.clone().cpu(), img.clone().cpu()

    prompts = {"A": ("A red fox standing in fresh snow at sunrise", 11),
               "B": (prompt_for_bucket(te, 2), 22)}
    torch.cuda.reset_peak_memory_stats()
    seq = []
    for name in ("A", "A", "B", "A"):
        t = time.time()
        seq.append((name, *generate(*prompts[name])))
        log(f"{name}: {time.time() - t:.2f}s reserved {torch.cuda.memory_reserved() / 2**30:.3f} GiB")
    torch.save({
        "seq": seq,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "counters": dict(fp.SIDE_GATE_UP_COUNTERS),
        "side_gate_up_blocks": n_on,
        "modes": modes,
    }, out_path)
    log("child done")


def parent(mode):
    import torch

    tmp = tempfile.mkdtemp()
    res = {}
    for tag, env_mode in (("off", "0"), ("on", mode)):
        out = os.path.join(tmp, f"{tag}.pt")
        env = dict(os.environ, TRTLLM_FLUX3_SIDE_GATE_UP=env_mode)
        log(f"=== {tag}: TRTLLM_FLUX3_SIDE_GATE_UP={env_mode}")
        rc = subprocess.run([sys.executable, __file__, CKPT, "--child", out], env=env).returncode
        if rc != 0:
            print(json.dumps({"passed": False, "error": f"{tag} child rc={rc}"}))
            return
        res[tag] = torch.load(out)
    off, on = res["off"], res["on"]
    per = []
    for i, ((n0, l0, i0), (n1, l1, i1)) in enumerate(zip(off["seq"], on["seq"])):
        per.append({"call": i, "prompt": n0, "latent_bitwise": torch.equal(l0, l1),
                    "image_bitwise": torch.equal(i0, i1),
                    "max_abs_px": int((i0.int() - i1.int()).abs().max())})
    within = {}
    for tag in ("off", "on"):
        s = res[tag]["seq"]
        within[tag] = all(torch.equal(s[0][1], s[j][1]) and torch.equal(s[0][2], s[j][2])
                          for j in (1, 3))
    mem_delta = on["peak_reserved_gib"] - off["peak_reserved_gib"]
    cross = all(p["latent_bitwise"] and p["image_bitwise"] for p in per)
    passed = (cross and all(within.values()) and mem_delta <= 0.1
              and on["counters"]["sync_fallback"] == 0 and on["side_gate_up_blocks"] > 0
              and off["side_gate_up_blocks"] == 0)
    print(json.dumps({
        "passed": bool(passed),
        "reference": "fresh graph-on process with TRTLLM_FLUX3_SIDE_GATE_UP=0 (main's sequential "
                     "gate_up), same prompts/seeds, A,A,B(T=160),A",
        "on_vs_off": per,
        "A_repeats_bitwise_within_process": within,
        "peak_reserved_gib": {"off": round(off["peak_reserved_gib"], 3),
                              "on": round(on["peak_reserved_gib"], 3),
                              "delta": round(mem_delta, 3)},
        "peak_allocated_gib": {"off": round(off["peak_allocated_gib"], 3),
                               "on": round(on["peak_allocated_gib"], 3)},
        "counters_on": on["counters"], "counters_off": off["counters"],
        "side_gate_up_blocks": {"off": off["side_gate_up_blocks"], "on": on["side_gate_up_blocks"]},
        "modes_on": on["modes"],
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent(sys.argv[2] if len(sys.argv) > 2 else "3")
