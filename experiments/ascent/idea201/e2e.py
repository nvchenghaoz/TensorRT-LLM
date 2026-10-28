"""Idea 201 acceptance 2: one process family (shared fresh TORCHINDUCTOR/TRITON cache dirs,
autotune off, ruling C) of four fresh graph-on pipeline processes:
  (TRTLLM_FLUX3_SM120_SWIGLU_SCHED, TRTLLM_FLUX3_SIDE_GATE_UP) in
  (static, 3), (dynamic, 3), (static, 0), (dynamic, 0).
Each generates A, A, B (T=160 bucket), B: first-pass per text length only (ruling on the
T80-after-T160 leak). Requires dynamic == static bitwise (latents and images) call by call at each
SIDE_GATE_UP, A2 == A1 and B2 == B1 in every process (repeated_execution_passed), and under mode 3
SIDE_GATE_UP_COUNTERS issued == joined > 0 with sync_fallback == 0.

usage: python e2e.py <checkpoint>                     (parent)
       python e2e.py <checkpoint> --child <out.pt>     (child: env decides the arm)
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

    import tensorrt_llm
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc
    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_gemm_config import autotune_context
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(HERE, "..", "..", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    tcc = dict(engine.pop("torch_compile_config", {}))
    tcc.update({"enable": True, "enable_fullgraph": False, "enable_autotune": False})
    args = VisualGenArgs(
        model=CKPT,
        pipeline_config=pipeline_config,
        torch_compile_config=tcc,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    sched = swc.tile_sched() if hasattr(swc, "tile_sched") else "static"
    blocks = [m for m in pipe.transformer.modules() if hasattr(m, "_sm120_side_gate_up")]
    modes = sorted({int(m._sm120_side_gate_up) for m in blocks})
    log(f"loaded {time.time() - t0:.0f}s trtllm={tensorrt_llm.__file__} sched={sched} "
        f"SIDE_GATE_UP={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP')} modes={modes}")
    te = pipe.text_encoder
    counters0 = dict(fp.SIDE_GATE_UP_COUNTERS)

    def generate(prompt, seed):
        with torch.inference_mode():
            ctx = pipe._encode_prompt(prompt, None)
        torch.cuda.synchronize()
        t = time.time()
        with torch.inference_mode(), autotune_context(pipe.device, pipe.rank, tune=False):
            latent, ids = pipe._prepare_t2i(
                ctx=ctx, height=1024, width=1024, num_inference_steps=16, seed=seed
            )
            torch.cuda.synchronize()
            t_den = time.time() - t
            img = pipe._package_output(pipe._decode_latents((latent, ids))).image
        torch.cuda.synchronize()
        return latent.clone().cpu(), img.clone().cpu(), t_den

    prompts = {"A": ("A red fox standing in fresh snow at sunrise", 11),
               "B": (prompt_for_bucket(te, 2), 22)}
    seq = []
    for name in ("A", "A", "B", "B"):
        lat, img, t_den = generate(*prompts[name])
        seq.append((name, lat, img))
        log(f"{name}: denoise {t_den * 1e3:.1f} ms")
    counters = {k: fp.SIDE_GATE_UP_COUNTERS[k] - counters0[k] for k in counters0}
    torch.save({"seq": seq, "sched": sched, "modes": modes, "counters": counters,
                "counters_total": dict(fp.SIDE_GATE_UP_COUNTERS)}, out_path)
    log("child done")


def parent():
    import torch

    tmp = tempfile.mkdtemp(prefix="idea201_e2e_")
    base_env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=os.path.join(tmp, "inductor"),
                    TRITON_CACHE_DIR=os.path.join(tmp, "triton"))
    arms = [("static", "3"), ("dynamic", "3"), ("static", "0"), ("dynamic", "0")]
    res = {}
    for sched, sgu in arms:
        tag = f"{sched}_sgu{sgu}"
        out = os.path.join(tmp, f"{tag}.pt")
        env = dict(base_env, TRTLLM_FLUX3_SM120_SWIGLU_SCHED=sched, TRTLLM_FLUX3_SIDE_GATE_UP=sgu)
        log(f"=== {tag}")
        rc = subprocess.run([sys.executable, __file__, CKPT, "--child", out], env=env).returncode
        if rc != 0:
            print(json.dumps({"passed": False, "error": f"{tag} child rc={rc}"}))
            return
        res[tag] = torch.load(out)
    passed = True
    report = {}
    for sgu in ("3", "0"):
        s, d = res[f"static_sgu{sgu}"], res[f"dynamic_sgu{sgu}"]
        per = []
        for i, ((n0, l0, i0), (n1, l1, i1)) in enumerate(zip(s["seq"], d["seq"])):
            per.append({"call": i, "prompt": n0, "latent_bitwise": torch.equal(l0, l1),
                        "image_bitwise": torch.equal(i0, i1),
                        "max_abs_px": int((i0.int() - i1.int()).abs().max())})
        cross = all(p["latent_bitwise"] and p["image_bitwise"] for p in per)
        rep = {}
        for tag, r in (("static", s), ("dynamic", d)):
            q = r["seq"]
            rep[tag] = (torch.equal(q[0][1], q[1][1]) and torch.equal(q[0][2], q[1][2])
                        and torch.equal(q[2][1], q[3][1]) and torch.equal(q[2][2], q[3][2]))
        ok = cross and all(rep.values()) and d["sched"] == "dynamic" and s["sched"] == "static"
        if sgu == "3":
            for r in (s, d):
                c = r["counters"]
                ok &= c["issued"] == c["joined"] and c["sync_fallback"] == 0 and c["issued"] > 0
        passed &= ok
        report[f"SIDE_GATE_UP={sgu}"] = {
            "dynamic_vs_static": per, "repeated_execution_passed": rep,
            "counters": {"static": s["counters"], "dynamic": d["counters"]},
            "modes": {"static": s["modes"], "dynamic": d["modes"]}, "ok": ok,
        }
    print(json.dumps({
        "passed": bool(passed),
        "repeated_execution_passed": all(all(v["repeated_execution_passed"].values())
                                         for v in report.values()),
        "reference": "fresh process with TRTLLM_FLUX3_SM120_SWIGLU_SCHED=static (main's module), "
                     "same SIDE_GATE_UP, same prompts/seeds A,A,B(T=160),B; one process family with "
                     "shared fresh inductor/triton caches, autotune off",
        "results": report,
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--child":
        child(sys.argv[3])
    else:
        parent()
