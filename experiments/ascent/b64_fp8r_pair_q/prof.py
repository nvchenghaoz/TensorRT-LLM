"""Branch 64 model-level checks (served-like pipeline via tail_pair/e2e_check.Runner).

  prof <ckpt> : warm 2 requests, then ONE steady request inside cudaProfilerStart/Stop (for
                profile_timeline capture=region) -> kernel launch counts of that request.
  ab <ckpt>   : one process, TORCH_LOGS=recompiles,graph_breaks. Flag primsts._FP8R_FUSED_SWIGLU_QUANT
                off (main's FP8R tail-pair path) vs on: A, B, A per arm, image + final latent bitwise
                off vs on at the same positions, and A==A within each arm; graph breaks counted.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tail_pair"))

from e2e_check import PROMPT_A, PROMPT_B, Runner, log  # noqa: E402


def prof(ckpt):
    import torch

    r = Runner(ckpt)
    for _ in range(2):
        log(f"[warm] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("request")
    log(f"[prof] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def ab(ckpt):
    import torch
    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts

    r = Runner(ckpt)
    seqs = {}
    for flag in (False, True):
        primsts._FP8R_FUSED_SWIGLU_QUANT = flag
        log(f"=== PHASE warm_{flag}")
        log(f"[warm {flag}] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
        log(f"=== PHASE aba_{flag}")
        seq = []
        for name, p in (("A", PROMPT_A), ("B", PROMPT_B), ("A", PROMPT_A)):
            img, lat, d, _ = r.gen(p)
            seq.append((name, img, lat))
            log(f"[aba {flag}] {name} denoise {d:.1f} ms")
        seqs[flag] = seq
    cmp = [{"pos": i, "prompt": a[0], "image_bitwise": bool(torch.equal(a[1], b[1])),
            "latent_bitwise": bool(torch.equal(a[2], b[2]))}
           for i, (a, b) in enumerate(zip(seqs[False], seqs[True]))]
    within = {str(f): bool(torch.equal(s[0][1], s[2][1]) and torch.equal(s[0][2], s[2][2]))
              for f, s in seqs.items()}
    log("RESULT " + json.dumps({"off_vs_on": cmp, "A_repeat_bitwise": within,
                                "eligible": r.eligible}))


def main():
    mode, ckpt = sys.argv[1], sys.argv[2]
    if mode == "prof":
        prof(ckpt)
        return
    if mode == "--inner":
        ab(ckpt)
        return
    import subprocess

    env = dict(os.environ, TORCH_LOGS="recompiles,graph_breaks")
    phase, counts, result = "load", {"load": {"graph_breaks": 0, "recompiles": 0}}, None
    proc = subprocess.Popen([sys.executable, __file__, "--inner", ckpt], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True)
    for line in proc.stdout:
        if line.startswith("=== PHASE "):
            phase = line[len("=== PHASE "):].strip()
            counts.setdefault(phase, {"graph_breaks": 0, "recompiles": 0})
        if "Graph break" in line:
            counts[phase]["graph_breaks"] += 1
            print(f"[graph break in {phase}] " + line.rstrip()[:300], flush=True)
        elif "Recompiling function" in line:
            counts[phase]["recompiles"] += 1
            print(f"[recompile in {phase}] " + line.rstrip()[:300], flush=True)
        elif line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        elif line.startswith(("[", "===", "Traceback")) or "Error" in line:
            print(line.rstrip()[:400], flush=True)
    proc.wait()
    passed = (proc.returncode == 0 and result is not None
              and all(c["image_bitwise"] and c["latent_bitwise"] for c in result["off_vs_on"])
              and all(result["A_repeat_bitwise"].values())
              and counts.get("aba_True", {}).get("graph_breaks", 1) == 0
              and counts.get("aba_True", {}).get("recompiles", 1) == 0)
    print(json.dumps({
        "passed": bool(passed),
        "compared": "full pipeline, one process: FP8R fused tail pair (flag on) vs main's FP8R "
                    "tail-pair path (flag off), A/B/A image + final latent bitwise; A==A per arm; "
                    "graph breaks / recompiles per phase (warm_True absorbs the flag-flip "
                    "recompile; aba_True must have none)",
        "counts_by_phase": counts,
        "result": result,
    }))


if __name__ == "__main__":
    main()
