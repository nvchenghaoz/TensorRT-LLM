"""Kernel list of one served-config request + recompile / graph-break / JIT counts per phase.

Pipeline built as the serving benchmark runs it (engine.json: FLASHINFER fp8/fp8 attention,
BF16 VAE decode, per-block torch.compile), via tail_pair/e2e_check.Runner.
Phases: load -> warm (2 requests) -> timed (A, B, A; A==A bitwise) -> profile (1 request
under torch.profiler, CUDA kernels aggregated by name -> $AUTO_DEV_ARTIFACTS/kernels.json).

usage: python profile_request.py <checkpoint>
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tail_pair"))
PHASE = "=== PHASE "
ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")


def inner(ckpt):
    import torch
    from e2e_check import PROMPT_A, PROMPT_B, Runner, log
    from torch.profiler import ProfilerActivity, profile

    r = Runner(ckpt)
    log(PHASE + "warm")
    for _ in range(2):
        log(f"[warm] denoise {r.gen(PROMPT_A)[2]:.1f} ms")
    log(PHASE + "timed")
    a1, b, a2 = r.gen(PROMPT_A), r.gen(PROMPT_B), r.gen(PROMPT_A)
    rep_ok = bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]))
    log(f"[timed] denoise {a1[2]:.1f}/{b[2]:.1f}/{a2[2]:.1f} ms; A==A bitwise {rep_ok}")
    log(PHASE + "profile")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = r.gen(PROMPT_A)
        torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0, 0.0])
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            agg[ev.name][0] += 1
            agg[ev.name][1] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
    rows = sorted(({"name": k, "calls": v[0], "total_us": round(v[1], 1)} for k, v in agg.items()),
                  key=lambda x: -x["total_us"])
    with open(os.path.join(ART, "kernels.json"), "w") as f:
        json.dump(rows, f, indent=1)
    log(f"[profile] denoise {out[2]:.1f} ms, {len(rows)} distinct kernels")
    log("RESULT " + json.dumps({"A_repeat_bitwise": rep_ok, "kernels": rows}))


def main():
    if sys.argv[1] == "--inner":
        inner(sys.argv[2])
        return
    ckpt = sys.argv[1]
    env = dict(os.environ, TORCH_LOGS="recompiles,graph_breaks")
    phase = "load"
    counts = collections.OrderedDict(load={"recompiles": 0, "graph_breaks": 0, "jit_build": 0})
    result = None
    proc = subprocess.Popen([sys.executable, __file__, "--inner", ckpt], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True)
    for line in proc.stdout:
        if line.startswith(PHASE):
            phase = line[len(PHASE):].strip()
            counts.setdefault(phase, {"recompiles": 0, "graph_breaks": 0, "jit_build": 0})
        if "Recompiling function" in line:
            counts[phase]["recompiles"] += 1
            print(f"[recompile in {phase}] " + line.rstrip()[:240], flush=True)
        elif "Graph break" in line:
            counts[phase]["graph_breaks"] += 1
            print(f"[graph break in {phase}] " + line.rstrip()[:240], flush=True)
        elif ("ninja" in line.lower() or "Building" in line or "JIT" in line) and phase != "load":
            counts[phase]["jit_build"] += 1
            print(f"[jit in {phase}] " + line.rstrip()[:240], flush=True)
        elif line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        elif line.startswith(("[", "===", "Traceback")) or "Error" in line:
            print(line.rstrip()[:400], flush=True)
    proc.wait()
    kernels = result["kernels"] if result else []
    attn_like = [k for k in kernels if any(s in k["name"].lower() for s in (
        "fmha", "qkv_", "quantize", "scalematrix", "computefp8", "copy", "memset", "memcpy",
        "fill", "triton"))]
    for k in attn_like:
        print(f"{k['calls']:6d} {k['total_us']:12.1f} us  {k['name'][:150]}", flush=True)
    passed = (proc.returncode == 0 and result is not None and result["A_repeat_bitwise"]
              and counts.get("timed", {}).get("recompiles", 1) == 0
              and counts.get("timed", {}).get("jit_build", 1) == 0
              and counts.get("profile", {}).get("recompiles", 1) == 0)
    print(json.dumps({
        "passed": bool(passed),
        "compared": "one served-config request's CUDA kernel list (torch.profiler); recompile/"
                    "graph-break/JIT counts per phase; A==A image+latent bitwise",
        "counts_by_phase": counts,
        "attention_related_kernels": attn_like[:40],
        "n_kernel_names": len(kernels),
    }))


if __name__ == "__main__":
    main()
