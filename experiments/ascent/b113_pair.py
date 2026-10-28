"""run_gpu_pair wrapper for branch 113 (idea 52, fused VAE decoder tail).

usage: python experiments/ascent/b113_pair.py <checkpoint_root> base|cand

1) experiments/ascent/vae_decode_bench.py time --decode-dtype bfloat16 --save-seeds
   /tmp/b113_pair/<arm>: CUDA-event median of 60 compiled bf16 decodes after 5 warmup,
   then the untimed repeat sequence (same warmup/iters, every output bitwise == iter 0,
   input unchanged, no recompiles), and per-seed (0-2) outputs of this tree's fp32 and
   bf16-trunk decoders.
2) experiments/ascent/b113_census.py: tail rows (last 10 kernels of a decode) and Dynamo
   counters (one frame, no graph break, unchanged).
3) cand arm only, gates per seed 0-2 (thresholds from THIS pair's main arm):
   (A) fp32 decoder: cand vs main u8_max_diff <= 2 and max_abs <= 1e-2.
   (B) bf16 vs R = main's fp32 decode: mean_abs <= 1.05x main_bf16's, PSNR >= main_bf16's
       - 0.3 dB, u8_max <= main_bf16's + 2. cand vs main_bf16 reported ungated.
Last stdout line: pair evidence JSON (batch/kernel_ms/noise_floor) + census + gates.
"""

import json
import os
import shutil
import subprocess
import sys

import torch

ROOT, ARM = sys.argv[1], sys.argv[2]
BASE_DIR = "/tmp/b113_pair"
MY = os.path.join(BASE_DIR, ARM)


def log(*a):
    print(*a, flush=True)


def to_u8(x):
    x = (x / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
    return (x * 255).round().to(torch.uint8)


def diff(a, b):
    d = (a.double() - b.double()).abs()
    du = (to_u8(a.float()).int() - to_u8(b.float()).int()).abs()
    mse = ((a.double().clamp(-1, 1) - b.double().clamp(-1, 1)) ** 2).mean().item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "u8_max_diff": du.max().item(),
            "u8_frac_diff": (du > 0).double().mean().item(),
            "psnr_db": float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()}


def run(cmd):
    log("running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{"):
            last = line
    rc = proc.wait()
    return rc, (json.loads(last) if last else None)


log("git:", subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout.strip())
shutil.rmtree(MY, ignore_errors=True)
rc, res = run([sys.executable, "experiments/ascent/vae_decode_bench.py", ROOT, "--decode-dtype", "bfloat16",
               "--save-seeds", MY])
if rc != 0 or res is None:
    print(json.dumps({"passed": False, "error": f"bench rc={rc}"}))
    sys.exit(1)
res["arm"] = ARM
res["output"]["class"] = "reordered"
res["output"]["note"] = ("both arms bf16 trunk; cand tail = one Triton kernel (conv2 bias + residual add, "
                         "norm_out affine + swish in fp32, conv_out tf32x3 fp32-level) vs main's bf16 apply "
                         "+ cuDNN TF32 conv_out")
rc2, cen = run([sys.executable, "experiments/ascent/b113_census.py", ROOT])
res["census"] = cen if rc2 == 0 else {"error": f"census rc={rc2}"}
ok = bool(res.get("passed")) and rc2 == 0 and bool(cen and cen.get("passed"))

if ARM == "cand":
    base = os.path.join(BASE_DIR, "base")
    per_seed = []
    if not os.path.exists(os.path.join(base, "fp32_s0.pt")):
        res["gate_error"] = "base arm outputs not found (arms on different hosts?)"
        ok = False
    else:
        for s in range(3):
            R = torch.load(os.path.join(base, f"fp32_s{s}.pt"))
            mb = torch.load(os.path.join(base, f"dec_s{s}.pt"))
            c32 = torch.load(os.path.join(MY, f"fp32_s{s}.pt"))
            cb = torch.load(os.path.join(MY, f"dec_s{s}.pt"))
            a = diff(c32, R)
            gate_a = a["u8_max_diff"] <= 2 and a["max_abs"] <= 1e-2
            m, c = diff(mb, R), diff(cb, R)
            gate_b = (c["mean_abs"] <= 1.05 * m["mean_abs"] and c["psnr_db"] >= m["psnr_db"] - 0.3
                      and c["u8_max_diff"] <= m["u8_max_diff"] + 2)
            rec = {"seed": s, "A_fp32_cand_vs_main": a, "A_pass": bool(gate_a), "B_main_bf16_vs_R": m,
                   "B_cand_bf16_vs_R": c, "B_pass": bool(gate_b), "info_cand_bf16_vs_main_bf16": diff(cb, mb)}
            log(json.dumps(rec))
            per_seed.append(rec)
            ok &= bool(gate_a and gate_b)
    res["gates_per_seed"] = per_seed
res["passed"] = bool(ok)
print(json.dumps(res))
