"""run_gpu_pair wrapper for branch 29 (sub-pixel Flux3ImageUpsample), gates #461/#468.

usage: python experiments/ascent/subpixel_pair.py <checkpoint_root> base|cand [--decode-dtype bfloat16]

Each arm runs the shared bench (experiments/ascent/vae_decode_bench.py time, served
decode dtype, default bfloat16) with --save-seeds /tmp/sp29_pair/<arm>, which saves this
tree's compiled fp32 decode and its bf16-trunk decode for seeds 0-2. The base arm (main)
runs first on the same slot; the cand arm then applies, per seed 0-2:
  (A) fp32: cand fp32 vs main fp32: u8_max_diff <= 2 and max_abs <= 1e-2.
  (B) bf16 vs R = MAIN's compiled fp32 decode:
      mean_abs(cand,R) <= 1.05 x mean_abs(main_bf16,R);
      PSNR(cand,R) >= PSNR(main_bf16,R) - 0.3 dB;
      u8_max(cand,R) <= u8_max(main_bf16,R) + 2.
      cand vs main_bf16 (max_abs, mean_abs, u8_max, u8_frac) reported ungated.
Last stdout line: the bench's pair JSON (batch/kernel_ms/noise_floor) plus the gates.
"""

import json
import os
import shutil
import subprocess
import sys

import torch

ROOT, ARM = sys.argv[1], sys.argv[2]
DD = "bfloat16"
if "--decode-dtype" in sys.argv:
    DD = sys.argv[sys.argv.index("--decode-dtype") + 1]
BASE_DIR = "/tmp/sp29_pair"
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


shutil.rmtree(MY, ignore_errors=True)
cmd = [sys.executable, "experiments/ascent/vae_decode_bench.py", ROOT, "--decode-dtype", DD, "--save-seeds", MY]
log("running:", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
last = None
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    if line.startswith("{"):
        last = line
rc = proc.wait()
if rc != 0 or last is None:
    print(json.dumps({"passed": False, "error": f"bench rc={rc}"}))
    sys.exit(1)
res = json.loads(last)
res["arm"] = ARM
res["output"]["class"] = "reordered"  # same decode dtype in both arms; change is a reordering
res["output"]["note"] = f"both arms decode with --decode-dtype {DD}; mode-switch error fields are vs each tree's own fp32"

if ARM == "cand":
    base = os.path.join(BASE_DIR, "base")
    per_seed, ok = [], True
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
            m = diff(mb, R)
            c = diff(cb, R)
            gate_b = (c["mean_abs"] <= 1.05 * m["mean_abs"] and c["psnr_db"] >= m["psnr_db"] - 0.3
                      and c["u8_max_diff"] <= m["u8_max_diff"] + 2)
            rec = {"seed": s, "A_fp32_cand_vs_main": a, "A_pass": bool(gate_a),
                   "B_main_bf16_vs_R": m, "B_cand_bf16_vs_R": c, "B_pass": bool(gate_b),
                   "info_cand_bf16_vs_main_bf16": diff(cb, mb)}
            log(json.dumps(rec))
            per_seed.append(rec)
            ok &= bool(gate_a and gate_b)
    res["gates_per_seed"] = per_seed
    res["gates_passed"] = bool(ok)
    res["passed"] = bool(res.get("passed", False) and ok)
print(json.dumps(res))
