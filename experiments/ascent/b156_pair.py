"""run_gpu_pair wrapper for branch 156 (idea 226: conv_transpose2d upsample).

usage: python experiments/ascent/b156_pair.py <checkpoint_root> base|cand

Each arm runs the shared bench (experiments/ascent/vae_decode_bench.py time
--decode-dtype bfloat16 --warmup 10 --iters 60: real weights, latent 1x128x64x64,
CUDA-event median/p10/p90, then the untimed repeat sequence with the same
warmup/iteration count: every output bitwise == iteration 0, latent unchanged, no
Dynamo recompile) with --save-seeds /tmp/b156_pair/<arm> (this tree's compiled fp32
decode and its bf16-trunk decode, seeds 0-2). Base (main) runs first on the same slot.
The cand arm applies, per seed 0-2 (declared before any run, never relaxed):
  (B) bf16 vs R = MAIN's compiled fp32 decode:
      mean_abs(cand,R) <= 1.05 x mean_abs(main_bf16,R);
      PSNR(cand,R) >= PSNR(main_bf16,R) - 0.3 dB;
      u8_max(cand,R) <= u8_max(main_bf16,R) + 2.
Reported rows, NOT gated (#1588): cand fp32 vs main fp32 cross-process (the fp32 path
is main's code; cross-process compiled decodes are not bitwise even on main, #1502;
gate A is the same-process check in b156_check.py), cand bf16 vs main bf16.
Last stdout line: the bench's pair JSON (batch/kernel_ms/noise_floor) plus the gates.
"""

import json
import os
import shutil
import subprocess
import sys

import torch

ROOT, ARM = sys.argv[1], sys.argv[2]
BASE_DIR = "/tmp/b156_pair"
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
            "u8_frac_diff": (du > 0).double().mean().item(), "bitwise": bool(torch.equal(a, b)),
            "psnr_db": float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()}


try:
    head = subprocess.run(["git", "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    anc = subprocess.run(["git", "merge-base", "--is-ancestor",
                          "1185e8f970fa1f56c64deb4efbaae6f5d939e02c", "HEAD"]).returncode
except Exception as ex:  # noqa: BLE001
    head, anc = f"git failed: {ex}", None
log("arm", ARM, "HEAD", head, "is_ancestor(1185e8f9) rc", anc)

shutil.rmtree(MY, ignore_errors=True)
cmd = [sys.executable, "experiments/ascent/vae_decode_bench.py", ROOT, "--decode-dtype", "bfloat16",
       "--warmup", "10", "--iters", "60", "--save-seeds", MY]
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
res["arm"], res["head"], res["is_ancestor_rc"] = ARM, head, anc
res["output"]["class"] = "reordered"
res["output"]["note"] = "both arms decode with --decode-dtype bfloat16; mode-switch fields are vs each tree's own fp32"

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
            m, c = diff(mb, R), diff(cb, R)
            gate_b = (c["mean_abs"] <= 1.05 * m["mean_abs"] and c["psnr_db"] >= m["psnr_db"] - 0.3
                      and c["u8_max_diff"] <= m["u8_max_diff"] + 2)
            rec = {"seed": s, "B_main_bf16_vs_R": m, "B_cand_bf16_vs_R": c, "B_pass": bool(gate_b),
                   "row_fp32_cand_vs_main_cross_process": diff(c32, R),
                   "row_cand_bf16_vs_main_bf16": diff(cb, mb)}
            log(json.dumps(rec))
            per_seed.append(rec)
            ok &= bool(gate_b)
    res["gates_per_seed"] = per_seed
    res["gates_passed"] = bool(ok)
    res["passed"] = bool(res.get("passed", False) and ok)
print(json.dumps(res))
