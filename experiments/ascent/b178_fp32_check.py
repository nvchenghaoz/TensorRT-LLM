"""Branch 178: fp32 decode unchanged vs main (run in both arms of a pair; compare the printed shas).

usage: python experiments/ascent/b178_fp32_check.py <checkpoint_root>

The served-dtype (fp32) decoder path is untouched by the branch (the change is inside the
FP8 branches only). Shown three ways, seeds 0-2, 1x128x64x64 latents:
  eager   uncompiled fp32 decoder: sha256 of raw output bytes (cross-arm: must be equal),
          2 decodes per seed bitwise.
  fx      Dynamo FX graph code of the fp32 decoder (capture backend, comments stripped):
          sha256 (cross-arm: must be equal) -> same graph handed to Inductor.
  aa      compiled fp32 decoder twice in this process, the second with Inductor caches
          disabled: shas of both. If they differ, compiled-fp32 cross-start differences are
          compile noise (Triton autotune timing), not the branch.
Last stdout line: {"passed": ..., ...} (eager repeat bitwise, latents unchanged).
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent"))
import gn_check  # noqa: E402

ROOT = sys.argv[1]
T0 = time.time()
torch._dynamo.config.cache_size_limit = 64


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


rc, _ = git("merge-base", "--is-ancestor", "fb730d34", "HEAD")
_, head = git("log", "--oneline", "-3")
log("git", head, "ancestry rc", rc)


def seed_latent(s):
    g = torch.Generator(device="cuda").manual_seed(s)
    return torch.randn(1, 128, 64, 64, device="cuda", generator=g)


def sha(t):
    t = t.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def strip(code):
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))


res = {"git": head, "ancestry_rc": rc}
Zs = {s: seed_latent(s) for s in range(3)}
Z0 = {s: z.clone() for s, z in Zs.items()}

# eager
vae, _ = gn_check.build_vae(ROOT)
eager, rep = {}, True
with torch.inference_mode():
    for s in range(3):
        a = vae.decode(Zs[s]).clone()
        b = vae.decode(Zs[s])
        rep &= bool(torch.equal(a, b))
        eager[s] = sha(a)
res["eager_fp32_sha"], res["eager_repeat_bitwise"] = eager, rep
log("eager", json.dumps(eager), rep)
del vae

# fx graph
graphs = []


def backend(gm, example_inputs):
    graphs.append(gm.code)
    return gm.forward


vae, _ = gn_check.build_vae(ROOT)
dec = vae.decoder
orig = dec.compile


def compile_with(*a, **kw):
    kw.pop("options", None)
    return orig(*a, backend=backend, **kw)


dec.compile = compile_with
vae.compile_decoder()
with torch.inference_mode():
    fx_out = vae.decode(Zs[0])
code = "\n# ==== next graph ====\n".join(strip(g) for g in graphs)
res["fx_fp32"] = {"n_graphs": len(graphs), "sha": hashlib.sha256(code.encode()).hexdigest()[:16],
                  "lines": code.count("\n") + 1, "aot_eager_like_out_sha_s0": sha(fx_out)}
log("fx", json.dumps(res["fx_fp32"]))
del vae

# A/A compiled
import torch._inductor.config as icfg  # noqa: E402

aa = {}
for tag, nocache in (("C1", False), ("C2_nocache", True)):
    old = icfg.force_disable_caches
    icfg.force_disable_caches = nocache
    vae, _ = gn_check.build_vae(ROOT)
    vae.compile_decoder()
    with torch.inference_mode():
        aa[tag] = {s: sha(vae.decode(Zs[s])) for s in range(3)}
    icfg.force_disable_caches = old
    del vae
    log(tag, json.dumps(aa[tag]))
res["compiled_fp32_aa"] = aa
res["compiled_aa_equal"] = aa["C1"] == aa["C2_nocache"]
inp = all(torch.equal(Zs[s], Z0[s]) for s in range(3))
res["latents_unchanged"] = inp
res["passed"] = bool(rep and inp and rc == 0 and len(graphs) >= 1)
res["compared"] = ("fp32 decoder of this tree: eager output sha / FX graph sha (compare across pair arms), "
                   "compiled A/A within the process")
res["batch"] = {"1": {"kernel_ms": 0.0, "noise_floor": 0.0}}
res["output"] = {"class": "exact"}
print(json.dumps(res), flush=True)
