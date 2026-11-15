"""Branch 2 probe: (1) dump flashinfer CuTe rmsnorm_fp4quant PTX (H from the
checkpoint) and summarize its sum-of-squares / predicate code; (2) count h,
FP4 payload and full swizzled-SF bytes where the fused kernel differs from
main's chain (rmsnorm_residual_add -> rmsnorm_fp4_quant) per SSQ mode, with
the checkpoint's real norm weights and gate_up input scales.

usage: mf2_ssq_probe.py <checkpoint> [modes=0,1,2] [reps=3]
"""
import glob
import json
import os
import re
import sys

ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/mf2_art")
DUMP = os.path.join(ART, "cute_dump")
os.makedirs(DUMP, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", DUMP)

import torch  # noqa: E402

CK = sys.argv[1]
MODES = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

cfg = json.load(open(os.path.join(CK, "config.json")))
tc = cfg.get("text_config", cfg)
H = int(tc["hidden_size"])
EPS = float(tc.get("rms_norm_eps", 1e-6))
NL = int(tc["num_hidden_layers"])
print(f"hidden={H} eps={EPS} layers={NL}", flush=True)

from safetensors import safe_open  # noqa: E402

idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]


def find(sub):
    ks = [k for k in idx if k.endswith(sub)]
    return ks


def load(key):
    with safe_open(os.path.join(CK, idx[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


LAYERS = sorted({0, 5, NL // 2, NL - 1})
W = {}
for li in LAYERS:
    pa = [k for k in find(f"layers.{li}.post_attention_layernorm.weight")]
    pf = [k for k in find(f"layers.{li}.pre_feedforward_layernorm.weight")]
    isc = [k for k in idx if f"layers.{li}.mlp." in k and k.endswith("input_scale")]
    wpa, wpf = load(pa[0]), load(pf[0])
    ivals = {k: load(k).float().item() for k in isc}
    iv = max(ivals.values())
    # TRT-LLM's input_scale convention for the quantizer is 448*6/amax; the
    # checkpoint stores amax/(448*6). Both chains get the same value.
    gs = 1.0 / iv if iv < 1.0 else iv
    print(f"layer {li}: {pa[0]} {wpa.dtype} mean={wpa.float().mean():.4f} | "
          f"{pf[0]} mean={wpf.float().mean():.4f} | input_scales={ivals} -> gs={gs:.4f}", flush=True)
    W[li] = (wpa.to(torch.bfloat16).cuda(), wpf.to(torch.bfloat16).cuda(),
             torch.tensor([gs], dtype=torch.float32, device="cuda"))

from tensorrt_llm._torch.kernels.fused_ops import rmsnorm_residual_add as rra  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant  # noqa: E402


def chain(x, r, w, w2, gs):
    h = rra.rmsnorm_residual_add(x, r, w, EPS)
    fp4, sf = rmsnorm_fp4_quant(h, w2, EPS, gs)
    return h, fp4, sf


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # heavy-tailed channels like a real residual stream
    chs = torch.exp(torch.randn(H, device="cuda", generator=g) * 1.0)
    x = (torch.randn(m, H, device="cuda", generator=g) * 4.0 * chs).to(torch.bfloat16)
    r = (torch.randn(m, H, device="cuda", generator=g) * 20.0 * chs.flip(0)).to(torch.bfloat16)
    return x, r


# ---- (1) CuTe PTX ----
x, r = make(256, 1)
wpa, wpf, gs = W[LAYERS[0]]
chain(x, r, wpa, wpf, gs)
torch.cuda.synchronize()
ptxs = []
for root in (DUMP, os.getcwd(), "/tmp"):
    ptxs += glob.glob(os.path.join(root, "**", "*.ptx"), recursive=True)
ptxs = sorted(set(p for p in ptxs if os.path.getsize(p) > 0), key=os.path.getmtime)
print("ptx files:", ptxs[-10:], flush=True)
summary = {}
for p in ptxs:
    txt = open(p).read()
    if "RMSNormFP4" not in txt and "rmsnorm" not in txt.lower():
        continue
    lines = txt.splitlines()
    cnt = {}
    for pat in ["fma.rn.f32", "add.f32", "add.rn.f32", "mul.f32", "mul.rn.f32", "div.rn.f32",
                "div.full.f32", "div.approx", "rsqrt.approx", "cp.async.ca", "cp.async.cg",
                "shfl.sync.bfly", "bar.sync", "st.shared", "ld.shared"]:
        cnt[pat] = sum(1 for ln in lines if pat in ln)
    summary[os.path.basename(p)] = cnt
    print("PTX", p, json.dumps(cnt), flush=True)
    # print the region from first cp.async to the first shfl (sum-of-squares)
    i0 = next((i for i, ln in enumerate(lines) if "cp.async" in ln), 0)
    i1 = next((i for i, ln in enumerate(lines) if "shfl.sync" in ln), i0 + 300)
    print("----- PTX cp.async .. first shfl (%d..%d) -----" % (i0, i1), flush=True)
    for ln in lines[max(0, i0 - 20): min(len(lines), i1 + 40)]:
        print(ln)
    i2 = next((i for i, ln in enumerate(lines) if "rsqrt" in ln), i1)
    print("----- PTX around rsqrt -----")
    for ln in lines[max(0, i2 - 30): i2 + 5]:
        print(ln)
    break

# ---- (2) byte diffs per mode ----
MS = [1, 27, 300, 1000, 4800, 8000, 8192]
res = {}
for mode in MODES:
    rra._FP4_SSQ_MODE = mode
    tot = {"h": 0, "fp4": 0, "sf": 0, "elems": 0}
    per = {}
    for m in MS:
        d = [0, 0, 0]
        for rep in range(REPS):
            for li in LAYERS:
                wpa, wpf, gs = W[li]
                x, r = make(m, 1000 * rep + m + li)
                hb, qb, sb = chain(x, r, wpa, wpf, gs)
                hc, qc, sc = rra.rmsnorm_residual_add_fp4_quant(x, r, wpa, EPS, wpf, EPS, gs)
                assert sc.shape == sb.shape and qc.shape == qb.shape
                d[0] += (hc.view(torch.int16) != hb.view(torch.int16)).sum().item()
                d[1] += (qc != qb).sum().item()
                d[2] += (sc != sb).sum().item()
                tot["elems"] += m * H
        per[m] = d
        tot["h"] += d[0]
        tot["fp4"] += d[1]
        tot["sf"] += d[2]
        print(f"mode {mode} M={m}: h_diff={d[0]} fp4_byte_diff={d[1]} sf_byte_diff={d[2]}", flush=True)
    res[mode] = {"total": tot, "per_m": per}
print(json.dumps({"passed": True, "ptx": summary, "diffs": res}))
