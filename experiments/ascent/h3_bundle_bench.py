"""H3 elementwise bundle microbenchmark (TP2 per-rank shapes, BF16 Linears, 1 GPU, synthetic tensors).

Times, in whichever tree runs it:
  boundary: base = compiled MLP gated residual (Inductor poi) + h3_norm_mod (next block's norm1);
            cand = h3_prev_gate_res_norm_mod (one kernel).
  tail:     base = compiled residual + eager norm_out (flashinfer RMSNorm, index_select, 1+scale, mul,
            add) + the two .to(fp32) the projections take;
            cand = h3_gate_res + the same RMSNorm + h3_mod_out (FP32 out; the casts become no-ops).
  qk:       trtllm::minimax_h3_qk_norm_rope with the tree's default launch.
kernel_ms = 49 * boundary + tail + 50 * qk (one denoise forward's worth of these ops).
Last stdout line: evidence JSON. Also writes per-op medians and output digests.
"""

import json
import os
import sys

import torch

from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln as fa
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_rope  # noqa: F401  (registers op)

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
M = int(os.environ.get("H3_M", "37966"))
NT = 2  # timesteps in the AdaLN table (rows = NT * 3 modalities)
WARM, ITERS = 5, 30
NEW = hasattr(fa, "h3_prev_gate_res_norm_mod")


def cfg():
    c = {"hidden_size": 5376, "num_attention_heads": 56, "attention_head_dim": 128, "rope_freq_dim": 16,
         "norm_eps": 1e-6, "final_norm_eps": 1e-6, "qk_norm_eps": 1e-6}
    if CKPT:
        with open(os.path.join(CKPT, "transformer", "config.json")) as f:
            c.update(json.load(f))
    return c


def make_inputs(c, m=M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    D = c["hidden_size"]
    dev = "cuda"
    bf = torch.bfloat16

    def rn(*s, scale=1.0):
        return (torch.randn(*s, generator=g, device=dev) * scale).to(bf)

    t = {}
    t["x"] = rn(1, m, D, scale=4.0)
    t["ff"] = rn(1, m, D, scale=2.0)
    t["mod_prev"] = rn(NT * 3, 6 * D, scale=0.5)
    t["mod"] = rn(NT * 3, 6 * D, scale=0.5)
    tidx = torch.randint(0, NT, (m,), generator=g, device=dev)
    tags = torch.randint(0, 3, (m,), generator=g, device=dev)
    t["tidx"] = tidx
    t["idx"] = tidx * 3 + tags
    t["w1"] = (1 + 0.1 * torch.randn(D, generator=g, device=dev)).to(bf)
    t["wout"] = (1 + 0.1 * torch.randn(D, generator=g, device=dev)).to(bf)
    t["out_table"] = rn(NT, 2 * D, scale=0.5)
    return t


def compiled_residual():
    def res(x, mod_prev, idx, ff):
        D = x.shape[-1]
        return x + mod_prev[:, 5 * D : 6 * D].index_select(0, idx) * ff

    return torch.compile(res, dynamic=False)


def make_ops(c, t):
    D = c["hidden_size"]
    eps = c["norm_eps"]
    norm = RMSNorm(hidden_size=D, eps=c["final_norm_eps"], dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        norm.weight.copy_(t["wout"])
    res = compiled_residual()

    def boundary_old():
        x1 = res(t["x"], t["mod_prev"], t["idx"], t["ff"])
        m = fa.h3_norm_mod(x1, t["w1"], t["mod"], t["idx"], eps, 1, 0)
        return x1, m

    def tail_old():
        x1 = res(t["x"], t["mod_prev"], t["idx"], t["ff"])
        shift, scale = t["out_table"].chunk(2, dim=-1)
        n = norm(x1.reshape(-1, D)).view(x1.shape)
        o = n * (1.0 + scale.index_select(0, t["tidx"])) + shift.index_select(0, t["tidx"])
        a = o.to(torch.float32)
        b = o.to(torch.float32)
        return a, b

    ops = {"boundary_old": boundary_old, "tail_old": tail_old}
    if NEW:

        def boundary_new():
            return fa.h3_prev_gate_res_norm_mod(
                t["x"], t["ff"], t["w1"], t["mod_prev"], t["mod"], t["idx"], eps, 5, 1, 0
            )

        def tail_new():
            x1 = fa.h3_gate_res(t["x"], t["ff"], t["mod_prev"], t["idx"], 5)
            n = norm(x1.reshape(-1, D)).view(x1.shape)
            o = fa.h3_mod_out(n, t["out_table"], t["tidx"], 1, 0, torch.float32)
            a = o.to(torch.float32)
            b = o.to(torch.float32)
            return a, b

        ops["boundary_new"] = boundary_new
        ops["tail_new"] = tail_new
    return ops


def make_qk(c, m=M):
    H = c["num_attention_heads"] // 2
    hd = c["attention_head_dim"]
    rfd = c["rope_freq_dim"]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    qkv = torch.randn(1, m, 3 * H * hd, generator=g, device=dev).to(torch.bfloat16)
    wq = (1 + 0.1 * torch.randn(hd, generator=g, device=dev)).to(torch.bfloat16)
    wk = (1 + 0.1 * torch.randn(hd, generator=g, device=dev)).to(torch.bfloat16)
    ar = torch.arange(m, device=dev)
    pos = torch.stack([ar // 4096, (ar // 64) % 64, ar % 64], -1)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 2 * rfd, 2, dtype=torch.float32, device=dev) / (2 * rfd)))
    fq = pos.float().unsqueeze(-1) * inv.view(1, 1, -1)
    fq = torch.cat(fq.unbind(dim=1), dim=-1)
    fq = torch.cat((fq, fq), dim=-1)
    cos, sin = fq.cos(), fq.sin()
    eps = c["qk_norm_eps"]
    tens = {"qkv": qkv, "wq": wq, "wk": wk, "cos": cos, "sin": sin}

    def qk():
        return torch.ops.trtllm.minimax_h3_qk_norm_rope(qkv, wq, wk, cos, sin, eps, H, hd)

    return qk, tens


def time_fn(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, (ts[int(0.9 * len(ts))] - ts[int(0.1 * len(ts))]) / med


def main():
    c = cfg()
    print(f"tree={'new' if NEW else 'base'} D={c['hidden_size']} M={M} cap={torch.cuda.get_device_capability()}", flush=True)
    torch.manual_seed(0)
    with torch.inference_mode():
        t = make_inputs(c)
        ops = make_ops(c, t)
        qk, _ = make_qk(c)
        arm = "new" if NEW else "old"
        r = {}
        for name in ("boundary_old", "tail_old") + (("boundary_new", "tail_new") if NEW else ()):
            r[name] = time_fn(ops[name])
            print(name, r[name], flush=True)
        r["qk"] = time_fn(qk)
        print("qk", r["qk"], flush=True)
        b, tl_, q = r[f"boundary_{arm}"], r[f"tail_{arm}"], r["qk"]
        total = 49 * b[0] + tl_[0] + 50 * q[0]
        noise = max(b[1], tl_[1], q[1])
    out = {
        "batch": {"1": {"kernel_ms": round(total, 4), "noise_floor": round(noise, 4)}},
        "output": {"class": "exact"},
        "assumes": "BF16 Linears, TP2 rank shapes (D=5376, 28 local heads), M=%d" % M,
        "parts_ms": {k: round(v[0], 4) for k, v in r.items()},
        "parts_spread": {k: round(v[1], 4) for k, v in r.items()},
        "arm": arm,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
