"""Idea 97 off-2048 shape check (evaluator round 2).

can_use() now requires NT = T // CFG_A['BN'] to be a power of two <= 64 or a multiple of 64 (kernel
B reduces the [T, NT] tile maxes in unmasked MPC = min(64, NT) chunks). For each shape this runs
eager Flux3ImageAttnBlock._attention_tc(fp8_pv=True) on channels_last bf16 q/k/v [1, 512, H, W]:
  rejected shapes : kernel census must show landing 25's path (no _qk_tilemax / _pv_prologue).
  accepted shapes : split path (B) vs landing 25's path (A, TLLM_FLUX3_VAE_SPLIT_ATTN off) within the
                    declared tolerance (<= 2 bf16 ulp of max|O|, >= 99.9% bitwise), m = max of the tile
                    maxes bitwise vs its own S's row max, kernel census B (A and B once, no softmax /
                    _scaled_mm), B repeat bitwise, inputs unchanged.
"""

import json
import math
import time
from collections import Counter

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

C = 512
T0 = time.time()
# (H, W): T, NT = T / 256
SHAPES = [
    (96, 96),  # T=9216, NT=36: rejected (not pow2, not mult of 64)
    (192, 192),  # T=36864, NT=144: rejected
    (72, 72),  # T=5184: rejected (T % 256 != 0)
    (64, 64),  # T=4096, NT=16: accepted (pow2 <= 64)
    (128, 128),  # T=16384, NT=64: accepted
    (256, 192),  # T=49152, NT=192: accepted (multiple of 64, not pow2: 3 MP chunks)
]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def inputs(h, w, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for amp in (1.5, 1.5, 1.0):
        x = (torch.randn(1, h, w, C, device="cuda", generator=g) * amp).to(torch.bfloat16)
        out.append(x.permute(0, 3, 1, 2))
    return out


def attn(q, k, v):
    return iv.Flux3ImageAttnBlock._attention_tc(q, k, v, True)


def census(fn, *a):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a)
        torch.cuda.synchronize()
    cnt = Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name[:90]] += 1
    return dict(cnt)


def _ord16(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def has(cen, sub):
    return sum(c for n, c in cen.items() if sub in n)


@torch.inference_mode()
def main():
    out, gates = {}, {}
    for h, w in SHAPES:
        t = h * w
        key = f"{h}x{w}_T{t}"
        qkv = inputs(h, w, 0)
        keep = [x.clone() for x in qkv]
        q2 = qkv[0].reshape(1, C, t)
        sa.ENABLED = True
        use = sa.can_use(q2)
        nt = t // sa.CFG_A["BN"] if t % sa.CFG_A["BN"] == 0 else None
        r = {"T": t, "NT": nt, "can_use": use}
        attn(*qkv)
        torch.cuda.synchronize()
        cen_b = census(attn, *qkv)
        r["census_enabled"] = cen_b
        if not use:
            g = has(cen_b, "qk_tilemax") == 0 and has(cen_b, "pv_prologue") == 0
            r["fallback_landing25"] = g
            r["has_softmax_rows_e4m3"] = has(cen_b, "_softmax_rows_e4m3") > 0
            gates[f"{key}_fallback"] = g
        else:
            y_b = attn(*qkv).clone()
            rep = all(torch.equal(attn(*qkv), y_b) for _ in range(5))
            sa.ENABLED = False
            y_a = attn(*qkv).clone()
            r["census_disabled"] = census(attn, *qkv)
            sa.ENABLED = True
            mx = y_a.float().abs().max().item()
            ulp = 2.0 ** (math.floor(math.log2(mx)) - 7)
            dd = (y_a.float() - y_b.float()).abs().max().item()
            u = (_ord16(y_a) - _ord16(y_b)).abs()
            share = (u == 0).float().mean().item()
            r["B_vs_A"] = {"max_abs": dd, "bound": 2 * ulp, "max|O|": mx, "bitwise_share": share,
                           "max_ulp": int(u.max().item())}
            qq = q2[0]
            kk = qkv[1].reshape(1, C, t)[0]
            s, mp = sa.qk_tilemax(qq, kk)
            m_ok = bool(torch.equal(mp.amax(dim=1).view(torch.int32), s.amax(dim=1).view(torch.int32)))
            del s, mp
            r["m_bitwise_vs_own_S"] = m_ok
            r["repeat_bitwise"] = rep
            gates[f"{key}_tol"] = dd <= 2 * ulp and share >= 0.999
            gates[f"{key}_m"] = m_ok
            gates[f"{key}_repeat"] = rep
            gates[f"{key}_census"] = (has(cen_b, "qk_tilemax") == 1 and has(cen_b, "pv_prologue") == 1
                                      and has(cen_b, "_softmax_rows_e4m3") == 0)
            del y_a, y_b
        gates[f"{key}_inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(qkv, keep))
        out[key] = r
        log(key, json.dumps(r))
        del qkv, keep
        torch.cuda.empty_cache()
    res = {"shapes": out, "gates": gates, "passed": bool(all(gates.values())),
           "output": {"class": "reordered"},
           "compared": "split path vs landing 25's path (TLLM_FLUX3_VAE_SPLIT_ATTN off) on the same "
                       "inputs; m vs own S; rejected shapes vs landing 25 census"}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
