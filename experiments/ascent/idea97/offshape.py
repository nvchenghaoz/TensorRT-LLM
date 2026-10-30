"""Idea 97 guard check (evaluator round 2, area lead / top lead 05:00 rulings for branch 62).

can_use() now also requires NT = T // 256 to be a power of two <= 64 or a multiple of 64 (kernel B
reduces the [T, NT] tile maxes in unmasked MPC = min(64, NT) chunks), c == 512 and bf16 q. The
kernel bodies and pinned configs are unchanged from 390ee0a8.

(1) Shape sweep, eager Flux3ImageAttnBlock._attention_tc(fp8_pv=True), seeded channels_last bf16
    q/k/v [1, 512, H, W], pristine copies, the allocator poisoned with NaN before every call.
    Per grid: can_use, kernel path (profiler names).
    accepted: split (TLLM_FLUX3_VAE_SPLIT_ATTN on) vs the SPLIT_ATTN=0 arm (landing 25) in the same
              process within the declared tolerance (<= 2 bf16 ulp of max|O|, >= 99.9% bitwise),
              outputs finite, m (max of tile maxes) bitwise vs its own S, census A/B once.
    rejected: runs, census = landing 25's path, bitwise equal to the SPLIT_ATTN=0 arm, finite.
(2) Served shape 256x256 (T=65536): pre-fix guard (390ee0a8's can_use, copied below) and the new
    guard both True; the dispatched output is bitwise equal to a direct launch of the unchanged
    kernels; 30 repeats bitwise; short interleaved eager timing vs SPLIT_ATTN=0 (>= 50 iters per
    arm, cold L2, A/A noise) for information (the submitted pair stays 768/769).
"""

import json
import math
import os
import random
import time
from collections import Counter

import torch

from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import image_vae as iv
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.image_vae import split_attn_triton as sa

C = 512
T0 = time.time()
ACCEPT = [(256, 256), (128, 128), (64, 64), (256, 192)]
REJECT = [(96, 96), (144, 144), (160, 160), (192, 192), (125, 125)]
ITERS = int(os.environ.get("ITERS", "50"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def old_can_use(q):
    """can_use() as of 390ee0a8 (pre-fix), for the served-shape dispatch comparison."""
    if not (sa.ENABLED and q.is_cuda and q.dim() == 3 and q.shape[0] == 1):
        return False
    c, t = q.shape[1], q.shape[2]
    return (t % max(sa.CFG_A["BM"], sa.CFG_A["BN"], sa.CFG_B["BM"], sa.CFG_B["BK"]) == 0
            and c % sa.CFG_A["BK"] == 0 and c % sa.CFG_B["BN"] == 0 and t >= sa.CFG_A["BN"])


def inputs(h, w, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for amp in (1.5, 1.5, 1.0):
        x = (torch.randn(1, h, w, C, device="cuda", generator=g) * amp).to(torch.bfloat16)
        out.append(x.permute(0, 3, 1, 2))
    return out


def poison(t):
    # NaN-fill free allocator blocks of the sizes the op allocates (O, S, pq), then release them.
    junk = [torch.full((t, C), float("nan"), device="cuda", dtype=torch.bfloat16),
            torch.full((t, C), float("nan"), device="cuda", dtype=torch.float32)]
    del junk


def attn(q, k, v):
    return iv.Flux3ImageAttnBlock._attention_tc(q, k, v, True)


def call(arm_on, qkv):
    sa.ENABLED = arm_on
    poison(qkv[0].shape[2] * qkv[0].shape[3])
    y = attn(*qkv).clone()
    torch.cuda.synchronize()
    sa.ENABLED = True
    return y


def census(arm_on, qkv):
    from torch.profiler import ProfilerActivity, profile

    sa.ENABLED = arm_on
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        attn(*qkv)
        torch.cuda.synchronize()
    sa.ENABLED = True
    cnt = Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[ev.name[:80]] += 1
    return dict(cnt)


def has(cen, sub):
    return sum(c for n, c in cen.items() if sub in n)


def _ord16(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


FLUSH = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)


def timed(arm_on, qkv):
    sa.ENABLED = arm_on
    FLUSH.fill_(1.0)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    attn(*qkv)
    e.record()
    e.synchronize()
    sa.ENABLED = True
    return s.elapsed_time(e)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2]
    p10, p90 = ts[int(0.1 * n)], ts[max(int(0.9 * n) - 1, 0)]
    return {"median_ms": round(med, 4), "p10_ms": round(p10, 4), "p90_ms": round(p90, 4),
            "noise_floor": round((p90 - p10) / med, 5), "n": n}


@torch.inference_mode()
def main():
    out, gates = {}, {}
    for (h, w), expect in [(s, True) for s in ACCEPT] + [(s, False) for s in REJECT]:
        t = h * w
        key = f"{h}x{w}"
        qkv = inputs(h, w, 0)
        keep = [x.clone() for x in qkv]
        q2 = qkv[0].reshape(1, C, t)
        use = sa.can_use(q2)
        r = {"T": t, "NT": t / 256, "can_use": use, "old_can_use": old_can_use(q2)}
        gates[f"{key}_can_use_as_expected"] = use == expect
        y_on = call(True, qkv)
        y_off = call(False, qkv)
        r["census_on"] = census(True, qkv)
        r["census_off"] = census(False, qkv)
        fin = bool(torch.isfinite(y_on).all().item() and torch.isfinite(y_off).all().item())
        r["finite"] = fin
        gates[f"{key}_finite"] = fin
        if expect:
            mx = y_off.float().abs().max().item()
            ulp = 2.0 ** (math.floor(math.log2(mx)) - 7)
            dd = (y_off.float() - y_on.float()).abs().max().item()
            u = (_ord16(y_off) - _ord16(y_on)).abs()
            share = (u == 0).float().mean().item()
            r["on_vs_off"] = {"max_abs": dd, "bound": 2 * ulp, "max|O|": mx, "bitwise_share": share,
                              "max_ulp": int(u.max().item())}
            gates[f"{key}_tol"] = dd <= 2 * ulp and share >= 0.999
            gates[f"{key}_census_split"] = (has(r["census_on"], "qk_tilemax") == 1
                                            and has(r["census_on"], "pv_prologue") == 1
                                            and has(r["census_on"], "_softmax_rows_e4m3") == 0)
            qq, kk = q2[0], qkv[1].reshape(1, C, t)[0]
            s, mp = sa.qk_tilemax(qq, kk)
            gates[f"{key}_m_bitwise_vs_own_S"] = bool(
                torch.equal(mp.amax(dim=1).view(torch.int32), s.amax(dim=1).view(torch.int32)))
            del s, mp
            if (h, w) == (256, 256):
                # (2) served shape: dispatch identical to 390ee0a8, output == direct launch
                gates["served_old_and_new_guard_true"] = bool(use and r["old_can_use"])
                v = qkv[2].reshape(1, C, t)[0].float()
                s_v = (v.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-30)
                vq = (v / s_v).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
                d = sa.split_attn_e4m3_direct(qq, kk, vq, s_v.reshape(C), C**-0.5)
                y_dir = d.t().reshape(1, C, h, w)
                gates["served_dispatch_eq_direct_kernels"] = bool(torch.equal(y_dir, y_on))
                del d, v, vq
                rep = True
                for i in range(30):
                    rep &= bool(torch.equal(call(True, qkv), y_on))
                gates["served_30_repeats_bitwise"] = rep
                rnd = random.Random(0)
                ts = {"off": [], "on": [], "off2": []}
                for _ in range(3):
                    timed(True, qkv), timed(False, qkv)
                for i in range(ITERS):
                    order = ["off", "on", "off2"]
                    rnd.shuffle(order)
                    for n in order:
                        ts[n].append(timed(n == "on", qkv))
                    if i % 10 == 0:
                        log(f"timing iter {i}")
                tim = {n: stats(v) for n, v in ts.items()}
                tim["aa_rel"] = round(tim["off2"]["median_ms"] / tim["off"]["median_ms"] - 1, 5)
                tim["on_vs_off_rel"] = round(tim["on"]["median_ms"] / tim["off"]["median_ms"] - 1, 5)
                r["timing_eager_info"] = tim
        else:
            gates[f"{key}_bitwise_vs_off"] = bool(torch.equal(y_on, y_off))
            gates[f"{key}_census_landing25"] = (has(r["census_on"], "qk_tilemax") == 0
                                                and has(r["census_on"], "pv_prologue") == 0
                                                and r["census_on"] == r["census_off"])
        gates[f"{key}_inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(qkv, keep))
        out[key] = r
        log(key, json.dumps(r))
        del qkv, keep, y_on, y_off
        torch.cuda.empty_cache()
    res = {"shapes": out, "gates": gates, "passed": bool(all(gates.values())),
           "output": {"class": "reordered"},
           "compared": "split path vs landing 25's path (TLLM_FLUX3_VAE_SPLIT_ATTN off) on the same "
                       "inputs; m vs own S; rejected shapes bitwise vs landing 25; served dispatch vs "
                       "direct launch of the unchanged 390ee0a8 kernels"}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
