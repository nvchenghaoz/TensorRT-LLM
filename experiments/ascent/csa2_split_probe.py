"""Probe DeepGEMM fp8_fp4_mqa_logits (SM120) for key-split replicated decode rows.

For n decode rows with per-row keys [n*W] and splits S padded to P rows per split,
compares the assembled [n, W] logits bitwise against one per-row call
(ks=0..ke=W at the row's base, clean_logits=True), and times both under CUDA
graph replay with an L2 flush before every replay.
"""

import inspect
import json
import sys

import torch

from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits

H, D2 = 32, 64


def log(*a):
    print(*a, flush=True)


def make(n, W, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(-128, 128, (n, H, D2), generator=g, dtype=torch.int32).to(torch.int8)
    qs = torch.randint(122, 129, (n, H, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    qs = qs.view(torch.int32).reshape(n, H)
    w = torch.randn((n, H), generator=g)
    k = torch.randint(-128, 128, (n * W, D2), generator=g, dtype=torch.int32).to(torch.int8)
    ks = torch.randint(118, 131, (n * W, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    ks = ks.view(torch.int32).reshape(-1)
    return [t.cuda() for t in (q, qs, w, k, ks)]


FLUSH = None


def timed(fn, iters=30):
    global FLUSH
    if FLUSH is None:
        FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = fn()
    torch.cuda.synchronize()
    ts = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        FLUSH.zero_()
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2], (ts[int(0.9 * iters)] - ts[int(0.1 * iters)]) / ts[len(ts) // 2], out, g


def main():
    log("device", torch.cuda.get_device_name(0))
    try:
        log("signature", str(inspect.signature(fp8_fp4_mqa_logits)))
    except Exception as e:  # pybind
        log("sig err", e)
    log("doc", (fp8_fp4_mqa_logits.__doc__ or "")[:2000])
    res = {}
    for n, W in ((1, 36864), (1, 18432), (1, 16384), (4, 36864), (4, 16384), (8, 18432)):
        q, qs, w, k, kscale = make(n, W, 100 + n + W)
        base = torch.arange(n, dtype=torch.int32, device="cuda") * W

        def ref_fn():
            outs = []
            for r in range(n):
                ks_ = base[r : r + 1]
                outs.append(
                    fp8_fp4_mqa_logits((q[r : r + 1], qs[r : r + 1]), (k, kscale), w[r : r + 1],
                                       ks_, ks_ + W, clean_logits=True)
                )
            return outs

        t_ref, nz_ref, outs, g = timed(ref_fn)
        torch.cuda.synchronize()
        log(f"n={n} W={W} ref out shape {tuple(outs[0].shape)} stride {outs[0].stride()} "
            f"dtype {outs[0].dtype}")
        ref = torch.stack([o[0, r * W : r * W + W] for r, o in enumerate(outs)])
        del g
        res[f"{n}x{W}"] = {"ref_us": t_ref, "ref_noise": nz_ref}
        log(f"  ref per-row {n} calls: {t_ref:.1f} us noise {nz_ref:.3f}")
        for S in (4, 8, 12, 16, 24, 32, 48, 64):
            if W % S or (W // S) % 64:
                continue
            if n * S > 400:
                continue
            chunk = W // S
            for P in (1, 2, 4, 8):
                rows = n * S * P
                if rows * n * W * 4 > (1 << 30):
                    continue
                rep = torch.arange(n, device="cuda").repeat_interleave(S * P)
                split = torch.arange(S, device="cuda").repeat_interleave(P).repeat(n)
                ks_ = (base.long()[rep] + split * chunk).int()
                ke_ = ks_ + chunk
                qr, qsr, wr = q[rep].contiguous(), qs[rep].contiguous(), w[rep].contiguous()

                def new_fn():
                    out = fp8_fp4_mqa_logits((qr, qsr), (k, kscale), wr, ks_, ke_,
                                             clean_logits=False)
                    stride = out.stride(0)
                    view = out.as_strided((n, S, chunk), (S * P * stride + W, P * stride + chunk, 1))
                    return view.reshape(n, W)

                try:
                    t, nz, got, g = timed(new_fn)
                except Exception as e:
                    log(f"  S={S} P={P} error {e}")
                    continue
                torch.cuda.synchronize()
                eq = torch.equal(got.view(torch.int32), ref.view(torch.int32))
                log(f"  S={S:3d} P={P} rows={rows} chunk={chunk}: {t:.1f} us noise {nz:.3f} "
                    f"bitwise={eq}")
                res[f"{n}x{W}"][f"S{S}P{P}"] = {"us": t, "eq": eq}
                del g
        # compressed output probe
        try:
            out = fp8_fp4_mqa_logits((q[:1], qs[:1]), (k, kscale), w[:1], base[:1], base[:1] + W,
                                     clean_logits=False, max_seqlen_k=1024)
            log("  max_seqlen_k supported: shape", tuple(out.shape))
        except Exception as e:
            log("  max_seqlen_k unsupported:", str(e)[:200])
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(main())
