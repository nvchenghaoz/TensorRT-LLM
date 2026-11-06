"""Branch 217 (idea 248): V4.1 final mHC collapse (_finalize_hc_state) as one exact Triton launch.

Usage:
  python final_hc_collapse_bench.py bench   # cold-L2 CUDA-graph chain, M = 1/2/4/8
  python final_hc_collapse_bench.py check   # bitwise checks + repeated graph sequence (candidate tree)

The chain is what main runs once per decode step at the end of the decoder stack:
  deferred HCState -> DeepseekV41Model._finalize_hc_state (mhc_post_mapping as main, then the collapse)
  -> final RMSNorm (tensorrt_llm RMSNorm module, as DeepseekV4LogitsProcessor).
Each tree runs its own _finalize_hc_state, so the base arm times main's mHC.collapse (12 ATen launches)
and the candidate arm the Triton kernel. No model load; dims come from the checkpoint config.json.
"""

import functools
import glob
import json
import os
import sys
import time
import types
from types import SimpleNamespace

import torch

CKPT = os.environ.get(
    "CKPT",
    "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/models/DeepSeek-V4.1-Flash-1017",
)
MS = (1, 2, 4, 8)
WARMUP = 20
ITERS = 300


def log(*a):
    print(*a, flush=True)


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def dims():
    with open(os.path.join(CKPT, "config.json")) as f:
        cfg = json.load(f)
    hidden = _find(cfg, "hidden_size")
    mult = _find(cfg, "hc_mult")
    eps = _find(cfg, "rms_norm_eps") or 1e-6
    log(f"[dims] hidden_size={hidden} hc_mult={mult} rms_norm_eps={eps}")
    assert hidden == 5120, hidden
    assert mult == 4, mult
    return hidden, mult, float(eps)


def setup():
    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Model
    from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState, mHC

    fake = SimpleNamespace()
    fake.layers = [
        SimpleNamespace(
            hc_attn=SimpleNamespace(post_mapping=functools.partial(mHC.post_mapping, SimpleNamespace(mult=4)))
        )
    ]
    fake._resolve_hc_state = types.MethodType(DeepseekV41Model._resolve_hc_state, fake)

    def finalize(st):
        return DeepseekV41Model._finalize_hc_state(fake, st)

    return finalize, HCState, mHC


def make_state(HCState, M, mult, hidden, gen, deferred=True, pre=None):
    dev = "cuda"
    residual = (torch.randn(M, mult, hidden, generator=gen, device=dev) * 2.0).to(torch.bfloat16)
    if pre is None:
        pre = torch.sigmoid(torch.randn(M, mult, generator=gen, device=dev)) + 1e-4
    pre = pre.contiguous().view(M, mult, 1)
    if not deferred:
        return HCState.resolved(residual, pre_mix=pre)
    x_prev = (torch.randn(M, hidden, generator=gen, device=dev)).to(torch.bfloat16)
    post = (2.0 * torch.sigmoid(torch.randn(M, mult, generator=gen, device=dev))).view(M, mult, 1)
    comb = torch.softmax(torch.randn(M, mult, mult, generator=gen, device=dev), dim=-1).contiguous()
    return HCState.deferred(residual, post, comb, x_prev, pre_mix=pre)


def sm_clock():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def bench():
    hidden, mult, eps = dims()
    finalize, HCState, mHC = setup()
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_((1.0 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16))
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
    # Elementwise dummy capture first (pitfall 2603).
    dummy = torch.zeros(16, device="cuda")
    gd = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(gd):
            dummy.add_(1.0)
    torch.cuda.synchronize()

    result = {}
    clocks = []
    for M in MS:
        st = make_state(HCState, M, mult, hidden, gen, deferred=True)

        def chain():
            return norm(finalize(st))

        # Eager warmup (JIT) outside capture, like the CUDA-graph runner's warmup.
        for _ in range(3):
            chain()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = chain()
        torch.cuda.synchronize()
        ref = chain().clone()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), "graph output differs from eager"
        for _ in range(WARMUP):
            flush.zero_()
            g.replay()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            flush.zero_()  # cold L2 (512 MB > 128 MB L2)
            starts[i].record()
            g.replay()
            ends[i].record()
        torch.cuda.synchronize()
        clocks.append(sm_clock())
        ts = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
        med = ts[len(ts) // 2]
        p10 = ts[int(0.1 * len(ts))]
        p90 = ts[int(0.9 * len(ts))]
        nf = (p90 - p10) / med
        log(f"[bench] M={M} chain median={med * 1e3:.2f} us p10={p10 * 1e3:.2f} p90={p90 * 1e3:.2f} noise={nf:.4f}")
        result[str(M)] = {"kernel_ms": med, "noise_floor": nf, "p10_ms": p10, "p90_ms": p90}
        del g
    log(f"[bench] sm_clock_mhz after each M: {clocks}")
    print(
        json.dumps(
            {
                "batch": result,
                "output": {"class": "exact"},
                "chain": "deferred HCState -> _finalize_hc_state (mhc_post_mapping + collapse) -> RMSNorm, CUDA graph, cold L2",
                "hidden": hidden,
                "hc_mult": mult,
                "sm_clock_mhz": clocks,
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def _err(a, b):
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp_min(1e-30)
    return float(d.max()), float(rel.max())


def real_rows(n_rows, hidden):
    """Real-checkpoint bf16 rows (the embedding table) to use as residual streams."""
    from safetensors import safe_open

    idx = os.path.join(CKPT, "model.safetensors.index.json")
    wm = {}
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
    else:
        for fn in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
            with safe_open(fn, "pt") as f:
                for k in f.keys():
                    wm[k] = os.path.basename(fn)
    cands = [k for k in wm if "embed" in k and k.endswith("weight")]
    cands.sort(key=lambda k: ("engram" in k, len(k)))
    log(f"[check] embed candidates: {cands[:6]}")
    for key in cands:
        with safe_open(os.path.join(CKPT, wm[key]), "pt") as f:
            sl = f.get_slice(key)
            shape = sl.get_shape()
            if len(shape) != 2 or shape[-1] != hidden:
                continue
            rows = sl[1000 : 1000 + n_rows].to(torch.bfloat16)
        log(f"[check] real rows from {key} {shape} in {wm[key]} dtype->bf16")
        return rows.cuda()
    raise RuntimeError("no [vocab, hidden] embedding tensor found in the checkpoint")


def check():
    hidden, mult, eps = dims()
    finalize, HCState, mHC = setup()
    from tensorrt_llm._torch.modules.mhc import mhc_final_collapse as fc
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mhc_post_mapping_cuda

    gen = torch.Generator(device="cuda").manual_seed(7)
    failures = []
    report = {}

    def main_ref(st):
        """Main's _finalize_hc_state: post_mapping (if deferred) then mHC.collapse."""
        r = st.residual
        if st.is_deferred:
            Mx = r.shape[0]
            r = mhc_post_mapping_cuda(r, st.x_prev.reshape(Mx, hidden), st.post_mix.view(Mx, mult),
                                      st.comb_mix.view(Mx, mult, mult), mult).view(Mx, mult, hidden)
        return mHC.collapse(r, st.pre_mix)

    def pre_variants(M):
        d = "cuda"
        oh = torch.zeros(M, mult, device=d)
        oh[:, 0] = 1.0
        big = torch.tensor([1e30, -3.0e-38, 7.5e-41, -1e20], device=d).repeat(M, 1)  # incl. fp32 denormal
        return {
            "sigmoid": torch.sigmoid(torch.randn(M, mult, generator=gen, device=d)) + 1e-4,
            "onehot_seed": oh,
            "negative": -torch.rand(M, mult, generator=gen, device=d) * 3.7,
            "mixed_sign_npot": torch.tensor([0.3, -1.7, 2.9, 1.0 / 3.0], device=d).repeat(M, 1),
            "large_dynamic_range": big,
            "huge_small": torch.exp((torch.randn(M, mult, generator=gen, device=d) * 20.0).clamp(-80.0, 80.0)),
        }

    real = real_rows(8 * mult + 8, hidden)
    # 1) Through _finalize_hc_state: deferred and resolved, random and real residuals, every pre variant.
    for M in MS:
        for pname, pre in pre_variants(M).items():
            for deferred in (True, False):
                for src in ("random", "real", "real_seed"):
                    st = make_state(HCState, M, mult, hidden, gen, deferred=deferred, pre=pre.clone())
                    if src == "real":
                        st.residual.copy_(real[: M * mult].view(M, mult, hidden))
                        if deferred:
                            st.x_prev.copy_(real[M * mult : M * mult + M])
                    elif src == "real_seed":  # _init_hc_state seed: 4 identical copies of an embedding row
                        st.residual.copy_(real[:M].unsqueeze(1).repeat(1, mult, 1))
                    if not fc.final_collapse_supported(
                        main_ref_res(st, mhc_post_mapping_cuda, mult, hidden), st.pre_mix
                    ):
                        failures.append(f"gate off M={M} deferred={deferred}")
                    got = finalize(st)
                    ref = main_ref(st)
                    ok = _bits_equal(got, ref)
                    if not ok:
                        failures.append(f"finalize M={M} pre={pname} deferred={deferred} src={src} err={_err(got, ref)}")
    log(f"[check] finalize matrix done, failures so far {len(failures)}")

    # 2) Kernel directly: hidden not a multiple of BLOCK, strided pre_mix, nonfinite-free extremes.
    for hid in (5120, 4000, 4104, 1000):
        for M in MS:
            x = (torch.randn(M, mult, hid, generator=gen, device="cuda") * 50).to(torch.bfloat16)
            base = torch.randn(M, 3 * mult + 8, generator=gen, device="cuda")
            pre_strided = base[:, 2 : 2 + mult].unsqueeze(-1)  # non-contiguous [M, 4, 1]
            for pre in (pre_strided, pre_strided.contiguous()):
                got = fc.mhc_final_collapse(x, pre)
                ref = mHC.collapse(x, pre)
                if not _bits_equal(got, ref):
                    failures.append(f"direct hid={hid} M={M} contig={pre.is_contiguous()} err={_err(got, ref)}")
    log(f"[check] direct kernel done, failures so far {len(failures)}")

    # 3) Gate falls back (keeps mHC.collapse) outside the decode shape.
    x16 = torch.randn(16, mult, hidden, device="cuda").to(torch.bfloat16)
    p16 = torch.rand(16, mult, 1, device="cuda")
    gate = {
        "M16": fc.final_collapse_supported(x16, p16),
        "noncontig_x": fc.final_collapse_supported(x16[:4].transpose(0, 1).contiguous().transpose(0, 1), p16[:4]),
        "fp32_x": fc.final_collapse_supported(x16[:4].float(), p16[:4]),
        "mult2": fc.final_collapse_supported(x16[:4, :2], p16[:4, :2]),
        "M4_ok": fc.final_collapse_supported(x16[:4].contiguous(), p16[:4]),
    }
    log(f"[check] gate {gate}")
    if gate["M16"] or gate["noncontig_x"] or gate["fp32_x"] or gate["mult2"] or not gate["M4_ok"]:
        failures.append(f"gate wrong {gate}")
    for M in (16, 64):
        st = make_state(HCState, M, mult, hidden, gen, deferred=True)
        if not _bits_equal(finalize(st), main_ref(st)):
            failures.append(f"fallback M={M} differs")

    # 4) Repeated graph sequence (same warmup / iteration counts as bench), NaN-poisoned output before each
    #    replay, deferred and resolved, fixed inputs and changing inputs (copied into static buffers).
    seq = {}
    for M in MS:
        for deferred in (True, False):
            for changing in (False, True):
                st = make_state(HCState, M, mult, hidden, gen, deferred=deferred)
                for _ in range(3):
                    finalize(st)  # eager warmup (JIT) outside capture
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    # Same compile key as the post_mapping output (fresh, aligned, contiguous).
                    in_capture_gate = fc.final_collapse_supported(st.residual, st.pre_mix)
                    out = finalize(st)
                torch.cuda.synchronize()
                if not in_capture_gate:
                    failures.append(f"gate off inside capture M={M}")
                max_abs, max_rel, bad = 0.0, 0.0, 0
                for it in range(WARMUP + ITERS):
                    if changing:
                        nst = make_state(HCState, M, mult, hidden, gen, deferred=deferred)
                        st.residual.copy_(nst.residual)
                        st.pre_mix.copy_(nst.pre_mix)
                        if deferred:
                            st.x_prev.copy_(nst.x_prev)
                            st.post_mix.copy_(nst.post_mix)
                            st.comb_mix.copy_(nst.comb_mix)
                    out.fill_(float("nan"))
                    g.replay()
                    ref = main_ref(st)
                    torch.cuda.synchronize()
                    if not _bits_equal(out, ref):
                        bad += 1
                    a, r = _err(out, ref)
                    if a != a:  # NaN left in out
                        bad += 1
                        a = float("inf")
                    max_abs, max_rel = max(max_abs, a), max(max_rel, r)
                key = f"M{M}_{'deferred' if deferred else 'resolved'}_{'changing' if changing else 'fixed'}"
                seq[key] = {"iters": WARMUP + ITERS, "mismatch_iters": bad, "max_abs": max_abs, "max_rel": max_rel}
                log(f"[check] seq {key}: {seq[key]}")
                if bad:
                    failures.append(f"seq {key} mismatches {bad}")
                del g

    # 5) Kill switch: with TRTLLM_FUSED_FINAL_HC=0 the gate is off (module re-read in a subprocess).
    import subprocess

    ks = subprocess.run(
        [sys.executable, "-c",
         "from tensorrt_llm._torch.modules.mhc import mhc_final_collapse as fc; print(fc.FUSED_FINAL_HC_ENV_ON)"],
        env={**os.environ, "TRTLLM_FUSED_FINAL_HC": "0"}, capture_output=True, text=True,
    )
    ks_off = ks.stdout.strip().splitlines()[-1] == "False" if ks.stdout.strip() else False
    log(f"[check] kill switch off -> FUSED_FINAL_HC_ENV_ON printed {ks.stdout.strip()[-20:]!r}")
    if not ks_off:
        failures.append("kill switch")

    report["sequence"] = seq
    report["gate"] = gate
    for f_ in failures[:20]:
        log("[FAIL]", f_)
    print(
        json.dumps(
            {
                "passed": not failures,
                "n_failures": len(failures),
                "compared_against": "main's _finalize_hc_state math: mhc_post_mapping_cuda + mHC.collapse (bitwise, int16 view)",
                "M": list(MS),
                "hidden": hidden,
                "warmup": WARMUP,
                "iters": ITERS,
                **report,
            }
        )
    )


def main_ref_res(st, post_map, mult, hidden):
    r = st.residual
    if st.is_deferred:
        Mx = r.shape[0]
        r = post_map(r, st.x_prev.reshape(Mx, hidden), st.post_mix.view(Mx, mult), st.comb_mix.view(Mx, mult, mult),
                     mult).view(Mx, mult, hidden)
    return r


if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    log(f"[start] mode={mode} gpu={torch.cuda.get_device_name()}")
    with torch.inference_mode():
        bench() if mode == "bench" else check()
    print(f"[done] {time.time() - t0:.1f}s", file=sys.stderr, flush=True)  # JSON stays the last stdout line
