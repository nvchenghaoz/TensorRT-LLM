"""Idea 60: Gemma4 MLP gate_up -> gelu_tanh_mul_fp4 chain, main vs M-chunked.

usage: bench.py <checkpoint> time|check|sweep

time:  cold-weight (256 MB flush before every iteration) CUDA-event time of
       the chain this tree runs: the candidate dispatches exactly as
       _Gemma4GeluQuantMLP does (gate_up_gelu_fp4_chunked when enabled and
       MIN_M <= M <= MAX_M, else main's chain); the base tree has no chunk
       path and runs main's chain. gate_up nvfp4_gemm autotuned in-process.
check: candidate tree only. Bitwise vs main's chain (nvfp4_gemm full M +
       gelu_tanh_mul_fp4_quant) at every protocol M, repeated A,B,A
       (8000,4800,8000) sequences with outputs landing on 0xA5-filled blocks,
       inputs unchanged, FP32 reference (b) error of the GEMM output, kill
       switch semantics, and tuner cache keys/picks equal main's.
sweep: candidate tree: chunked vs main chain over many M and C (diagnostic).
"""
import hashlib
import json
import os
import sys
import threading
import time

import torch


def _hb():
    t0 = time.time()
    while True:
        time.sleep(30)
        print(f"[heartbeat] {time.time() - t0:.0f}s", flush=True)


threading.Thread(target=_hb, daemon=True).start()

CK, MODE = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(CK, "config.json")))
tc = cfg.get("text_config", cfg)
H, I = int(tc["hidden_size"]), int(tc["intermediate_size"])
EPS = float(tc.get("rms_norm_eps", 1e-6))
N, K2 = 2 * I, H // 2
print(f"hidden_size={H} intermediate_size={I} N={N} K={H}", flush=True)
dev = "cuda"

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant  # noqa: E402
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_residual_add import rmsnorm_residual_add_fp4_quant  # noqa: E402

try:
    from tensorrt_llm._torch.kernels.fused_ops.gate_up_gelu_fp4_chunked import gate_up_gelu_fp4_chunked
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    CHUNK = dict(enabled=mg._MLP_CHUNK_ENABLED, rows=mg._MLP_CHUNK_ROWS, min_m=mg._MLP_CHUNK_MIN_M,
                 max_m=mg._MLP_CHUNK_MAX_M)
except ImportError:
    gate_up_gelu_fp4_chunked = None
    CHUNK = None
print("tree chunk path:", CHUNK, flush=True)
print("gpu:", torch.cuda.get_device_name(), flush=True)

BACKENDS = ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
MS = [300, 1000, 1200, 2000, 4000, 4800, 8000, 8100]


def make_layer(seed=11):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (N, K2), dtype=torch.uint8, device=dev, generator=g)
    sl = (torch.rand(N, H // 16, device=dev, generator=g) * 3.5 + 0.25).to(torch.float8_e4m3fn)
    ws = torch.ops.trtllm.block_scale_interleave(sl.view(torch.uint8))
    return dict(w=w, ws=ws, sl=sl, alpha=torch.tensor([3e-4], device=dev),
                gs_in=torch.tensor([448 * 6 / 30.0], device=dev),
                gs_down=torch.tensor([448 * 6 / 8.0], device=dev),
                nw1=(torch.rand(H, device=dev, generator=g) * 2 + 0.05).to(torch.bfloat16),
                nw2=(torch.rand(H, device=dev, generator=g) * 2 + 0.05).to(torch.bfloat16))


def make_x(m, lw, seed=1, heavy=False):
    """Gate_up input via main's producer (fused post-attn norm + FP4 quant)."""
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + m)
    x = torch.randn(m, H, device=dev, generator=g)
    if heavy:
        x = x * torch.exp2(torch.rand(m, 1, device=dev, generator=g) * 16 - 8)
    r = torch.randn(m, H, device=dev, generator=g) * 4
    _, fp4, sf = rmsnorm_residual_add_fp4_quant(x.to(torch.bfloat16), r.to(torch.bfloat16), lw["nw1"], EPS,
                                                lw["nw2"], EPS, lw["gs_in"])
    return fp4, sf


def tune(lw, fp4, sf):
    with autotune():
        torch.ops.trtllm.nvfp4_gemm(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16,
                                    allowed_backends=",".join(BACKENDS))
    torch.cuda.synchronize()


def main_chain(lw, fp4, sf):
    h1 = torch.ops.trtllm.nvfp4_gemm(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16,
                                     allowed_backends=",".join(BACKENDS))
    return gelu_tanh_mul_fp4_quant(h1, lw["gs_down"])


def chunk_applies(m):
    return CHUNK is not None and CHUNK["enabled"] and CHUNK["min_m"] <= m <= CHUNK["max_m"]


DISPATCH = {"chunked": 0, "main": 0}


def tree_chain(lw, fp4, sf, rows=None):
    """What _Gemma4GeluQuantMLP runs in this tree (rows overrides C for sweeps)."""
    m = fp4.shape[0]
    if rows is not None or chunk_applies(m):
        res = gate_up_gelu_fp4_chunked(fp4, sf, lw["w"], lw["ws"], lw["alpha"], torch.bfloat16, BACKENDS,
                                       lw["gs_down"], rows or CHUNK["rows"])
        if res is not None:
            DISPATCH["chunked"] += 1
            return res
    DISPATCH["main"] += 1
    return main_chain(lw, fp4, sf)


def pick_of(lw, fp4, sf):
    shapes = (fp4.size(), lw["w"].size(), sf.size(), lw["ws"].size(), lw["alpha"].size())
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    r = NVFP4GemmUnifiedRunner(0, torch.bfloat16, BACKENDS)
    hit, _, tac, t = AutoTuner.get().profiling_cache.search_cache(
        "trtllm::nvfp4_gemm::gemm", [r], shapes, NVFP4GemmUnifiedRunner.tuning_config)
    return f"{tac}{'' if hit else ' MISS'}"


FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
WARM, ITERS = 5, 30


def time_fn(fn):
    ts = []
    for it in range(WARM + ITERS):
        FLUSH.fill_(1.0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if it >= WARM:
            ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return ts[n // 2], ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]


def clocks():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)


def do_time(ms):
    print("clocks at start:", clocks(), flush=True)
    lw = make_layer()
    out = {}
    for m in ms:
        fp4, sf = make_x(m, lw)
        tune(lw, fp4, sf)
        d0 = dict(DISPATCH)
        med, p10, p90 = time_fn(lambda: tree_chain(lw, fp4, sf))
        disp = {k: DISPATCH[k] - d0[k] for k in DISPATCH}
        print(f"M={m:5d} pick={pick_of(lw, fp4, sf)} dispatch={disp} median={med*1000:8.1f}us "
              f"p10={p10*1000:8.1f} p90={p90*1000:8.1f}", flush=True)
        out[str(m)] = {"kernel_ms": med, "noise_floor": (p90 - p10) / med, "p10_ms": p10, "p90_ms": p90}
        del fp4, sf
    print("clocks at end:", clocks(), flush=True)
    print(json.dumps({"batch": out, "output": {"class": "exact"}}))


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def a5_prime(m):
    """Leave freed 0xA5 blocks of the output sizes in the caching allocator."""
    nkt = (I // 16 + 3) // 4
    a = torch.full((m, I // 2), 0xA5, dtype=torch.uint8, device=dev)
    b = torch.full((((m + 127) // 128) * 128 * nkt * 4,), 0xA5, dtype=torch.uint8, device=dev)
    pa, pb = a.data_ptr(), b.data_ptr()
    del a, b
    return pa, pb


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    device=dev)


def deq_rows(fp4, sf_lin, scale):
    lo, hi = (fp4 & 0xF).long(), (fp4 >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(fp4.shape[0], -1)
    return (v.reshape(fp4.shape[0], -1, 16) * sf_lin.float()[:, :, None]).reshape(fp4.shape[0], -1) * scale


def do_check():
    assert CHUNK is not None, "check runs on the candidate tree"
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import sf_swizzled_offsets
    res = {"bitwise": {}, "aba": {}, "fails": []}
    lw = make_layer()
    inputs = {}
    for m in MS:
        for seed, heavy in ((1, False), (2, False), (3, True)):
            fp4, sf = make_x(m, lw, seed, heavy)
            if seed == 1:
                tune(lw, fp4, sf)
            inputs[(m, seed)] = (fp4, sf, sha(fp4), sha(sf))
    keys0 = sorted(str(k) for k in AutoTuner.get().profiling_cache.cache.keys())
    picks0 = {str(k): str(v[1]) for k, v in AutoTuner.get().profiling_cache.cache.items()}
    # 1) bitwise vs main at every M, 3 inputs, from 0xA5 blocks
    for (m, seed), (fp4, sf, h0, h1) in inputs.items():
        ref = main_chain(lw, fp4, sf)
        a5_prime(m)
        got = tree_chain(lw, fp4, sf)
        ok = torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        res["bitwise"][f"{m}/s{seed}"] = ok
        if not ok:
            d = (got[0] != ref[0]).nonzero()
            res["fails"].append(f"M={m} s={seed}: payload diffs {d.shape[0]} first {d[:1].tolist()} "
                                f"sf_eq={torch.equal(got[1], ref[1])}")
        print(f"bitwise M={m} seed={seed} chunked={chunk_applies(m) and m > CHUNK['rows']} ok={ok}", flush=True)
    # 2) repeated A,B,A sequence (same call counts as timing), from 0xA5 blocks
    seq = [8000, 4800, 8000]
    firsts = {}
    for blk, m in enumerate(seq):
        fp4, sf, _, _ = inputs[(m, 1)]
        ref = main_chain(lw, fp4, sf)
        n_ok = 0
        on_a5 = 0
        for it in range(WARM + ITERS):
            pa, pb = a5_prime(m)
            o, s = tree_chain(lw, fp4, sf)
            on_a5 += (o.data_ptr() == pa) + (s.data_ptr() == pb)
            if m not in firsts:
                firsts[m] = (o.clone(), s.clone())
            ok = (torch.equal(o, firsts[m][0]) and torch.equal(s, firsts[m][1])
                  and torch.equal(o, ref[0]) and torch.equal(s, ref[1]))
            n_ok += ok
            del o, s
        res["aba"][f"{blk}:{m}"] = {"calls": WARM + ITERS, "equal_first_and_main": n_ok,
                                     "outputs_on_0xA5_blocks": int(on_a5)}
        print(f"A,B,A block {blk} M={m}: {n_ok}/{WARM + ITERS} equal; on 0xA5 blocks {on_a5}", flush=True)
    # 3) inputs unchanged
    res["inputs_unchanged"] = all(sha(v[0]) == v[2] and sha(v[1]) == v[3] for v in inputs.values())
    # 4) tuner cache keys / picks unchanged by the chunk path
    keys1 = sorted(str(k) for k in AutoTuner.get().profiling_cache.cache.keys())
    picks1 = {str(k): str(v[1]) for k, v in AutoTuner.get().profiling_cache.cache.items()}
    res["tuner_cache_unchanged"] = keys0 == keys1 and picks0 == picks1
    res["n_cache_keys"] = len(keys1)
    # 5) FP32 reference (b) of the gate_up output rows, chunked vs main (M=2000)
    m = 2000
    fp4, sf, _, _ = inputs[(m, 1)]
    nvec = H // 16
    a_sf = sf[sf_swizzled_offsets(m, nvec, dev)].view(torch.float8_e4m3fn).reshape(m, nvec)
    a = deq_rows(fp4, a_sf, 1.0)
    wd = deq_rows(lw["w"], lw["sl"], 1.0)
    ref_h = (a @ wd.t()) * lw["alpha"].item()
    del wd
    h_main = torch.ops.trtllm.nvfp4_gemm(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16,
                                         allowed_backends=",".join(BACKENDS)).float()
    from tensorrt_llm._torch.kernels.fused_ops.gate_up_gelu_fp4_chunked import full_m_gemm_tactic
    runner, tac = full_m_gemm_tactic(fp4, lw["w"], sf, lw["ws"], lw["alpha"], torch.bfloat16, BACKENDS)
    nb = (nvec + 3) // 4 * 512
    h_ch = torch.cat([runner.forward([fp4[r0:min(m, r0 + 1280)], lw["w"], sf[r0 // 128 * nb:
                                       (min(m, r0 + 1280) + 127) // 128 * nb], lw["ws"], lw["alpha"]],
                                     tactic=tac) for r0 in range(0, m, 1280)]).float()

    def err(x):
        return {"max_abs": (x - ref_h).abs().max().item(),
                "rel_fro": ((x - ref_h).norm() / ref_h.norm()).item()}
    res["ref_b_gemm_M2000"] = {"main": err(h_main), "chunked": err(h_ch), "tactic": str(tac),
                               "chunked_vs_main_max_abs": (h_ch - h_main).abs().max().item()}
    print("ref (b):", json.dumps(res["ref_b_gemm_M2000"]), flush=True)
    passed = (all(res["bitwise"].values()) and res["inputs_unchanged"] and res["tuner_cache_unchanged"]
              and all(v["equal_first_and_main"] == v["calls"] for v in res["aba"].values())
              and res["ref_b_gemm_M2000"]["chunked_vs_main_max_abs"] == 0.0)
    res["chunk_cfg"] = CHUNK
    res["passed"] = bool(passed)
    res["compared"] = ("chunked gate_up->gelu_tanh_mul_fp4 vs main full-M chain: FP4 payload + full SF bitwise; "
                       "A,B,A repeated; inputs; tuner cache; FP32 ref (b) of gate_up rows")
    print(json.dumps(res))


def do_sweep():
    assert CHUNK is not None
    lw = make_layer()
    rows = []
    for m in [1536, 1664, 1792, 2000, 2304, 2560, 3000, 3328, 4000, 4800, 6000, 8000, 8100, 8192]:
        fp4, sf = make_x(m, lw)
        tune(lw, fp4, sf)
        base = time_fn(lambda: main_chain(lw, fp4, sf))
        r = {"M": m, "pick": pick_of(lw, fp4, sf), "main_us": round(base[0] * 1000, 1)}
        for c in (1280, 1536):
            if c >= m:
                continue
            d0 = DISPATCH["chunked"]
            t = time_fn(lambda: tree_chain(lw, fp4, sf, rows=c))
            r[f"C{c}_us"] = round(t[0] * 1000, 1)
            r[f"C{c}_chunked_calls"] = DISPATCH["chunked"] - d0
        print(json.dumps(r), flush=True)
        rows.append(r)
        del fp4, sf
    print(json.dumps({"sweep": rows}))


if __name__ == "__main__":
    if MODE == "time":
        do_time(MS)
    elif MODE == "check":
        do_check()
    elif MODE == "sweep":
        do_sweep()
