"""Per-rank MiniMax-H3 DiT GEMM census on SM120 (TP2 shapes, synthetic tensors).

BF16 (F.linear, trtllm::cublas_mm, cuBLASLt heuristic sweep), FP8 per-tensor,
FP8_BLOCK_SCALES and NVFP4, each built through tensorrt_llm Linear with the
dynamic weight loader's quantizers (as the engine does), timed with CUDA events.
Reference: FP32 torch.matmul of the same BF16 inputs.
"""
import argparse
import ctypes
import json
import os
import statistics
import threading
import time

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUT_DIR = os.environ.get("ASCENT_ARTIFACTS", os.environ.get("ARTIFACTS_DIR", "."))


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# Clock sampler (pynvml, by UUID of the visible device)
# ----------------------------------------------------------------------------
class ClockSampler:

    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            if uuid.startswith("GPU-"):
                self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            else:
                self.h = pynvml.nvmlDeviceGetHandleByIndex(int(uuid or 0))
            self.nv = pynvml
            self.ok = True
        except Exception as e:  # noqa
            log("pynvml unavailable:", e)
        self.samples = []
        self._stop = threading.Event()

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        if self.ok:
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                clk = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                pw = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                self.samples.append((clk, pw))
            except Exception:
                pass
            time.sleep(0.01)

    def __exit__(self, *a):
        self._stop.set()
        if self.ok:
            self.t.join()

    def summary(self):
        if not self.samples:
            return None
        clks = sorted(s[0] for s in self.samples)
        pws = sorted(s[1] for s in self.samples)
        return {
            "sm_mhz_median": clks[len(clks) // 2],
            "sm_mhz_min": clks[0],
            "sm_mhz_max": clks[-1],
            "power_w_median": pws[len(pws) // 2],
            "n": len(clks),
        }


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med = statistics.median(ts)
    p10 = ts[max(0, int(0.1 * (len(ts) - 1)))]
    p90 = ts[min(len(ts) - 1, int(round(0.9 * (len(ts) - 1))))]
    return {
        "median_ms": med,
        "p10_ms": p10,
        "p90_ms": p90,
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "spread": (p90 - p10) / med if med > 0 else None,
    }


def repeat_check(fn, ref_out, inputs, warmup, iters):
    """Untimed rerun with the same warmup/iteration count: each output bitwise vs
    the first-run output; inputs bitwise unchanged."""
    pristine = [t.clone() for t in inputs]
    bad = []
    for i in range(warmup + iters):
        o = fn()
        if not torch.equal(o, ref_out):
            bad.append(i)
        for a, b in zip(inputs, pristine):
            if not torch.equal(a, b):
                bad.append(("input_changed", i))
    return {"iterations": warmup + iters, "mismatches": bad[:10], "passed": not bad}


def err(out, ref):
    o = out.float()
    d = o - ref
    return {
        "rel_l2": (d.norm() / ref.norm()).item(),
        "max_abs": d.abs().max().item(),
        "ref_max_abs": ref.abs().max().item(),
    }


# ----------------------------------------------------------------------------
# cuBLASLt heuristic sweep via ctypes
# ----------------------------------------------------------------------------
class HeurResult(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_uint64 * 8), ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


def _find_lib(name):
    with open("/proc/self/maps") as f:
        for line in f:
            if name in line:
                return line.split()[-1]
    return None


class LtSweep:

    def __init__(self):
        path = _find_lib("libcublasLt.so")
        if path is None:
            raise RuntimeError("libcublasLt not loaded")
        self.path = path
        L = ctypes.CDLL(path)
        self.L = L
        vp = ctypes.c_void_p
        L.cublasLtCreate.argtypes = [ctypes.POINTER(vp)]
        L.cublasLtMatmulDescCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int]
        L.cublasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
        L.cublasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int,
                                                 ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
        L.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(vp)]
        L.cublasLtMatmulPreferenceSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
        L.cublasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, ctypes.c_int,
                                                     ctypes.POINTER(HeurResult),
                                                     ctypes.POINTER(ctypes.c_int)]
        L.cublasLtMatmul.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
                                     ctypes.c_size_t, vp]
        L.cublasLtMatmulAlgoConfigGetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t,
                                                           ctypes.POINTER(ctypes.c_size_t)]
        self.h = vp()
        self._chk(L.cublasLtCreate(ctypes.byref(self.h)), "create")
        self.ws_bytes = 64 << 20
        self.ws = torch.empty(self.ws_bytes, dtype=torch.uint8, device="cuda")

    @staticmethod
    def _chk(st, what):
        if st != 0:
            raise RuntimeError(f"cublasLt {what} status {st}")

    def run(self, x, w, y, warmup=3, iters=10, max_algos=24):
        L = self.L
        vp = ctypes.c_void_p
        M, K = x.shape
        N = w.shape[0]
        desc = vp()
        self._chk(L.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0), "desc")  # COMPUTE_32F, R_32F
        opT = ctypes.c_int(1)
        self._chk(L.cublasLtMatmulDescSetAttribute(desc, 3, ctypes.cast(ctypes.byref(opT), vp), 4), "transa")
        A, B, C = vp(), vp(), vp()
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(A), 14, K, N, K), "A")
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(B), 14, K, M, K), "B")
        self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(C), 14, N, M, N), "C")
        pref = vp()
        self._chk(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        wsb = ctypes.c_uint64(self.ws_bytes)
        self._chk(L.cublasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.cast(ctypes.byref(wsb), vp), 8), "prefws")
        res = (HeurResult * max_algos)()
        n = ctypes.c_int(0)
        self._chk(L.cublasLtMatmulAlgoGetHeuristic(self.h, desc, A, B, C, C, pref, max_algos, res,
                                                   ctypes.byref(n)), "heur")
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        stream = torch.cuda.current_stream().cuda_stream
        out = []
        ref = F.linear(x, w)
        for i in range(n.value):
            r = res[i]
            if r.state != 0:
                continue
            ids = []
            for attr in (0, 1, 2, 8):  # ID, TILE_ID, SPLITK_NUM, STAGES_ID
                v = ctypes.c_int(0)
                wr = ctypes.c_size_t(0)
                L.cublasLtMatmulAlgoConfigGetAttribute(ctypes.cast(ctypes.byref(r.algo), vp), attr,
                                                       ctypes.cast(ctypes.byref(v), vp), 4,
                                                       ctypes.byref(wr))
                ids.append(v.value)

            def fn(r=r):
                st = L.cublasLtMatmul(self.h, desc, ctypes.cast(ctypes.byref(alpha), vp),
                                      vp(w.data_ptr()), A, vp(x.data_ptr()), B,
                                      ctypes.cast(ctypes.byref(beta), vp), vp(y.data_ptr()), C,
                                      vp(y.data_ptr()), C, ctypes.cast(ctypes.byref(r.algo), vp),
                                      vp(self.ws.data_ptr()), self.ws_bytes, vp(stream))
                if st != 0:
                    raise RuntimeError(f"matmul status {st}")
                return y

            try:
                t = time_fn(fn, warmup, iters)
                d = (y.float() - ref.float()).abs().max().item()
                out.append({"idx": i, "algo_id": ids[0], "tile_id": ids[1], "splitk": ids[2],
                            "stages_id": ids[3], "waves": r.wavesCount,
                            "ws": int(r.workspaceSize), "median_ms": t["median_ms"],
                            "spread": t["spread"], "maxdiff_vs_flinear": d})
            except Exception as e:  # noqa
                out.append({"idx": i, "error": str(e)[:200]})
        return out


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def make_act(M, K, kind, dist, gen):
    if kind == "swiglu":
        a = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32) * 1.5
        b = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32) * 1.5
        x = F.silu(a) * b
        del a, b
    elif kind == "temb":
        x = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32) * 2.0
        x = F.silu(x)
    else:  # AdaLN-modulated RMSNorm output (or attention output)
        x = torch.randn(M, K, generator=gen, device="cuda", dtype=torch.float32)
        col_scale = torch.exp(0.5 * torch.randn(K, generator=gen, device="cuda"))
        col_shift = 0.3 * torch.randn(K, generator=gen, device="cuda")
        x = x * col_scale + col_shift
    if dist == "outlier":
        idx = torch.randperm(K, generator=gen, device="cuda")[:8]
        x[:, idx] *= 50.0
    return x.to(torch.bfloat16).contiguous()


def make_weight(N, K, gen):
    w = torch.randn(N, K, generator=gen, device="cuda", dtype=torch.float32) / (K ** 0.5)
    w = w * torch.exp(0.3 * torch.randn(N, 1, generator=gen, device="cuda"))
    return w.to(torch.bfloat16).contiguous()


# ----------------------------------------------------------------------------
# Quantized Linear modules built like the engine
# ----------------------------------------------------------------------------
def build_linear(algo_name, w_bf16, bias_bf16):
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.visual_gen.quantization import ops as qops
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo
    N, K = w_bf16.shape
    algo = {"fp8": QuantAlgo.FP8, "fp8_block": QuantAlgo.FP8_BLOCK_SCALES,
            "nvfp4": QuantAlgo.NVFP4}[algo_name]
    qc = QuantConfig(quant_algo=algo, group_size=128) if algo_name == "fp8_block" \
        else QuantConfig(quant_algo=algo)
    mod = Linear(K, N, bias=bias_bf16 is not None, dtype=torch.bfloat16, quant_config=qc,
                 force_dynamic_quantization=True).cuda()
    if algo_name == "fp8":
        qw, s = qops.quantize_fp8_per_tensor(w_bf16)
        wd = {"weight": qw, "weight_scale": s}
    elif algo_name == "fp8_block":
        qw, s = qops.quantize_fp8_blockwise(w_bf16, block_size=128)
        wd = {"weight": qw, "weight_scale": s}
    else:
        qw, s, s2 = qops.quantize_nvfp4(w_bf16)
        wd = {"weight": qw, "weight_scale": s, "weight_scale_2": s2}
    if bias_bf16 is not None:
        wd["bias"] = bias_bf16
    mod.load_weights([wd])
    mod.process_weights_after_loading()
    mod.post_load_weights()
    return mod


# ----------------------------------------------------------------------------
def run_shape(name, M, N, K, act_kind, bias, dists, args, lt, results):
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm.quantization.utils.fp8_utils import per_token_quant_and_transform
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234 + N + K + M)
    w = make_weight(N, K, gen)
    b = (0.02 * torch.randn(N, generator=gen, device="cuda")).to(torch.bfloat16) if bias else None
    flop = 2.0 * M * N * K
    W, I = args.warmup, args.iters
    shape_res = {"M": M, "N": N, "K": K, "flop": flop, "dists": {}}
    results[name] = shape_res
    mods = {}
    for algo in ("fp8", "fp8_block", "nvfp4"):
        try:
            mods[algo] = build_linear(algo, w, b)
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            log(f"[{name}] build {algo} failed: {e}")
            shape_res.setdefault("build_errors", {})[algo] = str(e)[:300]
    for dist in dists:
        x = make_act(M, K, act_kind, dist, gen)
        ref = torch.matmul(x.float(), w.float().t())
        if b is not None:
            ref += b.float()
        R = {}
        shape_res["dists"][dist] = R

        def record(key, fn, quant_fn=None, gemm_fn=None, check=True, clocks=False):
            try:
                out0 = fn()
                torch.cuda.synchronize()
                if clocks:
                    with ClockSampler() as cs:
                        t = time_fn(fn, W, I)
                    t["clock"] = cs.summary()
                else:
                    t = time_fn(fn, W, I)
                t["tflops"] = flop / (t["median_ms"] * 1e-3) / 1e12
                t["err"] = err(out0, ref)
                if quant_fn is not None:
                    t["quant"] = time_fn(quant_fn, W, I)
                if gemm_fn is not None:
                    t["gemm"] = time_fn(gemm_fn, W, I)
                if check:
                    t["repeat"] = repeat_check(fn, out0, [x], W, I)
                R[key] = t
                q = t.get("quant", {}).get("median_ms")
                g = t.get("gemm", {}).get("median_ms")
                log(f"[{name}/{dist}] {key}: total {t['median_ms']:.4f} ms (spread {t['spread']:.3f})"
                    f" quant {q} gemm {g} | {t['tflops']:.1f} TFLOPS | relL2 {t['err']['rel_l2']:.3e}"
                    f" maxabs {t['err']['max_abs']:.3e} | repeat {t.get('repeat', {}).get('passed')}"
                    + (f" | clk {t['clock']}" if clocks else ""))
            except Exception as e:  # noqa
                import traceback
                R[key] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
                log(f"[{name}/{dist}] {key} FAILED: {e}")
                traceback.print_exc()

        # ---- BF16
        record("bf16_flinear", lambda: F.linear(x, w, b), clocks=True)
        record("bf16_cublas_mm",
               lambda: torch.ops.trtllm.cublas_mm(x, w.t(), b, None, 0, None))
        if lt is not None and dist == dists[0]:
            try:
                y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                xs = x
                if b is not None:
                    pass  # sweep measures the GEMM without bias
                sweep = lt.run(xs, w, y, warmup=3, iters=10 if M > 64 else 20)
                R["bf16_lt_sweep"] = sweep
                ok = [s for s in sweep if "median_ms" in s]
                if ok:
                    best = min(ok, key=lambda s: s["median_ms"])
                    log(f"[{name}] cuBLASLt sweep: {len(ok)} algos; heuristic#0 {ok[0]['median_ms']:.4f} ms"
                        f" (algo {ok[0]['algo_id']} tile {ok[0]['tile_id']}); best idx {best['idx']}"
                        f" {best['median_ms']:.4f} ms (algo {best['algo_id']} tile {best['tile_id']}"
                        f" splitk {best['splitk']}) maxdiff {best['maxdiff_vs_flinear']}")
                del y
            except Exception as e:  # noqa
                R["bf16_lt_sweep"] = {"error": str(e)[:300]}
                log(f"[{name}] lt sweep failed: {e}")

        # ---- FP8 per-tensor
        if "fp8" in mods:
            m = mods["fp8"]
            qx, sx = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
            sx = sx.to(torch.float32)
            record("fp8_linear", lambda: m(x),
                   quant_fn=lambda: torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x),
                   gemm_fn=lambda: torch.ops.trtllm.cublas_scaled_mm(
                       qx, m.weight.t(), scale_a=sx, scale_b=m.weight_scale, bias=None,
                       out_dtype=torch.bfloat16))
            if M <= 8:
                record("fp8_cuda_scaled_mm",
                       lambda: torch.ops.trtllm.cuda_scaled_mm(
                           qx, m.weight.t(), scale_a=sx, scale_b=m.weight_scale, bias=None,
                           out_dtype=torch.bfloat16) + b,
                       check=False)
                record("fp8_cublas_scaled_mm",
                       lambda: torch.ops.trtllm.cublas_scaled_mm(
                           qx, m.weight.t(), scale_a=sx, scale_b=m.weight_scale, bias=None,
                           out_dtype=torch.bfloat16) + b,
                       check=False)
            del qx

        # ---- FP8 block scales
        if "fp8_block" in mods:
            m = mods["fp8_block"]
            try:
                ax, asf = per_token_quant_and_transform(x)
                record("fp8_block_linear", lambda: m(x),
                       quant_fn=lambda: per_token_quant_and_transform(x),
                       gemm_fn=lambda: torch.ops.trtllm.fp8_block_scaling_gemm(
                           ax, m.weight, asf, m.weight_scale))
                del ax, asf
            except Exception as e:  # noqa
                R["fp8_block_linear"] = {"error": str(e)[:400]}
                log(f"[{name}/{dist}] fp8_block failed: {e}")

        # ---- NVFP4
        if "nvfp4" in mods:
            m = mods["nvfp4"]
            backends = ["all"] + list(m.nvfp4_allowed_backends)
            try:
                with autotune(skip_dynamic_tuning_buckets=True):
                    fp4, sf, alpha = m.quant_method._input_prepare(m, x)
                    for bk in backends:
                        try:
                            torch.ops.trtllm.nvfp4_gemm(
                                fp4, m.weight, sf, m.weight_scale, alpha, torch.bfloat16,
                                allowed_backends=",".join(m.nvfp4_allowed_backends) if bk == "all" else bk)
                        except Exception as e:  # noqa
                            log(f"[{name}] nvfp4 tune {bk} failed: {str(e)[:200]}")
                    m(x)
                torch.cuda.synchronize()

                def amax2():
                    return torch.amax(torch.abs(x)).float()

                def amax1():
                    mn, mx = torch.aminmax(x)
                    return torch.maximum(mn.abs(), mx.abs()).float()

                def quant_only():
                    a = torch.amax(torch.abs(x)).float()
                    return torch.ops.trtllm.fp4_quantize(x, 448.0 * 6.0 / a, 16, False)

                gs = (448.0 * 6.0 / torch.amax(torch.abs(x)).float())
                record("nvfp4_linear", lambda: m(x), quant_fn=quant_only,
                       gemm_fn=lambda: torch.ops.trtllm.nvfp4_gemm(
                           fp4, m.weight, sf, m.weight_scale, alpha, torch.bfloat16,
                           allowed_backends=",".join(m.nvfp4_allowed_backends)))
                R["nvfp4_amax_abs_amax"] = time_fn(amax2, W, I)
                R["nvfp4_amax_aminmax"] = time_fn(amax1, W, I)
                R["nvfp4_fp4_quantize"] = time_fn(
                    lambda: torch.ops.trtllm.fp4_quantize(x, gs, 16, False), W, I)
                try:
                    R["nvfp4_tunable_fp4_quantize"] = time_fn(
                        lambda: torch.ops.trtllm.tunable_fp4_quantize(x, gs, 16, False), W, I)
                except Exception as e:  # noqa
                    R["nvfp4_tunable_fp4_quantize"] = {"error": str(e)[:200]}
                log(f"[{name}/{dist}] nvfp4 amax(abs) {R['nvfp4_amax_abs_amax']['median_ms']:.4f}"
                    f" aminmax {R['nvfp4_amax_aminmax']['median_ms']:.4f}"
                    f" fp4_quantize {R['nvfp4_fp4_quantize']['median_ms']:.4f}"
                    f" tunable {R['nvfp4_tunable_fp4_quantize'].get('median_ms')}")
                per_bk = {}
                for bk in backends[1:]:
                    try:
                        t = time_fn(lambda: torch.ops.trtllm.nvfp4_gemm(
                            fp4, m.weight, sf, m.weight_scale, alpha, torch.bfloat16,
                            allowed_backends=bk), W, I)
                        per_bk[bk] = t
                    except Exception as e:  # noqa
                        per_bk[bk] = {"error": str(e)[:200]}
                R["nvfp4_backends"] = per_bk
                log(f"[{name}/{dist}] nvfp4 per-backend gemm: " + ", ".join(
                    f"{k}={v.get('median_ms', str(v.get('error', ''))[:60])}" for k, v in per_bk.items()))
                del fp4, sf
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                R["nvfp4_linear"] = {"error": str(e)[:400]}
        del x, ref
        torch.cuda.empty_cache()
    del w, mods
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="qkv,out,gate_up,down,adaln3,adaln6")
    ap.add_argument("--dists", default="realistic,outlier")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--text-rows", type=int, default=256)
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default="census.json")
    args = ap.parse_args()

    import tensorrt_llm  # noqa
    import tensorrt_llm._torch.custom_ops  # noqa
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import packing as pk

    nf = pk.align_num_frames(124)
    lat_t = pk.video_latent_num_frames(nf)
    aud = pk.audio_latent_num_frames(nf)
    video_rows = lat_t * (768 // 32) * (1344 // 32)
    audio_rows = pk.MINIMAX_H3_AUDIO_CHANNELS * aud
    M = args.M or (video_rows + audio_rows + args.text_rows)
    log(json.dumps({"aligned_frames": nf, "video_latent_frames": lat_t, "video_rows": video_rows,
                    "audio_latents": aud, "audio_rows": audio_rows, "text_rows_assumed": args.text_rows,
                    "M": M}))
    log("device", torch.cuda.get_device_name(), "SMs", torch.cuda.get_device_properties(0).multi_processor_count)

    hidden, heads, hd, ffn, temb = 5376, 56, 128, 14336, 2688
    tp = 2
    hl = heads // tp * hd  # 3584 per-rank attention width
    fl = ffn // tp  # 7168
    shapes = {
        "qkv": (M, 3 * hl, hidden, "mod", False),
        "out": (M, hidden, hl, "mod", False),
        "gate_up": (M, 2 * fl, hidden, "mod", False),
        "down": (M, hidden, fl, "swiglu", False),
        "adaln3": (3, 6 * hidden * 3, temb, "temb", True),
        "adaln6": (6, 6 * hidden * 3, temb, "temb", True),
    }
    _ = F.linear(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
                 torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
    lt = None
    if not args.no_sweep:
        try:
            lt = LtSweep()
            log("cublasLt:", lt.path)
        except Exception as e:  # noqa
            log("LtSweep unavailable:", e)
    results = {"M": M, "shapes": {}}
    dists = args.dists.split(",")
    for name in args.shapes.split(","):
        Mm, N, K, kind, bias = shapes[name]
        log(f"=== {name}: M={Mm} N={N} K={K}")
        run_shape(name, Mm, N, K, kind, bias, dists, args, lt, results["shapes"])
        with open(os.path.join(OUT_DIR, args.out), "w") as f:
            json.dump(results, f, indent=1, default=str)
    summ = {}
    for name, sr in results["shapes"].items():
        d0 = sr["dists"].get(dists[0], {})
        summ[name] = {k: round(v["median_ms"], 4) for k, v in d0.items()
                      if isinstance(v, dict) and "median_ms" in v}
    print(json.dumps({"M": M, "summary_ms": summ}))


if __name__ == "__main__":
    main()
