"""Idea 116 (branch 50): cold-L2 kernel pair for the NVFP4 tactic pins (run_gpu_pair, both arms).

usage: bench_pair.py --checkpoint CK [--reps 3]

Method 971: each arm runs REPS fresh processes; each warms the GPU (10 s of BF16 GEMMs), tunes as
the engine warmup does (one M=8192 trtllm::nvfp4_gemm call per shape under autotune(), synthetic
operands of the real shapes from config.json), then times at every served M with a 256 MiB L2
flush before each of 5 warmup + 30 timed calls (CUDA events): trtllm::nvfp4_gemm for gate_up and
down, and for gate_up at M >= 1664 the gate_up -> gelu_tanh_mul_fp4_quant chain as
_Gemma4GeluQuantMLP runs it (chunked when gate_up_gelu_fp4_chunked engages, else unchunked).
Per key: median of the process medians; noise_floor = max(within-process (p90-p10)/median,
across-process spread). Each process logs its tuner picks and the pin that applied.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nvfp4_dense"))
import bench  # noqa: E402

log = bench.log
GU_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000, 4800]
DN_M = [300, 1000, 1200, 1500, 1800, 2000, 2300, 4000, 4800]
CHAIN_M = [1800, 2000, 2300, 4000, 4800]
bench.WARMUP, bench.ITERS = 5, 30
FLUSH_BYTES = 256 * 1024 * 1024  # bench.timed zeroes this buffer before every call


def make_flush():
    return torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")


def one(args):
    dims = bench.read_dims(args.checkpoint)
    t = bench.tco()
    log(f"[gpu] {torch.cuda.get_device_name()} start {bench.gpu_state()}")
    log(f"[pins] available={hasattr(t, 'nvfp4_pinned_tactic')} "
        f"gate_up={getattr(t, 'NVFP4_GATEUP_PIN', None)} down={getattr(t, 'NVFP4_DOWN_PIN', None)}")
    weights = {}
    for i, name in enumerate(("gate_up", "down")):
        n, k = dims[name]
        weights[name] = bench.make_weight(n, k, 1000 + i)
    bench.warm_gpu()
    ts = bench.tune(weights, dims)
    flush = make_flush()
    from tensorrt_llm._torch.kernels.fused_ops import gate_up_gelu_fp4_chunked as gc
    from tensorrt_llm._torch.kernels.fused_ops.gelu_tanh_mul_fp4_quant import gelu_tanh_mul_fp4_quant
    dis = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    batch, picks = {}, {}
    pin_info = getattr(t, "nvfp4_pin_info", None)
    for name, ms in (("gate_up", GU_M), ("down", DN_M)):
        wq, wsf, wgs = weights[name]
        n, k = dims[name]
        for m in ms:
            xq, xsf, xgs = bench.make_act(m, k, 11)
            alpha = (1.0 / (xgs * wgs)).float()
            med, p10, p90 = bench.timed(lambda: bench.gemm(xq, xsf, wq, wsf, alpha), flush)
            key = f"{name}@{m}"
            pk = bench.picks_for(name, m, n, k)
            pin = pin_info(n, k, m) if pin_info else None
            picks[key] = {"tuner": pk, "pin": pin}
            batch[key] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med}
            log(f"[time] {key} {med * 1e3:.1f}us p10={p10 * 1e3:.1f} p90={p90 * 1e3:.1f} picks={pk} pin={pin}")
            if name == "gate_up" and m in CHAIN_M:
                engaged = gc.gate_up_gelu_fp4_chunked(xq, xsf, wq, wsf, alpha, torch.bfloat16,
                                                      bench.BACKENDS.split(","), dis, 1280) is not None

                def chain():
                    if gc.gate_up_gelu_fp4_chunked(xq, xsf, wq, wsf, alpha, torch.bfloat16,
                                                   bench.BACKENDS.split(","), dis, 1280) is None:
                        gelu_tanh_mul_fp4_quant(bench.gemm(xq, xsf, wq, wsf, alpha), dis)

                med, p10, p90 = bench.timed(chain, flush)
                ck = f"gate_up_chain@{m}"
                batch[ck] = {"kernel_ms": med, "p10_ms": p10, "p90_ms": p90, "noise_floor": (p90 - p10) / med}
                picks[ck] = {"chunk_engaged": engaged}
                log(f"[time] {ck} {med * 1e3:.1f}us chunk={engaged}")
    log(f"[gpu] end {bench.gpu_state()}")
    print(json.dumps({"batch": batch, "picks": picks, "tune_seconds": ts}))


def drive(args):
    reps = []
    for r in range(args.reps):
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", "one", "--checkpoint", args.checkpoint]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = None
        for line in p.stdout:
            sys.stdout.write(f"[rep{r}] {line}")
            sys.stdout.flush()
            if line.startswith("{"):
                last = line
        p.wait()
        reps.append(json.loads(last))
    batch = {}
    for key in reps[0]["batch"]:
        meds = sorted(rp["batch"][key]["kernel_ms"] for rp in reps)
        within = sorted(rp["batch"][key]["noise_floor"] for rp in reps)[len(reps) // 2]
        med = meds[len(meds) // 2]
        across = (meds[-1] - meds[0]) / med
        batch[key] = {"kernel_ms": med, "noise_floor": max(within, across), "within_process_noise": within,
                      "across_process_spread": across, "process_medians_ms": meds,
                      "picks_per_process": [rp["picks"][key] for rp in reps]}
        log(f"[agg] {key} med={med * 1e3:.1f}us procs={[round(x * 1e3, 1) for x in meds]} "
            f"noise={batch[key]['noise_floor']:.4f}")
    print(json.dumps({"batch": batch, "output": {"class": "reordered"}, "processes": args.reps}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pair")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    if args.mode == "one":
        return one(args)
    return drive(args)


if __name__ == "__main__":
    main()
