"""Idea 116 (branch 50): cold-L2 sweep of every NVFP4 tactic at the served M of the flip buckets.

usage: sweep.py [--cases gate_up:300,down:2000,...]

Synthetic operands of the real MLP shapes (dims printed from <checkpoint>/config.json), quantized
with trtllm::fp4_quantize. For each (shape, M): every TRT CUTLASS config ('cutlass', i) and every
cuBLASLt heuristic index at the actual M (list of actual M, as the 'gemm' op key runs it) and, for
gate_up M in the dense range, at the dense bucket's list (as 'gemm_dense' runs it). Per tactic:
kernel name + grid (torch.profiler), cold-L2 median (256 MiB flush before each of 5 warmup + 20
timed calls), the resolved cuBLASLt config, and int16 equality of the output against
('cutlass', 0) on the same inputs (seed 1).
"""
import json
import os
import sys
import tempfile
import time

import torch

CKPT = os.environ.get("CKPT") or \
    "/home/scratch.chengzhang/trtllm-dev-gemma4-31b-pf-20261114-163048/models/Gemma-4-31B-IT-NVFP4"
DEFAULT_CASES = ("gate_up:300,gate_up:2300,gate_up:4000,down:300,down:1000,down:1200,down:1500,"
                 "down:1800,down:2000,down:4000")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    cases = DEFAULT_CASES
    if len(sys.argv) > 2 and sys.argv[1] == "--cases":
        cases = sys.argv[2]
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    tc = cfg.get("text_config", cfg)
    h, i = tc["hidden_size"], tc["intermediate_size"]
    D = {"gate_up": (2 * i, h), "down": (h, i)}
    log(f"[dims] hidden={h} intermediate={i} -> {D}")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.utils import nvfp4_dense_bucket

    def quantize(x):
        gs = (448.0 * 6.0 / x.abs().amax().float()).reshape(1).to(torch.float32)
        q, sf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
        return q, sf, gs

    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    cut = tco.FP4GemmRunner(tco.fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, 0, torch.bfloat16)
    lt = tco.CublasLtFP4GemmRunner(0, torch.bfloat16)
    n_cut = cut.fp4_gemm_runner.get_num_configs()
    log(f"cutlass configs: {n_cut}")

    def cold(fn, warm=5, iters=20):
        s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for _ in range(warm):
            flush.zero_()
            fn()
        for j in range(iters):
            flush.zero_()
            s[j].record()
            fn()
            e[j].record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))
        return round(ts[len(ts) // 2], 1), round(ts[2], 1), round(ts[17], 1)

    def kname(fn):
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        p = tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(p)
        tr = json.load(open(p))
        os.remove(p)
        return [(ev["name"][:140], ev.get("args", {}).get("grid")) for ev in tr["traceEvents"]
                if ev.get("cat") == "kernel"]

    W = {}
    out = {}
    for case in cases.split(","):
        name, m = case.split(":")
        m = int(m)
        n, k = D[name]
        if name not in W:
            g = torch.Generator(device="cuda").manual_seed(100 + n)
            W[name] = quantize((torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16))
        wq, wsf, wgs = W[name]
        g = torch.Generator(device="cuda").manual_seed(1)
        xq, xsf, xgs = quantize(torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16))
        alpha = (1.0 / (xgs * wgs)).float()
        inputs = [xq, wq, xsf, wsf, alpha]
        ref = cut.forward(inputs, tactic=0)
        rows = []

        def add(label, fn, cfgv=None):
            try:
                o = fn()
                torch.cuda.synchronize()
                eq = bool(torch.equal(o.view(torch.int16), ref.view(torch.int16)))
                nd = int((o.view(torch.int16) != ref.view(torch.int16)).sum().item())
                t = cold(fn)
                kn = kname(fn)
                r = {"tactic": label, "cold_us": t, "int16_eq_cutlass0": eq, "n_diff": nd, "kernels": kn,
                     "cfg": cfgv}
            except Exception as ex:  # noqa: BLE001
                r = {"tactic": label, "error": repr(ex)[:200]}
            rows.append(r)
            log(f"  {name}@{m} {label} {r.get('cold_us')} eq={r.get('int16_eq_cutlass0')} nd={r.get('n_diff')} "
                f"cfg={cfgv} {[(a[:90], b) for a, b in r.get('kernels', [])]}")

        log(f"== {name} M={m} N={n} K={k}")
        for t in range(n_cut):
            add(("cutlass", t), lambda t=t: cut.forward(inputs, tactic=t))
        na = lt.cublaslt_runner.get_num_heuristic_algos(xq, wq, xsf, wsf)
        for t in range(na):
            cfgv = list(lt.cublaslt_runner.get_algo_config(xq, wq, xsf, wsf, t, -1))
            add(("cublaslt", t), lambda t=t: lt.forward(inputs, tactic=t), cfgv)
        rng = tco._nvfp4_dense_range(n, k)
        if rng is not None and rng[0] <= m <= rng[1]:
            b = nvfp4_dense_bucket(m, rng[0])
            ltb = tco.CublasLtFP4GemmRunner(0, torch.bfloat16, bucket_fn=lambda mm, b=b: b)
            hx = torch.empty((b, xq.shape[1]), dtype=xq.dtype, device="cuda")
            nb = lt.cublaslt_runner.get_num_heuristic_algos(hx, wq, xsf, wsf)
            for t in range(nb):
                cfgv = list(lt.cublaslt_runner.get_algo_config(xq, wq, xsf, wsf, t, b))
                add((f"cublaslt_b{b}", t), lambda t=t: ltb.forward(inputs, tactic=t), cfgv)
        out[f"{name}@{m}"] = rows
        ok = sorted([r for r in rows if "cold_us" in r], key=lambda r: r["cold_us"][0])
        log(f"-- {name}@{m} fastest: " + "; ".join(f"{r['tactic']} {r['cold_us'][0]}us eq={r['int16_eq_cutlass0']}"
                                                    for r in ok[:6]))
        del ref
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "sweep.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({"passed": True, "cases": list(out), "file": "sweep.json"}))


if __name__ == "__main__":
    main()
