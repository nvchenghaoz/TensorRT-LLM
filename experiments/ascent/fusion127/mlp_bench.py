"""Idea 127: GatedMLP.forward at H3's TP2 per-rank shapes (K 5376, local gate_up N 14336).

--mode bench  (run_gpu_pair, same script in both trees): times GatedMLP.forward at the chunk
              rows (9600, 9166) with an A/A arm, L2 flushed before every timed call, CUDA events,
              5 warmup + 20 timed per pass, 2 interleaved passes. Base tree: F.linear -> swiglu ->
              down; branch: fused SM120 gate_up+SwiGLU -> down.
--mode check  (branch only): untimed repeat with the same warmup/iteration counts; every
              iteration's output compared bitwise to a protected reference chain (F.linear ->
              torch.ops.trtllm.silu_and_mul -> F.linear); inputs/weights checked unchanged;
              compile count across 8 M values; torch.compile(fullgraph) of the module.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

H, I_LOCAL = 5376, 7168
WARM, ITERS, PASSES = 5, 20, 2


def log(*a):
    print(*a, flush=True)


def build():
    with torch.device("cuda"):
        mlp = GatedMLP(hidden_size=H, intermediate_size=I_LOCAL, bias=False,
                       dtype=torch.bfloat16, config=ModelConfig(), reduce_output=False,
                       layer_idx=0, fuse_bf16_gate_up_swiglu=True)
    g = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(
            (torch.randn(mlp.gate_up_proj.weight.shape, device="cuda", generator=g) * 0.02).bfloat16())
        mlp.down_proj.weight.copy_(
            (torch.randn(mlp.down_proj.weight.shape, device="cuda", generator=g) * 0.02).bfloat16())
    return mlp.eval()


def inputs(m, kind=0, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed + kind)
    x = torch.randn(m, H, device="cuda", generator=g)
    if kind == 1:
        x = x * torch.exp(torch.randn(1, H, device="cuda", generator=g))
    elif kind == 2:
        x[:, :8] *= 50
    return x.bfloat16().contiguous()


def reference(mlp, x):
    h1 = F.linear(x, mlp.gate_up_proj.weight)
    h2 = torch.ops.trtllm.silu_and_mul(h1)
    return F.linear(h2, mlp.down_proj.weight), h2


def bench(mlp):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    res = {}
    xs = {m: inputs(m) for m in (9600, 9166)}
    times = {(m, arm): [] for m in xs for arm in ("a", "a2")}
    with torch.inference_mode():
        for p in range(PASSES):
            for m, x in xs.items():
                for _ in range(WARM):
                    mlp(x)
            torch.cuda.synchronize()
            for it in range(ITERS):
                for m, x in xs.items():
                    for arm in ("a", "a2"):
                        flush.zero_()
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        mlp(x)
                        e.record()
                        e.synchronize()
                        times[(m, arm)].append(s.elapsed_time(e))
            log(f"pass {p} done")
    for m in xs:
        t = torch.tensor(times[(m, "a")])
        t2 = torch.tensor(times[(m, "a2")])
        med = t.median().item()
        nf = (t.quantile(0.9) - t.quantile(0.1)).item() / med
        res[str(m)] = {"kernel_ms": round(med, 4), "noise_floor": round(nf, 4),
                       "aa_ms": round(t2.median().item(), 4),
                       "aa_delta_pct": round(100 * (t2.median().item() / med - 1), 2),
                       "p10": round(t.quantile(0.1).item(), 4), "p90": round(t.quantile(0.9).item(), 4)}
        log(f"M={m}: {res[str(m)]}")
    fused = getattr(mlp, "fuse_bf16_gate_up_swiglu_sm120", False)
    return {"batch": res, "output": {"class": "exact"}, "fused_path": bool(fused)}


def check(mlp):
    from tensorrt_llm._torch.modules import gate_up_swiglu_sm120 as G
    W = mlp.gate_up_proj.weight
    W0 = W.detach().clone()
    Wd0 = mlp.down_proj.weight.detach().clone()
    out = {"iters": {}, "passed": True}
    with torch.inference_mode():
        # (1) untimed repeat: same warmup / iteration / pass structure as the timed run
        for m in (9600, 9166):
            for kind in (0, 1, 2):
                x = inputs(m, kind)
                x0 = x.clone()
                ref_y, ref_h2 = reference(mlp, x)
                fails = 0
                n = 0
                calls0 = G.FUSED_CALLS
                for p in range(PASSES):
                    for it in range(WARM + ITERS):
                        y = mlp(x)
                        h2 = torch.full_like(ref_h2, float("nan"))
                        G.gate_up_sm120(x, W, fused=True, out=h2)
                        ok = torch.equal(y, ref_y) and torch.equal(h2, ref_h2)
                        ro = torch.equal(x, x0) and torch.equal(W, W0) and torch.equal(
                            mlp.down_proj.weight, Wd0)
                        if not (ok and ro):
                            fails += 1
                            log(f"  FAIL M={m} kind={kind} pass={p} it={it} out_eq={ok} ro={ro}")
                        n += 1
                used = G.FUSED_CALLS - calls0
                out["iters"][f"{m}_k{kind}"] = {"n": n, "fails": fails, "fused_calls": used}
                if fails or used != n:
                    out["passed"] = False
                log(f"repeat M={m} kind={kind}: n={n} fails={fails} fused_calls={used}")
        # (2) one compiled kernel serves every M
        c0 = G.COMPILE_COUNT
        per_m = {}
        for m in (9472, 9296, 9600, 9166, 9442, 9445, 1001, 129):
            x = inputs(m, 0)
            h = torch.full((m, I_LOCAL), float("nan"), device="cuda", dtype=torch.bfloat16)
            G.gate_up_sm120(x, W, fused=True, out=h)
            ref = torch.ops.trtllm.silu_and_mul(F.linear(x, W))
            per_m[m] = bool(torch.equal(h, ref))
        torch.cuda.synchronize()
        out["compiles_before_sweep"] = c0
        out["compiles_total"] = G.COMPILE_COUNT
        out["bitwise_per_M"] = {str(k): v for k, v in per_m.items()}
        log(f"compile count total {G.COMPILE_COUNT} (before 8-M sweep {c0}); bitwise per M {per_m}")
        if G.COMPILE_COUNT != 1:
            out["passed"] = False
        if not all(per_m[m] for m in (9472, 9296, 9600, 9166, 9442, 9445)):
            out["passed"] = False
        # (3) dynamo: the op is opaque, fullgraph compile matches eager bitwise
        try:
            cm = torch.compile(lambda t: mlp(t), fullgraph=True, dynamic=True)
            x = inputs(9600, 0)
            yc = cm(x)
            ye, _ = reference(mlp, x)
            out["compiled_fullgraph_eq"] = bool(torch.equal(yc, ye))
            if not out["compiled_fullgraph_eq"]:
                out["passed"] = False
        except Exception as e:  # noqa
            out["compiled_fullgraph_eq"] = f"error: {e!r}"[:300]
            out["passed"] = False
        out["fused_calls_total"] = G.FUSED_CALLS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench")
    a = ap.parse_args()
    log("device", torch.cuda.get_device_name())
    t0 = time.time()
    mlp = build()
    log(f"built in {time.time() - t0:.1f}s; sm120 fused flag="
        f"{getattr(mlp, 'fuse_bf16_gate_up_swiglu_sm120', None)}")
    r = bench(mlp) if a.mode == "bench" else check(mlp)
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
