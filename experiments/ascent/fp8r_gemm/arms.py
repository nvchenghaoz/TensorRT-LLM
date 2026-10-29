"""STEP 0(b)/(c) of idea 122: build the FP8R CUTLASS arms, check exactness vs torch._scaled_mm,
and time torch vs each arm at the 4 image.0 shapes (M=4096), WARM and EVICTED, graph x10."""

import glob
import json
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0  # noqa: E402

M = 4096
SHAPES = step0.SHAPES
ARMS = [int(a) for a in os.environ.get("ARMS", "0,1,2").split(",")]
ORDERS = [int(a) for a in os.environ.get("ORDERS", "1,0").split(",")]
log = step0.log


def build():
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg

    done = {"ok": False}

    def hb():
        t0 = time.time()
        while not done["ok"]:
            time.sleep(30)
            if not done["ok"]:
                log(f"[build] still building... {time.time() - t0:.0f}s")

    th = threading.Thread(target=hb, daemon=True)
    th.start()
    t0 = time.time()
    mod = fg.module()
    done["ok"] = True
    log(f"[build] done in {time.time() - t0:.1f}s")
    try:
        from flashinfer.jit import env as jit_env

        sos = glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "flux3_fp8r_gemm_sm120" / "*.so"))
        for so in sos:
            r = subprocess.run(["cuobjdump", "--dump-resource-usage", so], capture_output=True,
                               text=True)
            for line in r.stdout.splitlines():
                if "REG" in line or "Function" in line:
                    log("[res] " + line.strip()[:300])
    except Exception as e:  # noqa: BLE001
        log(f"[res] cuobjdump failed {e!r}")
    for arm in ARMS:
        log(f"[arm {arm}] stages={mod.fp8r_stages(arm)} smem={mod.fp8r_smem(arm)} "
            f"ws={mod.fp8r_workspace_size(4096, 18432, 3072, arm)}")
    return fg


def real_inputs(m, n, k, gen):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import quantize_fp8r_weight

    x = (torch.randn((m, k), device="cuda", generator=gen) * 1.5).to(torch.bfloat16)
    # Outlier rows/cols like real activations.
    x[:, :8] *= 20
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    data, scales = quantize_fp8r(x)
    wd, ws = quantize_fp8r_weight(w)
    return data.contiguous(), wd, scales.float().reshape(-1).contiguous(), ws


def ref_mm(a, w, sa, sb):
    return torch._scaled_mm(a, w.t(), scale_a=sa.reshape(-1, 1), scale_b=sb.reshape(1, -1),
                            out_dtype=torch.bfloat16)


def compare(cand, ref):
    eq = cand.view(torch.int16) == ref.view(torch.int16)
    frac = eq.float().mean().item()
    d = (cand.float() - ref.float()).abs()
    tol = 2 ** -7 * ref.float().abs() + 2 ** -10 * ref.float().abs().max()
    ulp = (cand.view(torch.int16).int() - ref.view(torch.int16).int()).abs().max().item()
    return {"bitwise": bool(eq.all().item()), "bitwise_frac": frac, "max_abs": d.max().item(),
            "max_ulp": ulp, "within_tol": bool((d <= tol).all().item()),
            "nan": bool(torch.isnan(cand.float()).any().item())}


def main():
    step0.header()
    fg = build()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    exact = {}
    for name, (n, k) in SHAPES.items():
        for m in (4096, 4176, 4256):
            a, w, sa, sb = real_inputs(m, n, k, gen)
            ref = ref_mm(a, w, sa, sb)
            for arm in ARMS:
                for order in ORDERS:
                    out = torch.full((m, n), -1, dtype=torch.int16, device="cuda").view(torch.bfloat16)
                    fg.gemm(a, w, sa, sb, arm=arm, order=order, out=out)
                    torch.cuda.synchronize()
                    c = compare(out, ref)
                    exact[f"{name}/M{m}/arm{arm}/order{order}"] = c
                    log(f"[exact] {name} M{m} arm{arm} order{order} {c}")
    # Timing (M=4096).
    timing = {}
    for name, (n, k) in SHAPES.items():
        ns = step0.nsets_for(n, k)
        sets = step0.make_sets(n, k, ns, gen)
        row = {}
        fns = {"torch": lambda s: step0.scaled_mm(*s)}
        for arm in ARMS:
            fns[f"arm{arm}"] = (lambda s, arm=arm: fg.gemm(s[0], s[1], s[2], s[3], arm=arm))
        ki = {lab: step0.kernel_info(lambda f=f: f(sets[0])) for lab, f in fns.items()}
        for rep in range(2):  # interleaved, A/A
            for lab, f in fns.items():
                ev = step0.time_graph([(lambda s=sets[i % ns], f=f: f(s)) for i in range(step0.CALLS)])
                warm = step0.time_graph([(lambda s=sets[0], f=f: f(s)) for _ in range(step0.CALLS)])
                row.setdefault(lab, []).append({"evicted": ev, "warm": warm})
                log(f"[time] {name} {lab} rep{rep} EVICTED {ev['med']:.1f} "
                    f"(p10 {ev['p10']:.1f} p90 {ev['p90']:.1f}) WARM {warm['med']:.1f}")
        for lab in fns:
            log(f"[kern] {name} {lab} {ki[lab]}")
        timing[name] = {"rows": row, "kernels": ki}
        del sets
        torch.cuda.empty_cache()
    summary = {}
    for name, t in timing.items():
        summary[name] = {lab: round(sum(r["evicted"]["med"] for r in v) / len(v), 1)
                         for lab, v in t["rows"].items()}
    log(json.dumps({"exact": exact}))
    print(json.dumps({"summary_evicted_us": summary,
                      "exact_any_order": {k: v["bitwise"] for k, v in exact.items()}}), flush=True)


if __name__ == "__main__":
    main()
