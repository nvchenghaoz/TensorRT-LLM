"""Small-M gate for the gate_up gelu epilogue rebuild (area-lead ruling 3253): M <= 512 unchanged vs main.

One process, one GPU, no model load. Real _Gemma4GeluQuantMLP modules (NVFP4 QuantConfig) built from
synthetic checkpoint-format weights, 3 copies per arm (cold weights: 3 x (123 + 62) MB > L2):
  main = TRTLLM_GEMMA4_GATEUP_GELU_EPI=0 modules (main's forward, original gate_up rows)
  cand = TRTLLM_GEMMA4_GATEUP_GELU_EPI=1 modules (gate_up rows permuted, _gelu_epi_forward)
Tuning: engine-like autotune over MS_TUNE on a main module (both arms then serve the same picks).

TIMING, arms in A,B,A order (main, cand, main2). Before each arm: >= 2 s of BF16 8192^3 GEMMs (SM at
boost), then SM clock / temperature logged (nvidia-smi). Per arm, per M in MS: 5 warmup + 25 timed whole
MLP forwards (call i on copy i % 3) with CUDA events around the forward (whole-MLP GPU time incl. launch
gaps), plus host time per call (perf_counter around the forward, synchronized before), then a
torch.profiler pass of 15 calls giving per-kernel times: gate_up GEMM (kernels before the act), act
(name has 'gelu'), down GEMM (kernels after). Medians and p10/p90.

REPEATED-CALL CHECK (untimed, after timing): per M in MS, passes A (MS), B (MS reversed), A2 (MS), 35
calls per M per pass = 105 calls per M, call i on copy i % 3, no reset. Every call's down_proj input
(FP4 payload + full SF; every uint8 torch.empty in the chain modules 0xA5-filled) and down_proj output
must equal that (M, copy)'s call 1 and main's (EPI=0 module, same copy) bitwise; inputs unchanged.
usage: small_m_gate.py <checkpoint>
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time

T0 = time.time()
MS = [1, 16, 64, 300, 512]
MS_TUNE = [1, 16, 64, 300, 512, 1000, 2000]
WARMUP, ITERS, PROF_ITERS = 5, 25, 15


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(45)
        log("heartbeat")


def smi():
    try:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                              "clocks_throttle_reasons.active", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]


def stats(xs):
    return {"med": round(statistics.median(xs), 2), "p10": round(q(xs, 0.1), 2), "p90": round(q(xs, 0.9), 2)}


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ck = sys.argv[1]
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    H, I = int(tc["hidden_size"]), int(tc["intermediate_size"])
    log(f"hidden={H} intermediate={I}")
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.kernels.fused_ops import gate_up_gelu_fp4_chunked as gcm
    from tensorrt_llm._torch.kernels.fused_ops import gelu_tanh_mul_fp4_quant as gqm
    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    dev = torch.device("cuda")
    assert ge.get_gelu_gemm_module() is not None, ge._module_error
    g = torch.Generator(device=dev)
    g.manual_seed(512)

    def q_ckpt(w, amax):
        gs = (448 * 6 / amax).reshape(1).float()
        fp4, sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, False)
        return (fp4.view(torch.uint8).reshape(w.shape[0], w.shape[1] // 2),
                sf.view(torch.uint8).reshape(-1)[: w.shape[0] * (w.shape[1] // 16)]
                .view(torch.float8_e4m3fn).reshape(w.shape[0], w.shape[1] // 16))

    mc = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16))
    in_amax = torch.tensor(24.0, device=dev)
    copies = []
    for _ in range(3):
        wg = torch.randn(I, H, device=dev, generator=g).to(torch.bfloat16)
        wu = torch.randn(I, H, device=dev, generator=g).to(torch.bfloat16)
        wd = (torch.randn(H, I, device=dev, generator=g) * 0.05).to(torch.bfloat16)
        a_gu = torch.maximum(wg.float().abs().max(), wu.float().abs().max())
        a_d = wd.float().abs().max()
        gf, gsf = q_ckpt(wg, a_gu)
        uf, usf = q_ckpt(wu, a_gu)
        df, dsf = q_ckpt(wd, a_d)
        del wg, wu, wd
        ws2 = (a_gu / (448 * 6)).float()
        isc = (in_amax / (448 * 6)).float()
        copies.append(dict(gate=dict(weight=gf, weight_scale=gsf, weight_scale_2=ws2.clone(), input_scale=isc),
                           up=dict(weight=uf, weight_scale=usf, weight_scale_2=ws2.clone(), input_scale=isc),
                           down=dict(weight=df, weight_scale=dsf, weight_scale_2=(a_d / (448 * 6)).float(),
                                     input_scale=torch.tensor(0.0125, device=dev))))
    torch.cuda.empty_cache()

    def build(c, epi):
        os.environ["TRTLLM_GEMMA4_GATEUP_GELU_EPI"] = "1" if epi else "0"
        with torch.device("cuda"):
            m = mg._Gemma4GeluQuantMLP(hidden_size=H, intermediate_size=I, bias=False, activation=mg.gelu_tanh,
                                       dtype=torch.bfloat16, config=mc, layer_idx=0)
        cc = copies[c]
        m.gate_up_proj.load_weights([cc["gate"], cc["up"]])
        m.down_proj.load_weights([dict(cc["down"])])
        for sub in m.modules():
            f = getattr(sub, "post_load_weights", None)
            if f is not None and not getattr(sub, "_weights_removed", False):
                f()
        os.environ["TRTLLM_GEMMA4_GATEUP_GELU_EPI"] = "1"
        return m

    arms = {"main": [build(c, False) for c in range(3)], "cand": [build(c, True) for c in range(3)]}
    perm = ([m._gelu_epi_permuted for m in arms["cand"]], [m._gelu_epi_permuted for m in arms["main"]])
    log(f"modules: cand permuted {perm[0]}, main permuted {perm[1]}")
    acts = {}
    for M in sorted(set(MS_TUNE)):
        a = torch.randn(M, H, device=dev, generator=g).to(torch.bfloat16)
        fp4, sf = torch.ops.trtllm.fp4_quantize(a, arms["main"][0].gate_up_proj.input_scale, 16, False)
        acts[M] = (fp4, sf)
    snap = {M: (v[0].clone(), v[1].clone()) for M, v in acts.items()}

    def X(M):
        return Fp4QuantizedTensor(acts[M][0], acts[M][1])

    with torch.inference_mode(), autotune():
        for M in MS_TUNE:
            arms["main"][0](X(M))
    torch.cuda.synchronize()
    log("tuned")

    wa = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)

    def warm(sec=2.5):
        t = time.time()
        while time.time() - t < sec:
            for _ in range(10):
                wa @ wa
            torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    report = {"order": ["main", "cand", "main2"], "arms": {}}
    with torch.inference_mode():
        for arm_tag in ("main", "cand", "main2"):
            mods = arms["main" if arm_tag.startswith("main") else "cand"]
            warm()
            clk = smi()
            log(f"arm {arm_tag}: smi {clk}")
            arow = {"smi_at_start": clk, "M": {}}
            for M in MS:
                x = X(M)
                for i in range(WARMUP):
                    mods[i % 3](x)
                torch.cuda.synchronize()
                gpu, host = [], []
                for i in range(ITERS):
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    s.record()
                    mods[i % 3](x)
                    e.record()
                    host.append((time.perf_counter() - t0) * 1e6)
                    e.synchronize()
                    gpu.append(s.elapsed_time(e) * 1e3)
                torch.cuda.synchronize()
                per = {"gate_up": [], "act": [], "down": []}
                names = {"gate_up": set(), "act": set(), "down": set()}
                fwd = []
                for i in range(PROF_ITERS):  # one forward per profiler session: before act = gate_up, after = down
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        mods[i % 3](x)
                        torch.cuda.synchronize()
                    evs = sorted([ev for ev in prof.events()
                                  if ev.device_type is not None and "CUDA" in str(ev.device_type)
                                  and ev.device_time_total > 0 and "Memcpy" not in ev.name
                                  and "Memset" not in ev.name],
                                 key=lambda ev: ev.time_range.start)
                    cur, seen_act = {"gate_up": 0.0, "act": 0.0, "down": 0.0}, False
                    for ev in evs:
                        nm = ev.name
                        if "gelu" in nm.lower():
                            seen_act = True
                            k = "act"
                        else:
                            k = "down" if seen_act else "gate_up"
                        cur[k] += ev.device_time_total
                        names[k].add(nm[:90])
                    if seen_act:
                        fwd.append(cur)
                        for k in per:
                            per[k].append(cur[k])
                row = {"mlp_gpu_us": stats(gpu), "host_us": stats(host), "n_profiled_forwards": len(fwd)}
                for k in per:
                    row[f"{k}_us"] = stats(per[k]) if per[k] else None
                row["kernels"] = {k: sorted(v) for k, v in names.items()}
                arow["M"][str(M)] = row
                log(f"{arm_tag} M={M} mlp {row['mlp_gpu_us']} host {row['host_us']} gate_up {row['gate_up_us']} "
                    f"act {row['act_us']} down {row['down_us']} n={len(fwd)}")
            report["arms"][arm_tag] = arow

    # ---- summary: candidate vs the mean of the two main arms ----
    summ = {}
    for M in MS:
        r = {}
        for k in ("mlp_gpu_us", "gate_up_us", "act_us", "down_us", "host_us"):
            vals = [report["arms"][a]["M"][str(M)][k] for a in ("main", "cand", "main2")]
            if any(v is None for v in vals):
                continue
            mm = (vals[0]["med"] + vals[2]["med"]) / 2
            r[k] = {"main": vals[0]["med"], "cand": vals[1]["med"], "main2": vals[2]["med"],
                    "cand_vs_main_mean": round(vals[1]["med"] / mm - 1, 4),
                    "main_vs_main2": round(vals[0]["med"] / vals[2]["med"] - 1, 4)}
        summ[str(M)] = r
    report["summary"] = summ
    log(f"summary {json.dumps(summ)}")

    # ---- repeated-call check (untimed) ----
    class A5Torch:
        def __init__(self, t):
            self._t = t

        def __getattr__(self, n):
            return getattr(self._t, n)

        def empty(self, *a, **k):
            x = self._t.empty(*a, **k)
            if x.dtype == self._t.uint8:
                x.fill_(0xA5)
            return x

    for mod in (ge, gcm, gqm):
        mod.torch = A5Torch(torch)
    CAP = []

    def wrap(m):
        dp = m.down_proj
        orig = dp.forward

        def fwd(inp, *a, **k):
            out = orig(inp, *a, **k)
            CAP.append((inp.fp4_tensor, inp.scaling_factor, out))
            return out
        dp.forward = fwd

    for m in arms["main"] + arms["cand"]:
        wrap(m)

    def run(m, M):
        CAP.clear()
        m(X(M))
        r = CAP[0]
        CAP.clear()
        return r

    def eq(a, b):
        return bool(a[0].shape == b[0].shape and a[1].shape == b[1].shape and torch.equal(a[0], b[0])
                    and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))

    rep = {}
    ok = all(perm[0]) and not any(perm[1])
    with torch.inference_mode():
        refs = {(M, c): tuple(t.clone() for t in run(arms["main"][c], M)) for M in MS for c in range(3)}
        firsts = {}
        p0 = dict(ge.path_calls)
        for pname, order in (("A", MS), ("B", MS[::-1]), ("A2", MS)):
            for M in order:
                st = rep.setdefault(str(M), {"calls": 0, "bad_vs_first": 0, "bad_vs_main": 0})
                for i in range(WARMUP + 30):
                    key = (M, i % 3)
                    r = run(arms["cand"][i % 3], M)
                    if key not in firsts:
                        firsts[key] = tuple(t.clone() for t in r)
                    st["calls"] += 1
                    st["bad_vs_first"] += int(not eq(r, firsts[key]))
                    st["bad_vs_main"] += int(not eq(r, refs[key]))
            torch.cuda.synchronize()
            log(f"repeat pass {pname}: {rep}")
        paths = {k: ge.path_calls[k] - p0[k] for k in p0}
    inputs_ok = all(torch.equal(acts[M][0], snap[M][0]) and torch.equal(acts[M][1], snap[M][1]) for M in acts)
    rep_ok = all(v["calls"] == 105 and v["bad_vs_first"] == 0 and v["bad_vs_main"] == 0 for v in rep.values())
    ok = ok and rep_ok and inputs_ok
    print(json.dumps({"passed": bool(ok), "repeat": rep, "repeat_paths": paths, "inputs_unchanged": inputs_ok,
                      "compared": "cand (EPI=1, permuted rows) vs main (EPI=0) _Gemma4GeluQuantMLP: down_proj input "
                                  "FP4 + full 0xA5-prefilled SF + down output bitwise, 105 calls per M, A,B,A",
                      **report}), flush=True)


if __name__ == "__main__":
    main()
