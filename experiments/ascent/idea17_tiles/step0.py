"""Idea 17 STEP 0: SM120 NVFP4 / FP8R GEMM tiles at the 2048^2 workload (M ~ 16.9K / 33.3K).

Synthetic tensors at the real shapes (random NVFP4 bytes, E4M3 scale bytes in [0x30, 0x40)), no
model load. Per (shape, M): CUDA graph of CALLS calls alternating NSETS input sets (each set's
A + W + out exceeds the 128 MB L2 at M >= 16K), arms interleaved per round, median us/call and
(p90 - p10) / median.

Arms per shape:
  fi_t<idx>   FlashInfer stock SM120 fp4_gemm tactic idx (tile = idx // 4; idx % 4: 0 swap-AB DP,
              1 DP, 2 swap-AB StreamK, 3 StreamK). Tiles (FP4 elements): 0 128x32x128,
              1 128x32x256, 2 128x64x128, 3 128x64x256, 4 128x128x128, 5 128x128x256,
              6 256x128x128, 7 128x256x128. Bitwise vs t17 (128x128x128 DP = main's per-tile MMA).
  main_*      main's served kernels: qkv3 (Pingpong 128x128x128 CLC, L=3 planar), raster/swizzle
              sweeps of it, swiglu (Pingpong 128x128x128 CLC EPI_VEC), swiglu Cooperative
              256x128x128 builds (FLUX3_SCHED=0), lpt pair (down || attn_out, one launch).
  fp8r_arm<a> TRTLLM_FLUX3_FP8R_GEMM_ARM arms 0-4 and torch._scaled_mm (bitwise vs torch).

Env: STEP0_MS (default 16896,33280,4176), STEP0_SHAPES (qkv,gate_up,down,attn_out),
STEP0_PARTS (fi,main,fp8r), STEP0_TACTICS (default all DP + StreamK), STEP0_ROUNDS.
Last stdout line: JSON table.
"""

import json
import os
import time

import torch

CALLS = 6
NSETS = 2
ROUNDS = int(os.environ.get("STEP0_ROUNDS", "15"))
MS = [int(x) for x in os.environ.get("STEP0_MS", "16896,33280,4176").split(",")]
SHAPES = {"qkv": (9216, 3072), "gate_up": (18432, 3072), "down": (3072, 9216),
          "attn_out": (3072, 3072)}
SHAPE_SEL = os.environ.get("STEP0_SHAPES", "qkv,gate_up,down,attn_out").split(",")
PARTS = os.environ.get("STEP0_PARTS", "fi,main,fp8r").split(",")
TACTICS = [int(x) for x in os.environ.get("STEP0_TACTICS", ",".join(map(str, range(32)))).split(",")]
TILES = ["128x32x128", "128x32x256", "128x64x128", "128x64x256", "128x128x128", "128x128x256",
         "256x128x128", "128x256x128"]
KINDS = ["swapDP", "DP", "swapSK", "SK"]


def log(*a):
    print(*a, flush=True)


def sf_rows(m):
    return (m + 127) // 128 * 128


def fp4_set(m, n, k, g, dev):
    a = torch.randint(0, 256, (m, k // 2), generator=g, dtype=torch.uint8, device=dev)
    w = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8, device=dev)
    sfa = torch.randint(0x30, 0x40, (sf_rows(m) * (k // 16),), generator=g, dtype=torch.uint8,
                        device=dev)
    sfb = torch.randint(0x30, 0x40, (n * (k // 16),), generator=g, dtype=torch.uint8, device=dev)
    return a, w, sfa, sfb


def capture(fns):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    return g


def time_arms(arms, rounds=ROUNDS):
    """arms: {name: list of CALLS closures}. Interleaved graph replays; us/call stats."""
    graphs = {}
    for name, fns in arms.items():
        try:
            graphs[name] = capture(fns)
        except Exception as e:  # noqa: BLE001
            log(f"    [{name}] capture failed: {str(e)[:200]}")
            torch.cuda.synchronize()
    names = list(graphs)
    for n in names:
        graphs[n].replay()
    torch.cuda.synchronize()
    t = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        evs = []
        for n in order:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[n].replay()
            e1.record()
            evs.append((n, e0, e1))
        torch.cuda.synchronize()
        for n, e0, e1 in evs:
            t[n].append(e0.elapsed_time(e1) * 1000.0 / CALLS)
    res = {}
    for n, v in t.items():
        v = sorted(v)
        med = v[len(v) // 2]
        p10, p90 = v[int(0.1 * (len(v) - 1))], v[int(0.9 * (len(v) - 1))]
        res[n] = {"us": round(med, 2), "nf": round((p90 - p10) / med, 4)}
    del graphs
    torch.cuda.synchronize()
    return res


def bits_equal(x, y):
    return bool(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def run_fi_part(fp, m, name, n, k, dev, g, ws):
    mod = fp._sm120_fi_fp4_module()
    sets = [fp4_set(m, n, k, g, dev) for _ in range(NSETS)]
    alpha = torch.tensor([1.0 / 64], dtype=torch.float32, device=dev)
    outs = [torch.empty((m, n), dtype=torch.bfloat16, device=dev) for _ in range(NSETS)]
    ref = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    mod.fp4_gemm(*sets[0], alpha, ref, ws, 17)
    torch.cuda.synchronize()
    status = {}
    for t in TACTICS:
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=dev)
        try:
            mod.fp4_gemm(*sets[0], alpha, out, ws, t)
            torch.cuda.synchronize()
            eq = bits_equal(out, ref)
            st = "bitwise" if eq else ("reordered" if KINDS[t % 4].endswith("SK") else "DIFFERS")
            if not eq:
                d = (out.float() - ref.float()).abs()
                st += f" maxabs={float(d.max()):.3g}"
            status[f"fi_t{t}"] = st
        except Exception as e:  # noqa: BLE001
            status[f"fi_t{t}"] = f"error {str(e)[:120]}"
            torch.cuda.synchronize()
        del out
    arms = {}
    for t in TACTICS:
        if status[f"fi_t{t}"].startswith("error"):
            continue
        arms[f"fi_t{t}"] = [(lambda i=i, t=t: mod.fp4_gemm(*sets[i % NSETS], alpha, outs[i % NSETS],
                                                         ws, t)) for i in range(CALLS)]
    if "fi_t17" in arms:
        arms["fi_t17_AA"] = arms["fi_t17"]
    res = time_arms(arms)
    for kname in res:
        base = kname.replace("_AA", "")
        res[kname]["status"] = status.get(base, "")
        t = int(base[4:])
        res[kname]["cfg"] = f"{TILES[t // 4]} {KINDS[t % 4]}"
    del sets, outs, ref
    torch.cuda.empty_cache()
    return res


def run_main_part(fp, m, name, n, k, dev, g, ws):
    res = {}
    alpha = torch.tensor([1.0 / 64], dtype=torch.float32, device=dev)
    if name == "qkv":
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

        strided.preload(dev)
        mod = strided.qkv_pingpong_module()
        qws = strided.workspace(dev, 0)
        sets = [fp4_set(m, n, k, g, dev) for _ in range(NSETS)]  # n = 3 planes x 3072
        a3 = torch.tensor([1.0 / 64] * 3, dtype=torch.float32, device=dev)
        outs = [torch.empty(m * n, dtype=torch.bfloat16, device=dev) for _ in range(NSETS)]
        fi = fp._sm120_fi_fp4_module()
        ref = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        fi.fp4_gemm(*sets[0], alpha, ref, ws, 17)
        mod.gemm_qkv3_pingpong(*sets[0], a3, outs[0], qws, 0, 1)
        torch.cuda.synchronize()
        pl = n // 3
        eq = all(bits_equal(outs[0].view(3, m, pl)[p], ref[:, p * pl:(p + 1) * pl]) for p in range(3))
        arms = {}
        for r, s in [(0, 1), (1, 1), (2, 1), (0, 2), (0, 4), (0, 8), (1, 4), (2, 4)]:
            arms[f"main_qkv3_r{r}s{s}"] = [
                (lambda i=i, r=r, s=s: mod.gemm_qkv3_pingpong(*sets[i % NSETS], a3, outs[i % NSETS],
                                                              qws, r, s)) for i in range(CALLS)]
        arms["main_qkv3_r0s1_AA"] = arms["main_qkv3_r0s1"]
        res = time_arms(arms)
        for kk in res:
            res[kk]["status"] = (f"served (planes vs fi_t17 bitwise={eq})"
                                 if kk.startswith("main_qkv3_r0s1") else "raster sweep")
        del sets, outs, ref
    elif name == "gate_up":
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_swiglu_fp4_cutlass as swc

        mods = {"main_swiglu": swc.get_module()}
        for lab, flags in (("swiglu_coop256_dyn", ("-DFLUX3_SCHED=0", "-DFLUX3_TILESCHED=1")),
                           ("swiglu_coop256_dyn_ev", ("-DFLUX3_SCHED=0", "-DFLUX3_TILESCHED=1",
                                                      "-DFLUX3_EPI_VEC=1")),
                           ("swiglu_coop256_static", ("-DFLUX3_SCHED=0",))):
            if os.environ.get("STEP0_SWIGLU_VARIANTS", "1") != "1":
                break
            t0 = time.time()
            try:
                mods[lab] = swc._build("_i17_" + lab, flags)
                log(f"    built {lab} in {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                log(f"    build {lab} failed: {str(e)[-300:]}")
        sets = [fp4_set(m, n, k, g, dev) for _ in range(NSETS)]
        inv = torch.tensor([0.5], dtype=torch.float32, device=dev)
        bufs = [swc.allocate_outputs(m, n, dev) for _ in range(NSETS)]
        status = {}
        ref = None
        for lab, md in mods.items():
            o, os_ = swc.allocate_outputs(m, n, dev)
            o.fill_(0xAB)
            os_.fill_(0xCD)
            try:
                swc.swiglu_packed(sets[0][0], sets[0][1], sets[0][2], sets[0][3], alpha, inv,
                                  module=md, output=o, output_scale=os_)
                torch.cuda.synchronize()
                if ref is None:
                    ref = (o.clone(), os_.clone())
                    status[lab] = "served"
                else:
                    status[lab] = ("bitwise" if bits_equal(o, ref[0]) and bits_equal(os_, ref[1])
                                   else "DIFFERS")
            except Exception as e:  # noqa: BLE001
                status[lab] = f"error {str(e)[:120]}"
                torch.cuda.synchronize()
        arms = {}
        for lab, md in mods.items():
            if status[lab].startswith("error"):
                continue
            arms[lab] = [(lambda i=i, md=md: swc.swiglu_packed(
                sets[i % NSETS][0], sets[i % NSETS][1], sets[i % NSETS][2], sets[i % NSETS][3],
                alpha, inv, module=md, output=bufs[i % NSETS][0], output_scale=bufs[i % NSETS][1]))
                for i in range(CALLS)]
        arms["main_swiglu_AA"] = arms["main_swiglu"]
        res = time_arms(arms)
        for kk in res:
            res[kk]["status"] = status.get(kk.replace("_AA", ""), "")
        del sets, bufs
    elif name == "down":
        from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

        lpt.preload(dev)
        ds = [fp4_set(m, 3072, 9216, g, dev) for _ in range(NSETS)]
        at = [fp4_set(m, 3072, 3072, g, dev) for _ in range(NSETS)]
        od = [torch.empty((m, 3072), dtype=torch.bfloat16, device=dev) for _ in range(NSETS)]
        oa = [torch.empty((m, 3072), dtype=torch.bfloat16, device=dev) for _ in range(NSETS)]
        fi = fp._sm120_fi_fp4_module()
        rd = torch.empty((m, 3072), dtype=torch.bfloat16, device=dev)
        ra = torch.empty((m, 3072), dtype=torch.bfloat16, device=dev)
        fi.fp4_gemm(*ds[0], alpha, rd, ws, 17)
        fi.fp4_gemm(*at[0], alpha, ra, ws, 17)
        lpt.run(*ds[0], alpha, od[0], *at[0], alpha, oa[0])
        torch.cuda.synchronize()
        eq = bits_equal(od[0], rd) and bits_equal(oa[0], ra)

        def pair(i, mode):
            j = i % NSETS
            lpt.run(*ds[j], alpha, od[j], *at[j], alpha, oa[j], mode=mode)

        def fi2(i, t):
            j = i % NSETS
            fi.fp4_gemm(*ds[j], alpha, od[j], ws, t)
            fi.fp4_gemm(*at[j], alpha, oa[j], ws, t)

        arms = {"main_lpt_pair_dyn": [(lambda i=i: pair(i, 1)) for i in range(CALLS)],
                "lpt_pair_static": [(lambda i=i: pair(i, 0)) for i in range(CALLS)]}
        for t in (17, 21, 25, 29, 24):
            arms[f"fi_t{t}_down+attn"] = [(lambda i=i, t=t: fi2(i, t)) for i in range(CALLS)]
        arms["main_lpt_pair_dyn_AA"] = arms["main_lpt_pair_dyn"]
        res = time_arms(arms)
        for kk in res:
            res[kk]["status"] = f"served (vs fi_t17 bitwise={eq})" if kk.startswith("main_lpt") else ""
        del ds, at, od, oa
    torch.cuda.empty_cache()
    return res


def run_fp8r_part(m, name, n, k, dev, g):
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_fp8r_gemm as fg

    fg.preload(dev)

    def f8(shape):
        return (torch.randn(shape, device=dev, generator=g) * 2).clamp(-448, 448).to(
            torch.float8_e4m3fn)

    sets = []
    for _ in range(NSETS):
        sets.append((f8((m, k)), f8((n, k)), torch.rand(m, device=dev, generator=g) * 0.01 + 1e-3,
                     torch.rand(n, device=dev, generator=g) * 0.01 + 1e-3))
    outs = [torch.empty((m, n), dtype=torch.bfloat16, device=dev) for _ in range(NSETS)]

    def tmm(s):
        return torch._scaled_mm(s[0], s[1].t(), scale_a=s[2].reshape(-1, 1),
                                scale_b=s[3].reshape(1, -1), out_dtype=torch.bfloat16)

    ref = tmm(sets[0])
    status = {"torch_scaled_mm": "reference"}
    for arm in range(5):
        o = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=dev)
        try:
            fg.gemm(*sets[0], arm=arm, order=1, out=o)
            torch.cuda.synchronize()
            status[f"fp8r_arm{arm}"] = "bitwise" if bits_equal(o, ref) else "DIFFERS"
        except Exception as e:  # noqa: BLE001
            status[f"fp8r_arm{arm}"] = f"error {str(e)[:120]}"
            torch.cuda.synchronize()
    arms = {"torch_scaled_mm": [(lambda i=i: tmm(sets[i % NSETS])) for i in range(CALLS)]}
    for arm in range(5):
        if status[f"fp8r_arm{arm}"].startswith("error"):
            continue
        arms[f"fp8r_arm{arm}"] = [(lambda i=i, arm=arm: fg.gemm(*sets[i % NSETS], arm=arm, order=1,
                                                               out=outs[i % NSETS]))
                                  for i in range(CALLS)]
    if "fp8r_arm1" in arms:
        arms["fp8r_arm1_AA"] = arms["fp8r_arm1"]
    res = time_arms(arms)
    for kk in res:
        res[kk]["status"] = status.get(kk.replace("_AA", ""), "")
    del sets, outs, ref
    torch.cuda.empty_cache()
    return res


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    dev = torch.device("cuda")
    log(f"gpu={torch.cuda.get_device_name()} MS={MS} shapes={SHAPE_SEL} parts={PARTS}")
    t0 = time.time()
    fp._sm120_fi_fp4_module()
    log(f"fi module ready {time.time() - t0:.0f}s")
    ws = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(17)
    table = {}
    for m in MS:
        for name in SHAPE_SEL:
            n, k = SHAPES[name]
            key = f"M{m}/{name}"
            table[key] = {}
            for part in PARTS:
                t1 = time.time()
                if part == "fi":
                    r = run_fi_part(fp, m, name, n, k, dev, g, ws)
                elif part == "main":
                    if name == "attn_out":
                        continue
                    r = run_main_part(fp, m, name, n, k, dev, g, ws)
                elif part == "fp8r":
                    r = run_fp8r_part(m, name, n, k, dev, g)
                else:
                    continue
                table[key].update(r)
                log(f"[{key}] part={part} ({time.time() - t1:.0f}s)")
                for an, v in sorted(r.items(), key=lambda kv: kv[1]["us"]):
                    log(f"    {an:28s} {v['us']:10.2f} us  nf {v['nf']:.4f}  {v.get('cfg', '')} "
                        f"{v.get('status', '')}")
    print(json.dumps({"table": table}), flush=True)


if __name__ == "__main__":
    main()
