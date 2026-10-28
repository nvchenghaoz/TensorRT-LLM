"""Idea 200: batched L=3 QKV NVFP4 Pingpong GEMM, static vs CLC dynamic tile scheduler.

Modes:
  bench (default): graph-replay timing of the served call strided.gemm_qkv3 (whatever the tree
      routes), plus direct static / dynamic module arms when present, interleaved, S4176/S4096.
      Last stdout line: evidence JSON (batch "1" = M4176 routed call, us -> ms).
  check: untimed correctness. q/k/v planes of the dynamic kernel vs the static kernel, bitwise,
      at M 4096/4176/4256; 0xFF prefill; eager calls 0..3 vs call 0; graph capture + 3 replays
      (re-prefilled 0xFF before each); inputs unchanged; kill switch static -> main module.
"""

import json
import os
import subprocess
import sys
import time

import torch

MAIN_AT_START = "cec13c57"
N = 3072
K = 3072


def log(*a):
    print(*a, flush=True)


def header():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        rc = subprocess.run(["git", "merge-base", "--is-ancestor", MAIN_AT_START, "HEAD"]).returncode
    except Exception as e:  # noqa: BLE001
        head, rc = f"n/a ({e})", -1
    import tensorrt_llm

    log(f"HEAD={head.strip()} MAIN_AT_START={MAIN_AT_START} ancestry_rc={rc}")
    log(f"tensorrt_llm.__file__={tensorrt_llm.__file__}")


def sf_rows(m):
    return (m + 127) // 128 * 128


def make_inputs(m, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    a = torch.randint(0, 256, (m, K // 2), generator=g, dtype=torch.uint8).to(dev)
    w = torch.randint(0, 256, (3 * N, K // 2), generator=g, dtype=torch.uint8).to(dev)
    # E4M3 scale bytes in [0x30, 0x40): finite, ~2^-3..2^0 (no NaN 0x7F/0xFF).
    sfa = torch.randint(0x30, 0x40, (sf_rows(m) * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    sfb = torch.randint(0x30, 0x40, (3 * N * (K // 16),), generator=g, dtype=torch.uint8).to(dev)
    alpha = torch.tensor([0.75, 1.25, 0.5], dtype=torch.float32, device=dev)
    out = torch.empty(3 * m * N, dtype=torch.bfloat16, device=dev)
    return a, w, sfa, sfb, alpha, out


def arms(strided):
    ws = strided.workspace(torch.device("cuda"), 0)
    res = {"routed": lambda a, w, sa, sb, al, o: strided.gemm_qkv3(a, w, sa, sb, al, o)}
    st = strided.pingpong_module()
    res["static"] = lambda a, w, sa, sb, al, o: st.gemm_qkv3_pingpong(a, w, sa, sb, al, o, ws, 0, 1)
    if hasattr(strided, "pingpong_dyn_module"):
        dy = strided.pingpong_dyn_module()
        res["dynamic"] = lambda a, w, sa, sb, al, o: dy.gemm_qkv3_pingpong(
            a, w, sa, sb, al, o, ws, 0, 1
        )
    # A/A control: a second static arm.
    res["static_aa"] = res["static"]
    return res


def bench(strided):
    dev = torch.device("cuda")
    fns = arms(strided)
    log(f"arms={list(fns)} sched={strided.qkv_schedule()} "
        f"tilesched={getattr(strided, 'qkv_tilesched', lambda: 'n/a')()}")
    R = int(os.environ.get("QKV3_BENCH_R", "10"))
    ITERS = int(os.environ.get("QKV3_BENCH_ITERS", "60"))
    WARM = 5
    results = {}
    for m in (4176, 4096):
        inp = make_inputs(m, dev)
        a, w, sfa, sfb, alpha, out = inp
        if hasattr(strided, "pingpong_dyn_module"):
            dy = strided.pingpong_dyn_module()
            ws = strided.workspace(dev, 0)
            log(f"M{m}: dyn grid={int(dy.qkv3_grid(a, w, sfa, sfb, alpha, out, ws))} "
                f"tiles={((m + 127) // 128) * (N // 128) * 3}")
        graphs = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for name, fn in fns.items():
                for _ in range(3):
                    fn(a, w, sfa, sfb, alpha, out)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for name, fn in fns.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    fn(a, w, sfa, sfb, alpha, out)
            graphs[name] = g
        for name, g in graphs.items():
            for _ in range(WARM):
                g.replay()
        torch.cuda.synchronize()
        times = {n: [] for n in graphs}
        for it in range(ITERS):
            for name, g in graphs.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000.0 / R)
        row = {}
        for name, t in times.items():
            t = sorted(t)
            med = t[len(t) // 2]
            p10, p90 = t[int(0.1 * len(t))], t[int(0.9 * len(t)) - 1]
            row[name] = {"us": round(med, 2), "noise": round((p90 - p10) / med, 4)}
        log(f"M{m} us/call: " + ", ".join(f"{n}={v['us']} (nf {v['noise']})" for n, v in row.items()))
        results[m] = row
        del graphs
    r = results[4176]["routed"]
    return {
        "batch": {"1": {"kernel_ms": r["us"] / 1000.0, "noise_floor": r["noise"]}},
        "output": {"class": "exact"},
        "detail_us": {str(m): v for m, v in results.items()},
    }


def check(strided):
    dev = torch.device("cuda")
    st = strided.pingpong_module()
    dy = strided.pingpong_dyn_module()
    ws = strided.workspace(dev, 0)
    info = {
        "smem": [int(st.smem_pingpong()), int(dy.smem_pingpong())],
        "stages": [int(st.stages_pingpong()), int(dy.stages_pingpong())],
        "ws_static": int(st.qkv3_workspace_size(1 << 16, N, K)),
        "ws_dynamic": int(dy.qkv3_workspace_size(1 << 16, N, K)),
        "ws_cap": strided._WORKSPACE_BYTES,
        "routed_is_dyn": strided.qkv_pingpong_module() is dy,
    }
    log(f"info {info}")
    ok = info["ws_dynamic"] <= info["ws_cap"] and info["routed_is_dyn"]
    fails = []
    for m in (4096, 4176, 4256):
        inp = make_inputs(m, dev)
        a, w, sfa, sfb, alpha, out = inp
        pristine = [t.clone() for t in (a, w, sfa, sfb, alpha)]
        grid = int(dy.qkv3_grid(a, w, sfa, sfb, alpha, out, ws))
        out.view(torch.int16).fill_(-1)
        st.gemm_qkv3_pingpong(a, w, sfa, sfb, alpha, out, ws, 0, 1)
        torch.cuda.synchronize()
        ref = out.clone()
        assert not torch.isnan(ref.float()).any(), "static ref has NaN (unwritten)"
        call0 = None
        for i in range(4):
            out.view(torch.int16).fill_(-1)
            strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)  # routed = dynamic
            torch.cuda.synchronize()
            if call0 is None:
                call0 = out.clone()
            for p, nm in enumerate("qkv"):
                sl = slice(p * m * N, (p + 1) * m * N)
                if not torch.equal(out[sl].view(torch.int16), ref[sl].view(torch.int16)):
                    fails.append(f"M{m} eager call{i} plane {nm} != static")
            if not torch.equal(out.view(torch.int16), call0.view(torch.int16)):
                fails.append(f"M{m} eager call{i} != call0")
        # Graph capture + 3 replays.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)
        for r in range(3):
            out.view(torch.int16).fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            for p, nm in enumerate("qkv"):
                sl = slice(p * m * N, (p + 1) * m * N)
                if not torch.equal(out[sl].view(torch.int16), ref[sl].view(torch.int16)):
                    fails.append(f"M{m} replay{r} plane {nm} != static")
        for t, p0 in zip((a, w, sfa, sfb, alpha), pristine):
            if not torch.equal(t, p0):
                fails.append(f"M{m} input mutated")
        del g
        log(f"M{m}: grid={grid} tiles={((m + 127) // 128) * (N // 128) * 3} "
            f"fails_so_far={len(fails)}")
        last = (inp, ref)
    # Kill switch: static restores main's module and output.
    os.environ["TRTLLM_FLUX3_SM120_QKV_TILESCHED"] = "static"
    strided.qkv_tilesched.cache_clear()
    ks_mod = strided.qkv_pingpong_module() is st
    (a, w, sfa, sfb, alpha, out), ref = last
    out.view(torch.int16).fill_(-1)
    strided.gemm_qkv3(a, w, sfa, sfb, alpha, out)
    torch.cuda.synchronize()
    ks_eq = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if not (ks_mod and ks_eq):
        fails.append(f"kill switch: module_is_static={ks_mod} bitwise={ks_eq}")
    passed = ok and not fails
    return {
        "passed": passed,
        "compared": "dynamic (CLC) gemm_qkv3 vs main static gemm_qkv3_pingpong, q/k/v bitwise, "
        "M4096/4176/4256, eager x4 + graph x3, 0xFF prefill, kill switch",
        "info": info,
        "fails": fails[:20],
    }


def main():
    header()
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_strided_gemm as strided

    t0 = time.time()
    strided.preload(torch.device("cuda"))
    if mode == "check" or hasattr(strided, "pingpong_dyn_module"):
        strided.pingpong_module()
        if hasattr(strided, "pingpong_dyn_module"):
            strided.pingpong_dyn_module()
    log(f"modules ready in {time.time() - t0:.1f}s")
    res = check(strided) if mode == "check" else bench(strided)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
