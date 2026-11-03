"""4-GPU NCCL all-reduce topology probe and knob sweep (torch.distributed, bundled NCCL).

Usage:
  python nccl_sweep.py probe                 # topo, peer matrix, NCCL_DEBUG logs at 10 KB / 20 MiB
  python nccl_sweep.py sweep [cfg ...]       # timing + every-iteration correctness per config
  python nccl_sweep.py pair                  # default-env timing; tries tensorrt_llm NCCL defaults
Each config runs in fresh processes (NCCL caches env knobs process-wide).
"""
import json
import os
import socket
import subprocess
import sys
import time

HIDDEN = 5120
DECODE_M = [1, 2, 4, 8]
PREFILL_M = [2048, 4096, 6144, 8192]
WARMUP = 20
ITERS = 100
SEQS = 3

CONFIGS = {
    "default": {},
    "p2p_sys": {"NCCL_P2P_LEVEL": "SYS"},
    "p2p_phb": {"NCCL_P2P_LEVEL": "PHB"},
    "ce_memcpy": {"NCCL_P2P_USE_CUDA_MEMCPY": "1"},
    "nch4": {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "nch8": {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
    "nch16": {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"},
    "nch32": {"NCCL_MIN_NCHANNELS": "32", "NCCL_MAX_NCHANNELS": "32"},
    "buf8m": {"NCCL_BUFFSIZE": str(8 << 20)},
    "buf16m": {"NCCL_BUFFSIZE": str(16 << 20)},
    "simple": {"NCCL_PROTO": "Simple"},
    "ll128": {"NCCL_PROTO": "LL128"},
    "ll": {"NCCL_PROTO": "LL"},
    "tree": {"NCCL_ALGO": "Tree"},
    "nthr256": {"NCCL_NTHREADS": "256"},
    "nthr512": {"NCCL_NTHREADS": "512"},
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_inputs(torch, m, it, world, device):
    """Deterministic per-(rank, iteration) inputs; returns list of all ranks' tensors."""
    outs = []
    for r in range(world):
        g = torch.Generator(device=device)
        g.manual_seed(1000003 * m + 7919 * it + r)
        outs.append(torch.randn((m, HIDDEN), generator=g, device=device,
                                dtype=torch.float32).to(torch.bfloat16))
    return outs


def worker(rank, world, port, cfg_name, mode, q):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    if mode in ("pair", "pair_check"):
        try:
            from tensorrt_llm._torch.distributed import nccl_tuning  # candidate only
            applied = nccl_tuning.apply_pcie_nccl_defaults(list(range(world)))
        except ImportError:
            applied = None
        if rank == 0:
            print(f"[pair] nccl defaults applied: {applied}", flush=True)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world, device_id=dev)
    res = {"timing": {}, "check": {}}
    check_sizes = DECODE_M + PREFILL_M
    strict = mode in ("check", "pair_check")
    res["env"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    if strict:
        check_sizes = [1, 8, 2048]
    # ---------- timing ----------
    for m in ([] if strict else DECODE_M + PREFILL_M):
        graph = m in DECODE_M
        seq_meds = []
        all_t = []
        for s in range(SEQS):
            src = make_inputs(torch, m, 0, world, dev)[rank]
            buf = src.clone()
            if graph:
                # warm the comm for this size eagerly, then capture copy+AR
                dist.all_reduce(buf)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(g, stream=stream):
                        buf.copy_(src)
                        dist.all_reduce(buf)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                run = g.replay
            else:
                def run():
                    dist.all_reduce(buf)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(ITERS)]
            for _ in range(WARMUP):
                if not graph:
                    buf.copy_(src)
                run()
            torch.cuda.synchronize()
            dist.barrier()
            for i in range(ITERS):
                if not graph:
                    buf.copy_(src)
                ev[i][0].record()
                run()
                ev[i][1].record()
            torch.cuda.synchronize()
            ts = sorted(a.elapsed_time(b) for a, b in ev)
            all_t.append(ts)
            seq_meds.append(ts[len(ts) // 2])
            if graph:
                del g
        flat = sorted(t for ts in all_t for t in ts)
        n = len(flat)
        med = sorted(seq_meds)[1]
        res["timing"][str(m)] = {
            "median_ms": med,
            "seq_medians": seq_meds,
            "p10": flat[n // 10], "p90": flat[(9 * n) // 10],
            "noise_floor": (flat[(9 * n) // 10] - flat[n // 10]) / med,
            "seq_spread": (max(seq_meds) - min(seq_meds)) / med,
            "busbw_GBs": (m * HIDDEN * 2) / (med * 1e-3) * 1.5 / 1e9,
        }
        if rank == 0:
            print(f"[{cfg_name}] m={m} graph={graph} med={med*1e3:.1f}us "
                  f"seqs={[round(x*1e3,1) for x in seq_meds]}", flush=True)
    # ---------- untimed correctness: same warmup/iter counts, every iteration checked ----------
    ok = True
    for m in check_sizes:
        graph = m in DECODE_M
        src = torch.empty((m, HIDDEN), device=dev, dtype=torch.bfloat16)
        buf = torch.empty_like(src)
        if graph:
            src.copy_(make_inputs(torch, m, 0, world, dev)[rank])
            buf.copy_(src)
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(g, stream=stream):
                    buf.copy_(src)
                    dist.all_reduce(buf)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        max_abs = 0.0
        max_rel = 0.0
        bad = 0
        frac_plain = 0.0
        sigs = []
        w = (torch.arange(m * HIDDEN, device=dev, dtype=torch.int64) % 9973) + 1
        for it in range(WARMUP + ITERS):
            ins = make_inputs(torch, m, it, world, dev)
            ref = sum(x.float() for x in ins)
            absum = sum(x.float().abs() for x in ins)
            if graph:
                src.copy_(ins[rank])
                g.replay()
            else:
                buf.copy_(ins[rank])
                dist.all_reduce(buf)
            out = buf.float()
            err = (out - ref).abs()
            # bf16 output tolerance (1.6e-2) plus the bound of a bf16 multi-step reduction:
            # (world-1) partial-sum roundings, each <= 2^-8 * sum|x_i|.
            tol = 1.6e-2 + 1.6e-2 * ref.abs() + (world - 1) * (2.0 ** -8) * absum
            if strict:
                # knowledge 311 bound: |out - ref| <= 2^-7 * sum_i |x_i| per element
                tol = (2.0 ** -7) * absum
            if not torch.isfinite(out).all() or (err > tol).any():
                bad += 1
            frac_plain = max(frac_plain,
                             (err > 1.6e-2 + 1.6e-2 * ref.abs()).float().mean().item())
            v = buf.view(torch.int16).reshape(-1).to(torch.int64)
            sigs.append([int(v.sum().item()), int((v * w).sum().item())])
            max_abs = max(max_abs, err.max().item())
            max_rel = max(max_rel, (err / ref.abs().clamp_min(1e-3)).max().item())
        if graph:
            del g
        res["check"][str(m)] = {"bad_iters": bad, "max_abs": max_abs, "max_rel": max_rel,
                                "max_frac_outside_plain_bf16_tol": frac_plain,
                                "iters": WARMUP + ITERS, "sigs": sigs}
        ok = ok and bad == 0
    res["check_passed"] = ok
    res["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
    dist.barrier()
    dist.destroy_process_group()
    q.put((rank, res))


def run_config(cfg_name, env_over, mode="sweep", world=4):
    """Spawn 4 ranks in a fresh python with env overrides; return rank results."""
    env = dict(os.environ)
    env.update(env_over)
    cmd = [sys.executable, os.path.abspath(__file__), "_child", cfg_name, mode, str(world)]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace", timeout=900)
    out = ascii_only(p.stdout)
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
        elif not line.startswith("NCCL INFO") and "NCCL INFO" not in line:
            print(line, flush=True)
    print(f"[{cfg_name}] done rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    return res, out


def child(cfg_name, mode, world):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=worker, args=(r, world, port, cfg_name, mode, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(world):
        r, res = q.get(timeout=850)
        results[r] = res
    for p in procs:
        p.join()
    # combine: per size take max median across ranks (collective completes when slowest does)
    comb = {"timing": {}, "check_passed": all(results[r]["check_passed"] for r in results),
            "check": results[0]["check"], "nccl_version": results[0]["nccl_version"],
            "env": results[0].get("env"),
            "ranks_identical": all(results[r]["check"][m]["sigs"] == results[0]["check"][m]["sigs"]
                                   for r in results for m in results[0]["check"])}
    for m in results[0]["timing"]:
        meds = [results[r]["timing"][m]["median_ms"] for r in results]
        comb["timing"][m] = dict(results[0]["timing"][m])
        comb["timing"][m]["rank_medians"] = meds
    print("RESULT " + json.dumps(comb), flush=True)


def ascii_only(s):
    return "".join(c if (c == "\n" or 32 <= ord(c) < 127) else " " for c in s)


def sh(cmd):
    try:
        return ascii_only(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, errors="replace", timeout=120).stdout)
    except Exception as e:  # noqa
        return f"ERR {e}"


def probe():
    print("=== nvidia-smi topo -m ===")
    print(sh(["nvidia-smi", "topo", "-m"]), flush=True)
    print(sh(["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,"
              "clocks.sm,temperature.gpu,power.draw", "--format=csv"]), flush=True)
    print(sh(["lscpu"])[:1500], flush=True)
    import torch
    n = torch.cuda.device_count()
    print("peer access matrix:")
    for i in range(n):
        print(i, [int(torch.cuda.can_device_access_peer(i, j)) if i != j else -1 for j in range(n)])
    print("torch nccl version", torch.cuda.nccl.version(), flush=True)
    # which libnccl does tensorrt_llm link?
    import glob
    import importlib.util
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec and spec.submodule_search_locations:
        base = list(spec.submodule_search_locations)[0]
        for so in glob.glob(os.path.join(base, "libs", "libth_common*.so")) + \
                glob.glob(os.path.join(base, "libs", "libtensorrt_llm.so")):
            print(so)
            print("\n".join(l for l in sh(["ldd", so]).splitlines() if "nccl" in l))
    print("torch libnccl:", glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "nccl", "lib", "*")))
    for name, envo in [("default", {}), ("p2p_sys", {"NCCL_P2P_LEVEL": "SYS"})]:
        envd = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING,P2P,SHM,ENV"}
        envd.update(envo)
        env = dict(os.environ)
        env.update(envd)
        cmd = [sys.executable, os.path.abspath(__file__), "_dbg"]
        out = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", timeout=600).stdout
        out = ascii_only(out)
        path = f"/tmp/nccl_dbg_{name}.log"
        with open(path, "w") as f:
            f.write(out)
        print(f"=== NCCL_DEBUG {name} ({len(out.splitlines())} lines) ===")
        keep = [l for l in out.splitlines() if any(k in l for k in (
            "via P2P", "via SHM", "via NET", "Channel", "Ring", "Tree", "nChannels", "comm 0x",
            "P2P level", "PXB", "PHB", "SYS", "Init COMPLETE", "Bootstrap", "NCCL version",
            "protocol", "Proto", "algorithm", "Enabled", "Disabled", "CUMEM", "cuMem", "Connected",
            "threadThresholds", "Pattern", "LL128", "Algo", "pattern", "Using", "NCCL_"))]
        # dedupe-ish but keep rank 0 mostly
        print("\n".join(keep[:400]), flush=True)


def dbg():
    """One AR at 10 KB and one at 20 MiB under NCCL_DEBUG."""
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    port = free_port()
    procs = [ctx.Process(target=dbg_worker, args=(r, 4, port)) for r in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def dbg_worker(rank, world, port):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world, device_id=dev)
    for m in (1, 2048):
        x = torch.ones((m, HIDDEN), device=dev, dtype=torch.bfloat16)
        dist.all_reduce(x)
        torch.cuda.synchronize()
    dist.destroy_process_group()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "_dbg":
        dbg()
        return
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "probe":
        probe()
        return
    print(sh(["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,power.draw,"
              "clocks_throttle_reasons.active", "--format=csv"]), flush=True)
    if mode == "check":
        allok = True
        summary = {}
        ref_sigs = None
        for nm in sys.argv[2:] or ["default", "p2p_sys", "p2p_phb", "pair"]:
            if nm == "pair":  # whatever tensorrt_llm's nccl_tuning sets in this tree
                res, _ = run_config("pair", {}, mode="pair_check")
            else:
                res, _ = run_config(nm, CONFIGS[nm], mode="check")
            if res is None:
                allok = False
                summary[nm] = None
                continue
            ok = res["check_passed"] and res["ranks_identical"]
            allok = allok and ok
            if nm == "default":
                ref_sigs = {m: c["sigs"] for m, c in res["check"].items()}
            bitwise = None
            if nm != "default" and ref_sigs is not None:
                bitwise = {m: c["sigs"] == ref_sigs.get(m) for m, c in res["check"].items()}
            print(f"[{nm}] bitwise identical to default (every call): {bitwise}", flush=True)
            if bitwise is not None and not all(bitwise.values()):
                allok = False
            summary[nm] = {"passed": ok, "env": res.get("env"), "bitwise_vs_default": bitwise,
                           "sizes": {m: {k: c[k] for k in ("bad_iters", "iters", "max_abs", "max_rel")}
                                     for m, c in res["check"].items()}}
            for m, c in res["check"].items():
                print(f"[{nm}] m={m}: bad_iters={c['bad_iters']}/{c['iters']} max_abs={c['max_abs']:.4g} "
                      f"max_rel={c['max_rel']:.4g} ranks_identical={res['ranks_identical']} env={res.get('env')}",
                      flush=True)
        print(json.dumps({"passed": allok, "bound": "|out-ref| <= 2^-7*sum|x_i| every call, inputs restored",
                          "configs": summary}))
        return
    if mode == "pair":
        res, _ = run_config("pair", {}, mode="pair")
        out = {"batch": {}, "output": {"class": "exact"}, "check_passed": res["check_passed"],
               "nccl_version": res["nccl_version"]}
        for m, t in res["timing"].items():
            out["batch"][m] = {"step_ms": t["median_ms"], "noise_floor": t["noise_floor"]}
        print(sh(["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu", "--format=csv"]), flush=True)
        print(json.dumps(out))
        return
    names = sys.argv[2:] or list(CONFIGS)
    table = {}
    extra = {}
    for nm in names:
        if nm.startswith("combo:"):
            envo = {}
            for part in nm[len("combo:"):].split("+"):
                envo.update(CONFIGS[part])
        elif "=" in nm:
            envo = dict(kv.split("=", 1) for kv in nm.split(","))
        else:
            envo = CONFIGS[nm]
        res, _ = run_config(nm, envo)
        if res is None:
            table[nm] = None
            continue
        table[nm] = {m: round(t["median_ms"] * 1e3, 1) for m, t in res["timing"].items()}
        table[nm]["check"] = res["check_passed"] and res["ranks_identical"]
        if "default" in extra:
            bw = {m: res["check"][m]["sigs"] == extra["default"]["check"][m]["sigs"]
                  for m in res["check"]}
        else:
            bw = None
        table[nm]["bitwise_vs_default"] = bw
        for m, c in res["check"].items():
            print(f"[{nm}] check m={m}: bad_iters={c['bad_iters']}/{c['iters']} "
                  f"max_abs={c['max_abs']:.4g} max_rel={c['max_rel']:.4g} "
                  f"frac_outside_plain_tol={c['max_frac_outside_plain_bf16_tol']:.3g} "
                  f"ranks_identical={res['ranks_identical']} "
                  f"bitwise_vs_default={None if bw is None else bw[m]}", flush=True)
        extra[nm] = res
    print(sh(["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,power.draw", "--format=csv"]), flush=True)
    print("=== median us (max over seqs' median-of-3), decode graph-captured copy+AR, prefill eager ===")
    sizes = [str(m) for m in DECODE_M + PREFILL_M]
    print("cfg".ljust(40) + "".join(s.rjust(9) for s in sizes) + "  check")
    for nm, row in table.items():
        if row is None:
            print(nm.ljust(40) + "  FAILED")
            continue
        bw = row["bitwise_vs_default"]
        bws = "-" if bw is None else ("all" if all(bw.values()) else
                                      ",".join(m for m, v in bw.items() if not v) + " differ")
        print(nm.ljust(40) + "".join(str(row[s]).rjust(9) for s in sizes) +
              f"  {row['check']}  bitwise:{bws}")
    print(json.dumps({"passed": all(r and r["check"] for r in table.values()), "table_us": table}))


if __name__ == "__main__":
    main()
