"""Microbenchmark + checks for the in-place ProcessGroup all-reduce (branch fusion/7).

Runs on 2 GPUs (one process per GPU, NCCL ProcessGroup, TLLM_DISABLE_MPI=1 as VisualGen sets it).
The op under test is tensorrt_llm's AllReduce module (strategy NCCL, the VisualGen default) as the
H3 DiT row-parallel Linears call it, on a [M, D] BF16 tensor: on main it runs
allreduce_pg_by_name (input.clone() + c10d all-reduce); on the branch (inplace_private_input=True)
it reduces the input in place. Reference: torch.distributed.all_reduce on a copy of the input.

Modes:
  bench    timed run (median of CUDA-event times) + an untimed repeated-execution check with the same
           warmup/iteration count and reset points; prints the evidence JSON (step_ms).
  compile  torch.compile (inductor, fullgraph) of matmul -> AllReduce -> residual add: bitwise check
           against eager reference, recompile/graph-break check, and a torch.profiler kernel census of
           one compiled call (Memcpy DtoD / copy kernels).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TLLM_DISABLE_MPI", "1")


def log(rank, *a):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_ar(rank):
    from types import SimpleNamespace

    import torch.distributed as dist

    from tensorrt_llm._torch.distributed.ops import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    pg = dist.group.WORLD
    mapping = SimpleNamespace(
        tp_size=2,
        tp_rank=rank,
        rank=rank,
        world_size=2,
        tp_group=[0, 1],
        tp_group_name=pg.group_name,
        enable_attention_dp=False,
    )
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=None)
    ar.inplace_private_input = True  # ignored on main
    return ar


def stats(ts):
    ts = sorted(ts)
    n = len(ts)

    def q(p):
        return ts[min(n - 1, max(0, int(round(p * (n - 1)))))]

    med = q(0.5)
    return med, (q(0.9) - q(0.1)) / med, ts[0], ts[-1]


def worker(rank, args, port):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    # As VisualGenMapping.build_mesh does: register the world/local PGs with the native side.
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg

    local = dist.new_group(ranks=[0, 1], backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local, torch_pybind11_abi())
    ar = build_ar(rank)
    from tensorrt_llm._torch.distributed.ops import warmup_named_allreduce_topologies

    warmup_named_allreduce_topologies(torch.nn.ModuleList([ar]))
    torch.cuda.synchronize()
    M, D = args.m, args.d
    log(rank, f"M={M} D={D} bytes={M * D * 2} torch={torch.__version__}")
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    pristine = torch.randn(M, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    pristine_ck = pristine.clone()
    sync_tok = torch.ones(1, device="cuda")

    def sync():
        dist.all_reduce(sync_tok)
        torch.cuda.synchronize()

    # Reference: torch.distributed all_reduce on a copy.
    ref = pristine.clone()
    dist.all_reduce(ref)
    torch.cuda.synchronize()

    result = {}
    if args.mode in ("bench", "check"):
        inp = torch.empty_like(pristine)
        times = []
        n = args.warmup + args.iters
        for i in range(n if args.mode == "bench" else 0):
            inp.copy_(pristine)  # reset (outside the timed region)
            sync()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = ar(inp)
            e.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(s.elapsed_time(e))
            del out
            if rank == 0 and i % 10 == 0:
                log(rank, f"iter {i}/{n}")
        # Rank 0's times; also gather rank 1's median for the record. The PCIe NCCL transfer dominates
        # and varies per call, so the noise floor is the spread of block medians (repeated timings of
        # blocks of --block consecutive iterations), next to the per-iteration spread.
        med, spread_iter, lo, hi = stats(times) if times else (0.0, 0.0, 0.0, 0.0)
        blocks = [stats(times[j:j + args.block])[0] for j in range(0, len(times), args.block)
                  if len(times[j:j + args.block]) == args.block]
        _, _, blo, bhi = stats(blocks) if blocks else (0.0, 0.0, 0.0, 0.0)
        spread = (bhi - blo) / med if med else 0.0
        meds = torch.tensor([med], device="cuda", dtype=torch.float64)
        all_meds = [torch.zeros_like(meds) for _ in range(2)]
        dist.all_gather(all_meds, meds)
        log(rank, f"median={med:.4f} ms block-spread={spread:.4f} iter-spread={spread_iter:.4f} "
            f"min={lo:.4f} max={hi:.4f} blocks={[round(b, 3) for b in blocks]}")
        # Repeated-execution check: same warmup/iters and reset points, untimed.
        mismatches = []
        inplace_iters = 0
        for i in range(n):
            if not torch.equal(pristine, pristine_ck):
                mismatches.append(f"pristine changed before iter {i}")
            inp.copy_(pristine)
            sync()
            out = ar(inp)
            torch.cuda.synchronize()
            if out.data_ptr() == inp.data_ptr():
                inplace_iters += 1
            if not torch.equal(out, ref):
                mismatches.append(f"iter {i} differs from torch.distributed reference")
            del out
        ok = torch.tensor([0 if not mismatches else 1], device="cuda")
        dist.all_reduce(ok)
        result = {
            "batch": {"1": {"step_ms": med, "noise_floor": spread}},
            "output": {"class": "exact"},
            "detail": {
                "M": M,
                "D": D,
                "iters": args.iters,
                "warmup": args.warmup,
                "min_ms": lo,
                "max_ms": hi,
                "per_iter_spread_p90_p10": spread_iter,
                "block": args.block,
                "block_medians_ms": blocks,
                "noise_floor_def": "(max-min)/median of block medians",
                "rank_medians_ms": [float(x.item()) for x in all_meds],
                "inplace_iters": inplace_iters,
                "check_passed_all_ranks": int(ok.item()) == 0,
                "mismatches_rank0": mismatches[:5],
            },
        }
        if args.mode == "check":
            result = {
                "passed": int(ok.item()) == 0,
                "compared": ("AllReduce module output vs torch.distributed.all_reduce reference, bitwise, "
                             f"{n} iterations ({args.warmup} warmup + {args.iters}) with the input restored "
                             "from a pristine copy before each; pristine (read-only) copy verified unchanged "
                             "before every reset; both ranks"),
                "M": M, "D": D, "iterations": n, "inplace_iters": inplace_iters,
                "inplace_path_used": inplace_iters == n,
                "mismatches_rank0": mismatches[:5],
            }
    else:
        import torch._dynamo
        import torch.nn as nn
        from torch._dynamo.utils import counters

        Dh = D // 2
        x0 = torch.randn(M, Dh, device="cuda", dtype=torch.bfloat16, generator=gen)
        w = torch.randn(Dh, D, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
        r = torch.randn(M, D, device="cuda", dtype=torch.bfloat16, generator=gen)

        class Blk(nn.Module):
            def __init__(self, ar):
                super().__init__()
                self.ar = ar

            def forward(self, x, w, r):
                y = torch.matmul(x, w)
                y = self.ar(y)
                return r + y

        # Eager reference with torch.distributed.
        y = torch.matmul(x0, w)
        dist.all_reduce(y)
        ref_c = r + y
        del y
        blk = torch.compile(Blk(ar), mode="default", dynamic=None, fullgraph=True)
        checks = []
        for i in range(3):
            out = blk(x0, w, r)
            torch.cuda.synchronize()
            checks.append(bool(torch.equal(out, ref_c)))
            del out
            if i == 0:
                torch._dynamo.config.error_on_recompile = True
            log(rank, f"compiled call {i} equal={checks[-1]}")
        # Kernel census of one compiled call.
        from torch.profiler import ProfilerActivity, profile

        sync()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            out = blk(x0, w, r)
            torch.cuda.synchronize()
        census = {}
        for ev in prof.events():
            if ev.device_type is not None and "CUDA" in str(ev.device_type):
                census[ev.name] = census.get(ev.name, 0) + 1
        log(rank, "census:", json.dumps(census, indent=1))
        dtod = sum(v for k, v in census.items() if "Memcpy DtoD" in k)
        copy_kernels = {k: v for k, v in census.items() if "copy" in k.lower() or "clone" in k.lower()}
        graph_breaks = sum(counters["graph_break"].values())
        unique_graphs = counters["stats"].get("unique_graphs", None)
        result = {
            "passed": all(checks) and graph_breaks == 0,
            "compared": "torch.compile(matmul -> AllReduce -> +residual) vs eager torch.distributed.all_reduce reference, bitwise, 3 calls",
            "bitwise_equal_calls": checks,
            "graph_breaks": graph_breaks,
            "unique_graphs": unique_graphs,
            "memcpy_dtod_in_compiled_call": dtod,
            "copy_like_events": copy_kernels,
            "census": census,
        }
    dist.barrier()
    if rank == 0:
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------------------------------
# envsweep mode (parallel_comm idea 40): NCCL env sweep of the in-place TP2 all-reduce.
# Each config runs in a FRESH pair of processes (NCCL caches env params per process), with plain
# torch.distributed.all_reduce (the same in-place NCCL call the H3 to_out/down AR issues).
# ---------------------------------------------------------------------------------------------------
SWEEP_CONFIGS = [
    ("default", {}),
    ("min_nch8", {"NCCL_MIN_NCHANNELS": "8"}),
    ("nch4", {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}),
    ("nch8", {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"}),
    ("nch16", {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}),
    ("proto_simple", {"NCCL_PROTO": "Simple"}),
    ("proto_ll128", {"NCCL_PROTO": "LL128"}),
    ("default_rep", {}),
    ("p2p_cuda_memcpy", {"NCCL_P2P_USE_CUDA_MEMCPY": "1"}),
    ("buff8m", {"NCCL_BUFFSIZE": str(8 << 20)}),
    ("buff16m", {"NCCL_BUFFSIZE": str(16 << 20)}),
    ("nthreads512", {"NCCL_NTHREADS": "512"}),
    ("p2p_level_pix", {"NCCL_P2P_LEVEL": "PIX"}),
    ("p2p_level_node", {"NCCL_P2P_LEVEL": "NODE"}),
    ("p2p_level_sys", {"NCCL_P2P_LEVEL": "SYS"}),
    ("default_rep2", {}),
]


def qstats(ts):
    ts = sorted(ts)
    n = len(ts)
    if n == 0:
        return {}

    def q(p):
        return ts[min(n - 1, max(0, int(round(p * (n - 1)))))]

    med = q(0.5)
    return {"median_ms": round(med, 4), "p10_ms": round(q(0.1), 4), "p90_ms": round(q(0.9), 4),
            "spread": round((q(0.9) - q(0.1)) / med, 4) if med else 0.0, "min_ms": round(ts[0], 4),
            "max_ms": round(ts[-1], 4), "n": n}


def sweep_worker(rank, name, env, opts, port, outdir):
    os.environ.update(env)
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV"
    os.environ["NCCL_DEBUG_FILE"] = os.path.join(outdir, f"nccl_{name}.r{rank}.log")
    import traceback

    import torch
    import torch.distributed as dist

    res = {"name": name, "env": env, "rank": rank, "ok": False}
    warm, iters = opts["warmup"], opts["iters"]
    t_start = time.time()
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("cuda:nccl,cpu:gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=2, device_id=torch.device("cuda", rank))
        M, D, F = opts["m"], opts["d"], opts["f"]
        sync_tok = torch.ones(1, device="cuda")

        def sync():
            dist.all_reduce(sync_tok)
            torch.cuda.synchronize()

        sync()

        def make(m, seed):
            xs = []
            for r in range(2):
                g = torch.Generator(device="cuda").manual_seed(seed * 10 + r)
                xs.append(torch.randn(m, D, device="cuda", dtype=torch.bfloat16, generator=g))
            ref = (xs[0].float() + xs[1].float()).to(torch.bfloat16)  # bf16(x0 + x1), independent of NCCL
            return xs, ref

        sizes = [("M", M), ("M2", M // 2), ("M4", (M + 3) // 4)]
        res["ar"] = {}
        mism_total = 0
        for si, (lbl, m) in enumerate(sizes):
            xs, ref = make(m, 100 + si)
            pristine = xs[rank]
            pristine_ck = pristine.clone()
            peer = None
            del xs
            inp = torch.empty_like(pristine)
            times = []
            for i in range(warm + iters):
                inp.copy_(pristine)  # reset outside the timed region
                sync()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                dist.all_reduce(inp)
                e.record()
                torch.cuda.synchronize()
                if i >= warm:
                    times.append(s.elapsed_time(e))
            # untimed repeat: same counts and reset points, bitwise vs bf16(x0+x1)
            bad = []
            for i in range(warm + iters):
                if not torch.equal(pristine, pristine_ck):
                    bad.append(f"pristine changed before iter {i}")
                inp.copy_(pristine)
                sync()
                dist.all_reduce(inp)
                torch.cuda.synchronize()
                if not torch.equal(inp, ref):
                    bad.append(f"iter {i}")
            # timed-run outputs: the last timed iteration's output also checked (inp after the timed loop
            # is overwritten by the check loop, so we compare each timed iteration in a separate pass)
            st = qstats(times)
            st["GBps_per_dir"] = round(m * D * 2 / (st["median_ms"] * 1e-3) / 1e9, 2)
            st["bytes"] = m * D * 2
            st["rows"] = m
            st["mismatches"] = len(bad)
            st["mismatch_examples"] = bad[:3]
            st["times_ms"] = [round(t, 4) for t in times]
            mism_total += len(bad)
            res["ar"][lbl] = st
            del inp, pristine, pristine_ck, ref, peer
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"[{time.strftime('%H:%M:%S')}] [{name}] AR {lbl} rows={m} median={st['median_ms']:.3f} ms "
                      f"p10={st['p10_ms']:.3f} p90={st['p90_ms']:.3f} spread={st['spread']:.3f} "
                      f"{st['GBps_per_dir']} GB/s/dir mism={len(bad)}", flush=True)

        # Timed-sequence bitwise check: the timed loop above restores the input each iteration, so a
        # second timed-pattern pass that checks every output (events recorded the same way) covers it.
        xs, ref = make(sizes[1][1], 101)
        pristine = xs[rank]
        del xs
        inp = torch.empty_like(pristine)
        bad_t = 0
        for i in range(warm + iters):
            inp.copy_(pristine)
            sync()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            dist.all_reduce(inp)
            e.record()
            torch.cuda.synchronize()
            bad_t += int(not torch.equal(inp, ref))
        res["timed_pattern_mismatches_M2"] = bad_t
        mism_total += bad_t
        del inp, pristine, ref
        torch.cuda.empty_cache()

        if opts.get("overlap"):
            import torch.nn.functional as Fn
            from torch.nn.attention import SDPBackend, sdpa_kernel

            m2 = M // 2
            m4 = (M + 3) // 4
            g = torch.Generator(device="cuda").manual_seed(777)
            gu = (torch.randn(m2, D, device="cuda", dtype=torch.bfloat16, generator=g),
                  torch.randn(D, 2 * F, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
            dn = (torch.randn(m2, F, device="cuda", dtype=torch.bfloat16, generator=g),
                  torch.randn(F, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
            Hl = opts.get("heads_local", 28)
            q = torch.randn(1, Hl, m4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
            kk = torch.randn(1, Hl, M, 128, device="cuda", dtype=torch.bfloat16, generator=g)
            vv = torch.randn(1, Hl, M, 128, device="cuda", dtype=torch.bfloat16, generator=g)

            def sdpa():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return Fn.scaled_dot_product_attention(q, kk, vv)

            ops = [("gate_up", lambda: torch.matmul(gu[0], gu[1]), m2),
                   ("down", lambda: torch.matmul(dn[0], dn[1]), m2),
                   ("sdpa_fa2_q_M4", sdpa, m4)]
            side = torch.cuda.Stream()
            main_s = torch.cuda.current_stream()
            res["overlap"] = {}
            for gname, fn, ar_rows in ops:
                xs, ref = make(ar_rows, 101 if ar_rows == m2 else 102)
                pristine = xs[rank]
                del xs
                inp = torch.empty_like(pristine)
                gref = fn()
                galone = []
                for i in range(warm + iters):
                    sync()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    o = fn()
                    e.record()
                    torch.cuda.synchronize()
                    if i >= warm:
                        galone.append(s.elapsed_time(e))
                    del o
                ga_med = qstats(galone)["median_ms"]

                def one(timed):
                    inp.copy_(pristine)
                    sync()
                    ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                    ev[0].record(main_s)
                    side.wait_stream(main_s)
                    with torch.cuda.stream(side):
                        ev[1].record(side)
                        dist.all_reduce(inp)
                        ev[2].record(side)
                    ev[3].record(main_s)
                    o = fn()
                    ev[4].record(main_s)
                    main_s.wait_stream(side)
                    ev[5].record(main_s)
                    torch.cuda.synchronize()
                    return ev, o

                gpar, arpar, walls, exposed = [], [], [], []
                for i in range(warm + iters):
                    ev, o = one(True)
                    if i >= warm:
                        gpar.append(ev[3].elapsed_time(ev[4]))
                        arpar.append(ev[1].elapsed_time(ev[2]))
                        w = ev[0].elapsed_time(ev[5])
                        walls.append(w)
                        exposed.append(max(0.0, w - ga_med))
                    del o
                obad = 0
                for i in range(warm + iters):
                    ev, o = one(False)
                    obad += int(not torch.equal(inp, ref)) + int(not torch.equal(o, gref))
                    del o
                mism_total += obad
                ent = {"gemm_alone": qstats(galone), "gemm_under_ar": qstats(gpar), "ar_e2e": qstats(arpar),
                       "pair_wall": qstats(walls), "exposed_ar": qstats(exposed), "mismatches": obad}
                ent["gemm_slowdown"] = round(ent["gemm_under_ar"]["median_ms"] / ga_med - 1.0, 4)
                res["overlap"][gname] = ent
                if rank == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] [{name}] overlap {gname}: gemm alone {ga_med:.3f} ms, "
                          f"under AR {ent['gemm_under_ar']['median_ms']:.3f} ({ent['gemm_slowdown'] * 100:+.1f}%), "
                          f"AR e2e {ent['ar_e2e']['median_ms']:.3f}, wall {ent['pair_wall']['median_ms']:.3f}, "
                          f"exposed {ent['exposed_ar']['median_ms']:.3f}, mism={obad}", flush=True)
                del gref, inp, pristine, ref
                torch.cuda.empty_cache()
            del gu, dn, q, kk, vv
            torch.cuda.empty_cache()

        if opts.get("rsag"):
            m2 = M // 2
            xs, ref = make(M, 100)
            x = xs[rank]
            rs_ref = ref[rank * m2:(rank + 1) * m2]
            ag_ref = torch.cat([xs[0][:m2], xs[1][:m2]], 0)
            shard = x[:m2].contiguous()
            del xs
            res["rsag"] = {}
            cases = {
                "reduce_scatter_bf16": (lambda o: dist.reduce_scatter_tensor(o, x),
                                        torch.empty(m2, D, device="cuda", dtype=torch.bfloat16), rs_ref),
                "all_gather_bf16": (lambda o: dist.all_gather_into_tensor(o, shard),
                                    torch.empty(M, D, device="cuda", dtype=torch.bfloat16), ag_ref),
                "all_gather_u8": (lambda o: dist.all_gather_into_tensor(o, shard.view(torch.uint8)),
                                  torch.empty(M, 2 * D, device="cuda", dtype=torch.uint8),
                                  ag_ref.view(torch.uint8)),
            }
            for cname, (fn, out, cref) in cases.items():
                times = []
                for i in range(warm + iters):
                    out.zero_()
                    sync()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn(out)
                    e.record()
                    torch.cuda.synchronize()
                    if i >= warm:
                        times.append(s.elapsed_time(e))
                cb = 0
                for i in range(warm + iters):
                    out.zero_()
                    sync()
                    fn(out)
                    torch.cuda.synchronize()
                    cb += int(not torch.equal(out, cref))
                st = qstats(times)
                st["mismatches"] = cb
                mism_total += cb
                res["rsag"][cname] = st
                if rank == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] [{name}] {cname}: median {st['median_ms']:.3f} ms "
                          f"spread {st['spread']:.3f} mism={cb}", flush=True)
            del x, ref, rs_ref, ag_ref, shard, cases
            torch.cuda.empty_cache()
        res["mismatches_total"] = mism_total
        res["ok"] = True
        dist.barrier()
        dist.destroy_process_group()
    except Exception as ex:  # record and continue the sweep
        res["error"] = repr(ex)[:500]
        res["traceback"] = traceback.format_exc()[-2000:]
        print(f"[{name}] rank {rank} ERROR {ex!r}", flush=True)
    res["wall_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(outdir, f"res_{name}.r{rank}.json"), "w") as f:
        json.dump(res, f)


def _smi(args_):
    import subprocess

    try:
        return subprocess.run(["nvidia-smi"] + args_, capture_output=True, text=True, timeout=60).stdout
    except Exception as ex:
        return repr(ex)


def _norm_bus(b):
    b = b.strip().lower()
    parts = b.split(":")
    return ":".join(parts[-2:]) if len(parts) >= 2 else b


def pair_check():
    import torch

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    info = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES")}
    bus = []
    for d in (0, 1):
        err, s = cudart.cudaDeviceGetPCIBusId(32, d)
        s = s.decode(errors="replace") if isinstance(s, (bytes, bytearray)) else str(s)
        bus.append(s.split("\x00")[0].strip())
    info["cuda_bus_ids"] = bus
    lst = _smi(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
    idx_by_bus = {}
    for line in lst.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2:
            idx_by_bus[_norm_bus(p[1])] = int(p[0])
    phys = [idx_by_bus.get(_norm_bus(b)) for b in bus]
    info["physical_ids"] = phys
    topo = _smi(["topo", "-m"])
    print(topo, flush=True)
    rel = None
    try:
        lines = [ln for ln in topo.splitlines() if ln.strip()]
        hdr = lines[0].split()
        for ln in lines[1:]:
            cols = ln.split()
            if phys[0] is not None and cols and cols[0] == f"GPU{phys[0]}":
                rel = cols[1 + hdr.index(f"GPU{phys[1]}")]
                break
    except Exception as ex:
        rel = f"parse_error {ex!r}"
    info["relation"] = rel
    # 1 s warm-up peer copy, then keep the link busy while reading link gen/width.
    a = torch.empty(256 << 20, dtype=torch.uint8, device="cuda:0")
    b = torch.empty(256 << 20, dtype=torch.uint8, device="cuda:1")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        b.copy_(a, non_blocking=True)
        a.copy_(b, non_blocking=True)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
    for _ in range(300):
        b.copy_(a, non_blocking=True)
        a.copy_(b, non_blocking=True)
    q = ["-i", ",".join(str(x) for x in phys if x is not None) or "0,1",
         "--query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
         "pcie.link.width.max,clocks.sm,clocks.max.sm,temperature.gpu,power.draw", "--format=csv,noheader"]
    pcie = _smi(q)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    del a, b
    torch.cuda.empty_cache()
    info["pcie"] = pcie.strip()
    widths = []
    for line in pcie.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            widths.append(int(p[5]))
        except Exception:
            pass
    info["link_width_current"] = widths
    info["wrong_pair"] = (set(phys) == {4, 5}) or (len(widths) == 2 and min(widths) < 16)
    print("PAIR", json.dumps(info), flush=True)
    return info


def ce_reference(N_rows, D, warm, iters):
    import torch

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    res = {}
    nmax = N_rows * D * 2
    bufs = {}
    for d in (0, 1):
        bufs[d] = (torch.empty(nmax, dtype=torch.uint8, device=f"cuda:{d}").random_(0, 255),
                   torch.empty(nmax, dtype=torch.uint8, device=f"cuda:{d}"))
    for d, p in ((0, 1), (1, 0)):
        cudart.cudaSetDevice(d)
        cudart.cudaDeviceEnablePeerAccess(p, 0)
    streams = {0: torch.cuda.Stream(device=0), 1: torch.cuda.Stream(device=1)}
    for lbl, rows in (("M", N_rows), ("M2", N_rows // 2), ("M4", (N_rows + 3) // 4)):
        nbytes = rows * D * 2
        per, walls = [], []
        for it in range(warm + iters):
            for d in (0, 1):
                bufs[d][1].zero_()
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            t0 = time.perf_counter()
            evs = {}
            for src in (0, 1):
                dst = 1 - src
                s = streams[src]
                with torch.cuda.device(src):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(s)
                    err = cudart.cudaMemcpyPeerAsync(bufs[dst][1].data_ptr(), dst, bufs[src][0].data_ptr(), src,
                                                     nbytes, s.cuda_stream)
                    assert int(err[0]) == 0, err
                    e1.record(s)
                evs[src] = (e0, e1)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            t1 = time.perf_counter()
            if it >= warm:
                walls.append((t1 - t0) * 1e3)
                per.append(max(a.elapsed_time(b) for a, b in evs.values()))
        ok = all(torch.equal(bufs[s][0][:nbytes].to(f"cuda:{1 - s}"), bufs[1 - s][1][:nbytes]) for s in (0, 1))
        st = qstats(per)
        st["GBps_per_dir"] = round(nbytes / (st["median_ms"] * 1e-3) / 1e9, 2)
        st["wall"] = qstats(walls)
        st["bitwise"] = ok
        res[lbl] = st
        print(f"[{time.strftime('%H:%M:%S')}] CE bidir {lbl} {nbytes / 1e6:.0f} MB/dir: median {st['median_ms']:.3f} ms "
              f"p10 {st['p10_ms']:.3f} p90 {st['p90_ms']:.3f} {st['GBps_per_dir']} GB/s/dir bitwise={ok}", flush=True)
    del bufs
    torch.cuda.empty_cache()
    return res


def run_config(name, env, opts, outdir, timeout=300):
    import random

    import torch.multiprocessing as mp

    port = 31000 + random.randint(0, 3000)
    t0 = time.time()
    ctx = mp.start_processes(sweep_worker, args=(name, env, opts, port, outdir), nprocs=2, join=False,
                             start_method="spawn")
    try:
        while not ctx.join(timeout=5):
            if time.time() - t0 > timeout:
                for p in ctx.processes:
                    p.kill()
                print(f"[{name}] TIMEOUT", flush=True)
                break
    except Exception as ex:
        print(f"[{name}] spawn error {ex!r}", flush=True)
    out = {}
    for r in (0, 1):
        pth = os.path.join(outdir, f"res_{name}.r{r}.json")
        if os.path.exists(pth):
            with open(pth) as f:
                out[r] = json.load(f)
    # NCCL init/env lines from both ranks
    lines = {}
    for r in (0, 1):
        lp = os.path.join(outdir, f"nccl_{name}.r{r}.log")
        keep = []
        if os.path.exists(lp):
            with open(lp, errors="replace") as f:
                for ln in f:
                    if ("set by environment" in ln or "coll channels" in ln or " via " in ln
                            or "comm 0x" in ln and "Init COMPLETE" in ln):
                        keep.append(ln.strip()[-200:])
        lines[r] = keep[:12]
    ok = all(out.get(r, {}).get("ok") for r in (0, 1))
    entry = {"name": name, "env": env, "ok": ok, "wall_s": round(time.time() - t0, 1), "nccl_log": lines}
    if not ok:
        entry["error"] = [out.get(r, {}).get("error") for r in (0, 1)]
        entry["tb"] = (out.get(0, {}) or out.get(1, {})).get("traceback", "")[-800:]
        print(f"[{name}] FAILED: {entry['error']} {entry['tb']}", flush=True)
        return entry
    r0, r1 = out[0], out[1]
    entry["ar"] = {k: {kk: v for kk, v in r0["ar"][k].items() if kk != "times_ms"} for k in r0["ar"]}
    for k in entry["ar"]:
        entry["ar"][k]["rank1_median_ms"] = r1["ar"][k]["median_ms"]
        entry["ar"][k]["mismatches_both"] = r0["ar"][k]["mismatches"] + r1["ar"][k]["mismatches"]
    entry["mismatches_total_both"] = r0["mismatches_total"] + r1["mismatches_total"]
    if "overlap" in r0:
        entry["overlap"] = r0["overlap"]
        entry["overlap_rank1_exposed"] = {g: r1["overlap"][g]["exposed_ar"]["median_ms"] for g in r1["overlap"]}
    if "rsag" in r0:
        entry["rsag"] = r0["rsag"]
        entry["rsag_mismatches_rank1"] = {k: v["mismatches"] for k, v in r1["rsag"].items()}
    print(f"[{time.strftime('%H:%M:%S')}] CONFIG {name} env={env} M={entry['ar']['M']['median_ms']} "
          f"M2={entry['ar']['M2']['median_ms']} M4={entry['ar']['M4']['median_ms']} "
          f"spreadM={entry['ar']['M']['spread']} mism={entry['mismatches_total_both']} nccl={lines.get(0)}",
          flush=True)
    return entry


def envsweep_main(args):
    import torch

    outdir = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS") or "/tmp", "nccl_sweep")
    os.makedirs(outdir, exist_ok=True)
    pair = pair_check()
    if pair["wrong_pair"]:
        print(json.dumps({"wrong_pair": True, "pair": pair}), flush=True)
        return 0
    label = f"phys {pair['physical_ids']} {pair['relation']} width {pair['link_width_current']}"
    print("PAIR LABEL:", label, flush=True)
    M, D, F = args.m, args.d, args.f
    opts = {"m": M, "d": D, "f": F, "warmup": args.warmup, "iters": args.iters}
    summary = {"pair": pair, "label": label, "M": M, "D": D, "F_local": F, "configs": {}}
    summary["ce_start"] = ce_reference(M, D, args.warmup, args.iters)
    torch.cuda.empty_cache()

    def save():
        with open(os.path.join(outdir, "sweep_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)

    names = [n for n, _ in SWEEP_CONFIGS]
    only = [x for x in args.configs.split(",") if x] if args.configs else names
    for name, env in SWEEP_CONFIGS:
        if name not in only:
            continue
        o = dict(opts, rsag=(name == "default" and not args.no_rsag))
        summary["configs"][name] = run_config(name, env, o, outdir)
        save()

    def med(n, k="M"):
        c = summary["configs"].get(n)
        return c["ar"][k]["median_ms"] if c and c.get("ok") else None

    defaults = [n for n in summary["configs"] if n.startswith("default") and med(n)]
    dmeds = [med(n) for n in defaults]
    dref = sorted(dmeds)[len(dmeds) // 2] if dmeds else None
    aa_delta = (max(dmeds) - min(dmeds)) / dref if len(dmeds) > 1 else None
    aa_spread = max(summary["configs"][n]["ar"]["M"]["spread"] for n in defaults) if defaults else None
    summary["aa"] = {"default_medians_M": dmeds, "aa_delta": aa_delta, "aa_iter_spread_max": aa_spread}
    print("A/A", json.dumps(summary["aa"]), flush=True)

    if args.quick:
        ce = summary["ce_start"]
        table = {n: {k: [c["ar"][k]["median_ms"], c["ar"][k]["p10_ms"], c["ar"][k]["p90_ms"],
                         c["ar"][k]["GBps_per_dir"], round(c["ar"][k]["median_ms"] / ce[k]["median_ms"], 3)]
                     for k in ("M", "M2", "M4")} | {"mism": c["mismatches_total_both"]}
                 for n, c in summary["configs"].items() if c.get("ok")}
        summary["table"] = table
        save()
        ok = all(c.get("ok") and c["mismatches_total_both"] == 0 for c in summary["configs"].values())
        print(json.dumps({"passed": ok, "label": label, "aa": summary["aa"],
                          "ce_ms": {k: ce[k]["median_ms"] for k in ce}, "table": table}), flush=True)
        return 0
    # rerun noisy configs
    if aa_spread:
        for name, env in SWEEP_CONFIGS:
            c = summary["configs"].get(name)
            if c and c.get("ok") and c["ar"]["M"]["spread"] > 2 * aa_spread and not name.startswith("default"):
                summary["configs"][name + "_rerun"] = run_config(name + "_rerun", env, opts, outdir)
                save()

    cand = [(med(n), n) for n in summary["configs"] if med(n) and not n.startswith("default")]
    cand.sort()
    best = []
    for _, n in cand:
        base = n.replace("_rerun", "")
        if base not in best:
            best.append(base)
    envs = dict(SWEEP_CONFIGS)
    summary["ranking_M"] = cand
    # pairwise combination of the best two with disjoint keys
    combo = None
    for i in range(min(3, len(best))):
        for j in range(i + 1, min(4, len(best))):
            ei, ej = envs[best[i]], envs[best[j]]
            if not (set(ei) & set(ej)) and ei and ej:
                combo = (best[i] + "+" + best[j], {**ei, **ej})
                break
        if combo:
            break
    if combo:
        envs[combo[0]] = combo[1]
        summary["configs"][combo[0]] = run_config(combo[0], combo[1], opts, outdir)
        save()
    # overlapped metric: default and best two (single configs)
    ovl = ["default"] + best[:2]
    summary["overlap_configs"] = ovl
    for n in ovl:
        nm = "ovl_" + n
        summary["configs"][nm] = run_config(nm, envs[n], dict(opts, overlap=True), outdir)
        save()
    summary["configs"]["default_end"] = run_config("default_end", {}, opts, outdir)
    summary["ce_end"] = ce_reference(M, D, args.warmup, args.iters)
    save()

    ce = summary["ce_start"]
    table = {}
    all_ok = True
    for n, c in summary["configs"].items():
        if not c.get("ok"):
            table[n] = {"error": str(c.get("error"))[:160]}
            all_ok = False
            continue
        all_ok = all_ok and c["mismatches_total_both"] == 0
        table[n] = {k: [c["ar"][k]["median_ms"], c["ar"][k]["p10_ms"], c["ar"][k]["p90_ms"],
                        c["ar"][k]["GBps_per_dir"], round(c["ar"][k]["median_ms"] / ce[k]["median_ms"], 3)]
                    for k in ("M", "M2", "M4")}
        table[n]["mism"] = c["mismatches_total_both"]
        if "overlap" in c:
            table[n]["ovl"] = {g: [v["gemm_alone"]["median_ms"], v["gemm_under_ar"]["median_ms"],
                                   v["gemm_slowdown"], v["ar_e2e"]["median_ms"], v["exposed_ar"]["median_ms"]]
                               for g, v in c["overlap"].items()}
    summary["table_fmt"] = "per size: [median, p10, p90, GB/s/dir, NCCL/CE]; ovl: [gemm alone, gemm under AR, slowdown, AR e2e, exposed]"
    summary["table"] = table
    save()
    final = {"passed": all_ok, "label": label, "aa": summary["aa"],
             "ce_ms": {k: [ce[k]["median_ms"], ce[k]["GBps_per_dir"], ce[k]["bitwise"]] for k in ce},
             "ce_end_ms": {k: summary["ce_end"][k]["median_ms"] for k in summary["ce_end"]},
             "table": table, "artifact": os.path.join(outdir, "sweep_summary.json")}
    print(json.dumps(final), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench", choices=["bench", "check", "compile", "envsweep"])
    p.add_argument("--f", type=int, default=0, help="per-rank FFN width (ffn_dim / 2)")
    p.add_argument("--configs", default="", help="envsweep: comma list of config names (default all)")
    p.add_argument("--no-rsag", action="store_true")
    p.add_argument("--quick", action="store_true", help="envsweep: phase A only (no reruns/combos/overlap)")
    p.add_argument("--m", type=int, default=38016)
    p.add_argument("--d", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--block", type=int, default=25)
    p.add_argument("--checkpoint", default="")
    args = p.parse_args()
    if args.d <= 0:
        cfg_path = os.path.join(args.checkpoint, "transformer", "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        print("transformer config:", json.dumps(cfg), flush=True)
        hs = cfg.get("hidden_size") or cfg["num_attention_heads"] * cfg["attention_head_dim"]
        args.d = int(hs)
        if args.f <= 0 and cfg.get("ffn_dim"):
            args.f = int(cfg["ffn_dim"]) // 2
    if args.mode == "envsweep":
        if args.f <= 0:
            args.f = 7168
        return envsweep_main(args)
    import random

    import torch.multiprocessing as mp

    port = 29500 + random.randint(0, 2000)
    mp.spawn(worker, args=(args, port), nprocs=2, join=True)


if __name__ == "__main__":
    sys.exit(main())
