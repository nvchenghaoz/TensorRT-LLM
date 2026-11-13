"""2-GPU PCIe link + H3 collective analysis (parallel_comm idea 2).

Usage: python comm_analysis.py <checkpoint> [config,config,...]
No model load. Synthetic BF16 tensors at the H3 per-rank shapes.
"""
import datetime
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WARM = 5
ITERS = 20
OUTDIR = "/tmp/comm_analysis_out"


def log(*a):
    s = " ".join(str(x) for x in a)
    s = "".join(c if (c == "\n" or 32 <= ord(c) < 127) else " " for c in s)
    print(s, flush=True)


def stats(ms):
    s = sorted(ms)
    n = len(s)

    def q(p):
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    med = q(0.5)
    return {
        "median_ms": round(med, 4),
        "p10_ms": round(q(0.1), 4),
        "p90_ms": round(q(0.9), 4),
        "min_ms": round(s[0], 4),
        "max_ms": round(s[-1], 4),
        "noise": round((q(0.9) - q(0.1)) / med, 4) if med > 0 else None,
    }


# ----------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------
def shapes(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "transformer", "config.json")))
    hidden = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    D = cfg["attention_head_dim"]
    info = {k: cfg.get(k) for k in ("hidden_size", "num_attention_heads", "attention_head_dim",
                                    "ffn_dim", "num_layers", "patch_size")}
    try:
        from tensorrt_llm._torch.visual_gen.models.minimax_h3 import packing as pk
        nf = pk.align_num_frames(124)
        lat_f = pk.video_latent_num_frames(nf)
        aud = pk.audio_latent_num_frames(nf)
        ps = cfg.get("patch_size", [1, 2, 2])
        rows_pf = (768 // 16 // ps[1]) * (1344 // 16 // ps[2])
        video_rows = lat_f * rows_pf
        audio_rows = aud * pk.MINIMAX_H3_AUDIO_CHANNELS
        text = 256  # assumption: prompt-dependent, a few hundred tokens
        M = video_rows + audio_rows + text
        info.update(dict(num_frames=nf, latent_frames=lat_f, rows_per_frame=rows_pf,
                         video_rows=video_rows, audio_rows=audio_rows, text_rows_assumed=text))
    except Exception as e:  # pragma: no cover
        info["packing_error"] = repr(e)
        M = 38000
    M += M % 2
    info["M"] = M
    return M, hidden, H, D, info


# ----------------------------------------------------------------------------
# parent-side: topology + link floor
# ----------------------------------------------------------------------------
def topology():
    out = {}
    for name, cmd in (("topo", ["nvidia-smi", "topo", "-m"]),
                      ("pcie", ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,pcie.link.gen.current,"
                                "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
                                "clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                                "--format=csv"]),
                      ("p2p_r", ["nvidia-smi", "topo", "-p2p", "r"]),
                      ("p2p_w", ["nvidia-smi", "topo", "-p2p", "w"])):
        try:
            out[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:
            out[name] = repr(e)
        log(f"=== nvidia-smi {name} ===\n{out[name]}")
    out["can_access_peer_0_1"] = torch.cuda.can_device_access_peer(0, 1)
    out["can_access_peer_1_0"] = torch.cuda.can_device_access_peer(1, 0)
    out["device_name"] = torch.cuda.get_device_name(0)
    out["capability"] = torch.cuda.get_device_capability(0)
    log("can_device_access_peer 0->1", out["can_access_peer_0_1"], "1->0", out["can_access_peer_1_0"],
        out["device_name"], out["capability"])
    return out


def link_floor(N):
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    res = {}
    bufs = {}
    nbytes_list = [N // 4, N // 2, N]
    nmax = max(nbytes_list)
    for d in (0, 1):
        torch.cuda.set_device(d)
        bufs[d] = (torch.empty(nmax, dtype=torch.uint8, device=f"cuda:{d}").random_(0, 255),
                   torch.empty(nmax, dtype=torch.uint8, device=f"cuda:{d}"))
    for d, p in ((0, 1), (1, 0)):
        cudart.cudaSetDevice(d)
        err = cudart.cudaDeviceEnablePeerAccess(p, 0)
        res[f"enable_peer_{d}_{p}"] = repr(err[0])
    torch.cuda.set_device(0)
    streams = {0: torch.cuda.Stream(device=0), 1: torch.cuda.Stream(device=1)}

    def issue(src, nbytes):
        dst = 1 - src
        s = streams[src]
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(src):
            e0.record(s)
            err = cudart.cudaMemcpyPeerAsync(bufs[dst][1].data_ptr(), dst, bufs[src][0].data_ptr(), src,
                                             nbytes, s.cuda_stream)
            assert int(err[0]) == 0, err
            e1.record(s)
        return e0, e1

    for nbytes in nbytes_list:
        key = f"{nbytes / 1e6:.0f}MB"
        for mode in ("0to1", "1to0", "bidir"):
            per = {0: [], 1: []}
            walls = []
            for it in range(WARM + ITERS):
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                t0 = time.perf_counter()
                evs = {}
                if mode in ("0to1", "bidir"):
                    evs[0] = issue(0, nbytes)
                if mode in ("1to0", "bidir"):
                    evs[1] = issue(1, nbytes)
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                t1 = time.perf_counter()
                if it >= WARM:
                    walls.append((t1 - t0) * 1e3)
                    for k, (a, b) in evs.items():
                        per[k].append(a.elapsed_time(b))
            entry = {"wall": stats(walls)}
            for k in per:
                if per[k]:
                    st = stats(per[k])
                    st["GBps"] = round(nbytes / (st["median_ms"] * 1e-3) / 1e9, 2)
                    entry[f"src{k}"] = st
            ndir = 2 if mode == "bidir" else 1
            entry["aggregate_GBps_wall"] = round(ndir * nbytes / (entry["wall"]["median_ms"] * 1e-3) / 1e9, 2)
            res[f"{key}_{mode}"] = entry
            log(f"[link] {key} {mode}: {json.dumps(entry)}")
        # correctness of the copy
        ok = torch.equal(bufs[0][0][:nbytes].cpu(), bufs[1][1][:nbytes].cpu())
        res[f"{key}_copy_bitwise"] = ok
    del bufs
    torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def time_loop(fn, reset=None, warm=WARM, iters=ITERS):
    pairs = []
    for i in range(warm + iters):
        if reset is not None:
            reset()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        if i >= warm:
            pairs.append((s, e))
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in pairs]


def check_loop(fn, expect_fn, reset=None, readonly=(), warm=WARM, iters=ITERS):
    """Untimed repeat sequence, same iterations and reset points as timing."""
    bad = []
    snaps = [r.clone() for r in readonly]
    maxabs = 0.0
    for i in range(warm + iters):
        if reset is not None:
            reset()
        out = fn()
        torch.cuda.synchronize()
        ok, err = expect_fn(out)
        maxabs = max(maxabs, err)
        if not ok:
            bad.append(i)
        for k, (r, sn) in enumerate(zip(readonly, snaps)):
            if not torch.equal(r, sn):
                bad.append(f"readonly{k}@{i}")
    return {"iterations": warm + iters, "bad": bad[:10], "passed": not bad, "max_abs": maxabs}


def profile_grid(fn, tag, rank):
    """One profiled call; return kernel names with grid/block from the kineto trace."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = f"/tmp/trace_{tag}_{rank}_{os.getpid()}.json"
    prof.export_chrome_trace(path)
    out = []
    try:
        tr = json.load(open(path))
        for ev in tr.get("traceEvents", []):
            if ev.get("cat") == "kernel":
                a = ev.get("args", {})
                out.append({"name": ev.get("name", "")[:80], "grid": a.get("grid"), "block": a.get("block"),
                            "dur_us": ev.get("dur")})
    except Exception as e:
        out.append({"error": repr(e)})
    return out


def worker(rank, cfg_name, M, hidden, H, D, port, extra):
    torch.cuda.set_device(rank)
    dist.init_process_group("cuda:nccl,cpu:gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=180))
    P = 2
    Hp = H // P
    res = {"cfg": cfg_name, "rank": rank, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    N = M * hidden * 2
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    def gather(t):
        lst = [torch.empty_like(t) for _ in range(P)]
        dist.all_gather(lst, t)
        return lst

    def log0(*a):
        if rank == 0:
            log(f"[{cfg_name}]", *a)

    # ---------------- (a) TP2 all-reduce [M, hidden] ----------------
    x0 = rnd(M, hidden)
    buf = torch.empty_like(x0)
    xs = gather(x0)
    ref_a = (xs[0].float() + xs[1].float()).to(torch.bfloat16)
    del xs

    def reset_a():
        buf.copy_(x0)

    def ar():
        dist.all_reduce(buf)
        return buf

    ms = time_loop(ar, reset_a)
    st = stats(ms)
    st["bytes_per_dir"] = N
    st["GBps_per_dir"] = round(N / (st["median_ms"] * 1e-3) / 1e9, 2)
    res["a_allreduce"] = st

    def exp_a(out):
        eq = torch.equal(out, ref_a)
        return eq, 0.0 if eq else (out.float() - ref_a.float()).abs().max().item()

    res["a_check"] = check_loop(ar, exp_a, reset_a, readonly=(x0,))
    if extra.get("profile", True):
        reset_a()
        res["a_kernels"] = profile_grid(ar, f"{cfg_name}_a", rank)
    log0("a", json.dumps(res["a_allreduce"]), "check", json.dumps(res["a_check"]))

    # ---------------- (e) LOWPRECISION (default only, run with a's tensors) ----------------
    if extra.get("lowprecision"):
        try:
            # AllReduce(strategy=LOWPRECISION) under TLLM_DISABLE_MPI builds TorchDist, whose
            # setup_local_comm calls ray.is_initialized() and fails without ray (job 122). Bypass
            # it: hand IpcMemory a minimal tp_allgather and call the native op directly.
            from tensorrt_llm._torch.distributed import communicator as comm_mod
            from tensorrt_llm._torch.distributed.ops import allocate_low_presicion_allreduce_workspace
            from tensorrt_llm._utils import torch_pybind11_abi
            from tensorrt_llm.bindings.internal.process_group import init_pg
            from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
            from tensorrt_llm.mapping import Mapping

            class _MiniDist:
                def tp_allgather(self, obj, **kw):
                    out = [None, None]
                    dist.all_gather_object(out, obj)
                    return out

            comm_mod.Distributed.get = staticmethod(lambda m: _MiniDist())
            mapping = Mapping(world_size=2, rank=rank, gpus_per_node=2, tp_size=2)
            res["lp_mapping_type"] = type(mapping).__name__
            allocate_low_presicion_allreduce_workspace(mapping)
            init_pg(dist.group.WORLD, dist.group.WORLD, torch_pybind11_abi())
            gname = dist.group.WORLD.group_name

            def lp_run():
                o = torch.ops.trtllm.allreduce_pg_by_name(
                    x0, None, None, None, None, None, [0, 1], rank, gname,
                    int(AllReduceStrategy.LOWPRECISION), int(AllReduceFusionOp.NONE), 1e-5, False)
                return o[0] if isinstance(o, (list, tuple)) else o

            o = lp_run()
            torch.cuda.synchronize()
            ms = time_loop(lp_run)
            stl = stats(ms)
            err = (o.float() - ref_a.float())
            stl["max_abs"] = err.abs().max().item()
            stl["rel_l2"] = (err.norm() / ref_a.float().norm()).item()
            stl["bitwise_equal_nccl_ref"] = bool(torch.equal(o, ref_a))
            stl["repeat_deterministic"] = bool(torch.equal(lp_run(), o))
            res["e_lowprecision"] = stl
            res["e_kernels"] = profile_grid(lp_run, f"{cfg_name}_lp", rank)
        except Exception as e:
            import traceback
            res["e_lowprecision"] = {"error": repr(e), "tb": traceback.format_exc()[-3000:]}
        log0("e", json.dumps(res.get("e_lowprecision"))[:3000])
        log0("e kernels", json.dumps(res.get("e_kernels"))[:1500])
    del buf, ref_a, x0
    torch.cuda.empty_cache()

    # ---------------- (b) packed-QKV all-to-all ----------------
    from tensorrt_llm._torch.visual_gen.attention.backends.parallel import _all_to_all_5d_raw
    from tensorrt_llm._torch.distributed.ops import all_to_all_4d
    Sp = M // P
    qkv = rnd(1, Sp, 3, H, D)
    out_b = {}

    def b_full():
        out_b["o"] = _all_to_all_5d_raw(qkv, None)
        return out_b["o"]

    inp_perm = qkv.reshape(1, Sp, 3, P, Hp, D).permute(3, 0, 1, 2, 4, 5).contiguous()
    inp_flat = inp_perm.flatten()
    out_flat = torch.empty_like(inp_flat)

    def b_a2a():
        dist.all_to_all_single(out_flat, inp_flat)
        return out_flat

    bytes_b = qkv.numel() * 2 // P
    for key, fn in (("b_qkv_a2a_full", b_full), ("b_qkv_a2a_only", b_a2a)):
        st = stats(time_loop(fn))
        st["bytes_per_dir"] = bytes_b
        st["GBps_per_dir"] = round(bytes_b / (st["median_ms"] * 1e-3) / 1e9, 2)
        res[key] = st
        log0(key, json.dumps(st))
    # expected: raw[p] = qkv_p[..., rank heads]
    qs = gather(qkv)
    exp_b = torch.stack([q.reshape(1, Sp, 3, P, Hp, D)[:, :, :, rank] for q in qs], 0)
    del qs
    # all_to_all_4d (unfused path) reference, computed once
    ref4 = [all_to_all_4d(qkv[:, :, i].contiguous(), scatter_dim=2, gather_dim=1) for i in range(3)]
    raw = b_full()
    torch.cuda.synchronize()
    unpacked = raw.permute(1, 0, 2, 3, 4, 5).reshape(1, P * Sp, 3, Hp, D)
    res["b_vs_all_to_all_4d"] = all(torch.equal(unpacked[:, :, i], ref4[i]) for i in range(3))
    del ref4, unpacked, raw

    def exp_bf(out):
        eq = torch.equal(out, exp_b)
        return eq, 0.0 if eq else float("inf")

    res["b_check"] = check_loop(b_full, exp_bf, readonly=(qkv,))
    if extra.get("profile", True):
        res["b_kernels"] = profile_grid(b_a2a, f"{cfg_name}_b", rank)
    log0("b check", res["b_check"]["passed"], "vs a2a4d", res["b_vs_all_to_all_4d"])
    del qkv, exp_b, inp_perm, inp_flat, out_flat, out_b
    torch.cuda.empty_cache()

    # ---------------- (c) output all-to-all [1,M,H/2,D] -> [1,M/2,H,D] ----------------
    o = rnd(1, M, Hp, D)

    def c_full():
        return all_to_all_4d(o, scatter_dim=1, gather_dim=2)

    c_in = o.view(1, P, Sp, Hp, D).permute(1, 0, 2, 3, 4).contiguous().flatten()
    c_out = torch.empty_like(c_in)

    def c_a2a():
        dist.all_to_all_single(c_out, c_in)
        return c_out

    bytes_c = o.numel() * 2 // P
    for key, fn in (("c_out_a2a_full", c_full), ("c_out_a2a_only", c_a2a)):
        st = stats(time_loop(fn))
        st["bytes_per_dir"] = bytes_c
        st["GBps_per_dir"] = round(bytes_c / (st["median_ms"] * 1e-3) / 1e9, 2)
        res[key] = st
        log0(key, json.dumps(st))
    os_ = gather(o)
    exp_c = torch.cat([oo[:, rank * Sp:(rank + 1) * Sp] for oo in os_], dim=2)
    del os_

    def exp_cf(out):
        eq = torch.equal(out, exp_c)
        return eq, 0.0 if eq else float("inf")

    res["c_check"] = check_loop(c_full, exp_cf, readonly=(o,))
    log0("c check", res["c_check"]["passed"])
    del o, c_in, c_out, exp_c
    torch.cuda.empty_cache()

    # ---------------- (d) async Ulysses CE push via _SymmetricMemory ----------------
    if extra.get("async"):
        try:
            import tensorrt_llm  # noqa: F401  (loads torch.ops.trtllm)
            pg = dist.new_group([0, 1], backend="cuda:nccl,cpu:gloo")
            pgb = pg.boxed()
            x = rnd(1, Sp, H, D)
            side = torch.cuda.Stream()
            cur = torch.cuda.current_stream()
            last = {}

            def d_once(npush=1):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s0 = torch.cuda.Event(enable_timing=True)
                s1 = torch.cuda.Event(enable_timing=True)
                s2 = torch.cuda.Event(enable_timing=True)
                e0.record()
                handles = []
                recvs = []
                for _ in range(npush):
                    r, h = torch.ops.trtllm.ulysses_a2a_async_prepare(x, pgb)
                    handles.append(h)
                    recvs.append(r)
                e1.record()
                ev = torch.cuda.Event()
                ev.record()
                with torch.cuda.stream(side):
                    ev.wait()
                    s0.record()
                    for h in handles:
                        torch.ops.trtllm.ulysses_a2a_async_push(h, pgb)
                    s1.record()
                    for _ in handles:
                        torch.ops.trtllm.ulysses_a2a_async_barrier(pgb, 60000)
                    s2.record()
                    done = torch.cuda.Event()
                    done.record()
                cur.wait_event(done)
                e2.record()
                last["recv"] = recvs[-1]
                return (e0, e1, e2, s0, s1, s2)

            for npush, key in ((1, "d_async_push_x1"), (3, "d_async_push_x3")):
                evs = []
                for i in range(WARM + ITERS):
                    t = d_once(npush)
                    if i >= WARM:
                        evs.append(t)
                torch.cuda.synchronize()
                tot = [a.elapsed_time(c) for a, b, c, *_ in evs]
                prep = [a.elapsed_time(b) for a, b, *_ in evs]
                push = [s0.elapsed_time(s1) for *_, s0, s1, s2 in evs]
                bar = [s1.elapsed_time(s2) for *_, s0, s1, s2 in evs]
                ce = [s0.elapsed_time(s2) for *_, s0, s1, s2 in evs]
                nb = npush * x.numel() * 2 // P
                st = {"total": stats(tot), "prepare_kernel": stats(prep), "push_enqueue_span": stats(push),
                      "push_plus_barrier": stats(ce), "barrier": stats(bar), "bytes_per_dir": nb}
                st["GBps_per_dir_push_plus_barrier"] = round(
                    nb / (st["push_plus_barrier"]["median_ms"] * 1e-3) / 1e9, 2)
                st["GBps_per_dir_total"] = round(nb / (st["total"]["median_ms"] * 1e-3) / 1e9, 2)
                res[key] = st
                log0(key, json.dumps(st))
            # correctness: recv [P, 1, Sp, Hp, D]; recv[p] = x_p[:, :, rank heads]
            xs_ = gather(x)
            exp_d = torch.stack([xx[:, :, rank * Hp:(rank + 1) * Hp] for xx in xs_], 0)
            ref4 = all_to_all_4d(x, scatter_dim=2, gather_dim=1)
            del xs_
            bad = []
            vs4 = True
            for i in range(WARM + ITERS):
                d_once(1)
                torch.cuda.synchronize()
                r = last["recv"]
                if not torch.equal(r.reshape(exp_d.shape), exp_d):
                    bad.append(i)
                if i == 0:
                    vs4 = torch.equal(
                        r.reshape(P, 1, Sp, Hp, D).permute(1, 0, 2, 3, 4).reshape(1, M, Hp, D), ref4)
            res["d_check"] = {"passed": (not bad) and vs4, "bad": bad[:10], "vs_all_to_all_4d": vs4,
                              "iterations": WARM + ITERS}
            log0("d check", json.dumps(res["d_check"]))
        except Exception as e:
            import traceback
            res["d_error"] = {"error": repr(e), "tb": traceback.format_exc()[-3000:]}
            log0("d error", json.dumps(res["d_error"])[:3000])

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUTDIR, f"{cfg_name}_r{rank}.json"), "w"))
    dist.barrier()
    dist.destroy_process_group()


CONFIGS = {
    "default": {},
    "proto_simple": {"NCCL_PROTO": "Simple"},
    "proto_ll128": {"NCCL_PROTO": "LL128"},
    "nch2": {"NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"},
    "nch4": {"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "nch8": {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"},
    "nch16": {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"},
    "nch32": {"NCCL_MIN_NCHANNELS": "32", "NCCL_MAX_NCHANNELS": "32"},
    "simple_nch4": {"NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"},
    "buff4m": {"NCCL_BUFFSIZE": str(4 * 1024 * 1024)},
    "buff16m": {"NCCL_BUFFSIZE": str(16 * 1024 * 1024)},
    "algo_ring": {"NCCL_ALGO": "Ring"},
    "algo_tree": {"NCCL_ALGO": "Tree"},
    "p2p_level_sys": {"NCCL_P2P_LEVEL": "SYS"},
    "p2p_ce_memcpy": {"NCCL_P2P_USE_CUDA_MEMCPY": "1"},
    "p2p_disable_shm": {"NCCL_P2P_DISABLE": "1"},
    "default_rep": {},
}


def grep_nccl(cfg_name, limit):
    keys = ("NCCL version", "via ", "coll channels", "nRanks 2", "P2P/", "SHM/", "CUMEM", "Channel 00/",
            "Connected all", "threadThresholds", "Using network", "MAX_NCHANNELS", "P2P_LEVEL",
            "PROTO", "BUFFSIZE", "ALGO", "Ring 00", "Tree 0", "NVLS", "set by environment")
    lines = []
    for f in sorted(glob.glob(f"/tmp/nccl_{cfg_name}.*")):
        for ln in open(f, errors="replace"):
            if any(k in ln for k in keys):
                lines.append(os.path.basename(f) + ": " + ln.rstrip()[:220])
    seen, out = set(), []
    for ln in lines:
        k = ln.split(": ", 1)[1].split("NCCL INFO", 1)[-1]
        if k not in seen:
            seen.add(k)
            out.append(ln)
    return out[:limit]


def main():
    ckpt = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else list(CONFIGS)
    M, hidden, H, D, info = shapes(ckpt)
    log("SHAPES", json.dumps(info))
    N = M * hidden * 2
    log(f"N = M*hidden*2 = {N} bytes ({N / 2**20:.1f} MiB); H*D={H * D}")
    summary = {"shapes": info, "N_bytes": N}
    if "default" in only:
        summary["topology"] = topology()
        summary["link"] = link_floor(N)
    base_env = dict(os.environ)
    port = 29600
    for name in only:
        env = CONFIGS[name]
        os.environ.clear()
        os.environ.update(base_env)
        os.environ.update(env)
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,P2P,SHM,GRAPH,TUNING,ENV" if name == "default" else "INIT,P2P,SHM,ENV"
        os.environ["NCCL_DEBUG_FILE"] = f"/tmp/nccl_{name}.%h.%p.log"
        extra = {"profile": True, "async": name == "default", "lowprecision": name == "default"}
        port += 7
        log(f"===== config {name} env={env} =====")
        t0 = time.time()
        try:
            mp.spawn(worker, args=(name, M, hidden, H, D, port, extra), nprocs=2, join=True)
        except Exception as e:
            log(f"config {name} FAILED: {e!r}"[:3000])
            summary[name] = {"error": repr(e)[:2000]}
        log(f"config {name} took {time.time() - t0:.1f}s")
        for ln in grep_nccl(name, 60 if name == "default" else 14):
            log("  NCCL", ln)
        for r in (0, 1):
            p = os.path.join(OUTDIR, f"{name}_r{r}.json")
            if os.path.exists(p):
                summary.setdefault(name, {})[f"r{r}"] = json.load(open(p))
    os.environ.clear()
    os.environ.update(base_env)

    log("\n===== SUMMARY (rank0 median_ms, noise, GB/s per dir, rank1 median_ms) =====")
    keys = ["a_allreduce", "b_qkv_a2a_full", "b_qkv_a2a_only", "c_out_a2a_full", "c_out_a2a_only"]
    table = {}
    allpass = True
    for name in only:
        r0 = summary.get(name, {}).get("r0")
        r1 = summary.get(name, {}).get("r1")
        if not r0:
            table[name] = "missing"
            allpass = False
            log(name, "missing")
            continue
        row = {}
        for k in keys:
            if k in r0:
                row[k] = (r0[k]["median_ms"], r0[k]["noise"], r0[k]["GBps_per_dir"],
                          r1[k]["median_ms"] if r1 and k in r1 else None)
        checks = {k: r0[k]["passed"] for k in ("a_check", "b_check", "c_check", "d_check") if k in r0}
        checks.update({k + "_r1": r1[k]["passed"] for k in ("a_check", "b_check", "c_check", "d_check")
                       if r1 and k in r1})
        if "b_vs_all_to_all_4d" in r0:
            checks["b_vs_a2a4d"] = r0["b_vs_all_to_all_4d"]
        allpass &= all(checks.values())
        row["checks"] = checks
        ak = [k for k in r0.get("a_kernels", []) if "nccl" in k.get("name", "").lower()]
        bk = [k for k in r0.get("b_kernels", []) if "nccl" in k.get("name", "").lower()]
        row["a_kernel"] = ak[:2]
        row["b_kernel"] = bk[:2]
        table[name] = row
        log(name, json.dumps(row))
    for k in ("d_async_push_x1", "d_async_push_x3", "d_check", "d_error", "e_lowprecision",
              "lp_mapping_type", "lp_strategy_after_init", "e_kernels"):
        v = summary.get("default", {}).get("r0", {}).get(k)
        if v is not None:
            log(k, json.dumps(v)[:3000])
    json.dump(summary, open("/tmp/comm_analysis_summary.json", "w"))
    # compact final line: everything needed for the findings
    final = {"passed": bool(allpass), "M": M, "N_bytes": N, "shapes": {k: info.get(k) for k in (
        "hidden_size", "num_attention_heads", "attention_head_dim", "num_layers", "ffn_dim")}}
    topo = summary.get("topology")
    if topo:
        final["peer"] = [topo.get("can_access_peer_0_1"), topo.get("can_access_peer_1_0")]
        final["pcie_csv"] = "".join(c if 32 <= ord(c) < 127 else "|" for c in topo.get("pcie", ""))[:600]
        final["topo"] = "".join(c if 32 <= ord(c) < 127 else "|" for c in topo.get("topo", ""))[:500]
    link = summary.get("link")
    if link:
        final["link"] = {k: ({kk: (vv.get("median_ms"), vv.get("GBps"), vv.get("noise")) for kk, vv in v.items()
                              if kk.startswith("src")} | {"agg_wall_GBps": v.get("aggregate_GBps_wall")})
                         if isinstance(v, dict) else v for k, v in link.items()}
    final["table"] = {}
    for name, row in table.items():
        if not isinstance(row, dict):
            final["table"][name] = row
            continue
        r = {k: row[k] for k in row if k in keys}
        r["ok"] = all(row["checks"].values())
        r["agrid"] = [(k.get("grid"), k.get("block"), k.get("dur_us")) for k in row["a_kernel"][:1]]
        r["bgrid"] = [(k.get("grid"), k.get("block"), k.get("dur_us")) for k in row["b_kernel"][:1]]
        final["table"][name] = r
    r0 = summary.get("default", {}).get("r0", {})
    for k in ("d_async_push_x1", "d_async_push_x3"):
        v = r0.get(k)
        if v:
            final[k] = {"total": v["total"]["median_ms"], "total_noise": v["total"]["noise"],
                        "prep": v["prepare_kernel"]["median_ms"], "push+bar": v["push_plus_barrier"]["median_ms"],
                        "GBps_pb": v["GBps_per_dir_push_plus_barrier"], "GBps_tot": v["GBps_per_dir_total"],
                        "bytes": v["bytes_per_dir"]}
    final["d_check"] = r0.get("d_check")
    if "d_error" in r0:
        final["d_error"] = r0["d_error"]["error"][:600]
    e = r0.get("e_lowprecision")
    if e:
        final["lp"] = {k: e.get(k) for k in ("median_ms", "noise", "max_abs", "rel_l2", "bitwise_equal_nccl_ref",
                                              "repeat_deterministic", "error")}
        if "tb" in e:
            final["lp"]["tb_tail"] = e["tb"][-700:]
        final["lp_kernels"] = [(k.get("name"), k.get("grid"), k.get("dur_us")) for k in r0.get("e_kernels", [])][:4]
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
