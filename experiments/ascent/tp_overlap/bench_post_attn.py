"""2-GPU microbenchmark: MiniMax-H3 TP2 post-attention half, serialized vs token-chunked
all-reduce overlap.

Per-rank shapes (TP2, 768x1344x124f): M=37966 packed rows, hidden 5376, to_out K=3584,
gate_up N=14336 (local 2*7168), down K=7168.

Path per block (post attention):
  a  = o @ Wout^T                (row-parallel partial)  -> all-reduce
  x1, n = h3_gate_res_norm_mod(x, a, ...)
  h  = swiglu(n @ Wgu^T)
  f  = h @ Wd^T                  (row-parallel partial)  -> all-reduce
  y  = x1 + gate_mlp[idx] * f

serial  : as main runs it (clone + blocking c10d all_reduce, like allreduce_pg_by_name).
mlpK    : MLP rows split into K chunks, each chunk's all-reduce async on the PG stream.
fullK   : to_out also chunked; each chunk's MLP waits only for its own to_out AR.

Usage: python bench_post_attn.py [--envs "A=1,B=2;C=3"] [--ks 2,4,8] [--prio 0|1]
Prints one JSON line at the end.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M = 37966
D = 5376
DA = 3584
FF = 7168
N_MOD_ROWS = 6
WARMUP = 5
ITERS = 20


def log(*a):
    print(*a, flush=True)


def bounds_for(m, k, align=128):
    step = (m + k - 1) // k
    step = (step + align - 1) // align * align
    out = []
    r = 0
    while r < m:
        out.append((r, min(m, r + step)))
        r += step
    return out


def worker(rank, world, port, env, ks, prio, results_q):
    os.environ.update(env)
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    opts = None
    if prio:
        opts = dist.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = True
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev, pg_options=opts)
    pg = dist.group.WORLD

    from tensorrt_llm._torch.modules.swiglu import swiglu
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import h3_gate_res_norm_mod

    g = torch.Generator(device=dev)
    g.manual_seed(1234)  # shared activations
    o = torch.randn(1, M, DA, device=dev, dtype=torch.bfloat16, generator=g)
    x = torch.randn(1, M, D, device=dev, dtype=torch.bfloat16, generator=g)
    mod = (torch.randn(N_MOD_ROWS, 6 * D, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    idx = torch.randint(0, N_MOD_ROWS, (M,), device=dev, generator=g)
    norm_w = (1.0 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    gr = torch.Generator(device=dev)
    gr.manual_seed(5678 + rank)  # per-rank weight shards
    w_out = (torch.randn(D, DA, device=dev, generator=gr) / DA**0.5).to(torch.bfloat16)
    w_gu = (torch.randn(2 * FF, D, device=dev, generator=gr) / D**0.5).to(torch.bfloat16)
    w_d = (torch.randn(D, FF, device=dev, generator=gr) / FF**0.5).to(torch.bfloat16)
    eps = 1e-6
    gate_mlp = mod[:, 5 * D : 6 * D]
    pristine = [t.clone() for t in (o, x, mod, idx, norm_w, w_out, w_gu, w_d)]

    @torch.compile(dynamic=False)
    def final(x1, f, gate, idx):
        return x1 + gate.index_select(0, idx) * f

    @torch.compile(dynamic=False)
    def final_cat(x1s, fs, gate, idx):
        return torch.cat(x1s, dim=1) + gate.index_select(0, idx) * torch.cat(fs, dim=0).unsqueeze(0)

    def mlp(n2d):
        return torch.nn.functional.linear(swiglu(torch.nn.functional.linear(n2d, w_gu)), w_d)

    def ar_blocking(t):
        out = t.clone()
        dist.all_reduce(out, group=pg)
        return out

    def run_serial():
        a = ar_blocking(torch.nn.functional.linear(o, w_out))
        x1, n = h3_gate_res_norm_mod(x, a, norm_w, mod, idx, eps, 2, 4, 3)
        f = ar_blocking(mlp(n.view(-1, D))).view(1, M, D)
        return final(x1, f, gate_mlp, idx)

    def run_mlp_chunked(k):
        a = ar_blocking(torch.nn.functional.linear(o, w_out))
        x1, n = h3_gate_res_norm_mod(x, a, norm_w, mod, idx, eps, 2, 4, 3)
        n2 = n.view(-1, D)
        fs, works = [], []
        for r0, r1 in bounds_for(M, k):
            f = mlp(n2[r0:r1])
            works.append(dist.all_reduce(f, group=pg, async_op=True))
            fs.append(f)
        for w in works:
            w.wait()
        return final_cat([x1], fs, gate_mlp, idx)

    def run_full_chunked(k):
        bs = bounds_for(M, k)
        o2 = o.view(-1, DA)
        a_s, aw = [], []
        for r0, r1 in bs:
            a = torch.nn.functional.linear(o2[r0:r1], w_out)
            aw.append(dist.all_reduce(a, group=pg, async_op=True))
            a_s.append(a)
        x1s, fs, fw = [], [], []
        for i, (r0, r1) in enumerate(bs):
            aw[i].wait()
            x1, n = h3_gate_res_norm_mod(
                x[:, r0:r1], a_s[i].view(1, r1 - r0, D), norm_w, mod, idx[r0:r1], eps, 2, 4, 3
            )
            f = mlp(n.view(-1, D))
            fw.append(dist.all_reduce(f, group=pg, async_op=True))
            x1s.append(x1)
            fs.append(f)
        for w in fw:
            w.wait()
        return final_cat(x1s, fs, gate_mlp, idx)

    configs = [("serial", run_serial)]
    for k in ks:
        configs.append((f"mlp{k}", lambda k=k: run_mlp_chunked(k)))
    for k in ks:
        configs.append((f"full{k}", lambda k=k: run_full_chunked(k)))

    def timed(fn, warm=WARMUP, iters=ITERS):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med = statistics.median(ts)
        p10 = ts[max(0, int(0.1 * len(ts)))]
        p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))]
        return {"median_ms": med, "p10": p10, "p90": p90, "noise": (p90 - p10) / med}

    res = {"rank": rank, "env": env, "prio": prio, "timing": {}, "check": {}, "parts": {}}

    # Component timings (isolated)
    o2 = o.view(-1, DA)
    a_full = torch.nn.functional.linear(o, w_out)
    _, n_full = h3_gate_res_norm_mod(x, a_full, norm_w, mod, idx, eps, 2, 4, 3)
    n2 = n_full.view(-1, D)
    h_full = swiglu(torch.nn.functional.linear(n2, w_gu))
    parts = {
        "to_out": lambda: torch.nn.functional.linear(o, w_out),
        "norm": lambda: h3_gate_res_norm_mod(x, a_full, norm_w, mod, idx, eps, 2, 4, 3),
        "gate_up+swiglu": lambda: swiglu(torch.nn.functional.linear(n2, w_gu)),
        "down": lambda: torch.nn.functional.linear(h_full, w_d),
        "ar_clone_blocking": lambda: ar_blocking(a_full),
        "ar_inplace": lambda: dist.all_reduce(a_full.clone(), group=pg),
    }
    for name, fn in parts.items():
        res["parts"][name] = timed(fn)
        if rank == 0:
            log(f"[{env} prio={prio}] part {name}: {res['parts'][name]}")
    # AR of one chunk alone
    for k in ks:
        r0, r1 = bounds_for(M, k)[0]
        t = a_full.view(-1, D)[r0:r1].clone()
        res["parts"][f"ar_chunk_k{k}"] = timed(lambda t=t: dist.all_reduce(t, group=pg))
        if rank == 0:
            log(f"[{env} prio={prio}] part ar_chunk_k{k}: {res['parts'][f'ar_chunk_k{k}']}")
    # gate_up under a concurrent AR (side stream), to see GEMM slowdown
    big = a_full.clone()

    def gemm_with_ar():
        w = dist.all_reduce(big, group=pg, async_op=True)
        y = torch.nn.functional.linear(n2, w_gu)
        w.wait()
        return y

    res["parts"]["gate_up_plus_concurrent_ar"] = timed(gemm_with_ar)
    res["parts"]["gate_up_alone"] = timed(lambda: torch.nn.functional.linear(n2, w_gu))
    if rank == 0:
        log(f"[{env}] gate_up alone {res['parts']['gate_up_alone']['median_ms']:.3f} "
            f"with AR {res['parts']['gate_up_plus_concurrent_ar']['median_ms']:.3f}")
    del a_full, n_full, n2, h_full, big

    # Timed runs
    for name, fn in configs:
        res["timing"][name] = timed(fn)
        if rank == 0:
            log(f"[{env} prio={prio}] {name}: {res['timing'][name]}")

    # Untimed repeat: same warmup + iteration count; compare each iteration to the serial ref.
    ref = run_serial().clone()
    torch.cuda.synchronize()
    # GEMM row-chunk bitwise probe
    probe = {}
    full_out = torch.nn.functional.linear(o2, w_out)
    full_gu = torch.nn.functional.linear(pristine[1].view(-1, D), w_gu)
    for k in ks:
        eq_out = all(
            torch.equal(torch.nn.functional.linear(o2[r0:r1], w_out), full_out[r0:r1])
            for r0, r1 in bounds_for(M, k)
        )
        eq_gu = all(
            torch.equal(torch.nn.functional.linear(pristine[1].view(-1, D)[r0:r1], w_gu), full_gu[r0:r1])
            for r0, r1 in bounds_for(M, k)
        )
        probe[f"k{k}"] = {"to_out_rows_bitwise": eq_out, "gate_up_rows_bitwise": eq_gu}
    del full_out, full_gu
    res["gemm_chunk_probe"] = probe
    for name, fn in configs:
        bad = []
        maxdiff = 0.0
        for it in range(WARMUP + ITERS):
            y = fn()
            torch.cuda.synchronize()
            if not torch.equal(y, ref):
                d = (y.float() - ref.float()).abs().max().item()
                maxdiff = max(maxdiff, d)
                bad.append(it)
            for t, p in zip((o, x, mod, idx, norm_w, w_out, w_gu, w_d), pristine):
                if not torch.equal(t, p):
                    bad.append(f"input_changed@{it}")
        rel = None
        if bad:
            y = fn()
            rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        res["check"][name] = {"bitwise_all_iters": not bad, "bad_iters": bad[:10],
                              "max_abs": maxdiff, "rel_l2": rel}
        if rank == 0:
            log(f"[{env}] check {name}: {res['check'][name]}")
    # Determinism of the serial ref across ranks: outputs must be identical on both ranks.
    other = ref.clone()
    dist.broadcast(other, src=0)
    res["ranks_identical"] = bool(torch.equal(other, ref))
    results_q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="")
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--prio", default="0")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",") if k]
    envs = [{}]
    if args.envs:
        envs = []
        for spec in args.envs.split(";"):
            d = {}
            for kv in spec.split(","):
                if kv:
                    k, v = kv.split("=")
                    d[k] = v
            envs.append(d)
    prios = [int(p) for p in args.prio.split(",")]
    ctx = mp.get_context("spawn")
    all_res = []
    port = 29600
    for env in envs:
        for prio in prios:
            q = ctx.Queue()
            port += 1
            t0 = time.time()
            procs = [ctx.Process(target=worker, args=(r, 2, port, env, ks, prio, q)) for r in range(2)]
            for p in procs:
                p.start()
            got = [q.get(timeout=1200) for _ in range(2)]
            for p in procs:
                p.join()
            got.sort(key=lambda r: r["rank"])
            log(f"env {env} prio {prio} done in {time.time() - t0:.0f}s")
            all_res.append(got[0] | {"rank1_check": got[1]["check"], "rank1_timing": got[1]["timing"]})
    summary = []
    for r in all_res:
        ser = r["timing"]["serial"]["median_ms"]
        summary.append({
            "env": r["env"], "prio": r["prio"],
            "ms": {k: round(v["median_ms"], 3) for k, v in r["timing"].items()},
            "noise": {k: round(v["noise"], 4) for k, v in r["timing"].items()},
            "speedup_vs_serial": {k: round(ser / v["median_ms"], 4) for k, v in r["timing"].items()},
            "parts_ms": {k: round(v["median_ms"], 3) for k, v in r["parts"].items()},
            "check_bitwise": {k: v["bitwise_all_iters"] for k, v in r["check"].items()},
            "check_maxabs": {k: v["max_abs"] for k, v in r["check"].items()},
            "gemm_chunk_probe": r["gemm_chunk_probe"],
        })
    print(json.dumps({"results": summary}))


if __name__ == "__main__":
    main()
