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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bench", choices=["bench", "check", "compile"])
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
    import random

    import torch.multiprocessing as mp

    port = 29500 + random.randint(0, 2000)
    mp.spawn(worker, args=(args, port), nprocs=2, join=True)


if __name__ == "__main__":
    sys.exit(main())
