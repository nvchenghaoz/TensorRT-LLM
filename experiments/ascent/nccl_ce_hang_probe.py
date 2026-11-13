"""Probe: does an in-flight NCCL all-reduce under NCCL_P2P_USE_CUDA_MEMCPY=1 deadlock against a host-side
CUDA call on the issuing thread? One variant per process pair (a hang is killed by the caller).

  python nccl_ce_hang_probe.py <variant>
variants: plain (AR then wait), free (AR in flight, then free cached blocks: torch.cuda.empty_cache -> cudaFree),
          malloc (AR in flight, then a fresh 2 GiB allocation -> cudaMalloc), sync (AR in flight, then
          torch.cuda.synchronize on the issuing thread), async4 (4 back-to-back async ARs then waits).
Prints {"variant", "ok", "ms"} from rank 0.
"""
import json
import os
import sys
import time


def worker(rank, variant, port):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            device_id=torch.device("cuda", rank))
    x = torch.randn(9600, 5376, device="cuda", dtype=torch.bfloat16)
    dist.all_reduce(torch.ones(1, device="cuda"))
    torch.cuda.synchronize()
    for it in range(3):
        junk = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
        del junk  # cached block for 'free'
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        if variant == "async4":
            ws = [dist.all_reduce(x[i * 2400:(i + 1) * 2400], async_op=True) for i in range(4)]
            y = torch.matmul(x[:4096, :4096], x[:4096, :4096])
            for w in ws:
                w.wait()
        else:
            w = dist.all_reduce(x, async_op=True)
            if variant == "free":
                torch.cuda.empty_cache()
            elif variant == "malloc":
                big = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
                del big
            elif variant == "sync":
                torch.cuda.synchronize()
            w.wait()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[{variant}] iter {it} done in {(time.time() - t0) * 1e3:.1f} ms", flush=True)
    dist.barrier()
    if rank == 0:
        print(json.dumps({"variant": variant, "ok": True, "ms": (time.time() - t0) * 1e3}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import random

    import torch.multiprocessing as mp

    mp.spawn(worker, args=(sys.argv[1], 29000 + random.randint(0, 3000)), nprocs=2, join=True)
