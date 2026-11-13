"""Idea 68 acceptance 3: real MiniMax-H3 TP2 block (compiled, chunked TP overlap, NCCL co-run) with
the SM120 cublas_mm tile pin on vs off, in one 2-GPU process pair, with A/A arms.

Builds the block with experiments/ascent/tp_overlap/block_pair.py's build() (checkpoint config,
synthetic per-rank weights, M = 37,966, VANILLA attention, torch.compile mode=default), then times
arms in the order main, pin, pin_aa, main_aa (ABBA), where
  main = use_custom_cublas_mm False on qkv_proj / to_out / gate_up_proj / down_proj (F.linear, main),
  pin  = the flags as the model sets them on SM120 (trtllm::cublas_mm + sm120_bf16_algo_list).
Each arm: torch._dynamo.reset(), compile, 5 warmup + ITERS timed block forwards (CUDA events,
barrier before each), and its last output compared bitwise with the main arm's (both ranks).
Refuses (wrong_pair) unless the two visible GPUs are a PIX pair at x16.
usage: python block_pin_pair.py <checkpoint>
Last stdout line: JSON.
"""
import json
import os
import statistics
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tp_overlap"))

WARMUP = 5
ITERS = int(os.environ.get("BLOCK_PIN_ITERS", "20"))
ORDER = ["main", "pin", "pin_aa", "main_aa"]


def log(*a):
    print(*a, flush=True)


def pair_link():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,pcie.link.width.current,"
                        "pcie.link.width.max", "--format=csv,noheader"], capture_output=True, text=True).stdout
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    log("nvidia-smi gpus:\n" + q)
    log("topo -m:\n" + topo)
    rows = [[c.strip() for c in l.split(",")] for l in q.strip().splitlines()]
    by_uuid = {r[1]: r for r in rows}
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    sel = []
    for v in vis[:2]:
        r = by_uuid.get(v) or next((x for x in rows if x[0] == v), None)
        sel.append(r)
    if len(sel) != 2 or any(r is None for r in sel):
        raise RuntimeError(f"parser error: cannot map CUDA_VISIBLE_DEVICES={vis} to nvidia-smi rows")
    i0, i1 = int(sel[0][0]), int(sel[1][0])
    link = None
    for l in topo.splitlines():
        parts = l.split()
        if parts and parts[0] == f"GPU{i0}":
            link = parts[1 + i1]
            break
    if link is None:
        raise RuntimeError("parser error: topo row not found")
    widths = [(int(r[0]), r[2], r[3], r[4]) for r in sel]
    return {"gpus": [i0, i1], "link": link, "widths": widths}


def worker(rank, ckpt, q):
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29713")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    import block_pair as bp
    block, (x, temb, idx, rot), cfg = bp.build(ckpt, rank)
    lins = [getattr(block.attn, "qkv_proj", None), block.attn.to_out[0],
            getattr(block.ff, "gate_up_proj", None), block.ff.down_proj]
    model_flags = [bool(getattr(m, "use_custom_cublas_mm", False)) for m in lins]
    chunks = getattr(block, "_tp_overlap_chunks", 0)
    if rank == 0:
        log(f"model-set use_custom_cublas_mm {model_flags}; overlap chunks {chunks}")

    def call(fn):
        with torch.inference_mode():
            return fn(x, temb, idx, rot, None, None, valid_seq_len=None)

    from torch._dynamo.utils import counters
    res = {"rank": rank, "model_flags": model_flags, "chunks": chunks, "arms": {}}
    ref = None
    for arm in ORDER:
        on = arm.startswith("pin")
        for m, f in zip(lins, model_flags):
            m.use_custom_cublas_mm = bool(f and on)
        torch._dynamo.reset()
        counters.clear()
        fn = torch.compile(block, mode="default", dynamic=None)
        for _ in range(WARMUP):
            y = call(fn)
        torch.cuda.synchronize()
        stats_warm = dict(counters["stats"])
        ts = []
        for _ in range(ITERS):
            dist.barrier()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            y = call(fn)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        stats_end = dict(counters["stats"])
        ts.sort()
        med = statistics.median(ts)
        lo, hi = ts[int(0.1 * (len(ts) - 1))], ts[int(0.9 * (len(ts) - 1) + 0.5)]
        if ref is None:
            ref = y.clone()
        r = {"median_ms": med, "p10": lo, "p90": hi, "noise": (hi - lo) / med, "n": len(ts),
             "bitwise_vs_main": bool(torch.equal(y, ref)),
             "recompiled_after_warmup": stats_warm != stats_end,
             "graph_breaks": len(counters["graph_break"])}
        res["arms"][arm] = r
        if rank == 0:
            log(f"arm {arm}: {r}")
    q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ckpt = sys.argv[1]
    try:
        pl = pair_link()
    except RuntimeError as e:
        log(f"FAIL LOUD: {e}")
        print(json.dumps({"passed": False, "error": str(e)}))
        return
    log(f"pair: {pl}")
    x8 = any(w[2] != "16" for w in pl["widths"]) or 5 in pl["gpus"]
    if pl["link"] != "PIX" or x8:
        print(json.dumps({"wrong_pair": True, "pair": pl}))
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, ckpt, q)) for r in range(2)]
    for p in ps:
        p.start()
    outs = [q.get() for _ in ps]
    for p in ps:
        p.join()
    outs.sort(key=lambda o: o["rank"])
    r0 = outs[0]["arms"]
    main_ms = statistics.mean([r0["main"]["median_ms"], r0["main_aa"]["median_ms"]])
    pin_ms = statistics.mean([r0["pin"]["median_ms"], r0["pin_aa"]["median_ms"]])
    aa_main = abs(r0["main_aa"]["median_ms"] / r0["main"]["median_ms"] - 1)
    aa_pin = abs(r0["pin_aa"]["median_ms"] / r0["pin"]["median_ms"] - 1)
    noise = max(a["noise"] for a in r0.values())
    bit = all(a["bitwise_vs_main"] for o in outs for a in o["arms"].values())
    no_rc = not any(a["recompiled_after_warmup"] for o in outs for a in o["arms"].values())
    gb = {arm: [o["arms"][arm]["graph_breaks"] for o in outs] for arm in ORDER}
    summary = {"pair": pl, "main_ms": main_ms, "pin_ms": pin_ms, "delta": pin_ms / main_ms - 1,
               "aa_main": aa_main, "aa_pin": aa_pin, "max_arm_noise": noise,
               "bitwise_all_arms_both_ranks": bit, "no_recompile_after_warmup": no_rc,
               "graph_breaks_per_arm_rank": gb, "model_flags": outs[0]["model_flags"],
               "chunks": outs[0]["chunks"], "ranks": outs,
               "batch": {"1": {"step_ms": pin_ms, "noise_floor": noise}},
               "passed": bool(bit and no_rc and gb["pin"] == gb["main"])}
    log(f"block main {main_ms:.3f} ms, pin {pin_ms:.3f} ms ({100 * summary['delta']:+.2f}%), "
        f"A/A main {100 * aa_main:.2f}% pin {100 * aa_pin:.2f}%, bitwise {bit}")
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
