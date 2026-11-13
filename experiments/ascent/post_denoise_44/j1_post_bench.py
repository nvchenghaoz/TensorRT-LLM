"""J1 for idea 101 (branch 44): MiniMax-H3 uint8 post, main's eager form vs the branch's chunked form.

usage: python j1_post_bench.py --arm {baseline,candidate} [--iters 30]

baseline arm: times main's post (inline copy of pipeline_minimax_h3.py _decode_video's last two lines on main).
candidate arm: times the branch's _video_to_uint8 (imported from the worktree), and also runs every J1 diagnostic:
  bitwise over 5 back-to-back calls, input left unchanged, fp32 input, strides, interleaved main/branch/main timing,
  per-arm peak allocation and largest single block, and the server path's .cpu().numpy().reshape(-1) copy.
Last stdout line: evidence JSON.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

T, H, W, C = 124, 768, 1344, 3
PIXEL_MEAN = None
PIXEL_STD = None


def log(msg):
    print(f"[j1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_post(video, pixel_mean, pixel_std):
    # Verbatim from main's _decode_video (pipeline_minimax_h3.py ~1243-1244).
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    return (video.permute(0, 2, 3, 4, 1) * 255).round().to(torch.uint8)


def spread(xs):
    xs = sorted(xs)
    n = len(xs)
    p10 = xs[max(0, int(0.1 * (n - 1)))]
    p90 = xs[min(n - 1, int(round(0.9 * (n - 1))))]
    med = statistics.median(xs)
    return {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "min_ms": xs[0], "max_ms": xs[-1],
            "noise_floor": (p90 - p10) / med, "n": n}


def time_calls(fn, inp, iters, warmup=5):
    for _ in range(warmup):
        out = fn(inp)
        del out
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(inp)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
        del out
    return times


def mem_probe(fn, inp):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=100000)
    out = fn(inp)
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    peak = torch.cuda.max_memory_allocated() - base
    largest_alloc = 0
    largest_seg = 0
    n_alloc = 0
    dev = torch.cuda.current_device()
    for ev in snap["device_traces"][dev]:
        if ev["action"] == "alloc":
            n_alloc += 1
            largest_alloc = max(largest_alloc, ev["size"])
        elif ev["action"] == "segment_alloc":
            largest_seg = max(largest_seg, ev["size"])
    retained = torch.cuda.memory_allocated() - base
    del out
    return {"peak_alloc_bytes": peak, "largest_block_bytes": largest_alloc, "largest_new_segment_bytes": largest_seg,
            "num_allocs": n_alloc, "output_bytes": retained}


def numa_info():
    info = {"affinity": sorted(os.sched_getaffinity(0)), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        p = torch.cuda.get_device_properties(0)
        bus = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        info["gpu_pci"] = bus
        for cand in (bus, bus[4:]):
            path = f"/sys/bus/pci/devices/{cand.lower()}/numa_node"
            if os.path.exists(path):
                info["gpu_numa_node"] = open(path).read().strip()
    except Exception as exc:  # noqa: BLE001
        info["gpu_pci_err"] = repr(exc)
    nodes = {}
    try:
        for d in sorted(os.listdir("/sys/devices/system/node")):
            if d.startswith("node"):
                nodes[d] = open(f"/sys/devices/system/node/{d}/cpulist").read().strip()
    except Exception as exc:  # noqa: BLE001
        nodes["err"] = repr(exc)
    info["numa_cpulists"] = nodes
    aff = info["affinity"]
    info["affinity_summary"] = f"{len(aff)} cpus, {aff[0]}-{aff[-1]}" if aff else "none"
    return info


def strides(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "contiguous": t.is_contiguous()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    log(f"arm={args.arm} gpu={torch.cuda.get_device_name(0)}")
    numa = numa_info()
    log(f"affinity {numa['affinity_summary']} cvd={numa['cuda_visible_devices']} "
        f"gpu_pci={numa.get('gpu_pci')} gpu_numa={numa.get('gpu_numa_node')}")

    sys.path.insert(0, os.getcwd())
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=dev).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=dev).view(1, -1, 1, 1, 1)
    log(f"pixel_mean={MINIMAX_H3_PIXEL_MEAN} pixel_std={MINIMAX_H3_PIXEL_STD} dtype={pixel_mean.dtype}")

    g = torch.Generator(device="cpu").manual_seed(0)
    video = (torch.randn((1, C, T, H, W), generator=g, dtype=torch.float32) * 1.5).to(torch.float16).to(dev)
    log(f"input {tuple(video.shape)} {video.dtype}")

    def f_main(v):
        return main_post(v, pixel_mean, pixel_std)

    if args.arm == "baseline":
        f_arm = f_main
        result = {}
    else:
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
            _POST_CHUNK_FRAMES, _video_to_uint8)

        def f_arm(v):
            return _video_to_uint8(v, pixel_mean, pixel_std)

        result = {"chunk_frames": _POST_CHUNK_FRAMES}

    times = time_calls(f_arm, video, args.iters)
    arm_stats = spread(times)
    log(f"arm timing: {arm_stats}")
    result.update({"arm": args.arm, "arm_timing": arm_stats, "numa": {k: v for k, v in numa.items() if k != 'affinity'}})
    result["arm_memory"] = mem_probe(f_arm, video)
    log(f"arm memory: {result['arm_memory']}")

    passed = True
    if args.arm == "candidate":
        # ---- Bitwise, untimed, same warmup/iteration pattern as timing; input checked after every call. ----
        pristine = video.clone()
        ref = f_main(video)
        log(f"main out strides {strides(ref)}")
        assert torch.equal(video, pristine)
        bit = []
        first = None
        for i in range(5 + args.iters):
            out = f_arm(video)
            eq_ref = torch.equal(out, ref)
            eq_first = True if first is None else torch.equal(out, first)
            in_ok = torch.equal(video, pristine)
            if first is None:
                first = out.clone()
            bit.append(eq_ref and eq_first and in_ok)
            if not (eq_ref and eq_first and in_ok):
                log(f"call {i}: eq_ref={eq_ref} eq_first={eq_first} input_unchanged={in_ok}")
            del out
        result["bitwise_fp16"] = {"calls": len(bit), "all_equal_and_input_unchanged": all(bit),
                                  "first5_ok": all(bit[:5]), "main_strides": strides(ref),
                                  "branch_strides": strides(first),
                                  "clamp_edges": {"frac_0": float((ref == 0).float().mean()),
                                                  "frac_255": float((ref == 255).float().mean())}}
        passed &= all(bit)
        log(f"bitwise fp16: {result['bitwise_fp16']}")

        # ---- fp32 decoder output: copy=True must leave it unmodified. ----
        v32 = (torch.randn((1, C, 8, H, W), generator=g, dtype=torch.float32) * 1.5).to(dev)
        p32 = v32.clone()
        ok32 = []
        r32 = f_main(v32)
        for _ in range(5):
            o32 = f_arm(v32)
            ok32.append(torch.equal(o32, r32) and torch.equal(v32, p32))
        result["bitwise_fp32_input"] = {"calls": 5, "all_equal_and_input_unchanged": all(ok32)}
        passed &= all(ok32)
        log(f"fp32 input: {result['bitwise_fp32_input']}")
        del v32, p32, r32, o32

        # ---- Odd T (not a multiple of the chunk), non-unit batch dims stay correct. ----
        vodd = video[:, :, :7].clone()
        odd_ok = torch.equal(f_arm(vodd), f_main(vodd))
        result["bitwise_T7"] = odd_ok
        passed &= odd_ok
        del vodd

        # ---- Interleaved main / branch / main (A/A). ----
        inter = {"main_a": [], "branch": [], "main_b": []}
        for _ in range(3):
            f_main(video)
            f_arm(video)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            for name, fn in (("main_a", f_main), ("branch", f_arm), ("main_b", f_main)):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                o = fn(video)
                e.record()
                e.synchronize()
                inter[name].append(s.elapsed_time(e))
                del o
        result["interleaved"] = {k: spread(v) for k, v in inter.items()}
        log(f"interleaved: {json.dumps(result['interleaved'])}")
        result["main_memory"] = mem_probe(f_main, video)
        log(f"main memory: {result['main_memory']}")

        # ---- Server path: encode_video does video.cpu().numpy() then reshape(-1) (video is [T,H,W,C]). ----
        main_dev = ref[0]
        br_dev = first[0]
        cpu_t = {"main": [], "branch": []}
        rs_t = {"main": [], "branch": []}
        views = {}
        flat_eq = True
        for rep in range(12):
            for name, t in (("main", main_dev), ("branch", br_dev)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                npv = t.cpu().numpy()
                t1 = time.perf_counter()
                flat = npv.reshape(-1)
                t2 = time.perf_counter()
                cpu_t[name].append((t1 - t0) * 1e3)
                rs_t[name].append((t2 - t1) * 1e3)
                views[name] = {"np_strides": list(npv.strides), "reshape_is_view": bool(np.shares_memory(flat, npv))}
                if name == "main":
                    main_flat = flat
                else:
                    if rep == 0:
                        flat_eq = bool(np.array_equal(main_flat, flat))
                del npv, flat
        result["server_path"] = {"cpu_ms": {k: spread(v) for k, v in cpu_t.items()},
                                 "reshape_ms": {k: spread(v) for k, v in rs_t.items()},
                                 "layout": views, "bytes_equal": flat_eq,
                                 "torch_threads": torch.get_num_threads()}
        passed &= flat_eq
        log(f"server path: {json.dumps(result['server_path'])}")

    result["passed"] = bool(passed)
    result["batch"] = {"1": {"step_ms": arm_stats["median_ms"], "noise_floor": arm_stats["noise_floor"]}}
    result["output"] = {"class": "exact"}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
