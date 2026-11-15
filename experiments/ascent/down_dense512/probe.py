"""Idea 112 / branch 51 stage 0: down_proj NVFP4 GEMM at M 513-1023, TRT CUTLASS DP tactics.

No model load. Synthetic operands of the down shape (from <checkpoint>/config.json) plus, when the
shared capture set exists, real layer 0/30/59 down inputs captured at M=1000 (isl1k@1), sliced to
smaller M by unswizzling/reswizzling the activation SF.

Per M: main's in-process pick (tuned like the engine warmup: one M=8192 call under autotune()),
every TRT CUTLASS config: cold-L2 time (256 MiB flush before every call; 5 warmup + 20 timed) and
int16 mismatch vs main's pick output (synthetic seed 1/2/heavy and real slices).

usage: probe.py <checkpoint> [capture_dir]
"""
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nvfp4_dense"))
import bench  # noqa: E402

log = bench.log
M_LIST = [513, 520, 576, 600, 640, 704, 768, 832, 896, 900, 960, 1000, 1023]


def cold_us(fn, flush, warm=5, iters=20):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warm):
        flush.zero_()
        fn()
    for i in range(iters):
        flush.zero_()
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(st, en))
    q = lambda f: ts[int(round(f * (len(ts) - 1)))]  # noqa: E731
    return q(0.5), q(0.1), q(0.9)


def slice_rows(xq, xsf, m):
    rows = xq.shape[0]
    rows_pad = (rows + 127) // 128 * 128
    lin = torch.ops.trtllm.block_scale_interleave_reverse(xsf.view(torch.uint8).reshape(rows_pad, -1))
    sf = torch.ops.trtllm.block_scale_interleave(lin[:m].contiguous())
    return xq[:m].contiguous(), sf.reshape(-1)


def main():
    ck = sys.argv[1]
    cap = sys.argv[2] if len(sys.argv) > 2 else None
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    dims = bench.read_dims(ck)
    n, k = dims["down"]
    log(f"[gpu] {torch.cuda.get_device_name()} state={bench.gpu_state()}")
    log(f"[env] TLLM_AUTOTUNER_CACHE_PATH={os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')} "
        f"TRTLLM_CACHE_DIR={os.environ.get('TRTLLM_CACHE_DIR')}")
    t = bench.tco()
    log(f"[policy] dense_shapes={getattr(t, '_NVFP4_DENSE_SHAPES', None)}")
    wq, wsf, wgs = bench.make_weight(n, k, 1234)
    weights = {"synthetic": (wq, wsf, None)}
    reals = []
    if cap and os.path.exists(os.path.join(cap, "manifest.json")):
        with open(os.path.join(cap, "manifest.json")) as f:
            man = json.load(f)
        for e in man["entries"]:
            if e["shape"] == "down" and e["M"] == 1000:
                reals.append(e)
        log(f"[capture] {cap} commit={man['commit']} down M=1000 entries={len(reals)}")
        for e in reals:
            p = os.path.join(cap, e["file"])
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            log(f"[capture] {e['file']} L{e['layer']} sha_ok={h == man['files_sha256'][e['file']]}")
    else:
        log("[capture] none")
    bench.warm_gpu(8.0)
    # tune main like the engine warmup
    from tensorrt_llm._torch.autotuner import autotune
    t0 = time.time()
    with autotune():
        xq, xsf, xgs = bench.make_act(8192, k, 7)
        bench.gemm(xq, xsf, wq, wsf, (1.0 / (xgs * wgs)).float())
    torch.cuda.synchronize()
    log(f"[tune] {time.time() - t0:.1f}s")
    fp4 = torch.classes.trtllm.FP4GemmRunner(torch.bfloat16, int(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4))
    ncfg = fp4.get_num_configs()
    log(f"[cutlass] num_configs={ncfg}")
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    res = {"timing": {}, "bitwise": {}, "picks": {}}
    # timing + synthetic equality
    for m in M_LIST:
        res["picks"][m] = bench.picks_for("down", m, n, k)
        for variant, seed, heavy in (("seed1", 1, False), ("seed2", 2, False), ("heavy", 3, True)):
            xq, xsf, xgs = bench.make_act(m, k, seed * 7919 + m, heavy)
            alpha = (1.0 / (xgs * wgs)).float()
            ref = bench.gemm(xq, xsf, wq, wsf, alpha)
            row = {}
            for c in range(ncfg):
                fn = lambda c=c: fp4.run_gemm(xq, wq, xsf, wsf, alpha, 0, c, None, None)  # noqa: E731
                o = fn()
                torch.cuda.synchronize()
                row[f"cutlass{c}"] = bench.int16_mismatch(o, ref)[0]
                if variant == "seed1":
                    med, p10, p90 = cold_us(fn, flush)
                    res["timing"].setdefault(m, {})[f"cutlass{c}"] = [round(med, 2), round(p10, 2), round(p90, 2)]
            if variant == "seed1":
                med, p10, p90 = cold_us(lambda: bench.gemm(xq, xsf, wq, wsf, alpha), flush)
                res["timing"][m]["main_op"] = [round(med, 2), round(p10, 2), round(p90, 2)]
            res["bitwise"][f"{variant}|M{m}"] = row
        log(f"[M{m}] picks={res['picks'][m]} time={res['timing'][m]} "
            f"bitwise_vs_main={[res['bitwise'][f'{v}|M{m}'] for v in ('seed1', 'seed2', 'heavy')]}")
    # real captures
    for e in reals:
        tt = torch.load(os.path.join(cap, e["file"]))
        w = man["weights"][f"L{e['layer']}_down"]
        wt = torch.load(os.path.join(cap, w["file"]))
        rwq, rwsf = wt["weight"].cuda(), wt["weight_scale"].cuda()
        xq_full, xsf_full = tt["act_fp4"].cuda(), tt["act_sf"].cuda()
        alpha = tt["alpha"].cuda().float().reshape(-1)[:1]
        inmodel = tt.get("out")
        for m in [513, 576, 600, 640, 704, 768, 832, 896, 900, 960, 1000]:
            if m == 1000:
                xq, xsf = xq_full, xsf_full
            else:
                xq, xsf = slice_rows(xq_full, xsf_full, m)
            ref = bench.gemm(xq, xsf, rwq, rwsf, alpha)
            row = {}
            for c in range(ncfg):
                o = fp4.run_gemm(xq, rwq, xsf, rwsf, alpha, 0, c, None, None)
                row[f"cutlass{c}"] = bench.int16_mismatch(o, ref)[0]
            if m == 1000 and inmodel is not None:
                row["main_vs_inmodel"] = bench.int16_mismatch(ref, inmodel.cuda())[0]
            res["bitwise"][f"real_L{e['layer']}|M{m}"] = row
            log(f"[real L{e['layer']} M{m}] {row}")
    ok = all(v == 0 for r in res["bitwise"].values() for v in r.values())
    res["all_cutlass_int16_equal_main"] = ok
    log(f"[gpu] end {bench.gpu_state()}")
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"passed": ok, "timing": res["timing"], "picks": res["picks"]}))


if __name__ == "__main__":
    main()
