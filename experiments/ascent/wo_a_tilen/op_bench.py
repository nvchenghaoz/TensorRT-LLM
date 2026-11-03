"""wo_a op microbenchmark / check: torch.ops.trtllm.fp8_block_scaling_bmm_out on SM120 at the DeepSeek-V4.1-Flash
TP4 per-rank shape (2 groups x N=1024, K=4096), A from per_token_quant_and_transform(need_permute102=True).

timing mode (default; run in both pair arms):
  CUDA graph of NCOPIES calls over rotating weight copies (cold L2: 24 x 8.4 MB > 128 MB L2), out =
  o_lora[M,2,1024].transpose(0,1) (ldd=2048 != N, strideD=1024). 5 warmup replays + ITERS timed replays.
  Also prints sha256 of the outputs (strided and contiguous D) for a fixed seed, so arms can be compared bitwise.
check mode (--check): same graph, same 5 + ITERS replays, but every call writes its own NaN-poisoned output
  (strided and contiguous runs), and after EVERY replay every output is compared bitwise with the unmodified
  kernel (<TileM,128,4> SM120BlockScaledKernel compiled by harness.cu) on the same inputs.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

L, N, K = 2, 1024, 4096
NCOPIES = 24
WARMUP = 5
ITERS = int(os.environ.get("ITERS", "30"))
MS = [int(x) for x in os.environ.get("MS", "1,2,4,8,128,256,512,2048").split(",")]


def log(*a):
    print(*a, flush=True)


def make(M, seed):
    import tensorrt_llm  # noqa: F401  (registers trtllm ops)
    from tensorrt_llm.quantization.utils import fp8_utils

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(M, L, K, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
    a_fp8, a_scale = fp8_utils.per_token_quant_and_transform(x, need_permute102=True)
    bs, sfbs = [], []
    for _ in range(NCOPIES):
        b = (torch.randn(L, N, K, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        sfb = torch.randint(118, 134, (L, K // 512, N, 4), device="cuda", dtype=torch.uint8, generator=g)
        bs.append(b)
        sfbs.append(sfb.view(torch.int32).reshape(L, K // 512, N).contiguous())
    return a_fp8, a_scale, bs, sfbs


def new_out(M, strided):
    if strided:
        base = torch.empty(M, L, N, device="cuda", dtype=torch.bfloat16)
        return base, base.transpose(0, 1)
    base = torch.empty(L, M, N, device="cuda", dtype=torch.bfloat16)
    return base, base


def op(a, sa, b, sb, out):
    torch.ops.trtllm.fp8_block_scaling_bmm_out(a, b, sa, sb, out)


def timing():
    res = {}
    hashes = {}
    for M in MS:
        a, sa, bs, sfbs = make(M, 1000 + M)
        for strided in (False, True):
            base, out = new_out(M, strided)
            base.fill_(float("nan"))
            op(a, sa, bs[0], sfbs[0], out)
            torch.cuda.synchronize()
            hashes[f"M{M}_{'strided' if strided else 'contig'}"] = hashlib.sha256(
                base.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
        base, out = new_out(M, True)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for i in range(NCOPIES):
                op(a, sa, bs[i], sfbs[i], out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for i in range(NCOPIES):
                op(a, sa, bs[i], sfbs[i], out)
        for _ in range(WARMUP):
            gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) / NCOPIES)
        ts.sort()
        med = ts[len(ts) // 2]
        noise = (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med
        res[str(M)] = {"kernel_ms": round(med, 5), "noise_floor": round(noise, 4)}
        log(f"M={M} {med * 1000:.2f} us/call noise {noise:.4f}")
        del gr
    log("hashes", json.dumps(hashes))
    print(json.dumps({"batch": res, "output": {"class": "exact"}, "hashes": hashes}), flush=True)


def check():
    import bench

    mod = bench.build()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    allok = True
    detail = {}
    for M in MS:
        a, sa, bs, sfbs = make(M, 1000 + M)
        tm = 32 if M <= 64 else 64
        for strided in (False, True):
            # reference: unmodified kernel types (<tm,128,4>) per copy
            refs = []
            for i in range(NCOPIES):
                rb, ro = new_out(M, strided)
                rb.fill_(float("nan"))
                mod.run(tm, 128, a, bs[i], sa, sfbs[i], ro, nsm)
                refs.append(rb)
            torch.cuda.synchronize()
            finite = all(bool(torch.isfinite(r).all()) for r in refs)
            outs = [new_out(M, strided) for _ in range(NCOPIES)]
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for i in range(NCOPIES):
                    op(a, sa, bs[i], sfbs[i], outs[i][1])
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for i in range(NCOPIES):
                    op(a, sa, bs[i], sfbs[i], outs[i][1])
            nbad = 0
            maxdiff = 0.0
            for it in range(WARMUP + ITERS):
                for ob, _ in outs:  # reset point: poison every output before each replay
                    ob.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                for i in range(NCOPIES):
                    if not torch.equal(outs[i][0].view(torch.int16), refs[i].view(torch.int16)):
                        nbad += 1
                        maxdiff = max(maxdiff, (outs[i][0].float() - refs[i].float()).abs().nan_to_num(1e30).max().item())
            ok = nbad == 0 and finite
            allok &= ok
            key = f"M{M}_{'strided_ldd2048' if strided else 'contig'}"
            detail[key] = {"bitwise_vs_unmodified_kernel": ok, "replays": WARMUP + ITERS, "calls_per_replay": NCOPIES,
                           "mismatching_calls": nbad, "max_abs_diff": maxdiff, "ref_finite": finite}
            log(key, detail[key])
            del gr
    print(json.dumps({"passed": bool(allok), "compared": "fp8_block_scaling_bmm_out vs unmodified SM120 <TileM,128,4> "
                      "kernel, bitwise, every call of every replay", "detail": detail}), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        timing()
