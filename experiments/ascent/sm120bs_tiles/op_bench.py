"""Op-level pair bench / check for the SM120 FP8 block-scale tile heuristic (idea 235, branch 204).

Ops as the model calls them on SM120 (DeepSeek-V4.1-Flash TP4 rank shapes, pitfall 1430):
  gate_up : torch.ops.trtllm.fp8_block_scaling_gemm  [M,5120] x [2304,5120]
  down    : torch.ops.trtllm.fp8_block_scaling_gemm  [M,1152] x [5120,1152]
  wo_a    : torch.ops.trtllm.fp8_block_scaling_bmm_out 2 x [M,4096] x [1024,4096], out = o_lora[M,2,1024].transpose(0,1)
A from fp8_utils.per_token_quant_and_transform (as linear.py / mla.py do), B fp8 with packed UE8M0 int32 scales.

timing (default; both pair arms): per case a CUDA graph of NCOPIES calls over rotating weight copies (> 300 MB, cold
  L2), 5 warmup + ITERS timed replays, median us/call; the whole case list is run TWICE (A/A repeat) and noise_floor
  is max((p90-p10)/median of each pass, |pass0-pass1|/median). Also prints sha256 of every case's output for fixed
  seeds so the arms can be compared bitwise, and a build receipt (library path, mtime, symbol counts).
check (--check): untimed, same 5 + ITERS replays; every call writes its own output, NaN-poisoned before every replay,
  compared bitwise after EVERY replay with the unmodified production tile (<32,128,4> for M<=64, <64,128,4> else)
  compiled by harness.cu from the same headers; plus the M=128 shared chain (quant -> gate_up -> silu*up -> quant ->
  down) CUDA-graph replay vs eager, bitwise every replay.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SHAPES = {
    "gate_up": dict(L=1, N=2304, K=5120),
    "down": dict(L=1, N=5120, K=1152),
    "wo_a": dict(L=2, N=1024, K=4096),
}
WARMUP = 5
ITERS = int(os.environ.get("ITERS", "30"))
TIMING_CASES = [(s, m) for s in SHAPES for m in (128, 256, 512, 1024, 2048)]
CHECK_MS = [int(x) for x in os.environ.get("CMS", "1,2,4,8,16,64,96,128,200,256,512,1024,2048").split(",")]


def log(*a):
    print(*a, flush=True)


def receipt():
    import tensorrt_llm

    libdir = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs")
    pats = ["fused_pmap_gemm_fma_mtile_v3", "SM120BlockScaledBuilderILi96ELi128ELi2E",
            "SM120BlockScaledBuilderILi128ELi128ELi2E", "SM120BlockScaledBuilderILi32ELi64ELi4E",
            "SM120BlockScaledBuilderILi64ELi128ELi4E"]
    out = {}
    for name in ("libtensorrt_llm.so", "libth_common.so"):
        p = os.path.join(libdir, name)
        if not os.path.exists(p):
            out[name] = "missing"
            continue
        st = os.stat(p)
        counts = {k: 0 for k in pats}
        proc = subprocess.Popen(["strings", "-a", p], stdout=subprocess.PIPE, text=True, errors="replace")
        for line in proc.stdout:
            for k in pats:
                if k in line:
                    counts[k] += 1
        proc.wait()
        out[name] = {"path": os.path.realpath(p), "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z",
                                                                       time.localtime(st.st_mtime)),
                     "size": st.st_size, "strings_counts": counts}
    log("BUILD RECEIPT", json.dumps(out))
    return out


def make_case(shape, M, ncopies, seed):
    from tensorrt_llm.quantization.utils import fp8_utils

    sh = SHAPES[shape]
    L, N, K = sh["L"], sh["N"], sh["K"]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    sk = (K + 511) // 512
    sn = (N + 3) // 4 * 4
    if L == 1:
        x = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
        a, sa = fp8_utils.per_token_quant_and_transform(x)
    else:
        x = torch.randn(M, L, K, device="cuda", generator=g).to(torch.bfloat16)
        a, sa = fp8_utils.per_token_quant_and_transform(x, need_permute102=True)
    bs, sbs = [], []
    for _ in range(ncopies):
        b = (torch.randn((N, K) if L == 1 else (L, N, K), device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        sb = torch.randint(118, 134, (L, sk, sn, 4), device="cuda", dtype=torch.uint8, generator=g)
        sb = sb.view(torch.int32).reshape(L, sk, sn).contiguous()
        bs.append(b)
        sbs.append(sb[0] if L == 1 else sb)
    return a, sa, bs, sbs


def call(shape, a, sa, b, sb, out=None):
    """Returns the output base tensor (for wo_a: the [M,2,1024] o_lora buffer)."""
    if shape == "wo_a":
        torch.ops.trtllm.fp8_block_scaling_bmm_out(a, b, sa, sb, out.transpose(0, 1))
        return out
    return torch.ops.trtllm.fp8_block_scaling_gemm(a, b, sa, sb)


def new_out(shape, M):
    if shape == "wo_a":
        return torch.empty(M, 2, 1024, device="cuda", dtype=torch.bfloat16)
    return None


def ncopies_for(shape):
    sh = SHAPES[shape]
    return max(4, -(-300 * 2**20 // (sh["L"] * sh["N"] * sh["K"])))


def time_case(shape, M):
    nc = ncopies_for(shape)
    a, sa, bs, sbs = make_case(shape, M, nc, 1000 + M)
    out = new_out(shape, M)
    o = call(shape, a, sa, bs[0], sbs[0], out)
    torch.cuda.synchronize()
    h = hashlib.sha256(o.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(nc):
            call(shape, a, sa, bs[i], sbs[i], out)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(nc):
            call(shape, a, sa, bs[i], sbs[i], out)
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
        ts.append(e0.elapsed_time(e1) / nc)
    ts.sort()
    med = ts[len(ts) // 2]
    spread = (ts[int(0.9 * (len(ts) - 1))] - ts[int(0.1 * (len(ts) - 1))]) / med
    del gr
    return med, spread, h


def timing():
    import tensorrt_llm  # noqa: F401

    rec = receipt()
    passes = []
    hashes = {}
    for p in range(2):
        res = {}
        for shape, M in TIMING_CASES:
            med, spread, h = time_case(shape, M)
            key = f"{shape}_M{M}"
            res[key] = (med, spread)
            if p == 0:
                hashes[key] = h
            elif hashes[key] != h:
                hashes[key] = "UNSTABLE"
            log(f"pass{p} {key}: {med * 1000:.2f} us/call spread {spread:.4f} hash {h}")
            torch.cuda.empty_cache()
        passes.append(res)
    batch = {}
    for key in passes[0]:
        m0, s0 = passes[0][key]
        m1, s1 = passes[1][key]
        med = (m0 + m1) / 2
        batch[key] = {"kernel_ms": round(med, 5), "noise_floor": round(max(s0, s1, abs(m0 - m1) / med), 4),
                      "pass_ms": [round(m0, 5), round(m1, 5)]}

    def agg(name, keys):
        ms = sum(batch[k]["kernel_ms"] for k in keys)
        nf = max(batch[k]["noise_floor"] for k in keys)
        p = [round(sum(batch[k]["pass_ms"][i] for k in keys), 5) for i in range(2)]
        batch[name] = {"kernel_ms": round(ms, 5), "noise_floor": nf, "pass_ms": p, "of": keys}

    agg("L0_19_layer", ["gate_up_M1024", "down_M1024", "wo_a_M2048"])
    agg("L20_39_layer", ["gate_up_M128", "down_M128", "wo_a_M128"])
    agg("L20_39_shared_gate_up_down", ["gate_up_M128", "down_M128"])
    for k in ("L0_19_layer", "L20_39_layer", "L20_39_shared_gate_up_down"):
        log(k, batch[k])
    log("hashes", json.dumps(hashes))
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "hashes": hashes, "receipt": rec}), flush=True)


def check():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm.quantization.utils import fp8_utils

    import sweep

    rec = receipt()
    mod = sweep.build("A")
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    allok = True
    detail = {}
    NC = 4
    for shape in SHAPES:
        L, N, K = SHAPES[shape]["L"], SHAPES[shape]["N"], SHAPES[shape]["K"]
        for M in CHECK_MS:
            a, sa, bs, sbs = make_case(shape, M, NC, 7000 + M)
            tm = 32 if M <= 64 else 64
            refs = []
            for i in range(NC):  # unmodified production tile on the same inputs
                if shape == "wo_a":
                    rb = torch.full((M, L, N), float("nan"), device="cuda", dtype=torch.bfloat16)
                    mod.run(tm, 128, 4, a, bs[i], sa, sbs[i], rb.transpose(0, 1), nsm)
                else:
                    rb = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
                    mod.run(tm, 128, 4, a.view(1, M, K), bs[i].view(1, N, K), sa, sbs[i], rb, nsm)
                    rb = rb[0]
                refs.append(rb)
            torch.cuda.synchronize()
            finite = all(bool(torch.isfinite(r).all()) for r in refs)
            outs_in = [new_out(shape, M) for _ in range(NC)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(NC):
                    call(shape, a, sa, bs[i], sbs[i], outs_in[i])
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                outs = [call(shape, a, sa, bs[i], sbs[i], outs_in[i]) for i in range(NC)]
            nbad, maxdiff = 0, 0.0
            for it in range(WARMUP + ITERS):
                for o in outs:  # reset point: poison every output before each replay
                    o.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                for i in range(NC):
                    if not torch.equal(outs[i].view(torch.int16), refs[i].view(torch.int16)):
                        nbad += 1
                        maxdiff = max(maxdiff, (outs[i].float() - refs[i].float()).abs().nan_to_num(1e30).max().item())
            ok = nbad == 0 and finite
            allok &= ok
            key = f"{shape}_M{M}"
            detail[key] = {"bitwise_vs_unmodified_prod_tile": ok, "ref_tile": f"{tm}x128x4", "replays": WARMUP + ITERS,
                           "calls_per_replay": NC, "mismatching_calls": nbad, "max_abs_diff": maxdiff,
                           "ref_finite": finite}
            log(key, detail[key])
            del gr, outs
            torch.cuda.empty_cache()
    # ---- M=128 shared chain: CUDA-graph replay vs eager ----
    for M in (128, 1024):
        g = torch.Generator(device="cuda")
        g.manual_seed(99 + M)
        x = torch.randn(M, 5120, device="cuda", generator=g).to(torch.bfloat16)
        w1 = (torch.randn(2304, 5120, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        s1 = torch.randint(118, 130, (10, 2304, 4), device="cuda", dtype=torch.uint8,
                           generator=g).view(torch.int32).reshape(10, 2304).contiguous()
        w2 = (torch.randn(5120, 1152, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        s2 = torch.randint(118, 130, (3, 5120, 4), device="cuda", dtype=torch.uint8,
                           generator=g).view(torch.int32).reshape(3, 5120).contiguous()

        def chain():
            a1, sa1 = fp8_utils.per_token_quant_and_transform(x)
            h = torch.ops.trtllm.fp8_block_scaling_gemm(a1, w1, sa1, s1)
            gte, up = h.chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gte.float()) * up.float()).to(torch.bfloat16)
            a2, sa2 = fp8_utils.per_token_quant_and_transform(act.contiguous())
            return torch.ops.trtllm.fp8_block_scaling_gemm(a2, w2, sa2, s2)

        eager = chain().clone()
        e2 = chain().clone()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            chain()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gout = chain()
        nbad = 0
        for it in range(WARMUP + ITERS):
            gout.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            if not torch.equal(gout.view(torch.int16), eager.view(torch.int16)):
                nbad += 1
        ok = nbad == 0 and torch.equal(e2.view(torch.int16), eager.view(torch.int16)) and bool(
            torch.isfinite(eager).all())
        allok &= ok
        detail[f"shared_chain_M{M}_graph_vs_eager"] = {"bitwise": ok, "replays": WARMUP + ITERS, "bad": nbad,
                                                       "hash": hashlib.sha256(eager.view(torch.uint8).cpu().numpy()
                                                                              .tobytes()).hexdigest()[:16]}
        log(f"shared_chain_M{M}", detail[f"shared_chain_M{M}_graph_vs_eager"])
        del gr
    print(json.dumps({"passed": bool(allok), "compared": "fp8_block_scaling_gemm / fp8_block_scaling_bmm_out of the "
                      "built library vs the unmodified production SM120 <TileM,128,4> kernel (bitwise, every call of "
                      "every replay, NaN-poisoned outputs); M=128/1024 shared chain CUDA-graph replay vs eager",
                      "detail": detail, "receipt": rec}), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--both" in sys.argv:  # check first (untimed), then timing; the timing JSON is the last stdout line
        check()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        timing()
    else:
        timing()
