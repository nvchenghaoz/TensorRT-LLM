# Paired chain bench for gdn-fusion idea 69 (branch 27): the rope-gate kernel
# stops copying the output gate; the sigmoid-gate kernels read it as a strided
# view of the qkv projection. Runs unchanged in the base tree (copy path) and
# the branch tree (view path) through the public wrapper.
# Per pass: L2 flush, rope x16 layers (phase A), L2 flush, sigmoid_mul_nvfp4 x16
# (phase B) -- the flush stands in for the FMHA's L2 eviction between them.
# T<=128: each phase is one CUDA graph (PDL on). T>128: eager, as prefill.
# kernel_ms = median over iterations of (phase A + phase B) for 16 layers.
# Brace-free on purpose: passed verbatim via python -c to the base tree.
import os, sys, json, hashlib
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as K

CAND = hasattr(K, "qkv_output_gate_view")
NQ, NKV, HD, ROT = 24, 4, 256, 64
MSEC = (11, 11, 10)
QS = NQ * HD
W = 2 * QS + 2 * NKV * HD
L = 16
MAXPOS = 40960
WARM = int(os.environ.get("BENCH_WARM", "10"))
ITERS = int(os.environ.get("BENCH_ITERS", "60"))
TS = [int(v) for v in os.environ.get("BENCH_TS", "64,128,2111,8192").split(",")]
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


g0 = torch.Generator(device=dev).manual_seed(7)
inv = 1.0 / (1000000.0 ** (torch.arange(0, ROT, 2, device=dev).float() / ROT))
pos_f = torch.arange(MAXPOS, device=dev).float()
fr = pos_f[:, None] * inv[None, :]
cos_sin = torch.stack([fr.cos(), fr.sin()], dim=1).contiguous()
flush_buf = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)


def flush():
    flush_buf.zero_()


class Layer:
    def __init__(self, i, t):
        g = torch.Generator(device=dev).manual_seed(100 + i * 7919 + t)
        self.qkv = (torch.randn(t, W, generator=g, device=dev) * 1.5).to(torch.bfloat16)
        self.qw = (0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
        self.kw = (0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
        self.attn = (torch.randn(t, QS, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.gs = torch.tensor([6.0 / 0.02], dtype=torch.float32, device=dev)


def rope(ly, positions):
    return K.fused_qkv_gemma_rmsnorm_rope_gate(ly.qkv, ly.qw, ly.kw, cos_sin, positions, 1e-6,
                                               NQ, NKV, HD, ROT, MSEC)


def sig(ly, gate):
    return K.fused_sigmoid_mul_nvfp4(ly.attn, gate, ly.gs)


def hsh(ts):
    h = hashlib.sha256()
    for x in ts:
        h.update(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


res = dict()
detail = dict()
hashes = dict()
for t in TS:
    layers = [Layer(i, t) for i in range(L)]
    gp = torch.Generator(device=dev).manual_seed(5 + t)
    positions = torch.randint(30000, 36000, (3, t), generator=gp, device=dev, dtype=torch.int32)
    # eager outputs + hashes (identical math in both trees)
    ro = [rope(ly, positions) for ly in layers]
    so = [sig(layers[i], ro[i][1]) for i in range(L)]
    bf = []
    for i in range(L):
        w = layers[i].attn.clone()
        K.fused_sigmoid_mul_inplace(w, ro[i][1])
        bf.append(w)
    torch.cuda.synchronize()
    hashes[str(t)] = dict(qkv_out=hsh([r[0] for r in ro]), gate=hsh([r[1] for r in ro]),
                          fp4=hsh([s[0] for s in so]), sf=hsh([s[1] for s in so]), bf16=hsh(bf))
    gate_info = "stride=%s contiguous=%s" % (str(tuple(ro[0][1].stride())), ro[0][1].is_contiguous())
    log("T=%d cand=%s gate %s hashes %s" % (t, CAND, gate_info, json.dumps(hashes[str(t)])))
    gates = [r[1] for r in ro]
    graphed = t <= 128
    if graphed:
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                for i in range(L):
                    sig(layers[i], rope(layers[i], positions)[1])
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        ga = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga):
            gro = [rope(ly, positions) for ly in layers]
        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb):
            gso = [sig(layers[i], gro[i][1]) for i in range(L)]
        phase_a = ga.replay
        phase_b = gb.replay
    else:
        holder = dict()

        def phase_a():
            holder["r"] = [rope(ly, positions) for ly in layers]

        def phase_b():
            holder["s"] = [sig(layers[i], holder["r"][i][1]) for i in range(L)]
    for _ in range(WARM):
        flush()
        phase_a()
        flush()
        phase_b()
    torch.cuda.synchronize()
    evs = []
    for it in range(ITERS):
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        flush()
        e[0].record()
        phase_a()
        e[1].record()
        flush()
        e[2].record()
        phase_b()
        e[3].record()
        evs.append(e)
        if not graphed:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    ta = [e[0].elapsed_time(e[1]) for e in evs]
    tb = [e[2].elapsed_time(e[3]) for e in evs]
    tt = sorted(ta[i] + tb[i] for i in range(ITERS))
    ta = sorted(ta)
    tb = sorted(tb)

    def st(v):
        return v[len(v) // 2], v[len(v) // 10], v[(9 * len(v)) // 10]

    med, p10, p90 = st(tt)
    am, a10, a90 = st(ta)
    bm, b10, b90 = st(tb)
    same = None
    if graphed:
        same = all(torch.equal(gro[i][0], ro[i][0]) and torch.equal(gso[i][0], so[i][0])
                   and torch.equal(gso[i][1], so[i][1]) for i in range(L))
    detail[str(t)] = dict(total_ms=med, p10_ms=p10, p90_ms=p90, rope16_ms=am, rope16_p10=a10,
                          rope16_p90=a90, sig16_ms=bm, sig16_p10=b10, sig16_p90=b90,
                          graph_equals_eager=same, graphed=graphed, gate=gate_info)
    log("T=%d cand=%s total %.4f ms (p10 %.4f p90 %.4f) rope16 %.4f (%.4f-%.4f) sig16 %.4f (%.4f-%.4f) graph==eager %s" % (
        t, CAND, med, p10, p90, am, a10, a90, bm, b10, b90, same))
    res[str(t)] = dict(kernel_ms=med, noise_floor=(p90 - p10) / med, p10_ms=p10, p90_ms=p90,
                       samples=ITERS)
    if graphed:
        del ga, gb, gro, gso
    del layers, ro, so, bf, gates
    torch.cuda.empty_cache()

out = dict(batch=res)
out["output"] = dict(cand=CAND, hashes=hashes, detail=detail, warmup=WARM, iters=ITERS,
                     kernel_ms_definition="16 layers: rope-gate x16 (cold L2) + sigmoid_mul_nvfp4 x16 (cold L2)")
out["output"]["class"] = "exact"
print(json.dumps(out))
