import os, sys, json, hashlib, time
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.attention import attention as attn_mod
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
# Gated-attention qkv_proj (N,K)=(14336,5120) and o_proj (5120,6144): FP8 per-tensor,
# static input scale, BF16 in/out, as in Qwen3.8-27B (16 full-attention layers).
# Arm flags come from this tree's attention._attn_nvfp4_decode_requested (absent on main
# -> FP8 path). Cold L2: each CUDA graph replays L=16 distinct layers back to back
# (FP8 qkv 16 x 73.4 MB = 1.17 GB, o 16 x 31.5 MB = 0.50 GB; NVFP4 copies 0.66 / 0.28 GB).
# Graphs: Q = 16 qkv calls, O = 16 o calls, LAYER = 16 x (qkv, o). PDL on.
# NOTE: brace-free on purpose so it can be passed verbatim via python -c to both trees.
L, WARM, ITERS = 16, 10, 60
HID, QKV, OIN = 5120, 14336, 6144
MS = [1, 2, 4, 8]
GT = [17, 32, 2048]
XS_Q, XS_O = 0.02, 0.01
dev = torch.device("cuda")
def log(*a):
    print(*a, flush=True)
fn = getattr(attn_mod, "_attn_nvfp4_decode_requested", None)
FQ = bool(fn("TRTLLM_ATTN_QKV_NVFP4_DECODE")) if fn is not None else False
FO = bool(fn("TRTLLM_ATTN_O_NVFP4_DECODE")) if fn is not None else False
ARM = "cand" if (FQ or FO) else "main"
log("arm", ARM, "qkv flag", FQ, "o flag", FO, "gpu", torch.cuda.get_device_name())
def make(seed, n, k, xs, flag):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(k, n, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = flag
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(xs, device=dev))])
    lin.post_load_weights()
    return lin, w8, s
t0 = time.time()
qs, os_ = [], []
for i in range(L):
    lq, wq, sq = make(2000 + i, QKV, HID, XS_Q, FQ)
    lo, wo, so = make(3000 + i, HID, OIN, XS_O, FO)
    qs.append(lq)
    os_.append(lo)
    if i == 0:
        ref_q, ref_o = (wq, sq), (wo, so)
    else:
        del wq, wo
    torch.cuda.synchronize()
act_q = getattr(qs[0], "_nvfp4_decode_weight", None) is not None
act_o = getattr(os_[0], "_nvfp4_decode_weight", None) is not None
log("built %d layers in %.1fs; qkv copy %s o copy %s; mem GB %.2f" % (
    L, time.time() - t0, act_q, act_o, torch.cuda.memory_allocated() / 1e9))
def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
def err(o, r):
    d = (o.float() - r).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                rel_rms=((o.float() - r).norm() / r.norm()).item(),
                finite=bool(torch.isfinite(o.float()).all().item()))
def ref32(x, r):
    return x.float() @ (r[0].float() * r[1]).t()
def inputs(m, seed):
    g = torch.Generator(device=dev).manual_seed(seed + m)
    x = torch.randn(m, HID, generator=g, device=dev).to(torch.bfloat16)
    y = (torch.randn(m, OIN, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    return x, y
def ops(kind, x, y):
    if kind == "Q":
        return [q(x) for q in qs]
    if kind == "O":
        return [o(y) for o in os_]
    out = []
    for q, o in zip(qs, os_):
        out.append(q(x))
        out.append(o(y))
    return out
KINDS = ["Q", "O", "LAYER"]
res = dict()
errs = dict()
bitwise = dict()
dec_sha = dict()
for m in MS:
    x, y = inputs(m, 11)
    graphs, gouts, eager = dict(), dict(), dict()
    for kind in KINDS:
        eager[kind] = [t.clone() for t in ops(kind, x, y)]
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                ops(kind, x, y)
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o = ops(kind, x, y)
        graphs[kind] = gr
        gouts[kind] = o
        for _ in range(WARM):
            gr.replay()
    torch.cuda.synchronize()
    evs = dict((k, []) for k in KINDS)
    for it in range(ITERS):
        for kind in KINDS:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[kind].replay()
            e1.record()
            evs[kind].append((e0, e1))
    torch.cuda.synchronize()
    row = dict()
    for kind in KINDS:
        t = sorted(e0.elapsed_time(e1) / L for e0, e1 in evs[kind])
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        row[kind] = dict(median_ms=med, p10_ms=p10, p90_ms=p90, noise_floor=(p90 - p10) / med)
        ok = all(torch.equal(a, b) for a, b in zip(gouts[kind], eager[kind]))
        bitwise["%s_M%d" % (kind, m)] = ok
        log("M=%d %s %s per-layer median %.2f us p10 %.2f p90 %.2f replay==eager %s" % (
            m, ARM, kind, med * 1e3, p10 * 1e3, p90 * 1e3, ok))
    res[str(m)] = row
    errs["qkv_M%d" % m] = err(eager["Q"][0], ref32(x, ref_q))
    errs["o_M%d" % m] = err(eager["O"][0], ref32(y, ref_o))
    dec_sha["qkv_M%d" % m] = sha(eager["Q"][0])
    dec_sha["o_M%d" % m] = sha(eager["O"][0])
    log("M=%d err qkv %s o %s" % (m, json.dumps(errs["qkv_M%d" % m]), json.dumps(errs["o_M%d" % m])))
    del graphs, gouts, eager
gt_sha = dict()
for m in GT:
    x, y = inputs(m, 23)
    with torch.no_grad():
        gt_sha["qkv_M%d" % m] = sha(qs[0](x))
        gt_sha["o_M%d" % m] = sha(os_[0](y))
log("rows>16 sha", json.dumps(gt_sha))
batch = dict()
for m in MS:
    lay = res[str(m)]["LAYER"]
    batch[str(m)] = dict(kernel_ms=lay["median_ms"], noise_floor=lay["noise_floor"],
                         p90_ms=lay["p90_ms"], qkv_ms=res[str(m)]["Q"]["median_ms"],
                         qkv_noise=res[str(m)]["Q"]["noise_floor"],
                         o_ms=res[str(m)]["O"]["median_ms"], o_noise=res[str(m)]["O"]["noise_floor"])
out = dict(batch=batch)
out["output"] = dict(arm=ARM, qkv_copy=act_q, o_copy=act_o, kernel_ms_is="per-layer qkv+o chain",
                     per_graph=res, error_vs_fp32_fp8_weight_bf16_act_reference=errs,
                     replay_bitwise_eager=bitwise, all_replays_bitwise=all(bitwise.values()),
                     decode_sha=dec_sha, rows_gt16_sha=gt_sha, layers=L, warmup=WARM, iters=ITERS)
out["output"]["class"] = "precision" if (act_q or act_o) else "exact"
print(json.dumps(out))
