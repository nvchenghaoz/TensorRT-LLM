import os, sys, json
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm as RMSNormGated
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
# GDN gated-norm + out_proj chain (real Qwen3NextGatedDeltaNet._postprocess_gdn_output),
# (N,K)=(5120,6144), value heads 48 x 128, z a column slice of a 16384-wide qkvz.
# Cold L2: one CUDA graph replays C=15 distinct (norm, out_proj) copies back to back
# (FP8 15 x 31.5 MB = 472 MB; NVFP4 15 x 17.7 MB = 265 MB). PDL on.
# NOTE: brace-free on purpose so it can be passed verbatim via python -c to both trees.
HEADS, HD, HID, QKVZ = 48, 128, 5120, 16384
VD = HEADS * HD
CONV = QKVZ - VD
EPS, C, WARM, ITERS = 1e-6, 15, 10, 60
MS = [1, 2, 4, 8]
dev = torch.device("cuda")
BOTH = os.environ.get("BENCH_BOTH", "0") == "1"
def log(*a):
    print(*a, flush=True)
fn = getattr(gdn_mixer, "_gdn_out_proj_nvfp4_decode_requested", None)
CAND = bool(fn()) if fn is not None else False
class G:
    def __init__(self, norm, lin):
        self.norm = norm
        self.out_proj = lin
        self.head_v_dim = HD
        self.value_dim_per_tp = VD
    def __call__(self, a, z):
        return gdn_mixer.Qwen3NextGatedDeltaNet._postprocess_gdn_output(self, a, z)
def make(seed, flag):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(HID, VD, generator=g, device=dev) * 0.02
    w = w * (torch.rand(HID, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    nw = (1.0 + 0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
    lin = Linear(VD, HID, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = flag
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(0.04, device=dev))])
    lin.post_load_weights()
    norm = RMSNormGated(HD, eps=EPS, group_size=None, norm_before_gate=True,
                        device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(nw)
    sc = lin.quant_method.get_static_input_scale(lin)
    norm.fp8_scale = sc.detach() if sc is not None else None
    return G(norm, lin), (w8, s, nw)
def ref32(a, z, r):
    w8, s, nw = r
    x = a.reshape(-1, HD).float()
    zz = z.reshape(-1, HD).float()
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * nw.float()
    y = (y * zz * torch.sigmoid(zz)).view(-1, VD)
    return y @ (w8.float() * s).t()
def err(o, r):
    d = (o.float() - r).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                rel_rms=((o.float() - r).norm() / r.norm()).item())
arms = [("main", False), ("cand", True)] if BOTH else [("cand" if CAND else "main", CAND)]
mods = dict()
for name, flag in arms:
    mods[name] = []
    for i in range(C):
        mods[name].append(make(1000 + i, flag))
    act = getattr(mods[name][0][0].out_proj, "_nvfp4_decode_weight", None) is not None
    log("arm", name, "flag", flag, "nvfp4 copy active", act, "mem GB %.2f" % (torch.cuda.memory_allocated() / 1e9))
active = dict((name, getattr(mods[name][0][0].out_proj, "_nvfp4_decode_weight", None) is not None) for name, _ in arms)
res = dict((name, dict()) for name, _ in arms)
errs = dict((name, dict()) for name, _ in arms)
bitwise = dict((name, dict()) for name, _ in arms)
for m in MS:
    g = torch.Generator(device=dev).manual_seed(7 + m)
    a = (torch.randn(m, HEADS, HD, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    qkvz = (torch.randn(m, QKVZ, generator=g, device=dev) * 1.5).to(torch.bfloat16)
    z = qkvz[:, CONV:].view(m, HEADS, HD)
    graphs, outs, eager = dict(), dict(), dict()
    for name, _ in arms:
        ms_ = mods[name]
        eager[name] = ms_[0][0](a, z).clone()
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                o = [gm(a, z) for gm, _ in ms_]
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o = [gm(a, z) for gm, _ in ms_]
        graphs[name] = gr
        outs[name] = o
        for _ in range(WARM):
            gr.replay()
    torch.cuda.synchronize()
    evs = dict((name, []) for name, _ in arms)
    for it in range(ITERS):
        for name, _ in arms:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[name].replay()
            e1.record()
            evs[name].append((e0, e1))
    torch.cuda.synchronize()
    for name, _ in arms:
        t = sorted(e0.elapsed_time(e1) / C for e0, e1 in evs[name])
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        nbytes = (HID * VD // 2 + HID * VD // 16) if active[name] else HID * VD
        res[name][str(m)] = dict(kernel_ms=med, noise_floor=(p90 - p10) / med, p90_ms=p90,
                                 p10_ms=p10, tb_s_gemm_bytes=nbytes / (med * 1e-3) / 1e12, samples=ITERS)
        errs[name][str(m)] = err(outs[name][0], ref32(a, z, mods[name][0][1]))
        bitwise[name][str(m)] = bool(torch.equal(outs[name][0], eager[name]))
        log("M=%d arm=%s median %.2f us p10 %.2f p90 %.2f us chain-equivalent TB/s %.3f replay==eager %s err %s" % (
            m, name, med * 1e3, p10 * 1e3, p90 * 1e3, res[name][str(m)]["tb_s_gemm_bytes"],
            bitwise[name][str(m)], json.dumps(errs[name][str(m)])))
    if BOTH:
        sp = res["main"][str(m)]["kernel_ms"] / res["cand"][str(m)]["kernel_ms"]
        log("M=%d speedup main/cand %.3fx (%.1f%% faster)" % (m, sp, 100.0 * (1.0 - 1.0 / sp)))
    del graphs
if BOTH:
    out = dict(both=res, errors_vs_fp32_fp8_weight_reference=errs, replay_bitwise_eager=bitwise,
               nvfp4_copy_active=active, copies=C, warmup=WARM, iters=ITERS)
    print(json.dumps(out))
else:
    name = arms[0][0]
    cls = "precision" if active[name] else "exact"
    out = dict(batch=res[name])
    out["output"] = dict(nvfp4_out_proj_copy=active[name],
                         error_vs_fp32_fp8_weight_reference=errs[name],
                         replay_bitwise_eager=bitwise[name], copies=C, warmup=WARM, iters=ITERS)
    out["output"]["class"] = cls
    print(json.dumps(out))
