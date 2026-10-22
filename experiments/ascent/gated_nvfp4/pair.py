# Paired chain bench for quant-linear idea 11 (branch 23): producer kernels emit
# NVFP4 for the FP8 linears' W4A4 prefill copies at rows > 16.
#   GDN:  Qwen3NextGatedDeltaNet._postprocess_gdn_output (gated norm + out_proj)
#   attn: QKNormRoPEAttention.apply_output_gate + o_proj
# Both (N,K) = (5120,6144), real Linear(FP8) modules with the W4A4 copy built by
# fp8_prefill_nvfp4.maybe_build_prefill_nvfp4_copy. Runs unchanged in the base
# tree (main's path) and the branch tree (fused path).
# Layer pattern [G,G,G,A] x 6 = 24 distinct layers (18 GDN + 6 attn), ~425 MB of
# NVFP4 weights rotated per pass, so every GEMM sees a cold L2.
# M=32: one CUDA graph per pass (PDL on). M=2048/8192: eager, as serving prefill.
# Every attention chain restores its input from a pristine copy first (main's
# gate is in place), identically in both trees.
# Brace-free on purpose: passed verbatim via python -c to the base tree.
import os, sys, json, hashlib
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mamba import gdn_mixer, layernorm_gated
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm as RMSNormGated
from tensorrt_llm._torch.attention.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn
from tensorrt_llm._torch.autotuner import autotune, AutoTuner
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

CAND = hasattr(layernorm_gated, "rms_norm_gated_token_major_nvfp4")
HEADS, HD, HID, QKVZ = 48, 128, 5120, 16384
VD = HEADS * HD
CONV = QKVZ - VD
AH, AD = 24, 256
EPS = 1e-6
WARM, ITERS = 10, 60
MS = [int(v) for v in os.environ.get("BENCH_MS", "32,33,2048,8192").split(",")]
PATTERN = ["G", "G", "G", "A"] * 6
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


def make_lin(seed, env, in_scale):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(HID, VD, generator=g, device=dev) * 0.02
    w = w * (torch.rand(HID, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(VD, HID, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(in_scale, device=dev))])
    lin.post_load_weights()
    assert fpn.maybe_build_prefill_nvfp4_copy(lin, env)
    return lin


class G:
    def __init__(self, seed):
        g = torch.Generator(device=dev).manual_seed(seed + 77)
        self.out_proj = make_lin(seed, fpn.ENV_GDN_OUT, 0.04)
        self.norm = RMSNormGated(HD, eps=EPS, group_size=None, norm_before_gate=True,
                                 device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            self.norm.weight.copy_((1.0 + 0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16))
        self.norm.fp8_scale = self.out_proj.quant_method.get_static_input_scale(self.out_proj).detach()
        self.head_v_dim = HD
        self.value_dim_per_tp = VD
        self.seed = seed

    def inputs(self, m):
        g = torch.Generator(device=dev).manual_seed(self.seed * 131 + m)
        a = (torch.randn(m, HEADS, HD, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        a[::7, 5] = 0.0
        qkvz = (torch.randn(m, QKVZ, generator=g, device=dev) * 1.5).to(torch.bfloat16)
        z = qkvz[:, CONV:].view(m, HEADS, HD)
        return (a, qkvz, z)

    def run(self, inp):
        a, qkvz, z = inp
        return gdn_mixer.Qwen3NextGatedDeltaNet._postprocess_gdn_output(self, a, z)


class A:
    def __init__(self, seed):
        self.o_proj = make_lin(seed, fpn.ENV_ATTN_O, 0.01)
        self._fuse_qk_norm_rope_gate = True
        self.seed = seed

    def inputs(self, m):
        g = torch.Generator(device=dev).manual_seed(self.seed * 131 + m)
        pristine = (torch.randn(m, VD, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        pristine[::5, 256:512] = 0.0
        work = torch.empty_like(pristine)
        qg = (torch.randn(m, 2 * VD, generator=g, device=dev) * 2.0).to(torch.bfloat16)
        gate = qg[:, VD:].view(m, AH, AD)
        return (pristine, work, qg, gate)

    def run(self, inp):
        pristine, work, qg, gate = inp
        work.copy_(pristine)
        out = QKNormRoPEAttention.apply_output_gate(self, work, gate)
        return self.o_proj(out)


layers = []
for i, k in enumerate(PATTERN):
    layers.append(G(1000 + i) if k == "G" else A(2000 + i))
log("cand", CAND, "layers", len(layers), "mem GB %.2f" % (torch.cuda.memory_allocated() / 1e9))

CONFIGS = [("mixed", list(range(len(layers)))),
           ("gdn", [i for i, k in enumerate(PATTERN) if k == "G"]),
           ("attn", [i for i, k in enumerate(PATTERN) if k == "A"])]


def kernel_counts(fn):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cnt = dict()
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        n = e.name
        key = "other"
        if "quantize_with_block_size" in n:
            key = "quantize_with_block_size"
        elif "rms_norm_gated_nvfp4" in n:
            key = "rms_norm_gated_nvfp4"
        elif "rms_norm_gated" in n:
            key = "rms_norm_gated_bf16"
        elif "sigmoid_mul_nvfp4" in n:
            key = "sigmoid_mul_nvfp4"
        elif "sigmoid_mul" in n:
            key = "sigmoid_mul_bf16"
        elif "gemm" in n.lower() or "cutlass" in n.lower() or "nvjet" in n.lower() or "sm120" in n.lower():
            key = "gemm"
        elif "copy" in n.lower() or "elementwise" in n.lower():
            key = "copy"
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def tuner_entries():
    out = []
    for key, val in AutoTuner.get().profiling_cache.cache.items():
        ks = str(key)
        if "nvfp4" in ks.lower() or "fp4" in ks.lower():
            out.append("%s -> runner %s tactic %s" % (ks, str(val[0]), str(val[1])))
    return sorted(out)


res = dict()
detail = dict()
hashes = dict()
for m in MS:
    inps = [layers[i].inputs(m) for i in range(len(layers))]
    # Tune nvfp4_gemm at this M exactly as both trees see it.
    with autotune():
        for i in range(len(layers)):
            layers[i].run(inps[i])
    torch.cuda.synchronize()
    eager_out = [layers[i].run(inps[i]).clone() for i in range(len(layers))]
    torch.cuda.synchronize()
    hm = hashlib.sha256()
    for o in eager_out:
        hm.update(o.contiguous().view(torch.int16).cpu().numpy().tobytes())
    hashes[str(m)] = hm.hexdigest()
    detail[str(m)] = dict()
    for cname, idx in CONFIGS:
        if m <= 32:
            s0 = torch.cuda.Stream()
            s0.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s0):
                for _ in range(3):
                    for i in idx:
                        layers[i].run(inps[i])
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                gouts = [layers[i].run(inps[i]) for i in idx]
            step = gr.replay
        else:
            gouts = None

            def step(idx=idx):
                for i in idx:
                    layers[i].run(inps[i])
        for _ in range(WARM):
            step()
        torch.cuda.synchronize()
        evs = []
        for it in range(ITERS):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            step()
            e1.record()
            evs.append((e0, e1))
        torch.cuda.synchronize()
        t = sorted(e0.elapsed_time(e1) for e0, e1 in evs)
        med, p10, p90 = t[len(t) // 2], t[len(t) // 10], t[(9 * len(t)) // 10]
        same = None
        if gouts is not None:
            same = all(torch.equal(gouts[j], eager_out[idx[j]]) for j in range(len(idx)))
        cnt = kernel_counts(step)
        detail[str(m)][cname] = dict(pass_ms=med, p10_ms=p10, p90_ms=p90, noise=(p90 - p10) / med,
                                     per_layer_us=med * 1e3 / len(idx), graph_equals_eager=same,
                                     kernels_per_pass=cnt)
        log("M=%d %s cand=%s pass median %.4f ms p10 %.4f p90 %.4f per-layer %.2f us graph==eager %s kernels %s" % (
            m, cname, CAND, med, p10, p90, med * 1e3 / len(idx), same, json.dumps(cnt)))
        if cname == "mixed":
            # 64-layer step equivalent: 48 GDN + 16 attention chains.
            res[str(m)] = dict(kernel_ms=med * 64.0 / len(idx), noise_floor=(p90 - p10) / med,
                               p10_ms=p10 * 64.0 / len(idx), p90_ms=p90 * 64.0 / len(idx), samples=ITERS)
        if m <= 32:
            del gr
    log("M=%d output sha256 (24 layers, eager) %s" % (m, hashes[str(m)]))
    del inps, eager_out
    torch.cuda.empty_cache()

tun = tuner_entries()
th = hashlib.sha256("\n".join(tun).encode()).hexdigest()
for line in tun:
    log("tuner", line)
out = dict(batch=res)
out["output"] = dict(cand=CAND, output_sha256=hashes, detail=detail, tuner_nvfp4_entries=len(tun),
                     tuner_sha256=th, warmup=WARM, iters=ITERS,
                     kernel_ms_definition="mixed 18 GDN + 6 attn chain pass scaled x64/24")
out["output"]["class"] = "exact"
print(json.dumps(out))
