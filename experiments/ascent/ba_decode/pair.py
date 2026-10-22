import json, os, statistics, subprocess, sys, time
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
# Idea 3 pair window (brace-free; run inline in both trees). 48 GDN layers in one CUDA graph, decode order:
# gemma add+RMSNorm (PDL) -> production Qwen3NextGatedDeltaNet._compute_tokenwise_inputs under
# with_multi_stream(True) (in_proj_qkvz real FP8 Linear with its NVFP4 decode/W4A4 copies || in_proj_ba BF16
# Linear on the aux stream, the tree's own ba routing) -> causal_conv1d_update -> fused_sigmoid_gating_delta_rule_update.
# Arms in one process, alternating: noop (no ba, constant a/b) and tree. State reset + 512 MiB read flush before
# every replay, outside the timed events. kernel_ms = tree arm 48-layer median; exposure = tree - noop per layer.
for _a in sys.argv[1:]:
    print(_a, flush=True)
T0 = time.time()
def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
try:
    log("git_head", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
except Exception as e:
    log("git unavailable", repr(e))
import torch
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as FPN
from tensorrt_llm._torch.modules import small_n_gemv as SNG
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.utils import EventType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
dev = torch.device("cuda")
HID, QN, CONV, NBA, H, HV, DK, DV = 5120, 16384, 10240, 96, 16, 48, 128, 128
L, C = 48, 8
MS = [int(s) for s in os.environ.get("W_MS", "8,16,32,12,5").split(",")]
WARM, ITERS = 10, int(os.environ.get("W_ITERS", "100"))
log("gpu", torch.cuda.get_device_name(dev), "route table", SNG._MAX_M_BY_N, "MS", MS, "iters", ITERS)
def build_qkvz(i):
    g = torch.Generator(device=dev).manual_seed(1000 + i)
    w = torch.randn(QN, HID, generator=g, device=dev) * 0.02
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    q = Linear(HID, QN, bias=False, dtype=torch.bfloat16, tensor_parallel_mode=TensorParallelMode.COLUMN,
               quant_config=QuantConfig(quant_algo=QuantAlgo.FP8), reduce_output=False).to(dev)
    q._nvfp4_decode_copy = True
    q.load_weights([dict(weight=w8, weight_scale=s.reshape(()), input_scale=torch.tensor(8.0 / 448.0, device=dev))])
    q.post_load_weights()
    FPN.maybe_build_prefill_nvfp4_copy(q)
    return q
def build_ba(i):
    g = torch.Generator(device=dev).manual_seed(5000 + i)
    b = Linear(HID, NBA, bias=False, dtype=torch.bfloat16, tensor_parallel_mode=TensorParallelMode.COLUMN,
               reduce_output=False).to(dev)
    b.load_weights([dict(weight=(torch.randn(NBA, HID, generator=g, device=dev) * 0.02).to(torch.bfloat16))])
    b.post_load_weights()
    return b
QS = [build_qkvz(i) for i in range(C)]
BAS = [build_ba(i) for i in range(L)]
gen = torch.Generator(device=dev).manual_seed(77)
CW = [(torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16) for _ in range(L)]
A_LOG = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
DT_BIAS = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
WN = (0.1 * torch.randn(HID, generator=gen, device=dev)).to(torch.bfloat16)
FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device=dev)
AUX, SIDE = torch.cuda.Stream(), torch.cuda.Stream()
class FakeGDN:
    _compute_tokenwise_inputs = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs
    def __init__(self, q, b):
        self.in_proj_qkvz, self.in_proj_ba, self.aux_stream = q, b, AUX
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()), (EventType.Attention, torch.cuda.Event())])
        self.conv_dim_per_tp, self.num_v_heads_per_tp, self.head_v_dim = CONV, HV, DV
FAKES = [FakeGDN(QS[i % C], BAS[i]) for i in range(L)]
def make_fn(arm, m, xs, rs, cps, sps, idx, cu, abc):
    def fn():
        outs = []
        for li in range(L):
            x = xs[li]
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, rs[li], WN, 1e-6)
            if arm == "tree":
                with with_multi_stream(True):
                    mixed, z, a, b = FAKES[li]._compute_tokenwise_inputs(x)
            else:
                mixed = QS[li % C](x)[:, :CONV]
                a, b = abc[:, HV:], abc[:, :HV]
            mixed = causal_conv1d_update(mixed, cps[li], CW[li], None, "silu", conv_state_indices=idx)
            o = fused_sigmoid_gating_delta_rule_update(
                A_log=A_LOG, dt_bias=DT_BIAS, q=mixed[:, :H * DK].view(1, m, H, DK),
                k=mixed[:, H * DK:2 * H * DK].view(1, m, H, DK), v=mixed[:, 2 * H * DK:].view(1, m, HV, DV),
                a=a, b=b, initial_state_source=sps[li], initial_state_indices=idx, cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, output=None)
            outs.append(o)
        return outs
    return fn
def stats(v):
    s = sorted(v)
    n = len(s)
    return dict(med=statistics.median(s), p10=s[int(0.1 * n)], p90=s[int(0.9 * n) - 1], n=n)
def kernels_by_stream(gr, reset):
    from torch.profiler import ProfilerActivity, profile
    reset()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gr.replay()
        torch.cuda.synchronize()
    per = dict()
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        nm = e.name[:70]
        per[nm] = per.get(nm, 0) + 1
    return sorted(per.items(), key=lambda kv: -kv[1])[:12]
batch, detail = dict(), dict()
for m in MS:
    idx = torch.randperm(m, generator=torch.Generator().manual_seed(m)).to(torch.int32).to(dev)
    cu = torch.arange(m + 1, dtype=torch.long, device=dev)
    per = 2 * m * HID
    st0 = (torch.randn(L * per, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    cp0 = [torch.randn(m, CONV, 3, generator=gen, device=dev).to(torch.bfloat16) for _ in range(L)]
    sp0 = [(torch.randn(m, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16) for _ in range(L)]
    abc = torch.randn(m, NBA, generator=gen, device=dev).to(torch.bfloat16)
    with autotune():
        for i in range(C):
            QS[i](st0[:m * HID].view(m, HID).clone())
    torch.cuda.synchronize()
    graphs, resets, outs = dict(), dict(), dict()
    for arm in ("noop", "tree"):
        st = st0.clone()
        cps = [t.clone() for t in cp0]
        sps = [t.clone() for t in sp0]
        xs = [st[i * per: i * per + m * HID].view(m, HID) for i in range(L)]
        rs = [st[i * per + m * HID: (i + 1) * per].view(m, HID) for i in range(L)]
        fn = make_fn(arm, m, xs, rs, cps, sps, idx, cu, abc)
        def reset(st=st, cps=cps, sps=sps):
            st.copy_(st0)
            for a_, b_ in zip(cps, cp0):
                a_.copy_(b_)
            for a_, b_ in zip(sps, sp0):
                a_.copy_(b_)
        SIDE.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(SIDE):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(SIDE)
        torch.cuda.synchronize()
        reset()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn()
        torch.cuda.synchronize()
        graphs[arm], resets[arm], outs[arm] = gr, reset, out
    times = dict(noop=[], tree=[])
    for it in range(WARM + ITERS):
        for arm in (("noop", "tree") if it % 2 == 0 else ("tree", "noop")):
            resets[arm]()
            FLUSH.sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm].replay()
            e1.record()
            if it >= WARM:
                times[arm].append((e0, e1))
        if it % 40 == 39:
            torch.cuda.synchronize()
            log("M=%d iteration %d" % (m, it + 1))
    torch.cuda.synchronize()
    r = dict((arm, stats([a.elapsed_time(b) * 1e3 / L for a, b in times[arm]])) for arm in times)
    r["exposure_us_per_layer"] = r["tree"]["med"] - r["noop"]["med"]
    if m in (8, 16, 32):
        try:
            r["tree_kernels"] = kernels_by_stream(graphs["tree"], resets["tree"])
        except Exception as e:
            r["tree_kernels"] = repr(e)[:200]
    detail[str(m)] = r
    t = r["tree"]
    batch[str(m)] = dict(kernel_ms=t["med"] * L / 1e3, noise_floor=(t["p90"] - t["p10"]) / t["med"])
    log("M=%d tree med %.2f p10 %.2f p90 %.2f us/layer | noop med %.2f | ba exposure %+.3f us/layer" % (
        m, t["med"], t["p10"], t["p90"], r["noop"]["med"], r["exposure_us_per_layer"]))
    if "tree_kernels" in r:
        log("M=%d tree graph kernels:" % m, r["tree_kernels"])
    del graphs, resets, outs
    torch.cuda.empty_cache()
print(json.dumps(dict(batch=batch, output=dict([("class", os.environ.get("W_CLASS", "reordered"))]),
                      per_layer_us=detail, route_table=str(SNG._MAX_M_BY_N), layers=L)))
