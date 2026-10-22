# Idea 3 step 0 / acceptance window: GDN in_proj_ba decode exposure at M=8-32 (SM120).
# Window per layer, 48 layers in one CUDA graph, decode order:
#   gemma fused add+RMSNorm (PDL) -> fork: in_proj_qkvz (real FP8 Linear with its load-time
#   NVFP4 decode copies and W4A4 prefill copy; main's dispatch by rows) || in_proj_ba (aux)
#   -> causal_conv1d_update (in place on the qkvz view) -> fused_sigmoid_gating_delta_rule_update
#   (reads a/b; BF16 state, distinct per layer).
# Arms (one process, rotating ABBA order, state reset + 512 MiB READ flush before each replay,
# outside the timed event pair):
#   noop    : no ba; delta rule reads constant a/b
#   main    : production fork (maybe_execute_in_parallel order), join after qkvz, ba via Linear
#   late    : same ba kernel on aux, join moved to just before the delta-rule kernel
#   before  : ba on the main stream before qkvz
#   g_<cfg> / gl_<cfg> : Triton small_n_gemv with launch config cfg, early join / late join
#   tree    : the tree's GdnMixer._compute_tokenwise_inputs + its own join placement
#             (gdn_mixer._ba_late_join_rows if present), i.e. production code
# Exposure per layer = arm - noop.
# Brace-free; extra argv words are printed first.
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
for _a in sys.argv[1:]:
    print(_a, flush=True)
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


try:
    _h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    _d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    log("git_head=%s dirty=%d" % (_h.strip(), len(_d.splitlines())))
except Exception as _e:  # noqa: BLE001
    log("git unavailable", repr(_e))

import torch  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as FPN  # noqa: E402
from tensorrt_llm._torch.modules import small_n_gemv as SNG  # noqa: E402
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G  # noqa: E402
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update  # noqa: E402
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream  # noqa: E402
from tensorrt_llm._torch.utils import EventType  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

dev = torch.device("cuda")
EPS = 1e-6
HID, QKVZ_N, CONV, NBA = 5120, 16384, 10240, 96
H, HV, DK, DV = 16, 48, 128, 128
LAYERS = int(os.environ.get("W_LAYERS", "48"))
C = int(os.environ.get("W_COPIES", "8"))
MS = [int(s) for s in os.environ.get("W_MS", "32,24,16,12,8").split(",")]
WARM = int(os.environ.get("W_WARM", "10"))
ITERS = int(os.environ.get("W_ITERS", "60"))
IN_SCALE = 8.0 / 448.0
CFGS = [c for c in os.environ.get("W_CFGS", "").split(";") if c]
ARMS_ENV = os.environ.get("W_ARMS", "noop,main,late,before")
PROFILE_MS = [int(s) for s in os.environ.get("W_PROFILE_MS", "").split(",") if s]
log("gpu", torch.cuda.get_device_name(dev), "PDL", os.environ.get("TRTLLM_ENABLE_PDL"), "layers", LAYERS,
    "copies", C, "MS", MS, "warm", WARM, "iters", ITERS, "cfgs", CFGS, "arms", ARMS_ENV)


def build_qkvz(i):
    g = torch.Generator(device=dev).manual_seed(1000 + i)
    w = torch.randn(QKVZ_N, HID, generator=g, device=dev) * 0.02
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    q = Linear(HID, QKVZ_N, bias=False, dtype=torch.bfloat16,
               tensor_parallel_mode=TensorParallelMode.COLUMN,
               quant_config=QuantConfig(quant_algo=QuantAlgo.FP8), reduce_output=False).to(dev)
    q._nvfp4_decode_copy = True
    q.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                         input_scale=torch.tensor(IN_SCALE, device=dev))])
    q.post_load_weights()
    FPN.maybe_build_prefill_nvfp4_copy(q)
    return q


def build_ba(i):
    g = torch.Generator(device=dev).manual_seed(5000 + i)
    wb = (torch.randn(NBA, HID, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    b = Linear(HID, NBA, bias=False, dtype=torch.bfloat16,
               tensor_parallel_mode=TensorParallelMode.COLUMN, reduce_output=False).to(dev)
    b.load_weights([dict(weight=wb)])
    b.post_load_weights()
    return b


QS = [build_qkvz(i) for i in range(C)]
q0 = QS[0]
log("qkvz: decode copy", q0.__dict__.get("_nvfp4_decode_weight") is not None,
    "marlin max rows", q0.__dict__.get("_nvfp4_decode_max_rows"),
    "gemv rows", q0.__dict__.get("_nvfp4_decode_gemv_max_rows"),
    "prefill copy", FPN.get_prefill_nvfp4_copy(q0) is not None)
BAS = [build_ba(i) for i in range(LAYERS)]
log("ba quant_method", type(BAS[0].quant_method).__name__, "enable_cuda_core",
    getattr(BAS[0], "enable_cuda_core", None))
gen = torch.Generator(device=dev).manual_seed(77)
CW = [(torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16) for _ in range(LAYERS)]
A_LOG = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
DT_BIAS = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
WN = (0.1 * torch.randn(HID, generator=gen, device=dev)).to(torch.bfloat16)
FLUSH = torch.ones(128 * 1024 * 1024, dtype=torch.float32, device=dev)
AUX = torch.cuda.Stream()
SIDE = torch.cuda.Stream()
EV_MAIN = [torch.cuda.Event() for _ in range(LAYERS)]
EV_BA = [torch.cuda.Event() for _ in range(LAYERS)]
torch.cuda.synchronize()
log("built; GPU memory GB %.2f" % (torch.cuda.memory_allocated() / 1e9))


def parse_cfg(s):
    return tuple(int(v) for v in s.split(","))


class FakeGDN:
    _compute_tokenwise_inputs = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs

    def __init__(self, q, b):
        self.in_proj_qkvz = q
        self.in_proj_ba = b
        self.aux_stream = AUX
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()),
                                (EventType.Attention, torch.cuda.Event())])
        self.conv_dim_per_tp = CONV
        self.num_v_heads_per_tp = HV
        self.head_v_dim = DV


FAKES = [FakeGDN(QS[i % C], BAS[i]) for i in range(LAYERS)]
HAS_TREE_LATE = hasattr(G.Qwen3NextGatedDeltaNet, "_ba_join")


def make_fn(arm, m, xs, rs, cps, sps, idx, cu, ab_const):
    def ba_kernel(li, x):
        if arm in ("main", "late", "before"):
            return BAS[li](x)
        if arm == "empty":
            # fork/join floor: one trivial kernel on aux instead of the GEMV
            return ab_const.clone()
        cfg = parse_cfg(arm.split("_", 1)[1])
        return SNG.small_n_gemv(x, BAS[li].weight, cfg)

    def fn():
        outs = []
        for li in range(LAYERS):
            x = xs[li]
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, rs[li], WN, EPS)
            late = arm == "late" or arm.startswith("gl_")
            pending = None
            if arm == "tree":
                with with_multi_stream(True):
                    mixed, z, a, b = FAKES[li]._compute_tokenwise_inputs(x)
                y = None
            elif arm == "noop":
                y = QS[li % C](x)
                ba = ab_const
            elif arm == "before":
                ba = ba_kernel(li, x)
                y = QS[li % C](x)
            else:
                EV_MAIN[li].record()
                y = QS[li % C](x)
                with torch.cuda.stream(AUX):
                    EV_MAIN[li].wait()
                    ba = ba_kernel(li, x)
                    EV_BA[li].record()
                if late:
                    pending = EV_BA[li]
                else:
                    EV_BA[li].wait()
            if y is not None:
                mixed = y[:, :CONV]
                b = ba[:, :HV]
                a = ba[:, HV:]
            mixed = causal_conv1d_update(mixed, cps[li], CW[li], None, "silu", conv_state_indices=idx)
            if pending is not None:
                pending.wait()
            q = mixed[:, :H * DK].view(1, m, H, DK)
            k_ = mixed[:, H * DK:2 * H * DK].view(1, m, H, DK)
            v = mixed[:, 2 * H * DK:].view(1, m, HV, DV)
            o = fused_sigmoid_gating_delta_rule_update(
                A_log=A_LOG, dt_bias=DT_BIAS, q=q, k=k_, v=v, a=a, b=b,
                initial_state_source=sps[li], initial_state_indices=idx, cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, output=None)
            outs.append(o)
        return outs

    return fn


def stats(v):
    s = sorted(v)
    n = len(s)
    return dict(med=statistics.median(s), p10=s[int(0.1 * n)], p90=s[int(0.9 * n) - 1], n=n)


def profile_once(gr, reset, tag):
    from torch.profiler import ProfilerActivity, profile
    reset()
    FLUSH.sum()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gr.replay()
        torch.cuda.synchronize()
    path = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "trace_%s.json" % tag)
    prof.export_chrome_trace(path)
    with open(path) as fh:
        tr = json.load(fh)
    per = dict()
    for e in tr.get("traceEvents", []):
        if e.get("cat") != "kernel":
            continue
        a = e.get("args", dict())
        st = str(a.get("stream"))
        key = e.get("name", "")[:90] + " grid=" + str(a.get("grid"))
        d = per.setdefault(st, dict())
        c, tot = d.get(key, (0, 0.0))
        d[key] = (c + 1, tot + float(e.get("dur", 0.0)))
    for st in sorted(per):
        log("  [%s] stream %s" % (tag, st))
        for key, (c, tot) in sorted(per[st].items(), key=lambda kv: -kv[1][1]):
            log("     %3d x %8.2f us avg  %s" % (c, tot / c, key))


results = dict()
for m in MS:
    arms = [a for a in ARMS_ENV.split(",") if a]
    for c in CFGS:
        arms += ["g_" + c, "gl_" + c]
    idx = torch.randperm(m, generator=torch.Generator().manual_seed(m)).to(torch.int32).to(dev)
    cu = torch.arange(m + 1, dtype=torch.long, device=dev)
    per = 2 * m * HID
    st0 = (torch.randn(LAYERS * per, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    cp0 = [torch.randn(m, CONV, 3, generator=gen, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
    sp0 = [(torch.randn(m, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16)
           for _ in range(LAYERS)]
    ab_const = (torch.randn(m, NBA, generator=gen, device=dev)).to(torch.bfloat16)
    # autotune the W4A4 path (rows > 16) eagerly, as the engine warmup does
    with autotune():
        for i in range(C):
            QS[i](st0[:m * HID].view(m, HID).clone())
    torch.cuda.synchronize()
    graphs = dict()
    bufs = dict()
    outs = dict()
    for arm in arms:
        st = st0.clone()
        cps = [t.clone() for t in cp0]
        sps = [t.clone() for t in sp0]
        xs = [st[i * per: i * per + m * HID].view(m, HID) for i in range(LAYERS)]
        rs = [st[i * per + m * HID: (i + 1) * per].view(m, HID) for i in range(LAYERS)]
        fn = make_fn(arm, m, xs, rs, cps, sps, idx, cu, ab_const)

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
        graphs[arm] = gr
        bufs[arm] = (reset, sps)
        outs[arm] = out
    # equality after one replay from the same state
    snap = dict()
    for arm in arms:
        bufs[arm][0]()
        graphs[arm].replay()
        torch.cuda.synchronize()
        snap[arm] = (torch.stack([o.float().sum() for o in outs[arm]]).cpu(),
                     [o.clone() for o in outs[arm]])
    ref = "main" if "main" in arms else arms[0]
    eq = dict()
    maxdiff = dict()
    for arm in arms:
        eq[arm] = all(torch.equal(a_, b_) for a_, b_ in zip(snap[arm][1], snap[ref][1]))
        maxdiff[arm] = max(float((a_.float() - b_.float()).abs().max()) for a_, b_ in zip(snap[arm][1], snap[ref][1]))
    del snap
    log("M=%d outputs equal to %s:" % (m, ref), eq, "maxdiff", maxdiff)
    times = dict((arm, []) for arm in arms)
    order = list(arms)
    for it in range(WARM + ITERS):
        seq = order + order[::-1] if it % 2 == 0 else order[::-1] + order
        for arm in seq[:len(arms)]:
            bufs[arm][0]()
            FLUSH.sum()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[arm].replay()
            e1.record()
            if it >= WARM:
                times[arm].append((e0, e1))
        order = order[1:] + order[:1]
        if it % 20 == 19:
            torch.cuda.synchronize()
            log("M=%d iteration %d" % (m, it + 1))
    torch.cuda.synchronize()
    r = dict()
    for arm in arms:
        r[arm] = stats([e0.elapsed_time(e1) * 1e3 / LAYERS for e0, e1 in times[arm]])
    base = r.get("noop")
    for arm in arms:
        if base is not None:
            r[arm]["exposure_us"] = r[arm]["med"] - base["med"]
        r[arm]["equal_to_ref"] = eq[arm]
        log("M=%d %-22s med %7.2f p10 %7.2f p90 %7.2f us/layer  exposure %+6.2f  eq %s" % (
            m, arm, r[arm]["med"], r[arm]["p10"], r[arm]["p90"], r[arm].get("exposure_us", 0.0), eq[arm]))
    results[str(m)] = r
    if m in PROFILE_MS:
        for arm in arms:
            profile_once(graphs[arm], bufs[arm][0], "M%d_%s" % (m, arm))
    del graphs, bufs, outs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

summary = dict()
for m, r in results.items():
    summary[m] = dict((arm, round(v.get("exposure_us", 0.0), 3)) for arm, v in r.items())
log("EXPOSURE us/layer (arm - noop):", summary)
batch = dict()
for m, r in results.items():
    pick = "tree" if "tree" in r else ("main" if "main" in r else list(r)[0])
    d = r[pick]
    batch[m] = dict(kernel_ms=d["med"] * LAYERS / 1e3, noise_floor=(d["p90"] - d["p10"]) / d["med"])
print(json.dumps(dict(batch=batch, output=dict([("class", "exact")]), per_layer_us=results,
                      exposure=summary, layers=LAYERS)))
