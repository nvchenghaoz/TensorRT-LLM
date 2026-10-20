# Idea 198 / branch 118 paired chain bench (BRACE-FREE: run_gpu_pair passes this text inline as
# python -c, because the base tree has no copy of this file).
# 48 GDN decode layers, each the REAL Qwen3NextGatedDeltaNet.forward (called on a stand-in that
# carries exactly the attributes it reads; out_proj/gated norm stubbed out), preceded by the gemma
# fused add+RMSNorm (PDL producer) as in the model:
#   norm -> in_proj_qkvz (FP8 Linear, NVFP4 W4A16 GEMV decode copy) || in_proj_ba (aux stream)
#   -> join -> [causal_conv1d_update] -> T1 delta rule
# Real Linear modules built through load_weights/post_load_weights, distinct cold weights per
# layer, CUDA graphs with multi-stream on, PDL on. Every replay: restore x/r, conv pools and ssm
# pools from pristine copies, then a 256 MiB L2 read flush, all outside the timed events.
# Arms (rotated order): tree, tree2 (second capture = A/A floor), and on trees that have the
# kill switch: off (gdn_mixer._GDN_GEMV_CONV = False).
# batch[M].kernel_ms = the tree arm's 48-layer chain median (ms); M = 4 is a control (Marlin).
import json
import os
import subprocess
import sys
import time
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from torch import nn  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G  # noqa: E402
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream  # noqa: E402
from tensorrt_llm._torch.utils import EventType  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


for word in sys.argv[1:]:
    print(word, flush=True)
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=30).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=30).stdout.strip().splitlines()
    print("TREE_COMMIT=" + (head or "unknown") + " dirty_files=" + str(len(dirty)), flush=True)
except Exception as exc:  # noqa: BLE001
    print("TREE_COMMIT=unknown " + repr(exc), flush=True)


def smi(tag):
    try:
        q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        log("nvidia-smi %s: %s" % (tag, r))
    except Exception as e:  # noqa: BLE001
        log("nvidia-smi unavailable: %r" % (e,))


dev = torch.device("cuda")
HID, QKVZ, CONV, NBA = 5120, 16384, 10240, 96
H, HV, DK, DV = 16, 48, 128, 128
L = int(os.environ.get("B118_LAYERS", "48"))
SLOTS = 5
WARM = int(os.environ.get("B118_WARM", "10"))
ITERS = int(os.environ.get("B118_ITERS", "100"))
ROUNDS = int(os.environ.get("B118_ROUNDS", "3"))
MS = [int(v) for v in os.environ.get("B118_MS", "1,2,3,4").split(",")]
EPS = 1e-6
HAS = hasattr(G, "_GDN_GEMV_CONV")
log("arm=%s layers=%d MS=%s warm=%d iters=%d rounds=%d gpu=%s" % (
    "candidate" if HAS else "baseline", L, MS, WARM, ITERS, ROUNDS, torch.cuda.get_device_name()))
smi("start")


class FakeGDN(G.Qwen3NextGatedDeltaNet):
    """Carries exactly the attributes forward / forward_core / forward_decode read."""

    def __init__(self, li, qkvz, ba, conv_w, a_log, dt_bias, aux):
        nn.Module.__init__(self)
        self.in_proj_qkvz = qkvz
        self.in_proj_ba = ba
        self.conv1d = types.SimpleNamespace(weight=conv_w, bias=None)
        self.A_log = a_log
        self.dt_bias = dt_bias
        self.aux_stream = aux
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()),
                                (EventType.Attention, torch.cuda.Event())])
        self.layer_idx = li
        self.layer_idx_str = str(li)
        self.register_to_config = False
        self.activation = "silu"
        self.attn_tp_size = 1
        self.num_k_heads, self.num_v_heads = H, HV
        self.num_k_heads_per_tp, self.num_v_heads_per_tp = H, HV
        self.head_k_dim, self.head_v_dim = DK, DV
        self.key_dim = self.key_dim_per_tp = H * DK
        self.value_dim = self.value_dim_per_tp = HV * DV
        self.conv_dim = self.conv_dim_per_tp = CONV

    def _postprocess_gdn_output(self, attn_out, z, all_reduce_params=None):
        return attn_out


class FakeKVM:
    def __init__(self, conv, ssm):
        self.conv = conv
        self.ssm = ssm

    def mamba_layer_cache(self, layer_idx):
        return types.SimpleNamespace(conv=self.conv[layer_idx], temporal=self.ssm[layer_idx])

    def is_speculative(self):
        return False


def make_qkvz(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(QKVZ, HID, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(QKVZ, 1, generator=gen, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(HID, QKVZ, bias=False, dtype=torch.bfloat16,
                 tensor_parallel_mode=TensorParallelMode.COLUMN,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8), reduce_output=False).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(0.04, device=dev))])
    lin.post_load_weights()
    return lin


def make_ba(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    wb = (torch.randn(2 * HV, HID, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    ba = Linear(HID, 2 * HV, bias=False, dtype=torch.bfloat16,
                tensor_parallel_mode=TensorParallelMode.COLUMN, reduce_output=False).to(dev)
    ba.load_weights([dict(weight=wb)])
    ba.post_load_weights()
    return ba


AUX = torch.cuda.Stream()
SIDE = torch.cuda.Stream()
gen = torch.Generator(device=dev).manual_seed(118)
LAYERS, CONV0, SSM0 = [], [], []
for li in range(L):
    cw = (torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
    dtb = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
    LAYERS.append(FakeGDN(li, make_qkvz(1000 + li), make_ba(2000 + li), cw, a_log, dtb, AUX))
    CONV0.append(torch.randn(SLOTS, CONV, 3, generator=gen, device=dev).to(torch.bfloat16))
    SSM0.append((torch.randn(SLOTS, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16))
CONVS = [t.clone() for t in CONV0]
SSMS = [t.clone() for t in SSM0]
KVM = FakeKVM(CONVS, SSMS)
WN = (0.1 * torch.randn(HID, generator=gen, device=dev)).to(torch.bfloat16)
FLUSH = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
q0 = LAYERS[0].in_proj_qkvz
log("built %d layers: qkvz method %s, GEMV copy %s (rows <= %s), ba method %s; GPU mem %.2f GB" % (
    L, type(q0.quant_method).__name__, q0.__dict__.get("_nvfp4_decode_gemv") is not None,
    q0.__dict__.get("_nvfp4_decode_gemv_max_rows"), type(LAYERS[0].in_proj_ba.quant_method).__name__,
    torch.cuda.memory_allocated() / 1e9))


def metadata(m):
    idx = torch.tensor([3, 0, 4, 1][:m], dtype=torch.int32, device=dev)
    amd = types.SimpleNamespace(num_contexts=0, seq_lens=torch.ones(m, dtype=torch.int32),
                                num_ctx_tokens=0, num_tokens=m, kv_cache_manager=KVM)
    mmd = types.SimpleNamespace(has_initial_states=torch.ones(m, dtype=torch.bool, device=dev),
                                state_indices=idx,
                                query_start_loc=torch.arange(m + 1, dtype=torch.int32, device=dev),
                                query_start_loc_long=torch.arange(m + 1, dtype=torch.long, device=dev),
                                replay_work_items=None, replay_n_writes=None)
    return amd, mmd


def chain(xs, rs, amd, mmd):
    outs = []
    for li in range(L):
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(xs[li], rs[li], WN, EPS)
        outs.append(LAYERS[li].forward(xs[li], amd, mmd))
    return outs


def reset(st, st0):
    st.copy_(st0)
    for a, b in zip(CONVS, CONV0):
        a.copy_(b)
    for a, b in zip(SSMS, SSM0):
        a.copy_(b)


def capture(fn, st, st0, fused):
    if HAS:
        G._GDN_GEMV_CONV = fused
    with with_multi_stream(True):
        SIDE.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(SIDE):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(SIDE)
        torch.cuda.synchronize()
        reset(st, st0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn()
    torch.cuda.synchronize()
    if HAS:
        G._GDN_GEMV_CONV = True
    return gr, out


def snap(outs):
    return [o.clone() for o in outs] + [t.clone() for t in CONVS] + [t.clone() for t in SSMS]


def same(a, b):
    return all(torch.equal(u.view(torch.int16), v.view(torch.int16)) for u, v in zip(a, b))


def stats(v):
    t = sorted(v)
    n = len(t)
    return dict(med=t[n // 2], p10=t[n // 10], p90=t[(9 * n) // 10])


ARMS = ["tree", "tree2"] + (["off"] if HAS else [])
batch = dict()
results = dict()
for m in MS:
    amd, mmd = metadata(m)
    per = 2 * m * HID
    st0 = (torch.randn(L * per, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    st = st0.clone()
    xs = [st[i * per: i * per + m * HID].view(m, HID) for i in range(L)]
    rs = [st[i * per + m * HID: (i + 1) * per].view(m, HID) for i in range(L)]
    graphs = dict()
    outs = dict()
    for arm in ARMS:
        graphs[arm], outs[arm] = capture(lambda: chain(xs, rs, amd, mmd), st, st0, arm != "off")
    snaps = dict()
    for arm in ARMS:
        reset(st, st0)
        graphs[arm].replay()
        torch.cuda.synchronize()
        snaps[arm] = snap(outs[arm])
    bit = dict((arm, same(snaps[arm], snaps["tree"])) for arm in ARMS if arm != "tree")
    del snaps
    rounds = []
    for rnd in range(ROUNDS):
        times = dict((arm, []) for arm in ARMS)
        for it in range(WARM + ITERS):
            sh = it % len(ARMS)
            order = ARMS[sh:] + ARMS[:sh]
            if (it // len(ARMS)) % 2:
                order = order[::-1]
            for arm in order:
                reset(st, st0)
                FLUSH.sum()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm].replay()
                e1.record()
                if it >= WARM:
                    times[arm].append((e0, e1))
        torch.cuda.synchronize()
        r = dict((arm, stats([e0.elapsed_time(e1) for e0, e1 in times[arm]])) for arm in ARMS)
        rounds.append(r)
        line = "M=%d round %d chain ms: tree %.4f [%.4f-%.4f] tree2 %.4f" % (
            m, rnd, r["tree"]["med"], r["tree"]["p10"], r["tree"]["p90"], r["tree2"]["med"])
        if HAS:
            line += " off %.4f (tree-off %+.2f us/layer)" % (
                r["off"]["med"], (r["tree"]["med"] - r["off"]["med"]) * 1e3 / L)
        log(line + " | per layer tree %.3f us" % (r["tree"]["med"] * 1e3 / L))
    t = [x for rr in rounds for x in [rr["tree"]["med"]]]
    last = rounds[-1]["tree"]
    allt = []
    for rr in rounds:
        allt.append(rr["tree"])
    med = sorted(t)[len(t) // 2]
    batch[str(m)] = dict(kernel_ms=med, noise_floor=max((x["p90"] - x["p10"]) / x["med"] for x in allt))
    results[str(m)] = dict(rounds=rounds, bitwise_vs_tree=bit,
                           aa_floor_ms=max(abs(rr["tree"]["med"] - rr["tree2"]["med"]) for rr in rounds))
    log("M=%d bitwise vs tree arm: %s; kernel_ms %.4f noise %.4f" % (
        m, json.dumps(bit), batch[str(m)]["kernel_ms"], batch[str(m)]["noise_floor"]))
    del graphs, outs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

smi("end")
print(json.dumps(dict(batch=batch, output=dict([("class", "exact")]), results=results,
                      arm="candidate" if HAS else "baseline", layers=L)))
