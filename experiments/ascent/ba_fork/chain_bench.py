# GDN in_proj_qkvz + in_proj_ba 48-layer eager-prefill chain bench
# (Qwen3.8-27B-NVFP4, TP1, SM120). Branch 72 / idea 139.
# BRACE-FREE on purpose: run_gpu_pair passes this file's text inline as
#   python -c <text> BASE_COMMIT=... CAND_COMMIT=...
# because the baseline tree has no experiments/ directory.
#
# Chain = 48 distinct layers, each the real Qwen3NextGatedDeltaNet tokenwise-input
# methods called unbound on a stand-in carrying exactly the attributes they read:
#   in_proj_qkvz = real FP8 Linear(5120 -> 16384) with its load-time NVFP4 W4A4
#                  prefill copy (rows > 16: fp4_quantize + trtllm nvfp4_gemm),
#   in_proj_ba   = real BF16 Linear(5120 -> 96) (rows > 4: cuBLAS F.linear),
#   one aux stream shared by all layers (as the model does), per-layer events.
# Distinct [M, 5120] BF16 activations per layer. Every timed iteration: 256 MiB
# L2 flush (cold L2), torch.cuda._sleep, start event, 48 layers, end event.
# Arms, alternated in palindromic (ABBA) order in one process:
#   serial = gdn_mixer._PREFILL_BA_FORK_MIN_ROWS = 0 (main's serial order),
#   prod   = the tree's production dispatch (candidate: default threshold,
#            fork at rows >= 4096; baseline: main's _compute_tokenwise_inputs),
#   forced = candidate only, rows below the threshold: the fork method called
#            directly (information only; production does not fork there).
# batch[M].kernel_ms = the prod arm's 48-layer chain median (ms).
import json
import os
import statistics
import subprocess
import sys
import tempfile

for _a in sys.argv[1:] or ["TREE_COMMIT=unlabelled"]:
    print(_a, flush=True)
try:
    _h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    _d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    print("git_head=%s dirty=%d" % (_h.strip(), len(_d.splitlines())), flush=True)
except Exception as _e:  # noqa: BLE001
    print("git unavailable: %r" % (_e,), flush=True)


def log(*a):
    print(*a, flush=True)


def smi(tag):
    try:
        q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
        log("nvidia-smi %s: %s" % (tag, r))
    except Exception as e:  # noqa: BLE001
        log("nvidia-smi unavailable: %r" % (e,))


import torch  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as FPN  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G  # noqa: E402
from tensorrt_llm._torch.utils import EventType  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

HAS_FORK = hasattr(G, "_PREFILL_BA_FORK_MIN_ROWS")
DEFAULT_T = int(G._PREFILL_BA_FORK_MIN_ROWS) if HAS_FORK else 0
LAYERS = int(os.environ.get("CHAIN_LAYERS", "48"))
HID, QKVZ, CONV, NV, HD = 5120, 16384, 10240, 48, 128
MS = [int(s) for s in os.environ.get("CHAIN_MS", "2048,2049,4096,6145,8192").split(",")]
WARMUP = int(os.environ.get("CHAIN_WARMUP", "10"))
ITERS = int(os.environ.get("CHAIN_ITERS", "60"))
SLEEP_CYCLES = 1000000
IN_SCALE = 8.0 / 448.0
dev = torch.device("cuda")
log("arm=%s HAS_FORK=%s default_threshold=%d layers=%d MS=%s warmup=%d iters=%d" % (
    "candidate" if HAS_FORK else "baseline", HAS_FORK, DEFAULT_T, LAYERS, MS, WARMUP, ITERS))
log("torch %s device %s" % (torch.__version__, torch.cuda.get_device_name()))
smi("start")


class FakeGDN:
    """Carries exactly the attributes the tokenwise-input methods read."""

    _compute_tokenwise_inputs = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs

    def __init__(self, qkvz, ba, aux):
        self.in_proj_qkvz = qkvz
        self.in_proj_ba = ba
        self.aux_stream = aux
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()),
                                (EventType.Attention, torch.cuda.Event())])
        self.conv_dim_per_tp = CONV
        self.num_v_heads_per_tp = NV
        self.head_v_dim = HD


if HAS_FORK:
    FakeGDN._compute_tokenwise_inputs_routed = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs_routed
    FakeGDN._compute_tokenwise_inputs_prefill_fork = (
        G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs_prefill_fork)


def build_layer(i, aux):
    g = torch.Generator(device=dev).manual_seed(1000 + i)
    w = torch.randn(QKVZ, HID, generator=g, device=dev) * 0.02
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    qkvz = Linear(HID, QKVZ, bias=False, dtype=torch.bfloat16,
                  tensor_parallel_mode=TensorParallelMode.COLUMN,
                  quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
                  reduce_output=False).to(dev)
    qkvz.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                            input_scale=torch.tensor(IN_SCALE, device=dev))])
    qkvz.post_load_weights()
    built = FPN.maybe_build_prefill_nvfp4_copy(qkvz)
    wb = (torch.randn(2 * NV, HID, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    ba = Linear(HID, 2 * NV, bias=False, dtype=torch.bfloat16,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False).to(dev)
    ba.load_weights([dict(weight=wb)])
    ba.post_load_weights()
    return FakeGDN(qkvz, ba, aux), built


aux = torch.cuda.Stream()
layers = []
nbuilt = 0
for i in range(LAYERS):
    lay, b = build_layer(i, aux)
    layers.append(lay)
    nbuilt += int(bool(b))
torch.cuda.synchronize()
log("built %d layers, NVFP4 prefill copies %d, ba quant_method %s, qkvz quant_method %s" % (
    LAYERS, nbuilt, type(layers[0].in_proj_ba.quant_method).__name__,
    type(layers[0].in_proj_qkvz.quant_method).__name__))
assert nbuilt == LAYERS, "NVFP4 prefill copy missing"


def set_threshold(t):
    if HAS_FORK:
        G._PREFILL_BA_FORK_MIN_ROWS = t


def f_prod(lay, x):
    if HAS_FORK:
        return lay._compute_tokenwise_inputs_routed(x)
    return lay._compute_tokenwise_inputs(x)


def f_forced(lay, x):
    return lay._compute_tokenwise_inputs_prefill_fork(x)


ARMS = dict()
ARMS["serial"] = (0, f_prod)
ARMS["prod"] = (DEFAULT_T, f_prod)
if HAS_FORK:
    ARMS["forced"] = (DEFAULT_T, f_forced)


def chain(xs, arm):
    t, fn = ARMS[arm]
    set_threshold(t)
    out = None
    for li in range(LAYERS):
        out = fn(layers[li], xs[li])
    set_threshold(DEFAULT_T)
    return out


flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def one(xs, arm, ev):
    flush.zero_()
    torch.cuda._sleep(SLEEP_CYCLES)
    if ev is not None:
        ev[0].record()
    chain(xs, arm)
    if ev is not None:
        ev[1].record()


def stream_summary(xs, arm, tag):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        chain(xs, arm)
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as fh:
        tr = json.load(fh)
    os.remove(path)
    per = dict()
    for e in tr.get("traceEvents", []):
        if e.get("cat") != "kernel":
            continue
        a = e.get("args", dict())
        st = str(a.get("stream"))
        key = (e.get("name", "")[:100], str(a.get("grid")), str(a.get("block")))
        per.setdefault(st, dict())
        per[st][key] = per[st].get(key, 0) + 1
    for st in sorted(per):
        log("  [%s] stream %s: %d kernels" % (tag, st, sum(per[st].values())))
        for key, c in sorted(per[st].items(), key=lambda kv: -kv[1]):
            log("    %3d x %s grid=%s block=%s" % (c, key[0], key[1], key[2]))
    return len(per)


def stats(ts):
    s = sorted(ts)
    n = len(s)
    med = statistics.median(s)
    p10 = s[int(0.1 * n)]
    p90 = s[int(0.9 * n) - 1]
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90, noise_floor=(p90 - p10) / med,
                n=n, per_layer_us=med * 1000.0 / LAYERS)


batch = dict()
ab = dict()
streams = dict()
crit_ok = True
for M in MS:
    gx = torch.Generator(device=dev).manual_seed(7 + M)
    xs = [torch.randn(M, HID, generator=gx, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
    arms = ["serial", "prod"]
    if HAS_FORK and M < DEFAULT_T:
        arms.append("forced")
    with autotune():
        chain(xs, "serial")
    torch.cuda.synchronize()
    log("M=%d autotuned; arms %s" % (M, arms))
    for arm in arms:
        for _ in range(WARMUP):
            one(xs, arm, None)
    torch.cuda.synchronize()
    evs = dict()
    for arm in arms:
        evs[arm] = []
    order = list(arms)
    rounds = ITERS // 2
    for r in range(rounds):
        seq = order + order[::-1]
        for arm in seq:
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            one(xs, arm, ev)
            evs[arm].append(ev)
        order = order[1:] + order[:1]
    torch.cuda.synchronize()
    res = dict()
    for arm in arms:
        res[arm] = stats([e0.elapsed_time(e1) for e0, e1 in evs[arm]])
        log("M=%d %-6s median %.3f ms (%.1f us/layer) p10 %.3f p90 %.3f noise %.4f n=%d" % (
            M, arm, res[arm]["median_ms"], res[arm]["per_layer_us"], res[arm]["p10_ms"],
            res[arm]["p90_ms"], res[arm]["noise_floor"], res[arm]["n"]))
    p = res["prod"]
    batch[str(M)] = dict(kernel_ms=p["median_ms"], p10_ms=p["p10_ms"], p90_ms=p["p90_ms"],
                         noise_floor=p["noise_floor"])
    s = res["serial"]
    entry = dict(arms=res, fork_enabled=bool(HAS_FORK and M >= DEFAULT_T),
                 gain_prod_vs_serial=1.0 - p["median_ms"] / s["median_ms"])
    if entry["fork_enabled"]:
        entry["crit_gain_ge_4pct"] = entry["gain_prod_vs_serial"] >= 0.04
        entry["crit_fork_p90_lt_serial_p10"] = p["p90_ms"] < s["p10_ms"]
        crit_ok = crit_ok and entry["crit_gain_ge_4pct"] and entry["crit_fork_p90_lt_serial_p10"]
    if "forced" in res:
        entry["gain_forced_vs_serial"] = 1.0 - res["forced"]["median_ms"] / s["median_ms"]
    ab[str(M)] = entry
    log("M=%d gain prod vs serial %.4f %s" % (M, entry["gain_prod_vs_serial"],
                                               "(fork enabled)" if entry["fork_enabled"] else "(serial in both)"))
    # Per-stream kernel names and grids of one untimed chain per arm.
    streams[str(M)] = dict()
    for arm in arms:
        streams[str(M)][arm] = stream_summary(xs, arm, "M=%d %s" % (M, arm))
    del xs, evs
    torch.cuda.empty_cache()

smi("end")
summary = dict(batch=batch, output=dict([("class", "exact")]), ab=ab, streams=streams,
               arm="candidate" if HAS_FORK else "baseline", default_threshold=DEFAULT_T,
               layers=LAYERS, warmup=WARMUP, iters=ITERS, ab_criteria_met=bool(HAS_FORK and crit_ok))
print(json.dumps(summary))
