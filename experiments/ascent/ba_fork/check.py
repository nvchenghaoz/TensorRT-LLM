# Untimed correctness + dispatch check for the GDN eager-prefill in_proj_ba fork
# (branch 72 / idea 139). Same 48-layer stand-in as chain_bench.py: real
# FP8 Linear in_proj_qkvz with its NVFP4 W4A4 prefill copy, real BF16 Linear
# in_proj_ba, one shared aux stream, the real gdn_mixer methods called unbound.
#
# Per M in 2048/2049/4096/6145/8192:
#   ref     = main's serial order (gdn_mixer._PREFILL_BA_FORK_MIN_ROWS = 0),
#   forced  = _compute_tokenwise_inputs_prefill_fork called directly, every layer,
#   prod    = the production dispatch (_compute_tokenwise_inputs_routed, default
#             threshold), fork count per chain,
#   repeated execution, both arms (serial; fork = prod at M >= threshold, forced
#   below): 10 warmup + 50 iterations with the timing loop's reset points (L2
#   flush + sleep before each 48-layer chain); after each layer the main stream
#   compares all four outputs (mixed_qkv, z, a, b; bitwise as int16) with ref
#   without a host sync, i.e. a main-stream consumer reads the aux-stream
#   output exactly as forward_core does; inputs verified unchanged per arm.
# Dispatch: decode rows (1..16) never fork (even with threshold 1), 17 rows fork
# only with threshold <= 17, 4095 does not and 4096 does at the default; under
# with_multi_stream(True) and inside torch.cuda.graph capture the routed call
# runs main's _compute_tokenwise_inputs (graph replay bitwise equal to eager);
# TRTLLM_GDN_PREFILL_BA_FORK_MIN_ROWS=0 / unset parsed in a fresh process.
# Profiler: per-stream kernel lists of the production chain at every M.
import json
import os
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


import torch  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as FPN  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G  # noqa: E402
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream  # noqa: E402
from tensorrt_llm._torch.utils import EventType  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

DEFAULT_T = int(G._PREFILL_BA_FORK_MIN_ROWS)
LAYERS = int(os.environ.get("CHAIN_LAYERS", "48"))
HID, QKVZ, CONV, NV, HD = 5120, 16384, 10240, 48, 128
MS = [int(s) for s in os.environ.get("CHAIN_MS", "2048,2049,4096,6145,8192").split(",")]
WARMUP = int(os.environ.get("CHAIN_WARMUP", "10"))
ITERS = int(os.environ.get("CHAIN_ITERS", "50"))
SLEEP_CYCLES = 1000000
IN_SCALE = 8.0 / 448.0
dev = torch.device("cuda")
log("default_threshold=%d layers=%d MS=%s warmup=%d iters=%d" % (DEFAULT_T, LAYERS, MS, WARMUP, ITERS))

CNT = dict(fork=0, serial=0)
_orig_fork = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs_prefill_fork
_orig_serial = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs


def _counted_fork(self, x):
    CNT["fork"] += 1
    return _orig_fork(self, x)


def _counted_serial(self, x):
    CNT["serial"] += 1
    return _orig_serial(self, x)


class FakeGDN:
    """Carries exactly the attributes the tokenwise-input methods read."""

    _compute_tokenwise_inputs = _counted_serial
    _compute_tokenwise_inputs_prefill_fork = _counted_fork
    _compute_tokenwise_inputs_routed = G.Qwen3NextGatedDeltaNet._compute_tokenwise_inputs_routed

    def __init__(self, qkvz, ba, aux):
        self.in_proj_qkvz = qkvz
        self.in_proj_ba = ba
        self.aux_stream = aux
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()),
                                (EventType.Attention, torch.cuda.Event())])
        self.conv_dim_per_tp = CONV
        self.num_v_heads_per_tp = NV
        self.head_v_dim = HD


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
log("built %d layers, NVFP4 prefill copies %d" % (LAYERS, nbuilt))
failures = []
if nbuilt != LAYERS:
    failures.append("nvfp4 prefill copy missing")


def routed(lay, x, t):
    G._PREFILL_BA_FORK_MIN_ROWS = t
    try:
        return lay._compute_tokenwise_inputs_routed(x)
    finally:
        G._PREFILL_BA_FORK_MIN_ROWS = DEFAULT_T


def run_arm(lay, x, arm):
    if arm == "serial":
        return routed(lay, x, 0)
    if arm == "prod":
        return routed(lay, x, DEFAULT_T)
    return lay._compute_tokenwise_inputs_prefill_fork(x)


def as_bits(t):
    return t.view(torch.int16)


def ndiff_gpu(outs, ref):
    tot = None
    for o, r in zip(outs, ref):
        d = torch.ne(as_bits(o), as_bits(r)).sum()
        tot = d if tot is None else tot + d
    return tot


def maxabs(outs, ref):
    return max(float((o.float() - r.float()).abs().max()) for o, r in zip(outs, ref))


flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)


def stream_summary(xs, arm, tag):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for li in range(LAYERS):
            run_arm(layers[li], xs[li], arm)
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


report = dict()
for M in MS:
    gx = torch.Generator(device=dev).manual_seed(7 + M)
    xs = [torch.randn(M, HID, generator=gx, device=dev).to(torch.bfloat16) for _ in range(LAYERS)]
    xs0 = [x.clone() for x in xs]
    with autotune():
        for li in range(LAYERS):
            run_arm(layers[li], xs[li], "serial")
    torch.cuda.synchronize()
    r = dict()
    # Reference: main's serial order, one call per layer.
    CNT["fork"] = 0
    ref = []
    for li in range(LAYERS):
        ref.append([t.clone() for t in run_arm(layers[li], xs[li], "serial")])
    torch.cuda.synchronize()
    r["serial_forks"] = CNT["fork"]
    finite = all(bool(torch.isfinite(t.float()).all()) for rr in ref for t in rr)
    r["ref_finite"] = finite
    # Forced fork, every layer, bitwise vs serial.
    nd = 0
    ma = 0.0
    for li in range(LAYERS):
        outs = run_arm(layers[li], xs[li], "forced")
        d = int(ndiff_gpu(outs, ref[li]))
        nd += d
        if d:
            ma = max(ma, maxabs(outs, ref[li]))
    r["forced_fork_mismatch_elems"] = nd
    r["forced_fork_max_abs"] = ma
    # Production dispatch: fork count + bitwise.
    CNT["fork"] = 0
    CNT["serial"] = 0
    nd = 0
    for li in range(LAYERS):
        outs = run_arm(layers[li], xs[li], "prod")
        nd += int(ndiff_gpu(outs, ref[li]))
    r["prod_forks"] = CNT["fork"]
    r["prod_serial_calls"] = CNT["serial"]
    r["prod_mismatch_elems"] = nd
    exp_forks = LAYERS if M >= DEFAULT_T else 0
    if r["prod_forks"] != exp_forks or r["serial_forks"] != 0:
        failures.append("M=%d dispatch: prod forks %d expected %d, serial forks %d" % (
            M, r["prod_forks"], exp_forks, r["serial_forks"]))
    if r["forced_fork_mismatch_elems"] or r["prod_mismatch_elems"] or not finite:
        failures.append("M=%d bitwise: forced %d prod %d finite %s" % (
            M, r["forced_fork_mismatch_elems"], r["prod_mismatch_elems"], finite))
    # Repeated execution, both arms, timing loop's reset points.
    fork_arm = "prod" if M >= DEFAULT_T else "forced"
    rep = dict()
    for arm in ["serial", fork_arm]:
        bad_iters = []
        it1_eq_ref = None
        for it in range(WARMUP + ITERS):
            flush.zero_()
            torch.cuda._sleep(SLEEP_CYCLES)
            tot = None
            for li in range(LAYERS):
                outs = run_arm(layers[li], xs[li], arm)
                d = ndiff_gpu(outs, ref[li])
                tot = d if tot is None else tot + d
                del outs
            v = int(tot)
            if it == 0:
                it1_eq_ref = v == 0
            if v:
                bad_iters.append((it, v))
        inputs_ok = all(torch.equal(a, b) for a, b in zip(xs, xs0))
        rep[arm] = dict(iterations=WARMUP + ITERS, mismatching_iterations=len(bad_iters),
                        first_bad=bad_iters[:5], iter1_equals_ref=it1_eq_ref,
                        inputs_unchanged=inputs_ok)
        log("M=%d repeated %-6s: %d/%d iterations mismatch (iter1==ref %s, inputs unchanged %s)" % (
            M, arm, len(bad_iters), WARMUP + ITERS, it1_eq_ref, inputs_ok))
        if bad_iters or not inputs_ok:
            failures.append("M=%d repeated %s: %d bad iterations, inputs_ok %s" % (
                M, arm, len(bad_iters), inputs_ok))
    r["repeated"] = rep
    r["streams_prod"] = stream_summary(xs, "prod", "M=%d prod" % M)
    r["streams_serial"] = stream_summary(xs, "serial", "M=%d serial(threshold 0)" % M)
    exp_streams = 2 if M >= DEFAULT_T else 1
    if r["streams_prod"] != exp_streams or r["streams_serial"] != 1:
        failures.append("M=%d streams prod %d (expected %d) serial %d" % (
            M, r["streams_prod"], exp_streams, r["streams_serial"]))
    log("M=%d %s" % (M, json.dumps(r)))
    report[str(M)] = r
    del xs, xs0, ref
    torch.cuda.empty_cache()

# Dispatch checks on layer 0.
disp = dict()
lay = layers[0]


def forks_for(rows, t, ctx=None):
    x = torch.randn(rows, HID, device=dev).to(torch.bfloat16)
    CNT["fork"] = 0
    CNT["serial"] = 0
    if ctx is None:
        routed(lay, x, t)
    else:
        with ctx:
            routed(lay, x, t)
    torch.cuda.synchronize()
    return CNT["fork"], CNT["serial"]


cases = []
for rows in [1, 2, 3, 4, 8, 16]:
    cases.append(("decode rows=%d default" % rows, rows, DEFAULT_T, 0))
    cases.append(("decode rows=%d threshold 1" % rows, rows, 1, 0))
cases.append(("rows=17 threshold 1", 17, 1, 1))
cases.append(("rows=17 threshold 0", 17, 0, 0))
cases.append(("rows=2048 default", 2048, DEFAULT_T, 0))
cases.append(("rows=2049 default", 2049, DEFAULT_T, 0))
cases.append(("rows=4095 default", 4095, DEFAULT_T, 0))
cases.append(("rows=4096 default", 4096, DEFAULT_T, 1))
cases.append(("rows=6145 default", 6145, DEFAULT_T, 1))
cases.append(("rows=6145 threshold 0", 6145, 0, 0))
for name, rows, t, exp in cases:
    f, s = forks_for(rows, t)
    disp[name] = dict(forks=f, serial=s, expected_forks=exp)
    if f != exp or f + s != 1:
        failures.append("dispatch %s: forks %d serial %d expected forks %d" % (name, f, s, exp))
f, s = forks_for(6145, DEFAULT_T, with_multi_stream(True))
disp["rows=6145 under with_multi_stream(True)"] = dict(forks=f, serial=s, expected_forks=0)
if f != 0 or s != 1:
    failures.append("dispatch multi_stream: forks %d serial %d" % (f, s))

# CUDA-graph capture: routed call must take main's path; replay == eager serial.
for rows in [8, 4096]:
    name = "graph capture rows=%d threshold 1" % rows
    try:
        x = torch.randn(rows, HID, device=dev).to(torch.bfloat16)
        eager = [t.clone() for t in routed(lay, x, 0)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                routed(lay, x, 0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        CNT["fork"] = 0
        CNT["serial"] = 0
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = routed(lay, x, 1)
        cf, cs = CNT["fork"], CNT["serial"]
        g.replay()
        torch.cuda.synchronize()
        d = int(ndiff_gpu(outs, eager))
        disp[name] = dict(forks=cf, serial=cs, expected_forks=0, replay_mismatch_elems=d)
        if cf != 0 or cs != 1 or d != 0:
            failures.append("%s: forks %d serial %d mismatch %d" % (name, cf, cs, d))
        del g, outs
    except Exception as e:  # noqa: BLE001
        disp[name] = dict(error=repr(e)[:300])
        failures.append("%s raised %r" % (name, e))

# Env parsing in fresh processes.
code = ("from tensorrt_llm._torch.modules.mamba import gdn_mixer as G; "
        "print('THRESHOLD=%d' % G._PREFILL_BA_FORK_MIN_ROWS)")
for val in [None, "0"]:
    env = dict(os.environ)
    env.pop("TRTLLM_GDN_PREFILL_BA_FORK_MIN_ROWS", None)
    if val is not None:
        env["TRTLLM_GDN_PREFILL_BA_FORK_MIN_ROWS"] = val
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("THRESHOLD=")]
    got = lines[-1] if lines else "missing rc=%d %s" % (out.returncode, out.stderr[-300:])
    want = "THRESHOLD=%s" % ("4096" if val is None else val)
    disp["env %s" % ("unset" if val is None else "=" + val)] = got
    if got != want:
        failures.append("env %s: %s" % (val, got))

for k, v in disp.items():
    log("dispatch %s: %s" % (k, v))
for fl in failures:
    log("FAIL " + fl)
print(json.dumps(dict(passed=not failures, failures=failures, compared_against=(
    "main's serial _compute_tokenwise_inputs output (threshold 0) on the same inputs, bitwise as int16"),
    per_m=report, dispatch=disp, default_threshold=DEFAULT_T, layers=LAYERS,
    warmup=WARMUP, iters=ITERS)))
