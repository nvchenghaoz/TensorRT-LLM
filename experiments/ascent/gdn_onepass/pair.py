# GDN prefill front end in-chain pair + digest check (Qwen3.8-27B GDN, TP1, SM120).
# BRACE-FREE on purpose: run_gpu_pair passes this text inline as python -c <text> <mode>,
# because the base tree has no experiments/ascent/gdn_onepass directory.
# Each layer of the chain = L2 flush (128 MiB write) -> producer GEMM (FP8 per-tensor
# cublas_scaled_mm, in_proj_qkvz shape [T, 5120] x [5120, 16384] -> bf16 [T, 16384], so the
# activations are L2-warm as in serving) -> Qwen3NextGatedDeltaNet.forward_extend (the tree's
# own front end: main = transpose + conv + fused_gdn_post_conv; branch = one pass; then the
# FlashInfer chunk consumer with indexed state I/O on the production pool layout).
# mode bench: CUDA-graph chain of 4 layers, 10 warmup + 50 timed replays x 4 rounds, state
#   pool restored from a pristine copy before every replay; per-layer chain time.
# mode check: digests (sha256) of q/k/v/g/beta (captured at the consumer call), the
#   attention output, the conv pool and the ssm pool for every case; then 10 warmup + 50
#   NaN-poisoned repeats of every case, each compared bitwise with run 1.
#   argv[2] (optional) = expected digests "case=hash;case=hash" from the base tree.
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types

T0 = time.time()
MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
EXPECT = sys.argv[2] if len(sys.argv) > 2 else ""


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


try:
    _h = subprocess.run(["git", "rev-parse", "HEAD", "HEAD^" + chr(123) + "tree" + chr(125)],
                        capture_output=True, text=True).stdout.split()
    _d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    log("TREE_COMMIT", " ".join(_h), "dirty", len(_d.splitlines()))
except Exception as _e:  # noqa: BLE001
    log("git unavailable", repr(_e))

import torch  # noqa: E402

from tensorrt_llm._torch.modules.mamba import gdn_mixer  # noqa: E402

HAS_ONEPASS = hasattr(gdn_mixer, "gdn_prefill_onepass_enabled")
log("mode", MODE, "onepass module present", HAS_ONEPASS)

dev = torch.device("cuda")
DIM, QKVZ, HID = 10240, 16384, 5120
NKH, NVH, HD = 16, 48, 128
NB = 12
SSM_B = NVH * HD * HD * 2
CONV_B = DIM * 3 * 2
SLOTS = [3, 7, 1, 10, 5, 0, 8, 2, 4, 9, 6, 11]
CAP = 8


def make_case(lens, nd, seed, fresh=(), pad=()):
    # lens: prefill lengths; nd: decode rows (1-token sequences after the prefills)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    P = len(lens)
    n = P + nd
    Tp = sum(lens)
    T = Tp + nd
    c = dict(P=P, nd=nd, n=n, Tp=Tp, T=T)
    c["hidden"] = (torch.randn(T, HID, device=dev, generator=g) * 1.0).to(torch.bfloat16)
    c["ba"] = (torch.randn(T, 2 * NVH, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    pool = torch.empty(NB, SSM_B + CONV_B, dtype=torch.uint8, device=dev)
    ssm = pool[:, :SSM_B].view(torch.bfloat16).view(NB, NVH, HD, HD)
    conv = pool[:, SSM_B:].view(torch.bfloat16).view(NB, DIM, 3)
    ssm.copy_((torch.randn(NB, NVH, HD, HD, device=dev, generator=g) * 0.05).to(torch.bfloat16))
    conv.copy_(torch.randn(NB, DIM, 3, device=dev, generator=g).to(torch.bfloat16))
    c["pool"], c["ssm"], c["conv"] = pool, ssm, conv
    c["pristine"] = pool.clone()
    seq_lens = list(lens) + [1] * nd
    qsl = [0]
    for s in seq_lens:
        qsl.append(qsl[-1] + s)
    c["qsl"] = torch.tensor(qsl, dtype=torch.int32, device=dev)
    c["qsl_long"] = c["qsl"].to(torch.int64)
    idx = list(SLOTS[:n])
    for i in pad:
        idx[i] = -1
    c["idx"] = torch.tensor(idx, dtype=torch.int32, device=dev)
    c["hi"] = torch.tensor([i not in fresh for i in range(n)], dtype=torch.bool, device=dev)
    return c


def make_layer(seed):
    g = torch.Generator(device=dev)
    g.manual_seed(1000 + seed)
    w = (torch.randn(QKVZ, HID, device=dev, generator=g) * 0.02)
    ws = (w.abs().amax() / 448.0).reshape(())
    wq = (w / ws).to(torch.float8_e4m3fn)
    del w
    cw = (torch.randn(DIM, 4, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    A_log = torch.log(torch.rand(NVH, device=dev, generator=g) * 15 + 1)
    dt_bias = torch.rand(NVH, device=dev, generator=g) * 2 - 1
    me = types.SimpleNamespace()
    me.conv1d = types.SimpleNamespace(weight=cw, bias=None)
    me.activation = "silu"
    me.A_log = A_log
    me.dt_bias = dt_bias
    me.num_k_heads_per_tp = NKH
    me.num_v_heads_per_tp = NVH
    me.head_k_dim = HD
    me.head_v_dim = HD
    me.num_k_heads = NKH
    me.num_v_heads = NVH
    me.attn_tp_size = 1
    me._prefill_transpose_conv = types.MethodType(
        gdn_mixer.Qwen3NextGatedDeltaNet._prefill_transpose_conv, me)
    return dict(me=me, wq=wq, ws=ws.float(), in_scale=torch.tensor(4.0 / 448.0, device=dev))


WS = (torch.empty(CAP, NVH, HD, HD, dtype=torch.float32, device=dev),
      torch.empty(CAP, NVH, HD, HD, dtype=torch.float32, device=dev))


def producer(c, lay):
    q8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(c["hidden"], lay["in_scale"])
    return torch.ops.trtllm.cublas_scaled_mm(q8, lay["wq"].t(), scale_a=lay["in_scale"],
                                             scale_b=lay["ws"], bias=None,
                                             out_dtype=torch.bfloat16)


def front_and_consumer(c, lay, qkvz):
    P, nd, n = c["P"], c["nd"], c["n"]
    si = c["idx"]
    gdn_mixer.reset_recurrent_state_rows(c["ssm"], si[:P], c["hi"][:P], c["conv"])
    kw = dict(
        mixed_qkv=qkvz[:, :DIM],
        a=c["ba"][:, NVH:],
        b=c["ba"][:, :NVH],
        batch_size=n,
        has_initial_states=c["hi"],
        cache_indices=si,
        query_start_loc=c["qsl"],
        query_start_loc_long=c["qsl_long"],
        num_prefill_tokens=c["Tp"],
        num_decode_tokens=nd,
        state_indices_p=si[:P],
        state_indices_d=si[P:],
        num_prefill=P,
        num_decodes=nd,
        state_workspace=WS,
    )
    return gdn_mixer.Qwen3NextGatedDeltaNet.forward_extend(lay["me"], c["conv"], c["ssm"], **kw)


def layer_fn(c, lay):
    qkvz = producer(c, lay)
    return front_and_consumer(c, lay, qkvz)


def set_path(path):
    if HAS_ONEPASS:
        gdn_mixer.gdn_prefill_onepass_enabled = (lambda: True) if path == "onepass" else (
            lambda: False)


def digest(t):
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).view(-1).cpu().numpy().tobytes()).hexdigest()[:24]


CAPT = dict()
_orig_chunk = gdn_mixer.chunk_gated_delta_rule


def _capture_chunk(*args, **kwargs):
    for nm in ("q", "k", "v", "g", "beta"):
        CAPT[nm] = kwargs[nm].detach().clone()
    return _orig_chunk(*args, **kwargs)


CASES = [
    ("p1", [1], 0, (), ()),
    ("p2", [2], 0, (), ()),
    ("p3", [3], 0, (), ()),
    ("p85", [85], 0, (), ()),
    ("p2048", [2048], 0, (), ()),
    ("p2x2048", [2048, 2048], 0, (), ()),
    ("p4096", [4096], 0, (), ()),
    ("p8192", [8192], 0, (), ()),
    ("p4x2048", [2048, 2048, 2048, 2048], 0, (), ()),
    ("mix2048d1", [2048], 1, (), ()),
    ("mix2x2048d2", [2048, 2048], 2, (), ()),
    ("mix85d3", [85], 3, (), ()),
    ("ragged", [1000, 37, 2], 0, (1,), ()),
    ("ragged_d2", [1000, 37, 2], 2, (0,), ()),
    ("fresh2050", [2050], 0, (0,), ()),
]


def run_case_once(c, lay):
    c["pool"].copy_(c["pristine"])
    CAPT.clear()
    out = layer_fn(c, lay)
    torch.cuda.synchronize()
    d = []
    for nm in ("q", "k", "v", "g", "beta"):
        d.append(digest(CAPT[nm]))
    d.append(digest(out))
    d.append(digest(c["conv"]))
    d.append(digest(c["ssm"]))
    return "-".join(d), out, CAPT


def poison():
    junk = torch.full((768 * 1024 * 1024 // 4,), float("nan"), dtype=torch.float32, device=dev)
    del junk


def run_check():
    gdn_mixer.chunk_gated_delta_rule = _capture_chunk
    set_path("onepass")
    lay = make_layer(0)
    expected = dict()
    for item in EXPECT.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            expected[k] = v
    res = dict()
    allok = True
    dig_line = []
    for name, lens, nd, fresh, pad in CASES:
        c = make_case(lens, nd, seed=len(name) * 7 + sum(lens), fresh=fresh, pad=pad)
        d1, out1, cap1 = run_case_once(c, lay)
        finite = bool(torch.isfinite(out1).all().item()) and all(
            bool(torch.isfinite(cap1[k].float()).all().item()) for k in ("q", "k", "v", "g", "beta"))
        drift = 0
        for it in range(10 + 50):
            poison()
            dk, _, _ = run_case_once(c, lay)
            if dk != d1:
                drift += 1
        ok = finite and drift == 0
        match = None
        if name in expected:
            match = expected[name] == d1
            ok = ok and match
        allok = allok and ok
        res[name] = dict(digest=d1, finite=finite, repeats=60, drift_vs_run1=drift,
                         match_base=match, ok=ok)
        dig_line.append(name + "=" + d1)
        log("check", name, "T", c["T"], "digest", d1, "finite", finite, "drift", drift,
            "match_base", match)
        del c
        torch.cuda.empty_cache()
    gdn_mixer.chunk_gated_delta_rule = _orig_chunk
    log("DIGESTS", ";".join(dig_line))
    n_exp = sum(1 for v in res.values() if v["match_base"] is not None)
    return dict(passed=bool(allok and (not expected or n_exp == len(CASES))), cases=res,
                compared_against=("sha256 of q/k/v/g/beta at the consumer, attention output, "
                                  "conv pool, ssm pool from the BASE tree's stock kernels "
                                  "(digests passed in argv)" if expected else
                                  "self (digest emission run)"),
                n_compared=n_exp, onepass=HAS_ONEPASS)


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    p10, p90 = ts[int(0.1 * (n - 1))], ts[int(0.9 * (n - 1))]
    return dict(median_us=med, p10_us=p10, p90_us=p90, noise=(p90 - p10) / med)


LAYERS = 4
flush = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device=dev)


def build_graph(c, lays):
    def body():
        outs = []
        for lay in lays:
            flush.fill_(1.0)
            outs.append(layer_fn(c, lay))
        return outs
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            c["pool"].copy_(c["pristine"])
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        c["pool"].copy_(c["pristine"])
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        return ("graph", g.replay, g)
    except Exception as e:  # noqa: BLE001
        log("graph capture failed, timing eagerly:", repr(e)[:300])
        torch.cuda.synchronize()
        return ("eager", body, None)


def time_runner(c, run, iters=50, warm=10):
    ts = []
    for i in range(warm + iters):
        c["pool"].copy_(c["pristine"])
        torch.cuda._sleep(50000)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        run()
        e1.record()
        if i >= warm:
            ts.append((e0, e1))
    torch.cuda.synchronize()
    return [a.elapsed_time(b) * 1e3 / LAYERS for a, b in ts]


def run_bench():
    lays = [make_layer(i) for i in range(LAYERS)]
    paths = ["main"] + (["onepass"] if HAS_ONEPASS else [])
    active = "onepass" if HAS_ONEPASS else "main"
    out = dict(batch=dict(), output=dict([("class", "exact")]), extra=dict(active=active))
    for n in [int(v) for v in os.environ.get("ONEPASS_NS", "1,2,4").split(",")]:
        c = make_case([2048] * n, 0, seed=11 + n)
        runners = dict()
        for p in paths:
            set_path(p)
            runners[p] = build_graph(c, lays)
            log("n=%d path %s built (%s)" % (n, p, runners[p][0]))
        set_path(active)
        acc = dict((p, []) for p in paths)
        for rnd in range(4):
            order = paths if rnd % 2 == 0 else paths[::-1]
            for p in order:
                acc[p] += time_runner(c, runners[p][1])
        res = dict()
        for p in paths:
            st = stats(acc[p])
            res[p] = st
            log("n=%d T=%d %-8s chain per layer: median %8.2f us p10 %8.2f p90 %8.2f noise %.4f (%s)"
                % (n, 2048 * n, p, st["median_us"], st["p10_us"], st["p90_us"], st["noise"],
                   runners[p][0]))
        if "onepass" in res:
            sv = res["main"]["median_us"] - res["onepass"]["median_us"]
            res["saved_us_per_layer"] = sv
            log("n=%d in-process saved %.2f us/layer (%.2f ms per 48-layer prefill)"
                % (n, sv, sv * 48 / 1e3))
        prim = res[active]
        out["batch"][str(n)] = dict(kernel_ms=prim["median_us"] / 1e3, noise_floor=prim["noise"])
        out["extra"][str(n)] = res
        out["extra"]["mode_" + str(n)] = runners[active][0]
        del runners, c
        torch.cuda.empty_cache()
    return out


if MODE == "check":
    r = run_check()
else:
    r = run_bench()
print(json.dumps(r))
