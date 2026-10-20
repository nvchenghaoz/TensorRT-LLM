# Idea 198 / branch 118 STEP 0b: variant sweep of the conv epilogue (kern.py VAR 1..4) in the
# same 48-layer fork-window chain as step0.py (norm -> qkvz GEMV || ba -> join -> [conv] -> T1).
# Arms: base/base2 (production GEMV + production conv), v<VAR> (fused), v<VAR>b (second graph of
# the best-known variant), inl (inline VAR 0 + production conv), ceil (no conv).
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402

import kern  # noqa: E402

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

dev = torch.device("cuda")
EPS = 1e-6
K = 5120
NQKVZ, CONV, NBA = 16384, 10240, 96
H, HV, DK, DV = 16, 48, 128, 128
L = int(os.environ.get("B118_LAYERS", "48"))
SLOTS = 4
PAD = -1
G = 1.0 / 256.0
WARM = int(os.environ.get("B118_WARM", "10"))
ITERS = int(os.environ.get("B118_ITERS", "60"))
ROUNDS = int(os.environ.get("B118_ROUNDS", "2"))
MS = [int(v) for v in os.environ.get("B118_MS", "1,2,3").split(",")]
VARS = [int(v) for v in os.environ.get("B118_VARS", "1,2,3,4").split(",")]
ARMS = ["base", "base2", "inl", "ceil"] + ["v%d" % v for v in VARS]

ext = kern.build_ext(log)
for m in MS:
    for v in [0] + VARS:
        a = ext.attrs(m, v)
        log("attrs M=%d VAR=%d: regs %d local %d B smem %d B, blocks/SM %d" % (m, v, a[0], a[1], a[2], a[3]))

gen = torch.Generator(device=dev).manual_seed(118)
WQ, SF, CW, WBA, CP0, SP0 = [], [], [], [], [], []
for li in range(L):
    WQ.append(torch.randint(0, 256, (NQKVZ, K // 2), generator=gen, device=dev, dtype=torch.int32)
              .to(torch.uint8))
    SF.append(torch.randint(0x28, 0x40, (NQKVZ, K // 16), generator=gen, device=dev, dtype=torch.int32)
              .to(torch.uint8))
    CW.append((torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16))
    WBA.append((torch.randn(NBA, K, generator=gen, device=dev) * 0.02).to(torch.bfloat16))
    CP0.append((torch.randn(SLOTS, CONV, 3, generator=gen, device=dev)).to(torch.bfloat16))
    SP0.append((torch.randn(SLOTS, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16))
CP = [t.clone() for t in CP0]
SP = [t.clone() for t in SP0]
A_LOG = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
DT_BIAS = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
WN = (0.1 * torch.randn(K, generator=gen, device=dev)).to(torch.bfloat16)
FLUSH = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
AUX = torch.cuda.Stream()
SIDE = torch.cuda.Stream()
torch.cuda.synchronize()
log("built %d layers; GPU memory allocated GB %.2f" % (L, torch.cuda.memory_allocated() / 1e9))


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return torch.nn.functional.linear(x, w) if y is None else y


def make_fn(arm, m, xs, rs, idx, cu):
    ev0 = [torch.cuda.Event() for _ in range(L)]
    ev1 = [torch.cuda.Event() for _ in range(L)]
    var = int(arm[1:]) if arm.startswith("v") else 0

    def fn():
        outs = []
        for li in range(L):
            x = xs[li]
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, rs[li], WN, EPS)
            ev0[li].record()
            if arm in ("base", "base2", "ceil"):
                y = torch.ops.trtllm.nvfp4_w4a16_gemv(x, WQ[li], SF[li], G)
            else:
                y = ext.run(x, WQ[li], SF[li], G, CP[li], CW[li], idx, CONV, PAD, var)
            with torch.cuda.stream(AUX):
                ev0[li].wait()
                ba = in_proj_ba(x, WBA[li])
                ev1[li].record()
            ev1[li].wait()
            mixed = y[:, :CONV]
            if arm in ("base", "base2", "inl"):
                mixed = causal_conv1d_update(mixed, CP[li], CW[li], None, "silu", conv_state_indices=idx)
            q = mixed[:, :H * DK].view(1, m, H, DK)
            k_ = mixed[:, H * DK:2 * H * DK].view(1, m, H, DK)
            v = mixed[:, 2 * H * DK:].view(1, m, HV, DV)
            o = fused_sigmoid_gating_delta_rule_update(
                A_log=A_LOG, dt_bias=DT_BIAS, q=q, k=k_, v=v, a=ba[:, HV:], b=ba[:, :HV],
                initial_state_source=SP[li], initial_state_indices=idx, cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, output=None)
            outs.append((y, o))
        return outs

    return fn


def reset(st, st0):
    st.copy_(st0)
    for a, b in zip(CP, CP0):
        a.copy_(b)
    for a, b in zip(SP, SP0):
        a.copy_(b)


def capture(fn, st, st0):
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
    return gr, out


def snapshot(outs):
    return dict(y=[o[0].clone() for o in outs], o=[o[1].clone() for o in outs],
                cp=[t.clone() for t in CP], sp=[t.clone() for t in SP])


def compare(a, b):
    return all(torch.equal(u.view(torch.int16), v.view(torch.int16))
               for key in ("y", "o", "cp", "sp") for u, v in zip(a[key], b[key]))


def stats(v):
    t = sorted(v)
    n = len(t)
    return dict(med=t[n // 2], p10=t[n // 10], p90=t[(9 * n) // 10])


results = dict()
for m in MS:
    # Row 1 of M=3 is a PAD row (production pads with -1).
    idx = torch.tensor([2, 0, 3][:m], dtype=torch.int32, device=dev)
    cu = torch.arange(m + 1, dtype=torch.long, device=dev)
    per = 2 * m * K
    st0 = (torch.randn(L * per, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    st = st0.clone()
    xs = [st[i * per: i * per + m * K].view(m, K) for i in range(L)]
    rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(L)]
    graphs = dict()
    outs = dict()
    for arm in ARMS:
        graphs[arm], outs[arm] = capture(make_fn(arm, m, xs, rs, idx, cu), st, st0)
    snaps = dict()
    for arm in ARMS:
        if arm == "ceil":
            continue
        reset(st, st0)
        graphs[arm].replay()
        torch.cuda.synchronize()
        snaps[arm] = snapshot(outs[arm])
    chk = dict((arm, compare(snaps[arm], snaps["base"])) for arm in snaps if arm != "base")
    del snaps
    log("M=%d bitwise vs base: %s" % (m, json.dumps(chk)))
    rr = []
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
        r = dict((arm, stats([e0.elapsed_time(e1) * 1e3 / L for e0, e1 in times[arm]])) for arm in ARMS)
        base = r["base"]["med"]
        floor = abs(r["base"]["med"] - r["base2"]["med"])
        line = "M=%d round %d us/layer: base %.3f base2 %.3f inl %.3f ceil %.3f floor %.3f |" % (
            m, rnd, base, r["base2"]["med"], r["inl"]["med"], r["ceil"]["med"], floor)
        for v in VARS:
            a = "v%d" % v
            line += " v%d %.3f (gain %+.3f)" % (v, r[a]["med"], base - r[a]["med"])
        log(line)
        rr.append(r)
    results[str(m)] = dict(rounds=rr, bitwise=chk)
    del graphs, outs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

batch = dict()
for m in MS:
    f = results[str(m)]["rounds"][-1]["base"]
    batch[str(m)] = dict(kernel_ms=f["med"] * L / 1e3, noise_floor=(f["p90"] - f["p10"]) / f["med"])
print(json.dumps(dict(batch=batch, output={"class": "exact"}, results=results)))
