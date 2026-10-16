# Two checks for the Marlin persistent workspace + PDL change.
# Part 1 (timing diagnostic): the GDN decode fork. Per layer, in one CUDA graph:
#   gemma fused add RMSNorm (PDL) -> fork: main = Marlin in_proj_qkvz (16384,5120),
#   aux = BF16 in_proj_ba (96,5120) through the tree's own route (maybe_small_n_gemv on main
#   >= c3bb99c8, else cuBLAS F.linear as on base 2cc9d697) -> join ->
#   elementwise consumer of both (non-PDL, like causal_conv1d_update) -> gemma RMSNorm on Marlin out.
#   Mirrors maybe_execute_in_parallel. Arms A/W/B replayed alternately, cold L2 (6 copies).
# Part 2 (correctness): the lm_head shape N=248320, K=5120 at M=1/2/4, arms A/W/B: eager outputs
#   bitwise equal to arm A eager; graph chain RMSNorm(PDL) -> Marlin -> .float() with 10 warmup +
#   50 NaN-poisoned replays per arm interleaved with eager calls, each equal to iteration 1 and to
#   arm-A eager; every bound workspace slot all zero afterwards.
import json
import os
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

try:  # main >= c3bb99c8 routes BF16 in_proj_ba at M <= 4 to a Triton small-N GEMV (branch 10)
    from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402
except ImportError:  # older base: cuBLAS only
    def maybe_small_n_gemv(x, w):
        return None


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return F.linear(x, w) if y is None else y

dev = torch.device("cuda")
EPS = 1e-6
WARM, ITERS = 10, 50
ARMS = [("A", "0", "0"), ("W", "1", "0"), ("B", "1", "1")]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def set_arm(a):
    os.environ["TRTLLM_MARLIN_PERSISTENT_WORKSPACE"] = a[1]
    os.environ["TRTLLM_MARLIN_PDL"] = a[2]


def build(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16)


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k)


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


SIDE = torch.cuda.Stream()
AUX = torch.cuda.Stream()


def capture(fn):
    SIDE.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(SIDE):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(SIDE)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    return gr, out


# ---------------- Part 1: fork timing ----------------
N, K, C, NBA = 16384, 5120, 6, 96
base = build(N, K, 11 + N + K)
ws = [tuple(t.clone() for t in base) for _ in range(C)]
del base
g = torch.Generator(device=dev).manual_seed(5)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16) for _ in range(C)]
wn1 = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
wn2 = (0.1 * torch.randn(N, generator=g, device=dev)).to(torch.bfloat16)
fork_res = {}
for m in (1, 2, 4):
    per = 2 * m * K + m * N
    st0 = torch.zeros(C * per, device=dev, dtype=torch.bfloat16)
    for i in range(C):
        st0[i * per: i * per + 2 * m * K] = (torch.randn(2 * m * K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    graphs, keep = {}, []
    for a in ARMS:
        set_arm(a)
        st = st0.clone()
        xs = [st[i * per: i * per + m * K].view(m, K) for i in range(C)]
        r1s = [st[i * per + m * K: i * per + 2 * m * K].view(m, K) for i in range(C)]
        r2s = [st[i * per + 2 * m * K: (i + 1) * per].view(m, N) for i in range(C)]
        ev0 = [torch.cuda.Event() for _ in range(C)]
        ev1 = [torch.cuda.Event() for _ in range(C)]

        def fn(st=st, xs=xs, r1s=r1s, r2s=r2s, ev0=ev0, ev1=ev1):
            st.copy_(st0)
            outs = []
            for i in range(C):
                norm(xs[i], r1s[i], wn1)
                ev0[i].record()
                y = marlin(xs[i], ws[i], N, K)
                with torch.cuda.stream(AUX):
                    ev0[i].wait()
                    ba = in_proj_ba(xs[i], wba[i])
                    ev1[i].record()
                ev1[i].wait()
                z = y[:, :NBA] + ba
                norm(y, r2s[i], wn2)
                outs.append((y, z))
            return outs

        gr, out = capture(fn)
        graphs[a[0]] = gr
        keep.append((st, xs, r1s, r2s, out, fn))
        for _ in range(WARM):
            gr.replay()
    torch.cuda.synchronize()
    evs = {a[0]: [] for a in ARMS}
    for it in range(ITERS):
        for a in ARMS:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[a[0]].replay()
            e1.record()
            evs[a[0]].append((e0, e1))
    torch.cuda.synchronize()
    r = {}
    for a in ARMS:
        t = sorted(e0.elapsed_time(e1) * 1e3 / C for e0, e1 in evs[a[0]])
        r[a[0]] = {"us_med": t[len(t) // 2], "us_p10": t[len(t) // 10], "us_p90": t[(9 * len(t)) // 10]}
    r["W_minus_A_us"] = r["W"]["us_med"] - r["A"]["us_med"]
    r["B_minus_A_us"] = r["B"]["us_med"] - r["A"]["us_med"]
    r["B_minus_W_us"] = r["B"]["us_med"] - r["W"]["us_med"]
    fork_res[str(m)] = r
    log(f"fork M={m}: per-layer us A {r['A']['us_med']:.2f} W {r['W']['us_med']:.2f} B {r['B']['us_med']:.2f} "
        f"(W-A {r['W_minus_A_us']:+.2f}, B-A {r['B_minus_A_us']:+.2f}, B-W {r['B_minus_W_us']:+.2f})")
    del graphs, keep
    torch.cuda.synchronize()
del ws, wba
torch.cuda.empty_cache()

# ---------------- Part 2: lm_head shape correctness ----------------
N, K = 248320, 5120
w = build(N, K, 7)
wn1 = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
failures = []
replays = 0
for m in (1, 2, 4):
    x0 = (torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    r0 = (torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)

    def make(x0=x0, r0=r0):
        x = x0.clone()
        r = r0.clone()

        def fn():
            x.copy_(x0)
            r.copy_(r0)
            norm(x, r, wn1)
            y = marlin(x, w, N, K)
            return y, y.float()
        return fn

    eager = {}
    for a in ARMS:
        set_arm(a)
        y, yf = make()()
        eager[a[0]] = (y.clone(), yf.clone())
    torch.cuda.synchronize()
    for a in ARMS:
        if not (torch.equal(eager[a[0]][0], eager["A"][0]) and torch.equal(eager[a[0]][1], eager["A"][1])):
            failures.append(f"lmhead eager M={m} arm {a[0]} != A")
    graphs, outs = {}, {}
    for a in ARMS:
        set_arm(a)
        graphs[a[0]], outs[a[0]] = capture(make())
    set_arm(ARMS[2])
    efn = make()
    first = {}
    for it in range(WARM + ITERS):
        for a in ARMS:
            y, yf = outs[a[0]]
            y.fill_(float("nan"))
            yf.fill_(float("nan"))
            graphs[a[0]].replay()
            torch.cuda.synchronize()
            replays += 1
            if it == 0:
                first[a[0]] = (y.clone(), yf.clone())
            ok = (torch.equal(y, first[a[0]][0]) and torch.equal(yf, first[a[0]][1])
                  and torch.equal(y, eager["A"][0]) and torch.equal(yf, eager["A"][1]))
            if not ok:
                failures.append(f"lmhead replay M={m} arm {a[0]} it {it}")
            ey, eyf = efn()
            torch.cuda.synchronize()
            if not (torch.equal(ey, eager["A"][0]) and torch.equal(eyf, eager["A"][1])):
                failures.append(f"lmhead interleaved eager M={m} after arm {a[0]} it {it}")
    log(f"lmhead M={m}: {WARM + ITERS} NaN-poisoned replays x 3 arms + interleaved eager; failures so far {len(failures)}")
    del graphs, outs
torch.cuda.synchronize()
snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
sms = torch.cuda.get_device_properties(dev).multi_processor_count
ws_zero = bool((snap == 0).all().item()) and snap.numel() > 0
log("workspace bound slots", snap.numel() // sms, "all zero", ws_zero)
if not ws_zero:
    failures.append("workspace not all zero")
print(json.dumps({"passed": not failures, "failures": failures[:20], "n_failures": len(failures),
                  "lmhead_graph_replays_checked": replays, "bound_slots": snap.numel() // sms,
                  "workspace_all_zero": ws_zero, "fork_timing_us_per_layer": fork_res,
                  "compared": "lm_head (248320,5120) M=1/2/4: bitwise vs arm-A eager and iteration 1, "
                              "NaN-poisoned replays, interleaved eager"}))
