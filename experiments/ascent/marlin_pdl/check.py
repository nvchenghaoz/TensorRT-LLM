# Untimed correctness for the dense Marlin NVFP4 persistent workspace + PDL change.
# Same chain as bench.py (gemma fused add RMSNorm -> marlin_nvfp4_gemm -> gemma fused add
# RMSNorm, all mutable state restored by one in-graph copy), with:
#  0. a graph captured in arm B before any eager Marlin call (no pool yet -> at::zeros fallback);
#  1. per shape x M in 1,2,3,4,8,16: eager reference per arm (A = both switches 0, W = workspace
#     only, B = both on), all bitwise equal to arm A; a graph per arm (2 Marlin calls per graph,
#     sharing one slot, separated by the norms); 10 warmup + 50 replays per arm, alternated, with
#     every output / state buffer NaN-poisoned before each replay and an eager chain call
#     interleaved after each replay; every replay compared bitwise with its iteration 1 and with
#     the arm-A eager reference;
#  2. back-to-back Marlin -> Marlin on one slot (PDL edge between two Marlin grids) in a graph;
#  3. eager M=64 and 129 (multi-launch while(rest_m) loop on one slot) bitwise across arms;
#  4. every bound persistent-workspace slot reads back all zero.
import json
import os
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
EPS = 1e-6
WARM, ITERS = 10, 50
SHAPES = [("qkvz", 16384, 5120), ("outproj", 5120, 6144), ("qkv", 14336, 5120)]
MS_GRAPH = [1, 2, 3, 4, 8, 16]
MS_EAGER = [64, 129]
ARMS = [("A", "0", "0"), ("W", "1", "0"), ("B", "1", "1")]
COPIES = 2
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
    return _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16)


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k)


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


def snapshot():
    torch.cuda.synchronize()
    return torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))


class Chain:
    def __init__(self, n, k, m, ws, wn1, wn2, seed):
        self.n, self.k, self.m, self.ws, self.wn1, self.wn2 = n, k, m, ws, wn1, wn2
        c = len(ws)
        per = m * k * 2 + m * n
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 2 * m * k] = (
                torch.randn(2 * m * k, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.r2 = [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * k].view(m, k))
            self.r1.append(self.st[b + m * k: b + 2 * m * k].view(m, k))
            self.r2.append(self.st[b + 2 * m * k: b + per].view(m, n))
        self.ys = None

    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(self.ws)):
            norm(self.x[i], self.r1[i], self.wn1)
            y = marlin(self.x[i], self.ws[i], self.n, self.k)
            norm(y, self.r2[i], self.wn2)
            ys.append(y)
        return ys

    def state(self, ys):
        # st holds x (norm1 out), r1 (residual), r2 (= raw Marlin output); ys = norm2 outputs.
        return torch.cat([self.st.clone()] + [y.reshape(-1).clone() for y in ys])

    def poison(self, ys):
        self.st.fill_(float("nan"))
        for y in ys:
            y.fill_(float("nan"))


SIDE = torch.cuda.Stream()


def capture(ch):
    s0 = SIDE
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            ch.run()
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        ys = ch.run()
    ch.ys = ys  # keep graph outputs (and ch inputs) referenced
    return gr


failures = []
stats = {"graph_replays_checked": 0, "eager_interleaved_checked": 0, "cases": 0}
has = hasattr(torch.ops.trtllm, "marlin_nvfp4_persistent_workspace_snapshot")
log("has_switches", has, "gpu", torch.cuda.get_device_name())
assert has, "candidate tree required"

weights = {}
for sname, n, k in SHAPES:
    base = build(n, k, 11 + n + k)
    weights[sname] = [base] + [tuple(t.clone() for t in base) for _ in range(COPIES - 1)]
g = torch.Generator(device=dev).manual_seed(5)
normw = {}
for sname, n, k in SHAPES:
    normw[sname] = ((0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16),
                    (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16))

# 0. Capture before any eager Marlin call: the pool does not exist yet -> at::zeros fallback.
sname, n, k = SHAPES[1]
pre_snap = snapshot()
for width, wv in ((k, normw[sname][0]), (n, normw[sname][1])):
    for mm in (1, 4):
        x0 = torch.randn(mm, width, device=dev).to(torch.bfloat16)
        r0 = torch.randn(mm, width, device=dev).to(torch.bfloat16)
        norm(x0, r0, wv)  # JIT the norms outside capture (no Marlin call)
torch.cuda.synchronize()
set_arm(ARMS[2])
ch0 = Chain(n, k, 1, weights[sname], *normw[sname], seed=999)
gr0 = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr0):
    ys0 = ch0.run()
ch0.ys = ys0
mid_snap = snapshot()
ch0.poison(ys0)
gr0.replay()
torch.cuda.synchronize()
g0_out = ch0.state(ys0)
set_arm(ARMS[0])
che = Chain(n, k, 1, weights[sname], *normw[sname], seed=999)
e0_out = che.state(che.run())
torch.cuda.synchronize()
ok0 = pre_snap.numel() == 0 and mid_snap.numel() == 0 and torch.equal(g0_out, e0_out)
log("phase0 capture-before-pool: pool empty before/after capture",
    pre_snap.numel() == 0, mid_snap.numel() == 0, "replay == eager", torch.equal(g0_out, e0_out))
if not ok0:
    failures.append("phase0")

# 1. Graph chains per shape x M, three arms.
for sname, n, k in SHAPES:
    ws = weights[sname]
    wn1, wn2 = normw[sname]
    for m in MS_GRAPH:
        stats["cases"] += 1
        seed = 100 + m
        eager_ref = {}
        for a in ARMS:
            set_arm(a)
            ch = Chain(n, k, m, ws, wn1, wn2, seed)
            eager_ref[a[0]] = ch.state(ch.run())
        torch.cuda.synchronize()
        for a in ARMS:
            if not torch.equal(eager_ref[a[0]], eager_ref["A"]):
                failures.append(f"eager {sname} M={m} arm {a[0]} != A")
        ref = eager_ref["A"]
        chains, graphs = {}, {}
        for a in ARMS:
            set_arm(a)
            ch = Chain(n, k, m, ws, wn1, wn2, seed)
            graphs[a[0]] = capture(ch)
            chains[a[0]] = ch
        # eager chain interleaved between replays (runs in arm B mode, main stream slot)
        set_arm(ARMS[2])
        ech = Chain(n, k, m, ws, wn1, wn2, seed)
        first = {}
        worst = {a[0]: 0.0 for a in ARMS}
        for it in range(WARM + ITERS):
            for a in ARMS:
                ch = chains[a[0]]
                ch.poison(ch.ys)
                graphs[a[0]].replay()
                torch.cuda.synchronize()
                out = ch.state(ch.ys)
                if it == 0:
                    first[a[0]] = out
                eq1 = torch.equal(out, first[a[0]])
                eqr = torch.equal(out, ref)
                stats["graph_replays_checked"] += 1
                if not (eq1 and eqr):
                    d = (out.float() - ref.float()).abs()
                    worst[a[0]] = max(worst[a[0]], float(torch.nan_to_num(d, nan=float("inf")).max()))
                    failures.append(f"replay {sname} M={m} arm {a[0]} it {it} eq_iter1 {eq1} eq_ref {eqr}")
                eo = ech.state(ech.run())
                torch.cuda.synchronize()
                stats["eager_interleaved_checked"] += 1
                if not torch.equal(eo, ref):
                    failures.append(f"interleaved eager {sname} M={m} after arm {a[0]} it {it}")
        log(f"{sname} M={m}: {WARM + ITERS} poisoned replays x 3 arms + interleaved eager; "
            f"max err vs ref {worst}; failures so far {len(failures)}")
        del graphs, chains, ech
        torch.cuda.synchronize()

# 2. Marlin -> Marlin back to back on one slot (PDL edge between two Marlin grids).
for sname, n, k in SHAPES:
    ws = weights[sname]
    for m in (1, 16):
        x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
        set_arm(ARMS[0])
        refs = [marlin(x, w, n, k).clone() for w in ws]
        set_arm(ARMS[2])
        s0 = SIDE
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(3):
                [marlin(x, w, n, k) for w in ws]
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = [marlin(x, w, n, k) for w in ws]
        ok = True
        for it in range(WARM + ITERS):
            for o in outs:
                o.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            ok = ok and all(torch.equal(o, r) for o, r in zip(outs, refs))
            stats["graph_replays_checked"] += 1
        log(f"marlin->marlin {sname} M={m}: bitwise {ok}")
        if not ok:
            failures.append(f"marlin->marlin {sname} M={m}")
        del gr, outs

# 3. Eager M = 64 / 129 (multi-launch loop on one slot).
for sname, n, k in SHAPES:
    w = weights[sname][0]
    for m in MS_EAGER:
        x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
        res = {}
        for rep in range(3):
            for a in ARMS:
                set_arm(a)
                res.setdefault(a[0], []).append(marlin(x, w, n, k).clone())
        torch.cuda.synchronize()
        ok = all(torch.equal(o, res["A"][0]) for a in ARMS for o in res[a[0]])
        log(f"eager {sname} M={m}: arms x 3 reps bitwise {ok}")
        if not ok:
            failures.append(f"eager {sname} M={m}")

# 4. Workspace slots all zero.
snap = snapshot()
sms = torch.cuda.get_device_properties(dev).multi_processor_count
ws_zero = bool((snap == 0).all().item()) and snap.numel() > 0
log("persistent workspace: bound slots", snap.numel() // sms, "all zero", ws_zero)
if not ws_zero:
    failures.append("workspace not all zero")

passed = not failures
out = {"passed": passed, "failures": failures[:20], "n_failures": len(failures), **stats,
       "bound_slots": snap.numel() // sms, "workspace_all_zero": ws_zero,
       "phase0_capture_before_pool_ok": bool(ok0),
       "compared": "bitwise vs arm-A eager (both switches 0 = main's code path) and vs iteration 1; "
                   "outputs NaN-poisoned before each replay",
       "warmup": WARM, "iters": ITERS}
print(json.dumps(out))
