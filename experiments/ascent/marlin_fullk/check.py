# Untimed correctness for the full-K dense Marlin launch (idea 136): exact class, bitwise vs main's
# 188-CTA split-K launch (TRTLLM_MARLIN_FULLK=0) in the same process.
#  1. (5120,6144) eager, M = 1..8, 24 seeds over 4 weight families (normal, outlier output channels,
#     heavy-tailed, tiny) and per-seed activation patterns with outlier rows (x300 rows, x1e-3 rows,
#     zero rows, sparse 1e3 spikes): full-K output bitwise equal to the kill-switch output; FP32
#     dequant-reference error reported for both (identical by construction when bitwise).
#  2. (5120,6144) graphs, M = 1..8: norm -> Marlin -> norm chain (2 weight copies), arm F and arm A
#     graphs, 10 warmup + 50 replays each, every output / state buffer NaN-poisoned before each
#     replay; every replay compared bitwise with its iteration 1 and with the arm-A eager reference.
#  3. Rows outside the gate (M = 9..16, 64) and the other shapes ((16384,5120), (14336,5120)) at
#     M = 1, 4, 8: switch on == switch off bitwise (same split-K path).
#  4. Kernel actually launched per arm (profiler: name + grid) and persistent lock workspace all zero.
import json
import os
import subprocess
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
EPS = 1e-6
WARM, ITERS = 10, 50
N, K = 5120, 6144
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def git(args):
    try:
        return subprocess.run(["git"] + args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a {e}"


head = git(["rev-parse", "HEAD"])
dirty = git(["status", "--porcelain"])
print(f"TREE_COMMIT={head} dirty_files={len(dirty.splitlines()) if dirty else 0}", flush=True)


def set_arm(a):
    os.environ["TRTLLM_MARLIN_FULLK"] = "0" if a == "A" else "1"


def quant_only(w):
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    n, k = w.shape
    return _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16), (w_fp4, w_sf, gs)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float32)


def dequant_ref(w_fp4, w_sf, gs, n, k):
    # w_fp4: [n, k/2] uint8 (low nibble first); w_sf: swizzled 128x4 FP8 block scales.
    try:
        lut = E2M1.to(dev)
        b = w_fp4.view(torch.uint8).reshape(n, k // 2)
        lo = lut[(b & 0xF).long()]
        hi = lut[(b >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(n, k)
        sf = torch.ops.trtllm.block_scale_interleave_reverse(w_sf.view(torch.uint8).reshape(-1, k // 16)) \
            if hasattr(torch.ops.trtllm, "block_scale_interleave_reverse") else None
        if sf is None:
            return None
        sf = sf.view(torch.float8_e4m3fn).float()[:n, : k // 16]
        return vals * sf.repeat_interleave(16, dim=1) / gs.item()
    except Exception as e:  # noqa: BLE001
        log("dequant_ref unavailable:", repr(e)[:200])
        return None


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=n, size_k=k)


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


def weight_family(fam, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(N, K, generator=g, device=dev) * 0.02
    if fam == "outlier_ch":
        idx = torch.randint(0, N, (64,), generator=g, device=dev)
        w[idx] *= 50.0
    elif fam == "heavy":
        w = w * torch.exp(torch.randn(N, K, generator=g, device=dev) * 1.5)
    elif fam == "tiny":
        w = w * 1e-4
    return w.to(torch.bfloat16)


def act(m, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, K, generator=g, device=dev) * 0.5
    pat = seed % 4
    for r in range(m):
        kind = (r + pat) % 5
        if kind == 1:
            x[r] *= 300.0
        elif kind == 2:
            x[r] *= 1e-3
        elif kind == 3 and m > 2:
            x[r] = 0.0
        elif kind == 4:
            idx = torch.randint(0, K, (8,), generator=g, device=dev)
            x[r, idx] = 1e3 * torch.sign(torch.randn(8, generator=g, device=dev))
    return x.to(torch.bfloat16)


failures = []
stats = dict(eager_cases=0, graph_replays_checked=0, other_cases=0)
log("gpu", torch.cuda.get_device_name())

# 4a. kernel launched per arm
w0 = weight_family("normal", 7)
wt0, q0 = quant_only(w0)
x1 = act(1, 3)
launched = {}
for a in ("A", "F"):
    set_arm(a)
    marlin(x1, wt0, N, K)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        marlin(x1, wt0, N, K)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if "Marlin" in e.name or "marlin" in e.name]
    launched[a] = names[:2]
    log(f"arm {a} kernels: {names[:2]}")
fullk_in_F = any("8,1,true>" in nm.replace(" ", "") for nm in launched["F"])
fullk_in_A = any("8,1,true>" in nm.replace(" ", "") for nm in launched["A"])
log("full_k kernel in arm F:", fullk_in_F, "in arm A:", fullk_in_A)
if not fullk_in_F or fullk_in_A:
    failures.append(f"kernel selection F={launched['F']} A={launched['A']}")

# 1. eager bitwise, 24 seeds x M = 1..8
fams = ["normal", "outlier_ch", "heavy", "tiny"]
max_err = dict(A=0.0, F=0.0)
for fi, fam in enumerate(fams):
    w = weight_family(fam, 1000 + fi)
    wt, q = quant_only(w)
    ref_w = dequant_ref(q[0], q[1], q[2], N, K)
    for s in range(6):
        seed = 10 * fi + s
        for m in range(1, 9):
            x = act(m, seed * 100 + m)
            set_arm("A")
            ya = marlin(x, wt, N, K)
            set_arm("F")
            yf = marlin(x, wt, N, K)
            yf2 = marlin(x, wt, N, K)
            torch.cuda.synchronize()
            stats["eager_cases"] += 1
            if not (torch.equal(ya, yf) and torch.equal(yf, yf2)):
                d = (ya.float() - yf.float()).abs()
                failures.append(f"eager {fam} seed {seed} M={m} F!=A maxdiff {float(torch.nan_to_num(d, nan=1e30).max())}")
            if ref_w is not None and m in (1, 8):
                ref = x.float() @ ref_w.t()
                scale = ref.abs().amax().clamp_min(1e-30)
                max_err["A"] = max(max_err["A"], float(((ya.float() - ref).abs().amax() / scale)))
                max_err["F"] = max(max_err["F"], float(((yf.float() - ref).abs().amax() / scale)))
    log(f"eager family {fam}: 6 seeds x M=1..8 done; failures so far {len(failures)}; "
        f"rel max err vs FP32 ref A {max_err['A']:.3e} F {max_err['F']:.3e}")
    del w, wt, q, ref_w
    torch.cuda.empty_cache()

# 2. graph replays with NaN poisoning
base = quant_only(weight_family("normal", 77))[0]
ws = [base, tuple(t.clone() for t in base)]
g = torch.Generator(device=dev).manual_seed(5)
wn1 = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
wn2 = (0.1 * torch.randn(N, generator=g, device=dev)).to(torch.bfloat16)


class Chain:
    def __init__(self, m, seed):
        self.m = m
        c = len(ws)
        per = m * K * 2 + m * N
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + m * K] = act(m, seed + i).reshape(-1)
            st0[i * per + m * K: i * per + 2 * m * K] = act(m, seed + 50 + i).reshape(-1) * 0.01
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.r2 = [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * K].view(m, K))
            self.r1.append(self.st[b + m * K: b + 2 * m * K].view(m, K))
            self.r2.append(self.st[b + 2 * m * K: b + per].view(m, N))
        self.ys = None

    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(ws)):
            norm(self.x[i], self.r1[i], wn1)
            y = marlin(self.x[i], ws[i], N, K)
            norm(y, self.r2[i], wn2)
            ys.append(y)
        return ys

    def state(self, ys):
        return torch.cat([self.st.clone()] + [y.reshape(-1).clone() for y in ys])

    def poison(self, ys):
        self.st.fill_(float("nan"))
        for y in ys:
            y.fill_(float("nan"))


SIDE = torch.cuda.Stream()


def capture(ch):
    SIDE.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(SIDE):
        for _ in range(3):
            ch.run()
    torch.cuda.current_stream().wait_stream(SIDE)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        ys = ch.run()
    ch.ys = ys
    return gr


for m in range(1, 9):
    seed = 500 + m
    set_arm("A")
    che = Chain(m, seed)
    ref = che.state(che.run())
    torch.cuda.synchronize()
    graphs, chains = {}, {}
    for a in ("A", "F"):
        set_arm(a)
        ch = Chain(m, seed)
        graphs[a] = capture(ch)
        chains[a] = ch
    first = {}
    bad = 0
    for it in range(WARM + ITERS):
        for a in ("F", "A"):
            ch = chains[a]
            ch.poison(ch.ys)
            graphs[a].replay()
            torch.cuda.synchronize()
            out = ch.state(ch.ys)
            if it == 0:
                first[a] = out
            stats["graph_replays_checked"] += 1
            if not (torch.equal(out, first[a]) and torch.equal(out, ref)):
                bad += 1
                failures.append(f"replay M={m} arm {a} it {it}")
    log(f"graph M={m}: {WARM + ITERS} NaN-poisoned replays x 2 arms; mismatches {bad}")
    del graphs, chains, che
    torch.cuda.synchronize()

# 3. outside the gate: switch on == off
for m in list(range(9, 17)) + [64]:
    x = act(m, 900 + m)
    set_arm("A")
    ya = marlin(x, base, N, K)
    set_arm("F")
    yf = marlin(x, base, N, K)
    torch.cuda.synchronize()
    stats["other_cases"] += 1
    if not torch.equal(ya, yf):
        failures.append(f"outproj M={m} outside gate differs")
for sname, n, k in (("qkvz", 16384, 5120), ("qkv", 14336, 5120)):
    g2 = torch.Generator(device=dev).manual_seed(n)
    wt = quant_only((torch.randn(n, k, generator=g2, device=dev) * 0.02).to(torch.bfloat16))[0]
    for m in (1, 4, 8):
        x = (torch.randn(m, k, generator=g2, device=dev) * 0.5).to(torch.bfloat16)
        set_arm("A")
        ya = marlin(x, wt, n, k)
        set_arm("F")
        yf = marlin(x, wt, n, k)
        torch.cuda.synchronize()
        stats["other_cases"] += 1
        if not torch.equal(ya, yf):
            failures.append(f"{sname} M={m} differs")
    del wt
log(f"outside-gate checks done; failures so far {len(failures)}")

# 4b. persistent workspace all zero
torch.cuda.synchronize()
snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
ws_zero = bool((snap == 0).all().item()) and snap.numel() > 0
log("persistent workspace all zero", ws_zero, "ints", snap.numel())
if not ws_zero:
    failures.append("workspace not zero")

passed = not failures
print(json.dumps(dict(passed=passed, failures=failures[:20], n_failures=len(failures), **stats,
                      kernels=launched, fp32_ref_rel_max_err=max_err, workspace_all_zero=ws_zero,
                      compared="full-K (TRTLLM_MARLIN_FULLK=1) vs main's split-K launch (=0), bitwise, same process; "
                               "graph replays NaN-poisoned, vs iteration 1 and arm-A eager",
                      warmup=WARM, iters=ITERS, tree=head)))
