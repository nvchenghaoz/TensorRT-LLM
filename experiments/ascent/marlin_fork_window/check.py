# Exactness check for idea 137 (TRTLLM_MARLIN_FORK_FRESH_WORKSPACE default "1" -> "0").
# GDN in_proj_qkvz shape (N=16384, K=5120), M = 1, 2, 4, 8:
#   1. eager: Marlin workspace_mode 0 output bitwise equal to workspace_mode 1 (main's mode at M >= 2),
#      on normed activations, for every weight copy.
#   2. forked CUDA graph, production order, LAYERS layers cycling 4 distinct weights:
#      x/r reset -> gemma fused add+RMSNorm (PDL) -> ev0 -> Marlin (mode from production code:
#      linear._marlin_workspace_mode on a module marked by mark_marlin_fork_sites, env unset)
#      || aux: in_proj_ba (maybe_small_n_gemv) -> ev1 -> join -> z = y[:, :96] + ba.
#      10 warmup + 50 replays; before each replay every output is NaN-poisoned; after each replay
#      every output (y, z) equals iteration 1 and the mode-1 eager reference, and the mutable
#      x/r state equals the eager norm result (the reset sits inside the graph).
#   3. marlin_nvfp4_persistent_workspace_snapshot all zero afterwards.
# Brace-free (extra argv words are printed first).
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
os.environ.pop("TRTLLM_MARLIN_FORK_FRESH_WORKSPACE", None)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402

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
log("linear.py:", L.__file__)

dev = torch.device("cuda")
EPS = 1e-6
N, K, NBA = 16384, 5120, 96
NW = 4
LAYERS = 12
WARM, ITERS = 10, 50
MS = (1, 2, 4, 8)


def build(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    del w
    return L._pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), N, K, 16)


ws = [build(100 + i) for i in range(NW)]
g = torch.Generator(device=dev).manual_seed(9)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16) for _ in range(NW)]
wn = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
log("built", NW, "distinct qkvz weights")

root = torch.nn.Module()
root.in_proj_qkvz = torch.nn.Module()
root.in_proj_ba = torch.nn.Module()
root.in_proj_qkvz._nvfp4_decode_weight = ws[0][0]
n_marked = L.mark_marlin_fork_sites(root)
QKVZ = root.in_proj_qkvz
modes = [L._marlin_workspace_mode(QKVZ, m) for m in MS]
log("env unset: marked", n_marked, "modes at M=1/2/4/8:", "/".join(str(x) for x in modes))


def marlin(x, wt, mode):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None,
                                              out_dtype=torch.bfloat16, size_n=N, size_k=K,
                                              workspace_mode=mode)


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return F.linear(x, w) if y is None else y


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


SIDE = torch.cuda.Stream()
AUX = torch.cuda.Stream()
failures = []
replays = 0
eager_pairs = 0

# Create the persistent workspace pool with one eager mode-0 call (as production warmup does).
torch.cuda.synchronize()
_ = marlin((torch.randn(2, K, generator=g, device=dev) * 0.5).to(torch.bfloat16), ws[0], 0)
torch.cuda.synchronize()

for m in MS:
    mode = L._marlin_workspace_mode(QKVZ, m)
    x0 = [(torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16) for _ in range(LAYERS)]
    r0 = [(torch.randn(m, K, generator=g, device=dev) * 0.5).to(torch.bfloat16) for _ in range(LAYERS)]
    # eager mode-1 reference of the whole chain, and eager mode-0 vs mode-1 bitwise
    ref = []
    for i in range(LAYERS):
        x = x0[i].clone()
        r = r0[i].clone()
        norm(x, r, wn)
        y1 = marlin(x, ws[i % NW], 1)
        y0 = marlin(x, ws[i % NW], 0)
        ba = in_proj_ba(x, wba[i % NW])
        z = y1[:, :NBA] + ba
        torch.cuda.synchronize()
        eager_pairs += 1
        if not torch.equal(y0, y1):
            failures.append("eager M=%d layer %d mode0 != mode1 (maxabs %.3g)"
                            % (m, i, (y0.float() - y1.float()).abs().max().item()))
        if torch.isnan(y1).any() or torch.isinf(y1).any():
            failures.append("eager M=%d layer %d mode1 has NaN/Inf" % (m, i))
        ref.append((x.clone(), r.clone(), y1.clone(), z.clone()))

    xs = [t.clone() for t in x0]
    rs = [t.clone() for t in r0]
    ev0 = [torch.cuda.Event() for _ in range(LAYERS)]
    ev1 = [torch.cuda.Event() for _ in range(LAYERS)]

    def fn():
        outs = []
        for i in range(LAYERS):
            xs[i].copy_(x0[i])
            rs[i].copy_(r0[i])
            norm(xs[i], rs[i], wn)
            ev0[i].record()
            y = marlin(xs[i], ws[i % NW], mode)
            with torch.cuda.stream(AUX):
                ev0[i].wait()
                ba = in_proj_ba(xs[i], wba[i % NW])
                ev1[i].record()
            ev1[i].wait()
            z = y[:, :NBA] + ba
            outs.append((y, z))
        return outs

    SIDE.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(SIDE):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(SIDE)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs = fn()
    torch.cuda.synchronize()
    first = None
    worst = 0.0
    for it in range(WARM + ITERS):
        for y, z in outs:
            y.fill_(float("nan"))
            z.fill_(float("nan"))
        for i in range(LAYERS):
            xs[i].fill_(float("nan"))
            rs[i].fill_(float("nan"))
        gr.replay()
        torch.cuda.synchronize()
        replays += 1
        if first is None:
            first = [(y.clone(), z.clone()) for y, z in outs]
        for i in range(LAYERS):
            y, z = outs[i]
            rx, rr, ry, rz = ref[i]
            ok_first = torch.equal(y, first[i][0]) and torch.equal(z, first[i][1])
            ok_ref = torch.equal(y, ry) and torch.equal(z, rz)
            ok_state = torch.equal(xs[i], rx) and torch.equal(rs[i], rr)
            if not ok_ref:
                worst = max(worst, (y.float() - ry.float()).abs().nan_to_num(1e30).max().item())
            if not (ok_first and ok_ref and ok_state):
                failures.append("graph M=%d it %d layer %d first %s ref %s state %s"
                                % (m, it, i, ok_first, ok_ref, ok_state))
    log("M=%d mode %d: %d NaN-poisoned replays x %d layers, max |y - mode1 eager| %.3g, failures so far %d"
        % (m, mode, WARM + ITERS, LAYERS, worst, len(failures)))
    del gr, outs
    torch.cuda.synchronize()

torch.cuda.synchronize()
snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
sms = torch.cuda.get_device_properties(dev).multi_processor_count
ws_zero = bool((snap == 0).all().item()) and snap.numel() > 0
log("persistent workspace bound slots", snap.numel() // sms, "all zero", ws_zero)
if not ws_zero:
    failures.append("persistent workspace not all zero (or no pool)")
print(json.dumps(dict(passed=not failures, failures=failures[:20], n_failures=len(failures),
                      modes_env_unset=modes, graph_replays=replays, layers_per_graph=LAYERS,
                      eager_mode0_vs_mode1_pairs=eager_pairs, bound_slots=snap.numel() // sms,
                      workspace_all_zero=ws_zero,
                      compared="qkvz (16384,5120) M=1/2/4/8: Marlin mode 0 (env-unset production mode) "
                               "bitwise vs mode 1 eager (main's mode at M>=2) and vs iteration 1 over "
                               "10+50 NaN-poisoned forked graph replays; x/r state vs eager norm")))
