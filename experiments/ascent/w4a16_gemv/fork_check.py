# Idea 177 / branch 94: repeated-execution check (untimed) for fork_window.py's timed loop.
# Same graphs as the timed "step" graph (48 GDN fork windows + 16 attention windows in model order,
# production FP8QDQLinearMethod.apply on _build_fp8_nvfp4_decode_copy copies), same warmup (10),
# iteration count (60) and reset points (st.copy_(st0) then the 256 MiB READ flush before every
# replay). Per M in 1..4 and per arm (tree = default routing, marlin = INPROJ=0 copies):
#  - before every replay every output is NaN-poisoned;
#  - after every replay every output equals iteration 1 and the eager run of the same fn from st0,
#    bitwise; the mutable x/r state equals the eager post-norm state (the norm's in-place update
#    is the only state evolution; the reset restores st0 before the next replay);
#  - tree vs FP64 reference error <= 1.25x marlin's + 1 BF16 ulp (reordered class), where the
#    reference is x_normed(FP64) @ dequant(NVFP4 copy)(FP64), sliced like the consumer.
# Rows above the cap: tree outputs bitwise equal marlin outputs (M=4 qkvz/qkv, M=3 qkv).
import json
import os
import subprocess
import sys
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
INPROJ_ENV = "TRTLLM_NVFP4_DECODE_GEMV_INPROJ"
os.environ.pop(INPROJ_ENV, None)
os.environ.pop("TRTLLM_NVFP4_DECODE_GEMV", None)
import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules import linear as L  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm._torch.modules.small_n_gemv import maybe_small_n_gemv  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402


def log(*a):
    print(*a, flush=True)


for word in sys.argv[1:]:
    log(word)
try:
    log("TREE_COMMIT=" + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        timeout=30).stdout.strip())
except Exception as exc:  # noqa: BLE001
    log("TREE_COMMIT=unknown " + repr(exc))

dev = torch.device("cuda")
EPS = 1e-6
K = 5120
NQKVZ, NQKV, NBA = 16384, 14336, 96
NGDN, NATT = 48, 16
NDIST = 4  # distinct weight copies cycled (correctness only; the timed bench uses 48/16)
WARM, ITERS = 10, 60
MS = (1, 2, 3, 4)
CAPS = getattr(L, "_NVFP4_DECODE_GEMV_ROW_CAPS", dict())
log("caps", CAPS)


def make_lin(n, seed, inproj_env):
    if inproj_env is None:
        os.environ.pop(INPROJ_ENV, None)
    else:
        os.environ[INPROJ_ENV] = inproj_env
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, K, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(n, 1, generator=gen, device=dev) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    lin = Linear(K, n, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(0.04, device=dev))])
    lin.post_load_weights()
    os.environ.pop(INPROJ_ENV, None)
    return lin


def dequant(gemv, n):
    wq, sf, gs = gemv
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                       device=dev, dtype=torch.float64)
    v = torch.stack([lut[(wq & 0xF).long()], lut[(wq >> 4).long()]], dim=-1).reshape(n, K)
    return v * sf.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1) * gs


METHOD = None
NS = dict(tree=dict(), marlin=dict())
WREF = dict()
for name, n, seed in (("qkvz", NQKVZ, 11), ("qkv", NQKV, 12)):
    for src, env in (("tree", None), ("marlin", "0")):
        lst = []
        for c in range(NDIST):
            lin = make_lin(n, seed + 100 * c, env)
            METHOD = lin.quant_method
            ns = types.SimpleNamespace()
            ns.out_features, ns.in_features = lin.out_features, lin.in_features
            ns._nvfp4_decode_weight = lin._nvfp4_decode_weight
            ns._nvfp4_decode_scale = lin._nvfp4_decode_scale
            ns._nvfp4_decode_global_scale = lin._nvfp4_decode_global_scale
            ns._nvfp4_decode_gemv = lin.__dict__.get("_nvfp4_decode_gemv")
            ns._nvfp4_decode_gemv_max_rows = lin.__dict__.get("_nvfp4_decode_gemv_max_rows", 0)
            if src == "tree":
                # FP64 reference rows actually read by the consumer (first 2*96 output columns).
                if ns._nvfp4_decode_gemv is not None:
                    WREF[(name, c)] = dequant(ns._nvfp4_decode_gemv, n)[:2 * NBA].clone()
                else:
                    WREF[(name, c)] = None
            lst.append(ns)
            del lin
        NS[src][name] = lst
    log("built", name, "tree cap", NS["tree"][name][0]._nvfp4_decode_gemv_max_rows,
        "marlin-arm gemv copy", NS["marlin"][name][0]._nvfp4_decode_gemv is not None)
torch.cuda.empty_cache()

g = torch.Generator(device=dev).manual_seed(5)
wba = [(torch.randn(NBA, K, generator=g, device=dev) * 0.02).to(torch.bfloat16) for _ in range(NDIST)]
wn = (0.1 * torch.randn(K, generator=g, device=dev)).to(torch.bfloat16)
flush = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
LAYERS = []
for b in range(NATT):
    LAYERS += [("g", 3 * b), ("g", 3 * b + 1), ("g", 3 * b + 2), ("a", b)]
NL = len(LAYERS)
SIDE = torch.cuda.Stream()
AUX = torch.cuda.Stream()


def in_proj_ba(x, w):
    y = maybe_small_n_gemv(x, w)
    return torch.nn.functional.linear(x, w) if y is None else y


def make_fn(arm, xs, rs):
    qkvz_ns, qkv_ns = NS[arm]["qkvz"], NS[arm]["qkv"]
    ev0 = [torch.cuda.Event() for _ in LAYERS]
    ev1 = [torch.cuda.Event() for _ in LAYERS]

    def fn():
        zs = []
        for j, (t, idx) in enumerate(LAYERS):
            x = xs[j]
            torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, rs[j], wn, EPS)
            c = idx % NDIST
            if t == "g":
                ev0[j].record()
                y = METHOD.apply(qkvz_ns[c], x, None)
                with torch.cuda.stream(AUX):
                    ev0[j].wait()
                    ba = in_proj_ba(x, wba[c])
                    ev1[j].record()
                ev1[j].wait()
                zs.append(y[:, :NBA] + ba)
            else:
                y = METHOD.apply(qkv_ns[c], x, None)
                zs.append(y[:, :NBA] + y[:, NBA:2 * NBA])
        return zs

    return fn


def bits(t):
    return t.contiguous().view(torch.int16)


def ulp(v):
    e = torch.floor(torch.log2(torch.tensor(abs(v), dtype=torch.float64).clamp(min=1e-30)))
    return float(2.0 ** (e - 7))


passed = True
report = dict()
for m in MS:
    per = 2 * m * K
    st0 = (torch.randn(NL * per, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    rec = dict()
    eager_out = dict()
    eager_state = dict()
    graph_out = dict()
    for arm in ("tree", "marlin"):
        st = st0.clone()
        xs = [st[i * per: i * per + m * K].view(m, K) for i in range(NL)]
        rs = [st[i * per + m * K: (i + 1) * per].view(m, K) for i in range(NL)]
        fn = make_fn(arm, xs, rs)
        # Eager reference from st0 (multi-stream fork outside a graph, same order).
        st.copy_(st0)
        ref = [z.clone() for z in fn()]
        torch.cuda.synchronize()
        eager_out[arm] = ref
        eager_state[arm] = st.clone()
        # Capture (warm on a side stream first, as fork_window.py does).
        SIDE.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(SIDE):
            for _ in range(3):
                st.copy_(st0)
                fn()
        torch.cuda.current_stream().wait_stream(SIDE)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = fn()
        torch.cuda.synchronize()
        first = None
        stable = True
        eq_eager = True
        state_ok = True
        max_dev = 0.0
        for it in range(WARM + ITERS):
            for z in outs:
                z.fill_(float("nan"))
            st.copy_(st0)  # reset point (same as the timed loop)
            flush.sum()  # READ flush (same as the timed loop)
            gr.replay()
            torch.cuda.synchronize()
            cur = [z.clone() for z in outs]
            if first is None:
                first = cur
            else:
                for a, b in zip(cur, first):
                    if not torch.equal(bits(a), bits(b)):
                        stable = False
                        max_dev = max(max_dev, (a.float() - b.float()).abs().nan_to_num(1e30).max().item())
            for a, b in zip(cur, ref):
                eq_eager &= bool(torch.equal(bits(a), bits(b)))
            # State before the next reset: the in-place norm's expected evolution.
            state_ok &= bool(torch.equal(bits(st), bits(eager_state[arm])))
        graph_out[arm] = first
        rec[arm] = dict(replays=WARM + ITERS, stable=stable, eq_eager=eq_eager, state_ok=state_ok,
                        max_dev=max_dev)
        passed &= stable and eq_eager and state_ok
        del gr, outs
    # Error vs FP64 reference (tree vs marlin) and rows above cap.
    st = eager_state["tree"]
    xs = [st[i * per: i * per + m * K].view(m, K) for i in range(NL)]
    worst = dict(qkvz=dict(tree=0.0, marlin=0.0, ulp=0.0, eq=0, n=0),
                 qkv=dict(tree=0.0, marlin=0.0, ulp=0.0, eq=0, n=0))
    for j, (t, idx) in enumerate(LAYERS):
        name = "qkvz" if t == "g" else "qkv"
        c = idx % NDIST
        wref = WREF[(name, c)]
        zt, zm = graph_out["tree"][j], graph_out["marlin"][j]
        d = worst[name]
        d["eq"] += int((bits(zt) == bits(zm)).sum().item())
        d["n"] += zt.numel()
        if wref is None:
            continue
        y = xs[j].double() @ wref.t()
        if t == "g":
            # the ba branch is identical in both arms; compare the projection part only
            ba = in_proj_ba(xs[j], wba[c]).double()
            refz = y[:, :NBA] + ba
        else:
            refz = y[:, :NBA] + y[:, NBA:2 * NBA]
        d["tree"] = max(d["tree"], (zt.double() - refz).abs().max().item())
        d["marlin"] = max(d["marlin"], (zm.double() - refz).abs().max().item())
        d["ulp"] = max(d["ulp"], ulp(refz.abs().max().item()))
    for name in ("qkvz", "qkv"):
        d = worst[name]
        cap = NS["tree"][name][0]._nvfp4_decode_gemv_max_rows
        d["bitwise_frac"] = d.pop("eq") / max(d.pop("n"), 1)
        d["cap"] = cap
        if m <= cap:
            d["err_ok"] = d["tree"] <= 1.25 * d["marlin"] + d["ulp"]
        else:
            d["err_ok"] = d["bitwise_frac"] == 1.0  # above the cap: bitwise Marlin
        passed &= bool(d["err_ok"])
    rec["error"] = worst
    report[str(m)] = rec
    log("[fork_check] M=%d %s" % (m, json.dumps(rec)))

print(json.dumps(dict(passed=bool(passed), warm=WARM, iters=ITERS,
                      compared="fork-window step graph (48 qkvz fork + 16 qkv windows): replays vs "
                               "iteration 1 and eager bitwise, post-norm state vs eager, tree error vs "
                               "FP64 NVFP4 reference <= 1.25x Marlin + 1 ulp, rows above cap bitwise Marlin",
                      results=report)))
