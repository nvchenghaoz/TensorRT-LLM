"""Idea 98 Stage 1 Q3: cuDNN eng7 runtime-fusion (GN-apply+swish prologue, bias/residual
epilogue) 3x3 conv vs main's compiled unfused chain, at the 6 real FLUX.3 VAE shapes.

Arms per shape (bf16 NHWC, batch 1):
  U        main's chain, torch.compile(dynamic=False, layout_optimization=False,
           emulate_precision_casts=True): GN apply (group_norm_channel_stats math, bf16 out)
           -> swish (bf16) -> F.conv2d(+bias) (+residual) -> next consumer: per-channel fp32
           var_mean (the next GN's stats kernel).  conv2-tail: apply+swish -> conv (no bias), no consumer.
  F        cuDNN FE graph (best eng7 plan) writing y = conv(swish(x*a+b)) + bias (+res), then the
           same compiled var_mean consumer on y.
  pieces   compiled apply alone, aten conv alone, FE plain conv / prologue-only / epilogue-only /
           prologue+epilogue at EVERY plan index.
  D98      compiled per-channel a=gamma*rstd, b=beta-mean*a derivation from (mean_c, var_c) incl.
           group combine, vs main's combine producing (mean, rstd).
Prints JSON lines; writes $AUTO_DEV_ARTIFACTS/probe_q3.jsonl and Inductor output code.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

import cudnn

ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/probe_q3")
os.makedirs(ART, exist_ok=True)
OUT = open(os.path.join(ART, "probe_q3.jsonl"), "w")


def emit(d):
    s = json.dumps(d)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


torch.manual_seed(0)
dev = "cuda"
torch._dynamo.config.cache_size_limit = 256
torch.backends.cudnn.benchmark = False

emit(
    {
        "versions": {
            "cudnn_backend": cudnn.backend_version(),
            "fe": getattr(cudnn, "__version__", "?"),
            "torch": torch.__version__,
            "torch_cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
        }
    }
)

# name, cin, cout, hw, n_conv1 (bias), n_conv2 (bias+res), n_tail (prologue only)
SHAPES = [
    ("S1", 512, 512, 128, 5, 5, 0),
    ("S2", 512, 512, 256, 3, 3, 0),
    ("S3", 512, 256, 512, 1, 0, 0),
    ("S4", 256, 256, 512, 2, 3, 0),
    ("S5", 256, 128, 1024, 1, 0, 0),
    ("S6", 128, 128, 1024, 2, 2, 1),
]
only = os.environ.get("Q3_SHAPES")
if only:
    SHAPES = [s for s in SHAPES if s[0] in only.split(",")]
ROUNDS = int(os.environ.get("Q3_ROUNDS", "3"))
ITERS = int(os.environ.get("Q3_ITERS", "20"))
OPTS = {"layout_optimization": False, "emulate_precision_casts": True}
G = 32
EPS = 1e-6

BF16 = cudnn.data_type.BFLOAT16
F32 = cudnn.data_type.FLOAT
handle = cudnn.create_handle()
cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)


# ---------------------------------------------------------------- main's math (compiled)
def _apply(x, mean, rstd, gamma, beta):
    xf = x.float()
    y = (xf - mean) * rstd
    y = y * gamma.float().view(1, -1, 1, 1) + beta.float().view(1, -1, 1, 1)
    y = y.to(x.dtype)
    return y * torch.sigmoid(y)


def u_apply(x, mean, rstd, gamma, beta):
    return _apply(x, mean, rstd, gamma, beta)


def u_conv1(x, mean, rstd, gamma, beta, w, b):
    h = F.conv2d(_apply(x, mean, rstd, gamma, beta), w, b, padding=1)
    var, m = torch.var_mean(h.float(), dim=(2, 3), correction=0)
    return var, m


def u_conv2(x, mean, rstd, gamma, beta, w, b, res):
    h = F.conv2d(_apply(x, mean, rstd, gamma, beta), w, b, padding=1)
    y = res + h
    var, m = torch.var_mean(y.float(), dim=(2, 3), correction=0)
    return var, m


def u_tail(x, mean, rstd, gamma, beta, w):
    return F.conv2d(_apply(x, mean, rstd, gamma, beta), w, None, padding=1)


def c_stats(y):
    var, m = torch.var_mean(y.float(), dim=(2, 3), correction=0)
    return var, m


def comb_main(var_c, mean_c):
    c = mean_c.shape[1]
    mc = mean_c.view(1, G, c // G)
    vc = var_c.view(1, G, c // G)
    mg = mc.mean(-1, keepdim=True)
    vg = vc.mean(-1, keepdim=True) + (mc - mg).square().mean(-1, keepdim=True)
    rg = torch.rsqrt(vg + EPS)
    mean = mg.expand(1, G, c // G).reshape(1, c, 1, 1).float()
    rstd = rg.expand(1, G, c // G).reshape(1, c, 1, 1).float()
    return mean, rstd


def comb_ab(var_c, mean_c, gamma, beta):
    mean, rstd = comb_main(var_c, mean_c)
    a = rstd * gamma.float().view(1, -1, 1, 1)
    b = beta.float().view(1, -1, 1, 1) - mean * a
    return a, b


def compile_get_code(fn, args, tag):
    cfn = torch.compile(fn, dynamic=False, options=OPTS)
    from torch._inductor.utils import run_and_get_code

    try:
        _, codes = run_and_get_code(cfn, *args)
        code = "\n\n# ======\n\n".join(codes)
        with open(os.path.join(ART, f"code_{tag}.py"), "w") as f:
            f.write(code)
        n_triton = code.count("@triton.jit")
        externs = [ln.strip() for ln in code.splitlines() if "extern_kernels." in ln and "=" in ln]
        emit({"code": tag, "triton_kernels": n_triton, "externs": externs[:8]})
    except Exception as e:  # noqa: BLE001
        emit({"code": tag, "error": repr(e)[:300]})
        cfn(*args)
    return cfn


# ---------------------------------------------------------------- cuDNN FE graphs
def nhwc(c, h, w):
    return [c * h * w, 1, w * c, c]


def build_graph(cin, cout, hw, prologue, bias, residual):
    g = cudnn.pygraph(
        io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32, handle=handle
    )
    T = {}
    T["X"] = g.tensor(name="X", dim=[1, cin, hw, hw], stride=nhwc(cin, hw, hw), data_type=BF16)
    T["W"] = g.tensor(name="W", dim=[cout, cin, 3, 3], stride=nhwc(cin, 3, 3), data_type=BF16)
    a = T["X"]
    if prologue:
        T["S"] = g.tensor(name="S", dim=[1, cin, 1, 1], stride=[cin, 1, cin, cin], data_type=F32)
        T["B"] = g.tensor(name="B", dim=[1, cin, 1, 1], stride=[cin, 1, cin, cin], data_type=F32)
        a = g.mul(a=a, b=T["S"], compute_data_type=F32, name="gn_scale")
        a = g.add(a=a, b=T["B"], compute_data_type=F32, name="gn_shift")
        try:
            a = g.swish(input=a, compute_data_type=F32, name="swish")
        except TypeError:
            a = g.swish(a)
        a.set_data_type(BF16)
    y = g.conv_fprop(
        image=a,
        weight=T["W"],
        padding=[1, 1],
        stride=[1, 1],
        dilation=[1, 1],
        compute_data_type=F32,
        name="conv",
    )
    if bias:
        T["bias"] = g.tensor(
            name="bias", dim=[1, cout, 1, 1], stride=[cout, 1, cout, cout], data_type=BF16
        )
        y = g.add(a=y, b=T["bias"], compute_data_type=F32, name="bias_add")
    if residual:
        T["R"] = g.tensor(name="R", dim=[1, cout, hw, hw], stride=nhwc(cout, hw, hw), data_type=BF16)
        y = g.add(a=y, b=T["R"], compute_data_type=F32, name="res_add")
    y.set_output(True).set_data_type(BF16)
    T["Y"] = y
    t0 = time.perf_counter()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    t1 = time.perf_counter()
    g.build_plans(cudnn.build_plan_policy.ALL)
    t2 = time.perf_counter()
    n = g.get_execution_plan_count()
    names, ws = [], []
    for i in range(n):
        try:
            names.append(g.get_plan_name_at_index(i))
        except Exception as e:  # noqa: BLE001
            names.append(f"?{type(e).__name__}")
        try:
            ws.append(int(g.get_workspace_size_plan_at_index(i)))
        except Exception:  # noqa: BLE001
            ws.append(int(g.get_workspace_size()))
    return g, T, {"graph_s": t1 - t0, "build_plans_s": t2 - t1, "n_plans": n, "names": names, "ws": ws}


class FEArm:
    def __init__(self, g, T, info, tensors):
        self.g, self.T, self.info = g, T, info
        self.vp = {T[k]: v for k, v in tensors.items() if k in T and k != "Y"}
        self.out = tensors["Y"]
        self.vp[T["Y"]] = self.out
        wsz = max(info["ws"] + [1])
        self.ws = torch.empty(wsz, dtype=torch.uint8, device=dev)
        self.ok = []
        for i in range(info["n_plans"]):
            try:
                self.run(i)
                torch.cuda.synchronize()
                self.ok.append(i)
            except Exception as e:  # noqa: BLE001
                emit({"plan_exec_fail": i, "err": repr(e)[:200]})

    def run(self, i):
        self.g.execute_plan_at_index(self.vp, self.ws, i, handle=handle)


# ---------------------------------------------------------------- timing
def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def bench(fns, rounds=ROUNDS, iters=ITERS, warm=5):
    res = {k: [] for k in fns}
    for f in fns.values():
        for _ in range(warm):
            f()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for k, f in fns.items():
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for s, e in evs:
                s.record()
                f()
                e.record()
            torch.cuda.synchronize()
            res[k] += [s.elapsed_time(e) for s, e in evs]
    out = {}
    for k, v in res.items():
        out[k] = {"med": pct(v, 0.5), "p10": pct(v, 0.1), "p90": pct(v, 0.9), "n": len(v)}
    return out


def hw2(t):
    return (t["p90"] - t["p10"]) / 2


summary = {"shapes": {}}
V_total = 0.0
V_delta_total = 0.0
unc_total = 0.0

for name, cin, cout, hw, n1, n2, nt in SHAPES:
    print(f"=== {name} {cin}->{cout} @{hw}^2", flush=True)
    x = (torch.randn(1, cin, hw, hw, device=dev) * 1.5 + 0.3).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    gamma = (1 + 0.2 * torch.randn(cin, device=dev)).to(torch.bfloat16)
    beta = (0.2 * torch.randn(cin, device=dev)).to(torch.bfloat16)
    w = (torch.randn(cout, cin, 3, 3, device=dev) / math.sqrt(9 * cin)).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    b = (0.1 * torch.randn(cout, device=dev)).to(torch.bfloat16)
    res = torch.randn(1, cout, hw, hw, device=dev).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    var_c, mean_c = torch.var_mean(x.float(), dim=(2, 3), correction=0)
    mean, rstd = comb_main(var_c, mean_c)
    a_s, b_s = comb_ab(var_c, mean_c, gamma, beta)
    a_s, b_s = a_s.contiguous(), b_s.contiguous()
    b4 = b.view(1, cout, 1, 1)
    h_bf = _apply(x, mean, rstd, gamma, beta).contiguous(memory_format=torch.channels_last)
    y_out = torch.empty(1, cout, hw, hw, device=dev, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)

    # compiled main arms
    t0 = time.perf_counter()
    cu_apply = compile_get_code(u_apply, (x, mean, rstd, gamma, beta), f"{name}_apply")
    cu_c1 = compile_get_code(u_conv1, (x, mean, rstd, gamma, beta, w, b), f"{name}_conv1")
    cu_c2 = compile_get_code(u_conv2, (x, mean, rstd, gamma, beta, w, b, res), f"{name}_conv2") if n2 else None
    cu_tail = compile_get_code(u_tail, (x, mean, rstd, gamma, beta, w), f"{name}_tail") if nt else None
    c_st = compile_get_code(c_stats, (y_out,), f"{name}_stats")
    c_cm = compile_get_code(comb_main, (var_c, mean_c), f"{name}_comb_main")
    c_ab = compile_get_code(comb_ab, (var_c, mean_c, gamma, beta), f"{name}_comb_ab")
    emit({"shape": name, "compile_s": time.perf_counter() - t0})

    # FE graphs
    tens = {"X": x, "W": w, "S": a_s, "B": b_s, "bias": b4, "R": res, "Y": y_out}
    variants = {
        "plain": (False, False, False),
        "P": (True, False, False),
        "PE_b": (True, True, False),
        "E_b": (False, True, False),
    }
    if n2:
        variants["PE_br"] = (True, True, True)
        variants["E_br"] = (False, True, True)
    arms = {}
    for vn, (p, bb, rr) in variants.items():
        try:
            g, T, info = build_graph(cin, cout, hw, p, bb, rr)
            xin = dict(tens)
            if not p:
                xin["X"] = h_bf
            xin["Y"] = torch.empty_like(y_out)
            arm = FEArm(g, T, info, xin)
            arms[vn] = arm
            emit({"shape": name, "graph": vn, **info, "ok_plans": arm.ok})
        except Exception as e:  # noqa: BLE001
            emit({"shape": name, "graph": vn, "build_error": repr(e)[:400]})

    # pieces: every plan index
    fns = {
        "U_apply": lambda: cu_apply(x, mean, rstd, gamma, beta),
        "aten_conv_nobias": lambda: F.conv2d(h_bf, w, None, padding=1),
        "aten_conv_bias": lambda: F.conv2d(h_bf, w, b, padding=1),
        "stats": lambda: c_st(y_out),
        "comb_main": lambda: c_cm(var_c, mean_c),
        "comb_ab": lambda: c_ab(var_c, mean_c, gamma, beta),
    }
    for vn, arm in arms.items():
        for i in arm.ok:
            fns[f"FE_{vn}_p{i}"] = (lambda arm=arm, i=i: arm.run(i))
    t_pieces = bench(fns)
    emit({"shape": name, "pieces": t_pieces})

    def best(vn):
        if vn not in arms:
            return None
        ks = [k for k in t_pieces if k.startswith(f"FE_{vn}_p")]
        if not ks:
            return None
        k = min(ks, key=lambda k: t_pieces[k]["med"])
        return int(k.rsplit("_p", 1)[1])

    # chains through the next consumer
    chains = {"U_conv1": lambda: cu_c1(x, mean, rstd, gamma, beta, w, b)}
    bi = best("PE_b")
    if bi is not None:
        A = arms["PE_b"]
        chains["F_conv1"] = lambda: (A.run(bi), c_st(A.out))
    if n2:
        chains["U_conv2"] = lambda: cu_c2(x, mean, rstd, gamma, beta, w, b, res)
        bj = best("PE_br")
        if bj is not None:
            A2 = arms["PE_br"]
            chains["F_conv2"] = lambda: (A2.run(bj), c_st(A2.out))
    if nt:
        chains["U_tail"] = lambda: cu_tail(x, mean, rstd, gamma, beta, w)
        bp = best("P")
        if bp is not None:
            chains["F_tail"] = lambda: arms["P"].run(bp)
    t_chain = bench(chains)
    emit({"shape": name, "chains": t_chain, "best_plans": {v: best(v) for v in arms}})

    # numerics (fp64 reference of the same math, x' unrounded) + repeat bitwise
    num = {}
    with torch.no_grad():
        x64 = x.double()
        h64 = x64 * a_s.double() + b_s.double()
        h64 = h64 * torch.sigmoid(h64)
        c64 = F.conv2d(h64, w.double(), b.double(), padding=1)
        for vn, rr in (("PE_b", False), ("PE_br", True)):
            if vn not in arms or best(vn) is None:
                continue
            A = arms[vn]
            ref = c64 + (res.double() if rr else 0)
            A.run(best(vn))
            torch.cuda.synchronize()
            o0 = A.out.clone()
            rep = True
            for _ in range(3):
                A.run(best(vn))
                torch.cuda.synchronize()
                rep = rep and torch.equal(o0, A.out)
            hm = _apply(x, mean, rstd, gamma, beta)
            um = F.conv2d(hm, w, b, padding=1)
            if rr:
                um = res + um
            num[vn] = {
                "F_vs_U_max": (o0.double() - um.double()).abs().max().item(),
                "F_vs_64_max": (o0.double() - ref).abs().max().item(),
                "U_vs_64_max": (um.double() - ref).abs().max().item(),
                "F_vs_64_mean": (o0.double() - ref).abs().mean().item(),
                "U_vs_64_mean": (um.double() - ref).abs().mean().item(),
                "repeat_bitwise": rep,
            }
        del x64, h64, c64
    emit({"shape": name, "numerics": num})

    # V for this shape
    D98 = t_pieces["comb_ab"]["med"]
    D98_delta = max(0.0, t_pieces["comb_ab"]["med"] - t_pieces["comb_main"]["med"])
    rows = {}
    for var_name, n, uk, fk, has_d in (("conv1", n1, "U_conv1", "F_conv1", True), ("conv2", n2, "U_conv2", "F_conv2", True), ("tail", nt, "U_tail", "F_tail", True)):
        if not n or uk not in t_chain or fk not in t_chain:
            continue
        U, Fc = t_chain[uk], t_chain[fk]
        v = U["med"] - Fc["med"] - D98
        vd = U["med"] - Fc["med"] - D98_delta
        unc = hw2(U) + hw2(Fc) + hw2(t_pieces["comb_ab"])
        rows[var_name] = {"n": n, "U": U["med"], "F": Fc["med"], "D98": D98, "D98_delta": D98_delta, "U_minus_F_minus_D98": v, "with_delta": vd, "unc_each": unc}
        V_total += n * v
        V_delta_total += n * vd
        unc_total += n * unc
    emit({"shape": name, "V_rows": rows})
    summary["shapes"][name] = rows
    del arms, fns, chains
    torch.cuda.empty_cache()

summary.update({"V98_ms": V_total, "V98_ms_D98delta": V_delta_total, "unc_ms": unc_total})
emit({"summary": summary})
print(json.dumps({"V98_ms": V_total, "V98_ms_D98delta": V_delta_total, "unc_ms": unc_total}))
