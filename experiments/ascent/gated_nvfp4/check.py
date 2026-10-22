# Correctness check for quant-linear idea 11 (branch 23). Runs in the branch
# tree. Everything is compared bitwise against main's two-kernel path, whose
# code is unchanged in this tree (rms_norm_gated_token_major,
# fused_sigmoid_mul_inplace, fp4_quantize, prefill_nvfp4_apply).
#  1. Part A op: rms_norm_gated_token_major_nvfp4 vs norm(BF16) + fp4_quantize:
#     FP4 bytes and the full swizzled SF buffer (allocator poisoned first, so
#     unzeroed SF padding would show).
#  2. Part B op: fused_sigmoid_mul_nvfp4 vs fused_sigmoid_mul_inplace +
#     fp4_quantize; attention_output must stay unmodified.
#  3. Module chains (_postprocess_gdn_output + out_proj, apply_output_gate +
#     o_proj): fused path vs main's path (flags patched off) bitwise, kernel
#     names from torch.profiler prove which path ran.
#  4. Repeated execution: 10 warmup + 50 calls (eager and CUDA-graph replays at
#     M=32) each compared with call 1 and the reference.
#  5. Kill switches in a fresh process (env =0) reproduce main's path.
#  6. An Fp4QuantizedTensor at rows <= 16 is refused (never reaches Marlin).
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

from tensorrt_llm._torch.attention import qk_norm_attention  # noqa: E402
from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import (  # noqa: E402
    fused_sigmoid_mul_inplace, fused_sigmoid_mul_nvfp4)
from tensorrt_llm._torch.attention.qk_norm_attention import QKNormRoPEAttention  # noqa: E402
from tensorrt_llm._torch.modules import fp8_prefill_nvfp4 as fpn  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer  # noqa: E402
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm as RMSNormGated  # noqa: E402
from tensorrt_llm._torch.modules.mamba.layernorm_gated import (  # noqa: E402
    rms_norm_gated_token_major, rms_norm_gated_token_major_nvfp4)
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

HEADS, HD, HID, QKVZ = 48, 128, 5120, 16384
VD = HEADS * HD
CONV = QKVZ - VD
AH, AD = 24, 256
EPS = 1e-6
dev = torch.device("cuda")
WARM, ITERS = 10, 50
fails = []


def log(*a):
    print(*a, flush=True)


def poison(ref=None):
    # Refill freed blocks of the output sizes (and a large one) with 0xAB so
    # the next same-size allocations start dirty.
    bufs = [torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)]
    if ref is not None:
        for t in ref:
            for _ in range(4):
                bufs.append(torch.empty(t.numel(), dtype=torch.uint8, device=dev))
    for b in bufs:
        b.fill_(0xAB)
    del bufs


def gdn_inputs(m, seed, scale=0.3, zscale=1.5):
    g = torch.Generator(device=dev).manual_seed(seed * 1009 + m)
    a = (torch.randn(m, HEADS, HD, generator=g, device=dev) * scale).to(torch.bfloat16)
    a[::7, 5] = 0.0
    if m > 3:
        a[3, 7, :8] = 30.0  # outliers in one block
    qkvz = (torch.randn(m, QKVZ, generator=g, device=dev) * zscale).to(torch.bfloat16)
    z = qkvz[:, CONV:].view(m, HEADS, HD)
    return a, qkvz, z


def norm_weight(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (1.0 + 0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)


def ref_a(a, z, w, gs, act="silu"):
    y = rms_norm_gated_token_major(a.reshape(-1, HD), z, w, EPS, None, act)
    return torch.ops.trtllm.fp4_quantize(y.view(z.shape[0], VD), gs, 16, False)


def eq_pair(p, q):
    return torch.equal(p[0], q[0]) and torch.equal(p[1], q[1])


def ndiff(p, q):
    return int((p[0] != q[0]).sum().item()) + int((p[1] != q[1]).sum().item())


# ---------------------------------------------------------------- 1. Part A op
log("== Part A op ==")
w = norm_weight(5)
n_a = 0
for m in [17, 18, 31, 32, 33, 64, 100, 127, 128, 129, 255, 2016, 2048, 2079, 4096, 8192, 8223]:
    for s in [0.04, 0.004, 1.0]:
        for act in ["silu", "sigmoid"]:
            a, qkvz, z = gdn_inputs(m, 11)
            gs = torch.tensor([6.0 / s], dtype=torch.float32, device=dev)
            r = ref_a(a, z, w, gs, act)
            poison(r)
            c = rms_norm_gated_token_major_nvfp4(a.reshape(-1, HD), z, w, EPS, gs, act)
            ok = eq_pair(c, r) and c[0].shape == r[0].shape and c[1].shape == r[1].shape
            n_a += 1
            if not ok:
                fails.append("A op m=%d s=%g act=%s diff=%d" % (m, s, act, ndiff(c, r)))
                log("FAIL A op", m, s, act, ndiff(c, r))
log("Part A op cases", n_a, "fails", len(fails))

# repeated calls, 10 + 50
for m in [17, 31, 32, 64, 2048, 2079, 8223]:
    a, qkvz, z = gdn_inputs(m, 12)
    gs = torch.tensor([6.0 / 0.04], dtype=torch.float32, device=dev)
    r = ref_a(a, z, w, gs)
    first = None
    bad = 0
    for it in range(WARM + ITERS):
        poison(r)
        c = rms_norm_gated_token_major_nvfp4(a.reshape(-1, HD), z, w, EPS, gs)
        c = (c[0].clone(), c[1].clone())
        if first is None:
            first = c
        if not (eq_pair(c, first) and eq_pair(c, r)):
            bad += 1
    if bad:
        fails.append("A repeat m=%d bad=%d" % (m, bad))
    log("Part A repeat m=%d calls=%d bad=%d" % (m, WARM + ITERS, bad))


# ---------------------------------------------------------------- 2. Part B op
def attn_inputs(m, seed, three_d=True):
    g = torch.Generator(device=dev).manual_seed(seed * 1013 + m)
    att = (torch.randn(m, VD, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    att[::5, 256:512] = 0.0
    if m > 2:
        att[2, 1000:1010] = 40.0
    qg = (torch.randn(m, 2 * VD, generator=g, device=dev) * 2.0).to(torch.bfloat16)
    gate = qg[:, VD:].view(m, AH, AD) if three_d else qg[:, VD:].contiguous()
    return att, qg, gate


def ref_b(att, gate, gs):
    r = att.clone()
    fused_sigmoid_mul_inplace(r, gate)
    return torch.ops.trtllm.fp4_quantize(r, gs, 16, False)


log("== Part B op ==")
nb0 = len(fails)
n_b = 0
for m in [1, 16, 17, 32, 33, 127, 128, 129, 2016, 2048, 4096, 8192]:
    for s in [0.01, 0.001, 1.0]:
        for three_d in [True, False]:
            att, qg, gate = attn_inputs(m, 21, three_d)
            att0 = att.clone()
            gs = torch.tensor([6.0 / s], dtype=torch.float32, device=dev)
            r = ref_b(att, gate, gs)
            poison(r)
            c = fused_sigmoid_mul_nvfp4(att, gate, gs)
            ok = eq_pair(c, r) and torch.equal(att, att0)
            n_b += 1
            if not ok:
                fails.append("B op m=%d s=%g 3d=%s diff=%d" % (m, s, three_d, ndiff(c, r)))
                log("FAIL B op", m, s, three_d, ndiff(c, r))
log("Part B op cases", n_b, "fails", len(fails) - nb0)
for m in [17, 32, 33, 2016, 2048, 8192]:
    att, qg, gate = attn_inputs(m, 22)
    gs = torch.tensor([6.0 / 0.01], dtype=torch.float32, device=dev)
    r = ref_b(att, gate, gs)
    first = None
    bad = 0
    for it in range(WARM + ITERS):
        poison(r)
        c = fused_sigmoid_mul_nvfp4(att, gate, gs)
        c = (c[0].clone(), c[1].clone())
        if first is None:
            first = c
        if not (eq_pair(c, first) and eq_pair(c, r)):
            bad += 1
    if bad:
        fails.append("B repeat m=%d bad=%d" % (m, bad))
    log("Part B repeat m=%d calls=%d bad=%d" % (m, WARM + ITERS, bad))


# ---------------------------------------------------------- 3. module chains
def make_lin(seed, env, in_scale):
    g = torch.Generator(device=dev).manual_seed(seed)
    wt = torch.randn(HID, VD, generator=g, device=dev) * 0.02
    wt = wt * (torch.rand(HID, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (wt.abs().amax() / 448.0).float()
    w8 = (wt / s).to(torch.float8_e4m3fn)
    lin = Linear(VD, HID, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(in_scale, device=dev))])
    lin.post_load_weights()
    assert fpn.maybe_build_prefill_nvfp4_copy(lin, env)
    return lin


class G:
    def __init__(self, seed):
        self.out_proj = make_lin(seed, fpn.ENV_GDN_OUT, 0.04)
        self.norm = RMSNormGated(HD, eps=EPS, group_size=None, norm_before_gate=True,
                                 device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            self.norm.weight.copy_(norm_weight(seed + 1))
        self.norm.fp8_scale = self.out_proj.quant_method.get_static_input_scale(
            self.out_proj).detach()
        self.head_v_dim = HD
        self.value_dim_per_tp = VD

    def run(self, a, z):
        return gdn_mixer.Qwen3NextGatedDeltaNet._postprocess_gdn_output(self, a, z)


class A:
    def __init__(self, seed):
        self.o_proj = make_lin(seed, fpn.ENV_ATTN_O, 0.01)
        self._fuse_qk_norm_rope_gate = True

    def run(self, work, att, gate):
        work.copy_(att)
        return self.o_proj(QKNormRoPEAttention.apply_output_gate(self, work, gate))


def set_flags(on):
    gdn_mixer.GDN_GATED_NORM_NVFP4_ENABLED = on
    qk_norm_attention.ATTN_GATE_NVFP4_ENABLED = on


def kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()
             if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(quant=sum("quantize_with_block_size" in n for n in names),
                norm_nvfp4=sum("rms_norm_gated_nvfp4" in n for n in names),
                norm_bf16=sum(("rms_norm_gated" in n and "nvfp4" not in n) for n in names),
                gate_nvfp4=sum("sigmoid_mul_nvfp4" in n for n in names),
                gate_bf16=sum(("sigmoid_mul" in n and "nvfp4" not in n) for n in names),
                total=len(names))


KILL = len(sys.argv) > 1 and sys.argv[1] == "--kill"
g_layer = G(3000)
a_layer = A(4000)
GM = [8, 16, 17, 31, 32, 33, 64, 2048, 2079, 8223]
AM = [8, 16, 17, 32, 33, 2016, 2048, 8192]
hashes = dict()
log("== module chains (kill=%s) ==" % KILL)
for m in GM:
    a, qkvz, z = gdn_inputs(m, 31)
    if not KILL:
        set_flags(False)
        main_out = g_layer.run(a, z).clone()
        km = kernels(lambda: g_layer.run(a, z))
        set_flags(True)
        cand_out = g_layer.run(a, z).clone()
        kc = kernels(lambda: g_layer.run(a, z))
        ok = torch.equal(main_out, cand_out)
        fused_expected = m > 16
        path_ok = (kc["norm_nvfp4"] == (1 if fused_expected else 0)
                   and kc["quant"] == 0 if fused_expected else kc == km)
        if not ok or not path_ok:
            fails.append("G module m=%d equal=%s path_ok=%s" % (m, ok, path_ok))
        log("G m=%d bitwise=%s main_kernels=%s cand_kernels=%s" % (
            m, ok, json.dumps(km), json.dumps(kc)))
        out = cand_out
    else:
        out = g_layer.run(a, z).clone()
        k = kernels(lambda: g_layer.run(a, z))
        if k["norm_nvfp4"] != 0:
            fails.append("kill G m=%d fused ran" % m)
        log("kill G m=%d kernels=%s" % (m, json.dumps(k)))
    hashes["G%d" % m] = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
for m in AM:
    att, qg, gate = attn_inputs(m, 41)
    work = torch.empty_like(att)
    if not KILL:
        set_flags(False)
        main_out = a_layer.run(work, att, gate).clone()
        km = kernels(lambda: a_layer.run(work, att, gate))
        set_flags(True)
        cand_out = a_layer.run(work, att, gate).clone()
        kc = kernels(lambda: a_layer.run(work, att, gate))
        ok = torch.equal(main_out, cand_out)
        fused_expected = m > 16
        path_ok = (kc["gate_nvfp4"] == (1 if fused_expected else 0)
                   and kc["quant"] == 0 if fused_expected else kc == km)
        if not ok or not path_ok:
            fails.append("A module m=%d equal=%s path_ok=%s" % (m, ok, path_ok))
        log("A m=%d bitwise=%s main_kernels=%s cand_kernels=%s" % (
            m, ok, json.dumps(km), json.dumps(kc)))
        out = main_out
    else:
        out = a_layer.run(work, att, gate).clone()
        k = kernels(lambda: a_layer.run(work, att, gate))
        if k["gate_nvfp4"] != 0:
            fails.append("kill A m=%d fused ran" % m)
        log("kill A m=%d kernels=%s" % (m, json.dumps(k)))
    hashes["A%d" % m] = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()

if KILL:
    print(json.dumps(dict(hashes=hashes, fails=fails)))
    sys.exit(0)

# ---------------------------------------------- 4. repeated, CUDA graph M=32
log("== repeated execution, M=32 CUDA graph and eager ==")
set_flags(False)
m = 32
a, qkvz, z = gdn_inputs(m, 51)
att, qg, gate = attn_inputs(m, 52)
work = torch.empty_like(att)
ref_g = g_layer.run(a, z).clone()
ref_at = a_layer.run(work, att, gate).clone()
set_flags(True)
s0 = torch.cuda.Stream()
s0.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s0):
    for _ in range(3):
        g_layer.run(a, z)
        a_layer.run(work, att, gate)
torch.cuda.current_stream().wait_stream(s0)
torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    go = g_layer.run(a, z)
    ao = a_layer.run(work, att, gate)
kg = kernels(gr.replay)
log("M=32 graph replay kernels (1 GDN + 1 attn chain): %s" % json.dumps(kg))
if kg["quant"] != 0:
    fails.append("graph M=32 still launches quantize_with_block_size")
first = None
bad = 0
for it in range(WARM + ITERS):
    gr.replay()
    torch.cuda.synchronize()
    cur = (go.clone(), ao.clone())
    if first is None:
        first = cur
    if not (torch.equal(cur[0], first[0]) and torch.equal(cur[1], first[1])
            and torch.equal(cur[0], ref_g) and torch.equal(cur[1], ref_at)):
        bad += 1
if bad:
    fails.append("graph repeat bad=%d" % bad)
log("graph M=32 replays=%d bad=%d" % (WARM + ITERS, bad))
del gr
for m in [17, 2048, 8223]:
    a, qkvz, z = gdn_inputs(m, 61)
    set_flags(False)
    rg = g_layer.run(a, z).clone()
    set_flags(True)
    bad = 0
    first = None
    for it in range(WARM + ITERS):
        o = g_layer.run(a, z).clone()
        if first is None:
            first = o
        if not (torch.equal(o, first) and torch.equal(o, rg)):
            bad += 1
    if bad:
        fails.append("G eager repeat m=%d bad=%d" % (m, bad))
    log("G eager repeat m=%d calls=%d bad=%d" % (m, WARM + ITERS, bad))
for m in [17, 2048, 8192]:
    att, qg, gate = attn_inputs(m, 62)
    work = torch.empty_like(att)
    set_flags(False)
    ra = a_layer.run(work, att, gate).clone()
    set_flags(True)
    bad = 0
    first = None
    for it in range(WARM + ITERS):
        o = a_layer.run(work, att, gate).clone()
        if first is None:
            first = o
        if not (torch.equal(o, first) and torch.equal(o, ra)):
            bad += 1
    if bad:
        fails.append("A eager repeat m=%d bad=%d" % (m, bad))
    log("A eager repeat m=%d calls=%d bad=%d" % (m, WARM + ITERS, bad))

# ------------------------------------------- 6. Fp4QuantizedTensor at rows<=16
lin = g_layer.out_proj
fake = Fp4QuantizedTensor(torch.zeros(8, VD // 2, dtype=torch.uint8, device=dev),
                          torch.zeros(128 * 384, dtype=torch.uint8, device=dev))
refused = False
try:
    lin.quant_method.apply(lin, fake, None)
except AssertionError:
    refused = True
if not refused:
    fails.append("Fp4QuantizedTensor at rows 8 was not refused")
log("Fp4QuantizedTensor at rows<=16 refused:", refused)

# ---------------------------- 7. routing helper honours a per-module decode cap
# (as linear.fp8_nvfp4_decode_max_rows on main 7f09143a: Marlin up to the cap).
import tensorrt_llm._torch.modules.linear as _linear_mod  # noqa: E402

had = hasattr(_linear_mod, "fp8_nvfp4_decode_max_rows")
saved = getattr(_linear_mod, "fp8_nvfp4_decode_max_rows", None)
_linear_mod.fp8_nvfp4_decode_max_rows = lambda module: 32
fpn._decode_max_rows_fn = None
lin.__dict__["_nvfp4_decode_weight"] = torch.empty(1, device=dev)
route = dict((r, fpn.uses_prefill_nvfp4_copy(lin, r) is not None) for r in [16, 17, 32, 33, 2048])
del lin.__dict__["_nvfp4_decode_weight"]
route_nodec = dict((r, fpn.uses_prefill_nvfp4_copy(lin, r) is not None) for r in [16, 17, 32, 33])
if had:
    _linear_mod.fp8_nvfp4_decode_max_rows = saved
else:
    del _linear_mod.fp8_nvfp4_decode_max_rows
fpn._decode_max_rows_fn = None
exp = {16: False, 17: False, 32: False, 33: True, 2048: True}
exp_nodec = {16: False, 17: True, 32: True, 33: True}
if route != exp or route_nodec != exp_nodec:
    fails.append("routing helper with cap 32: %s / %s" % (route, route_nodec))
log("helper with decode cap 32 (decode copy):", route, "(no decode copy):", route_nodec)

# ------------------------------------------------------ 5. kill switch process
env = dict(os.environ)
env["TRTLLM_GDN_GATED_NORM_NVFP4"] = "0"
env["TRTLLM_ATTN_GATE_NVFP4"] = "0"
p = subprocess.run([sys.executable, os.path.abspath(__file__), "--kill"], env=env,
                   capture_output=True, text=True, timeout=1200)
kill_ok = False
try:
    kres = json.loads(p.stdout.strip().splitlines()[-1])
    # main-path hashes for comparison: G uses cand_out (== main), A uses main_out
    kill_ok = kres["hashes"] == hashes and not kres["fails"]
    if not kill_ok:
        log("kill mismatch", json.dumps(kres)[:2000])
except Exception as e:  # noqa: BLE001
    log("kill subprocess failed", repr(e), p.stdout[-2000:], p.stderr[-4000:])
for line in p.stdout.strip().splitlines()[:-1]:
    log("  [kill] " + line)
if not kill_ok:
    fails.append("kill switch process does not reproduce main")
log("kill switch reproduces main:", kill_ok)

print(json.dumps(dict(passed=not fails, fails=fails[:50], part_a_op_cases=n_a, part_b_op_cases=n_b,
                     compared_against="main's unchanged two-kernel path (norm/gate BF16 + "
                     "fp4_quantize + nvfp4_gemm) in this tree; bitwise",
                     warmup=WARM, iters=ITERS)))
