# Idea 198 / branch 118: untimed correctness for the fused in_proj_qkvz GEMV + conv1d decode path,
# run through the REAL Qwen3NextGatedDeltaNet.forward (stand-in object as in chain_pair.py) inside
# CUDA graphs (multi-stream on, PDL on). Reference = the same layers with the kill switch
# (gdn_mixer._GDN_GEMV_CONV = False), i.e. main's [GEMV -> causal_conv1d_update] path, plus an
# independent eager reference for the conv: main's ops called directly.
# Items:
#  A) 120-step stateful in-graph window per M in 1..3 (outputs feed nothing back; each step
#     replays the graph on fresh x written into the static input buffer): after EVERY step the
#     fused graph's outputs (T1 out, qkvz output z part), conv pools and ssm pools equal the
#     kill-switch graph's, bitwise. PAD rows (-1) included at M=2/3; untouched slots stay pristine.
#  B) reset-based repeated replays: 10 warm + 50 replays from pristine state each equal replay 1.
#  C) rows >= 4 (Marlin) and mixed (num_contexts > 0 -> forward_extend is not exercised here, but
#     the routing predicate must return None) steps: the fused route is not taken; the kill switch
#     makes M=1..3 take main's path. Checked by the routing predicate and by kernel names.
#  D) decode graph kernel count per layer at M=1..3: fused = kill-switch - 1 (48 per 48 layers).
import json
import os
import sys
import types

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from torch import nn  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer as G  # noqa: E402
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update  # noqa: E402
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream  # noqa: E402
from tensorrt_llm._torch.utils import EventType  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402


def log(*a):
    print(*a, flush=True)


dev = torch.device("cuda")
HID, QKVZ, CONV = 5120, 16384, 10240
H, HV, DK, DV = 16, 48, 128, 128
L = int(os.environ.get("B118_CHECK_LAYERS", "4"))
SLOTS = 6
STEPS = 120
EPS = 1e-6


class FakeGDN(G.Qwen3NextGatedDeltaNet):
    def __init__(self, li, qkvz, ba, conv_w, a_log, dt_bias, aux):
        nn.Module.__init__(self)
        self.in_proj_qkvz = qkvz
        self.in_proj_ba = ba
        self.conv1d = types.SimpleNamespace(weight=conv_w, bias=None)
        self.A_log = a_log
        self.dt_bias = dt_bias
        self.aux_stream = aux
        self.event_dict = dict([(EventType.Main, torch.cuda.Event()),
                                (EventType.Attention, torch.cuda.Event())])
        self.layer_idx = li
        self.layer_idx_str = str(li)
        self.register_to_config = False
        self.activation = "silu"
        self.attn_tp_size = 1
        self.num_k_heads, self.num_v_heads = H, HV
        self.num_k_heads_per_tp, self.num_v_heads_per_tp = H, HV
        self.head_k_dim, self.head_v_dim = DK, DV
        self.key_dim = self.key_dim_per_tp = H * DK
        self.value_dim = self.value_dim_per_tp = HV * DV
        self.conv_dim = self.conv_dim_per_tp = CONV
        self.last_z = None

    def _postprocess_gdn_output(self, attn_out, z, all_reduce_params=None):
        # Return both so the check sees the z columns of the qkvz output too.
        return attn_out, z


class FakeKVM:
    def __init__(self, conv, ssm):
        self.conv = conv
        self.ssm = ssm

    def mamba_layer_cache(self, layer_idx):
        return types.SimpleNamespace(conv=self.conv[layer_idx], temporal=self.ssm[layer_idx])

    def is_speculative(self):
        return False


def make_qkvz(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(QKVZ, HID, generator=gen, device=dev) * 0.02
    w = w * (torch.rand(QKVZ, 1, generator=gen, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    lin = Linear(HID, QKVZ, bias=False, dtype=torch.bfloat16,
                 tensor_parallel_mode=TensorParallelMode.COLUMN,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8), reduce_output=False).to(dev)
    lin._nvfp4_decode_copy = True
    lin.load_weights([dict(weight=w8, weight_scale=s.reshape(()),
                           input_scale=torch.tensor(0.04, device=dev))])
    lin.post_load_weights()
    return lin


def make_ba(seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    wb = (torch.randn(2 * HV, HID, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    ba = Linear(HID, 2 * HV, bias=False, dtype=torch.bfloat16,
                tensor_parallel_mode=TensorParallelMode.COLUMN, reduce_output=False).to(dev)
    ba.load_weights([dict(weight=wb)])
    ba.post_load_weights()
    return ba


AUX = torch.cuda.Stream()
SIDE = torch.cuda.Stream()
gen = torch.Generator(device=dev).manual_seed(198)
LAYERS = []
CONV0 = []
SSM0 = []
for li in range(L):
    cw = (torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, generator=gen, device=dev) * 15.0 + 1.0).float()
    dtb = (torch.randn(HV, generator=gen, device=dev) * 0.5).float()
    LAYERS.append(FakeGDN(li, make_qkvz(3000 + li), make_ba(4000 + li), cw, a_log, dtb, AUX))
    # Strided pool like the V2 manager's coalesced pages: slot stride > CONV * 3.
    raw = torch.randn(SLOTS, CONV * 3 + 96, generator=gen, device=dev).to(torch.bfloat16)
    CONV0.append(raw)
    SSM0.append((torch.randn(SLOTS, HV, DV, DK, generator=gen, device=dev) * 0.1).to(torch.bfloat16))
WN = (0.1 * torch.randn(HID, generator=gen, device=dev)).to(torch.bfloat16)


class Pools:
    def __init__(self):
        self.raw = [t.clone() for t in CONV0]
        self.conv = [r[:, :CONV * 3].view(SLOTS, CONV, 3) for r in self.raw]
        self.ssm = [t.clone() for t in SSM0]
        self.kvm = FakeKVM(self.conv, self.ssm)

    def reset(self):
        for a, b in zip(self.raw, CONV0):
            a.copy_(b)
        for a, b in zip(self.ssm, SSM0):
            a.copy_(b)

    def snap(self):
        return [t.clone() for t in self.raw] + [t.clone() for t in self.ssm]


def metadata(m, idx_list, kvm, num_contexts=0):
    idx = torch.tensor(idx_list, dtype=torch.int32, device=dev)
    amd = types.SimpleNamespace(num_contexts=num_contexts, seq_lens=torch.ones(m, dtype=torch.int32),
                                num_ctx_tokens=0, num_tokens=m, kv_cache_manager=kvm)
    mmd = types.SimpleNamespace(has_initial_states=torch.ones(m, dtype=torch.bool, device=dev),
                                state_indices=idx,
                                query_start_loc=torch.arange(m + 1, dtype=torch.int32, device=dev),
                                query_start_loc_long=torch.arange(m + 1, dtype=torch.long, device=dev),
                                replay_work_items=None, replay_n_writes=None)
    return amd, mmd


def chain(xs, rs, amd, mmd):
    outs = []
    for li in range(L):
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(xs[li], rs[li], WN, EPS)
        o, z = LAYERS[li].forward(xs[li], amd, mmd)
        outs += [o, z]
    return outs


def capture(fn, fused, pools):
    G._GDN_GEMV_CONV = fused
    with with_multi_stream(True):
        SIDE.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(SIDE):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(SIDE)
        torch.cuda.synchronize()
        pools.reset()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn()
    torch.cuda.synchronize()
    G._GDN_GEMV_CONV = True
    return gr, out


def same(a, b):
    return all(torch.equal(u.view(torch.int16), v.view(torch.int16)) for u, v in zip(a, b))


def kernel_names(fn, fused):
    from torch.profiler import ProfilerActivity, profile
    G._GDN_GEMV_CONV = fused
    torch.cuda.synchronize()
    with with_multi_stream(True):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    G._GDN_GEMV_CONV = True
    names = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            names.append(e.name)
    return names


PATTERNS = {1: [5], 2: [2, -1], 3: [4, 0, -1]}
report = dict()
ok_all = True
for m in (1, 2, 3):
    idx_list = PATTERNS[m]
    pf, pr = Pools(), Pools()
    amd_f, mmd_f = metadata(m, idx_list, pf.kvm)
    amd_r, mmd_r = metadata(m, idx_list, pr.kvm)
    # Routing predicate
    x_probe = torch.zeros(m, HID, dtype=torch.bfloat16, device=dev)
    route = LAYERS[0]._gemv_conv_operands(x_probe, amd_f, mmd_f, None) is not None
    per = 2 * m * HID
    xin = torch.zeros(L * per, dtype=torch.bfloat16, device=dev)
    xin_r = torch.zeros(L * per, dtype=torch.bfloat16, device=dev)
    xs = [xin[i * per: i * per + m * HID].view(m, HID) for i in range(L)]
    rs = [xin[i * per + m * HID: (i + 1) * per].view(m, HID) for i in range(L)]
    xs_r = [xin_r[i * per: i * per + m * HID].view(m, HID) for i in range(L)]
    rs_r = [xin_r[i * per + m * HID: (i + 1) * per].view(m, HID) for i in range(L)]
    gf, of = capture(lambda: chain(xs, rs, amd_f, mmd_f), True, pf)
    gr_, orf = capture(lambda: chain(xs_r, rs_r, amd_r, mmd_r), False, pr)

    # D) kernel counts (eager, one chain each, same inputs as graphs)
    kn_f = kernel_names(lambda: chain(xs, rs, amd_f, mmd_f), True)
    kn_r = kernel_names(lambda: chain(xs_r, rs_r, amd_r, mmd_r), False)
    conv_f = sum(1 for n in kn_f if "causal_conv1d" in n)
    conv_r = sum(1 for n in kn_r if "causal_conv1d" in n)
    fused_k = sum(1 for n in kn_f if "GemvConv" in n)
    count_ok = (len(kn_r) - len(kn_f) == L) and conv_f == 0 and conv_r == L and fused_k == L

    # A) 120-step stateful window, fresh inputs each step, compare after every step.
    pf.reset()
    pr.reset()
    g2 = torch.Generator(device=dev).manual_seed(500 + m)
    step_fail = []
    for step in range(STEPS):
        new = (torch.randn(L * per, generator=g2, device=dev) * 0.5).to(torch.bfloat16)
        xin.copy_(new)
        xin_r.copy_(new)
        gf.replay()
        gr_.replay()
        torch.cuda.synchronize()
        if not (same(of, orf) and same(pf.snap(), pr.snap())):
            step_fail.append(step)
    used = set(v for v in idx_list if v >= 0)
    untouched = all(torch.equal(pf.raw[li][s].view(torch.int16), CONV0[li][s].view(torch.int16))
                    for li in range(L) for s in range(SLOTS) if s not in used)
    pad_bytes = all(torch.equal(pf.raw[li][:, CONV * 3:].view(torch.int16),
                                CONV0[li][:, CONV * 3:].view(torch.int16)) for li in range(L))
    # Independent eager reference for one step from pristine state: main's ops directly.
    pf.reset()
    first = (torch.randn(L * per, generator=g2, device=dev) * 0.5).to(torch.bfloat16)
    xin.copy_(first)
    gf.replay()
    torch.cuda.synchronize()
    fused_y = [t.clone() for t in of]
    fused_pool = pf.snap()
    pe = Pools()
    xe = first.clone()
    eager_ok = True
    for li in range(L):
        x = xe[li * per: li * per + m * HID].view(m, HID)
        r = xe[li * per + m * HID: (li + 1) * per].view(m, HID)
        torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, WN, EPS)
        gv = LAYERS[li].in_proj_qkvz.__dict__["_nvfp4_decode_gemv"]
        y = torch.ops.trtllm.nvfp4_w4a16_gemv(x.contiguous(), *gv)
        mixed = causal_conv1d_update(y[:, :CONV], pe.conv[li], LAYERS[li].conv1d.weight, None, "silu",
                                     conv_state_indices=torch.tensor(idx_list, dtype=torch.int32, device=dev))
        z = y[:, CONV:].reshape(m, HV, DV)
        eager_ok &= torch.equal(z.view(torch.int16), fused_y[2 * li + 1].reshape(m, HV, DV).view(torch.int16))
        eager_ok &= torch.equal(pe.raw[li].view(torch.int16), fused_pool[li].view(torch.int16))
        del mixed
    torch.cuda.synchronize()

    # B) 10 warm + 50 replays from pristine state, each equal replay 1.
    xin.copy_(first)
    rep_fail = []
    ref1 = None
    for it in range(60):
        pf.reset()
        xin.copy_(first)
        gf.replay()
        torch.cuda.synchronize()
        cur = [t.clone() for t in of] + pf.snap()
        if ref1 is None:
            ref1 = cur
        elif not same(cur, ref1):
            rep_fail.append(it)
    ok = (route and count_ok and not step_fail and untouched and pad_bytes and eager_ok and not rep_fail)
    ok_all = ok_all and ok
    report["M%d" % m] = dict(slots=idx_list, fused_route=route, kernels_fused=len(kn_f), kernels_killswitch=len(kn_r),
                             conv_kernels_fused=conv_f, conv_kernels_killswitch=conv_r,
                             gemv_conv_kernels=fused_k, count_ok=count_ok,
                             stateful_steps=STEPS, stateful_fail_steps=step_fail, untouched_slots=untouched,
                             pool_padding_untouched=pad_bytes, eager_main_ops_equal=eager_ok,
                             replays=60, replay_fail=rep_fail, ok=ok)
    log("M=%d %s" % (m, json.dumps(report["M%d" % m])))
    del gf, gr_
    torch.cuda.synchronize()

# C) routing: rows >= 4, mixed step, spec verify, kill switch -> no fused route.
pc = Pools()
amd4, mmd4 = metadata(4, [0, 1, 2, 3], pc.kvm)
r4 = LAYERS[0]._gemv_conv_operands(torch.zeros(4, HID, dtype=torch.bfloat16, device=dev), amd4, mmd4, None)
amdm, mmdm = metadata(2, [0, 1], pc.kvm, num_contexts=1)
rmix = LAYERS[0]._gemv_conv_operands(torch.zeros(2, HID, dtype=torch.bfloat16, device=dev), amdm, mmdm, None)


class SpecKVM(FakeKVM):
    def is_speculative(self):
        return True


amds, mmds = metadata(1, [0], SpecKVM(pc.conv, pc.ssm))
rspec = LAYERS[0]._gemv_conv_operands(torch.zeros(1, HID, dtype=torch.bfloat16, device=dev), amds, mmds,
                                      types.SimpleNamespace())
G._GDN_GEMV_CONV = False
amd1, mmd1 = metadata(1, [0], pc.kvm)
rkill = LAYERS[0]._gemv_conv_operands(torch.zeros(1, HID, dtype=torch.bfloat16, device=dev), amd1, mmd1, None)
G._GDN_GEMV_CONV = True
rok = LAYERS[0]._gemv_conv_operands(torch.zeros(1, HID, dtype=torch.bfloat16, device=dev), amd1, mmd1, None)
# rows >= 4 eager chain: candidate's forward equals kill-switch forward bitwise (both main's path).
p4a, p4b = Pools(), Pools()
a4a, m4a = metadata(4, [0, 1, 2, 3], p4a.kvm)
a4b, m4b = metadata(4, [0, 1, 2, 3], p4b.kvm)
x4 = (torch.randn(4, HID, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
with with_multi_stream(True):
    oa = LAYERS[0].forward(x4.clone(), a4a, m4a)
    G._GDN_GEMV_CONV = False
    ob = LAYERS[0].forward(x4.clone(), a4b, m4b)
    G._GDN_GEMV_CONV = True
torch.cuda.synchronize()
rows4_equal = same(list(oa), list(ob)) and same(p4a.snap(), p4b.snap())
routing = dict(rows4_none=r4 is None, mixed_none=rmix is None, spec_none=rspec is None,
               killswitch_none=rkill is None, rows1_fused=rok is not None, rows4_bitwise_main=rows4_equal)
ok_all = ok_all and all(routing.values())
log("routing %s" % json.dumps(routing))
print(json.dumps(dict(passed=ok_all, layers=L,
                      compared="REAL Qwen3NextGatedDeltaNet.forward in CUDA graphs: fused GEMV+conv vs "
                               "kill switch (main's GEMV -> causal_conv1d_update) bitwise on T1 output, "
                               "qkvz z columns, conv pool (strided, incl. padding) and ssm pool after every "
                               "one of 120 stateful steps; 60 reset replays equal replay 1; eager main-op "
                               "reference; routing for rows>=4 / mixed / spec / kill switch",
                      cases=report, routing=routing)))
sys.exit(0 if ok_all else 1)
