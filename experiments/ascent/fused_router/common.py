"""Shared pieces for idea 67 (DSv4.1 decode fused router) benches and checks.

K1 stand-in: like _mhc_lagged_layer_input_kernel (grid M x 128 threads, PDL launch, griddepcontrol.wait then
launch_dependents as its first statements), spins SPIN ns, then writes x (copy of a pristine source).
Aux: 188-CTA (or all-SM) spinning kernel for contention / no-hang checks.
"""
import os
import sys
import time

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HIDDEN, EXPERTS, LAYERS, TOPK = 5120, 384, 40, 6
E_LOCAL, INTER = 96, 2304
SCALE = 1.5
SPIN_C1_NS, SPIN_C4_NS = 4900, 3300  # K1 post-trigger time: router starts ~0.6 us after the trigger (fact 2199)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


@triton.jit
def _gtime():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,~{memory}", [], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r,~{memory}", [], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _gdc_trigger():
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r,~{memory}", [],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _k1_standin_kernel(src, dst, SPIN_NS: tl.constexpr, HID: tl.constexpr, BLOCK: tl.constexpr):
    _gdc_wait()
    _gdc_trigger()
    pid = tl.program_id(0)
    t0 = _gtime()
    t = t0
    while t - t0 < SPIN_NS:
        t = _gtime()
    for k in tl.static_range(0, HID, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        v = tl.load(src + pid * HID + offs)
        tl.store(dst + pid * HID + offs, v)


def k1_standin(src, dst, spin_ns):
    _k1_standin_kernel[(src.shape[0],)](src, dst, SPIN_NS=spin_ns, HID=HIDDEN, BLOCK=1024, num_warps=4,
                                        launch_pdl=True)


@triton.jit
def _aux_kernel(buf, out, NBUF: tl.constexpr, DUR_NS: tl.constexpr, BLOCK: tl.constexpr, NCTA: tl.constexpr):
    pid = tl.program_id(0)
    t0 = _gtime()
    t = t0
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    i = 0
    while t - t0 < DUR_NS:
        offs = ((pid + i * NCTA) * BLOCK) % NBUF + tl.arange(0, BLOCK)
        acc += tl.load(buf + offs)
        i += 1
        t = _gtime()
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


class Aux:
    def __init__(self, ncta=188, num_warps=8):
        self.buf = torch.ones(64 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        self.out = torch.empty(ncta * 1024, dtype=torch.float32, device="cuda")
        self.ncta, self.num_warps = ncta, num_warps

    def launch(self, dur_ns):
        _aux_kernel[(self.ncta,)](self.buf, self.out, NBUF=self.buf.numel(), DUR_NS=dur_ns, BLOCK=1024,
                                  NCTA=self.ncta, num_warps=self.num_warps)


EMPTY_I = None


def router_main(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def gate_main(logits, bias):
    global EMPTY_I
    if EMPTY_I is None:
        EMPTY_I = torch.empty(0, dtype=torch.int32, device="cuda")
    m = logits.shape[0]
    ow = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
    oi = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.gate_forward(logits, bias, EMPTY_I, EMPTY_I, ow, oi, TOPK, SCALE, False, None, None)
    return oi, ow


def has_fused():
    return hasattr(torch.ops.trtllm, "dsv4_fused_router")


def fused(x, w, bias, counter):
    ids, wts, logits = torch.ops.trtllm.dsv4_fused_router(x, w, bias, counter, SCALE)
    return ids, wts, logits


def make_decode_state():
    """Real Sm12xW4a8DecodeState of EP4 rank 0 (96 local experts, slot_start 0) on synthetic MXFP4 weights."""
    from tensorrt_llm._torch.moe.fused_moe.cutlass import w4a8_mxfp4_mxfp8_sm12x_decode as dec

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4a8_fi_chain"))
    from common import make_weights, trtllm_layout  # noqa: E402

    ok, why = dec.fi_decode_available()
    assert ok, why
    w1, s1, w2, s2 = make_weights(seed=0)
    w31, s31, w2v, s2s = trtllm_layout(w1, s1, w2, s2)
    del w1, w2
    state = dec.Sm12xW4a8DecodeState(w3_w1_weight=w31, w3_w1_weight_scale=s31, w2_weight=w2v,
                                     w2_weight_scale=s2s, num_local_experts=E_LOCAL, slot_start=0,
                                     swiglu_limit=10.0, enable_pdl=True)
    assert dec._route_fuses_prologue()
    return state, dec


def router_weights(gen, n):
    return [(torch.randn((EXPERTS, HIDDEN), generator=gen, device="cuda") * HIDDEN ** -0.5).to(torch.bfloat16)
            for _ in range(n)]


def biases(gen, n, scale=0.05):
    return [(torch.randn((EXPERTS,), generator=gen, device="cuda") * scale).float() for _ in range(n)]


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        fn()
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g, out
