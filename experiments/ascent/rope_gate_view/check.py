# Untimed correctness check for gdn-fusion idea 69 (branch 27).
# For T in 1/33/64/128/2111/8161/8192:
#   reference = main's copy path: rope-gate op with the gate copy
#   (trtllm::fused_qkv_gemma_rmsnorm_rope_gate), then fused_sigmoid_mul_nvfp4 /
#   fused_sigmoid_mul_inplace on the contiguous gate copy. The gate copy itself
#   is also compared against a pure-torch slice of qkv.
#   candidate = public wrapper (default TRTLLM_ROPE_GATE_VIEW=1): no-gate op and a
#   strided gate view of qkv feeding both sigmoid kernels.
# Candidate chain is captured in a CUDA graph and replayed WARM+ITERS times; before
# every replay all outputs are poisoned (BF16 NaN, FP4/SF 0xFF incl. SF pad rows)
# and the in-place BF16 work buffer is reset from pristine; after every replay
# all outputs are compared bitwise with the reference and the qkv input with its
# pristine copy. One eager call is checked the same way. A subprocess with
# TRTLLM_ROPE_GATE_VIEW=0 must return a separate contiguous gate (copy path)
# with identical bytes.
import json
import os
import subprocess
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch

from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as K

NQ, NKV, HD, ROT = 24, 4, 256, 64
MSEC = (11, 11, 10)
QS = NQ * HD
W = 2 * QS + 2 * NKV * HD
MAXPOS = 40960
WARM, ITERS = 10, 50
TS = [int(v) for v in os.environ.get("CHECK_TS", "1,33,64,128,2111,8161,8192").split(",")]
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


def make_inputs(t, use_mrope):
    g = torch.Generator(device=dev).manual_seed(11 + t)
    qkv = (torch.randn(t, W, generator=g, device=dev) * 1.5).to(torch.bfloat16)
    qkv[::9, :HD] = 0.0  # zero q head rows
    qkv[1::13, HD:2 * HD] = 30.0  # saturating gate
    qkv[2::17, 3 * HD:4 * HD] = -30.0
    if t > 3:
        qkv[3, 5 * HD + 7] = float("inf")  # a gate inf: sigmoid -> 1
    qw = (0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
    kw = (0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
    attn = (torch.randn(t, QS, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    attn[::7, 256:512] = 0.0
    gs = torch.tensor([6.0 / 0.02], dtype=torch.float32, device=dev)
    if use_mrope:
        buf = torch.randint(0, 36000, (3, 1, max(t, 1) + 64), generator=g, device=dev,
                            dtype=torch.int32)
        positions = buf[:, 0, :t].reshape(3, -1)  # strided decode-style slice
    else:
        positions = torch.randint(0, 36000, (t,), generator=g, device=dev, dtype=torch.int32)
    return qkv, qw, kw, attn, gs, positions


inv = 1.0 / (1000000.0**(torch.arange(0, ROT, 2, device=dev).float() / ROT))
fr = torch.arange(MAXPOS, device=dev).float()[:, None] * inv[None, :]
cos_sin = torch.stack([fr.cos(), fr.sin()], dim=1).contiguous()


def reference(qkv, qw, kw, attn, gs, positions, msec):
    use_mrope = msec is not None
    s1, s2 = (msec[1], msec[2]) if use_mrope else (0, 0)
    qkv_out, gate = K._fused_qkv_gemma_rmsnorm_rope_gate(qkv, qw, kw, cos_sin, positions, 1e-6, NQ,
                                                         NKV, HD, ROT, use_mrope, s1, s2)
    fp4, sf = K.fused_sigmoid_mul_nvfp4(attn, gate, gs)
    bf = attn.clone()
    K.fused_sigmoid_mul_inplace(bf, gate)
    return qkv_out, gate, fp4, sf, bf


def candidate(qkv, qw, kw, attn, gs, positions, msec, work):
    qkv_out, gate = K.fused_qkv_gemma_rmsnorm_rope_gate(qkv, qw, kw, cos_sin, positions, 1e-6, NQ,
                                                        NKV, HD, ROT, msec)
    fp4, sf = K.fused_sigmoid_mul_nvfp4(attn, gate, gs)
    K.fused_sigmoid_mul_inplace(work, gate)
    return qkv_out, gate, fp4, sf


def is_qkv_view(gate, qkv):
    return (gate.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
            and gate.data_ptr() == qkv.data_ptr() + HD * qkv.element_size()
            # stride(0) of a size-1 dim is arbitrary in torch views
            and (qkv.shape[0] == 1 or gate.stride(0) == qkv.stride(0))
            and tuple(gate.stride())[1:] == (2 * HD, 1)
            and tuple(gate.shape) == (qkv.shape[0], NQ, HD))


def eq(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                              b.contiguous().view(torch.uint8))


assert K._ROPE_GATE_VIEW, "default must be the view path"
results = []
ok_all = True
for t in TS:
    for use_mrope in (True, False):
        msec = MSEC if use_mrope else None
        qkv, qw, kw, attn, gs, positions = make_inputs(t, use_mrope)
        pristine = qkv.clone()
        r_qkv, r_gate, r_fp4, r_sf, r_bf = reference(qkv, qw, kw, attn, gs, positions, msec)
        torch.cuda.synchronize()
        gate_slice_ok = eq(r_gate, qkv[:, :2 * QS].reshape(t, NQ, 2 * HD)[:, :, HD:])
        # torch sanity of BF16 gate math (tolerance, not the exactness criterion)
        tref = (attn.float() * torch.sigmoid(r_gate.reshape(t, QS).float())).to(torch.bfloat16)
        bf_torch_maxerr = (r_bf.float() - tref.float()).abs().max().item() if t else 0.0
        rec = dict(T=t, mrope=use_mrope, gate_copy_equals_slice=gate_slice_ok,
                   bf16_vs_torch_maxerr=bf_torch_maxerr)
        # eager candidate
        work = attn.clone()
        c_qkv, c_gate, c_fp4, c_sf = candidate(qkv, qw, kw, attn, gs, positions, msec, work)
        torch.cuda.synchronize()
        rec["eager_ok"] = (eq(c_qkv, r_qkv) and eq(c_fp4, r_fp4) and eq(c_sf, r_sf) and eq(work, r_bf)
                           and eq(c_gate, r_gate) and eq(qkv, pristine))
        rec["gate_is_qkv_view"] = is_qkv_view(c_gate, qkv)
        # graph candidate, NaN-poisoned replays
        work = attn.clone()
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(2):
                work.copy_(attn)
                candidate(qkv, qw, kw, attn, gs, positions, msec, work)
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            g_qkv, g_gate, g_fp4, g_sf = candidate(qkv, qw, kw, attn, gs, positions, msec, work)
        rec["graph_gate_is_qkv_view"] = is_qkv_view(g_gate, qkv)
        bad = []
        first = None
        for it in range(WARM + ITERS):
            g_qkv.fill_(float("nan"))
            g_fp4.fill_(255)
            g_sf.fill_(255)
            work.copy_(attn)
            gr.replay()
            torch.cuda.synchronize()
            cur = (g_qkv.clone(), g_fp4.clone(), g_sf.clone(), work.clone())
            if first is None:
                first = cur
            ok = (eq(g_qkv, r_qkv) and eq(g_fp4, r_fp4) and eq(g_sf, r_sf) and eq(work, r_bf)
                  and eq(qkv, pristine) and all(eq(a, b) for a, b in zip(cur, first)))
            if not ok:
                bad.append(it)
        rec["graph_replays"] = WARM + ITERS
        rec["graph_bad_iters"] = bad[:10]
        rec["graph_ok"] = not bad
        passed = (gate_slice_ok and rec["eager_ok"] and rec["gate_is_qkv_view"]
                  and rec["graph_gate_is_qkv_view"] and rec["graph_ok"] and bf_torch_maxerr <= 0.0625)
        rec["passed"] = passed
        ok_all = ok_all and passed
        log(json.dumps(rec))
        results.append(rec)
        del gr, g_qkv, g_gate, g_fp4, g_sf, first, cur
        torch.cuda.empty_cache()

# Kill switch: TRTLLM_ROPE_GATE_VIEW=0 restores the copy path.
child = r"""
import json, torch
from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as K
dev = torch.device('cuda')
t, NQ, NKV, HD, ROT = 64, 24, 4, 256, 64
W = 2 * NQ * HD + 2 * NKV * HD
g = torch.Generator(device=dev).manual_seed(3)
qkv = torch.randn(t, W, generator=g, device=dev).to(torch.bfloat16)
qw = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
kw = torch.zeros(HD, dtype=torch.bfloat16, device=dev)
fr = torch.arange(1024, device=dev).float()[:, None] * torch.ones(ROT // 2, device=dev)[None, :]
cs = torch.stack([fr.cos(), fr.sin()], dim=1).contiguous()
pos = torch.arange(t, device=dev, dtype=torch.int32)
o, gate = K.fused_qkv_gemma_rmsnorm_rope_gate(qkv, qw, kw, cs, pos, 1e-6, NQ, NKV, HD, ROT, None)
o2, gate2 = K.fused_qkv_gemma_rmsnorm_rope_gate(qkv, qw, kw, cs, pos, 1e-6, NQ, NKV, HD, ROT, None, gate_view=True)
res = dict(flag=K._ROPE_GATE_VIEW, contiguous=gate.is_contiguous(),
           separate_storage=gate.untyped_storage().data_ptr() != qkv.untyped_storage().data_ptr(),
           same_bytes=torch.equal(gate.view(torch.int16), gate2.contiguous().view(torch.int16)),
           same_qkv=torch.equal(o.view(torch.int16), o2.view(torch.int16)))
print(json.dumps(res))
"""
env = dict(os.environ)
env["TRTLLM_ROPE_GATE_VIEW"] = "0"
p = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
log("kill-switch stdout:", p.stdout.strip()[-2000:])
if p.returncode != 0:
    log("kill-switch stderr:", p.stderr[-4000:])
ks = None
try:
    ks = json.loads(p.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    ks = None
ks_ok = bool(ks) and ks["flag"] is False and ks["contiguous"] and ks["separate_storage"] and ks[
    "same_bytes"] and ks["same_qkv"]
log("kill switch ok", ks_ok)
ok_all = ok_all and ks_ok
print(json.dumps(dict(passed=ok_all, kill_switch=ks, kill_switch_ok=ks_ok,
                      cases=len(results), failed=[(r["T"], r["mrope"]) for r in results if not r["passed"]],
                      compared="candidate view path vs main copy path (rope-gate op with gate copy + "
                      "sigmoid kernels on contiguous gate): qkv_out, FP4, SF incl pad, BF16 in-place; "
                      "qkv input unchanged; 60 NaN-poisoned graph replays per case")))
