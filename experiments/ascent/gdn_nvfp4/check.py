# Correctness check for the decode-only NVFP4 W4A16 copy of an FP8 per-tensor
# Linear (GDN in_proj_qkvz, 16384 x 5120), FP8QDQLinearMethod.apply dispatch.
#  A. Copy construction: the module's Marlin tensors equal an independent
#     re-quantization (fp4_quantize with gs = s / g, g = 2^ceil(log2(amax/2688)))
#     packed by the helper; zero / subnormal E4M3 block scales counted (weights
#     include tiny-magnitude rows that force subnormal block scales).
#  B. Marlin output vs an independent FP32 reference built from the
#     dequantized NVFP4 operands (LUT E2M1 * E4M3 SF * g) at M = 1,2,4,8,16,
#     overall and on rows whose block scales are subnormal; plus error vs the
#     original math (FP8 weight * scale, FP32) next to main's FP8 path error.
#  C. Repeated execution with the timed loop's structure (8 copies in one CUDA
#     graph, 10 warmup + 60 replays): every replay bitwise equal to replay 1
#     and to the eager first call; FP8 weights and Marlin tensors hashed
#     before/after (unchanged).
#  D. Dispatch: rows > 16 (17, 2048) and the env kill-switch give outputs
#     bitwise identical to the original FP8 path (static quant + cuda/cublas
#     scaled mm); 3-D input reshapes correctly.
#  E. Refactor: _pack_nvfp4_weight_for_marlin equals the pre-change body of
#     MarlinNVFP4LinearMethod.transform_weights bitwise (padded and unpadded).
import json, os, torch
import torch.nn.functional as F
from tensorrt_llm._torch.modules import linear as L
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils import fp4_utils, marlin_utils

N, K, C, WARM, ITERS = 16384, 5120, 8, 10, 60
MS = [1, 2, 4, 8, 16]
dev = torch.device("cuda")
IN_SCALE = 0.05
def log(*a): print(*a, flush=True)
fails = []
def check(cond, msg):
    log(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)

def make_w(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(N, K, generator=g, device=dev) * 0.02
    w = w * (torch.rand(N, 1, generator=g, device=dev) * 1.5 + 0.25)
    w[:64] *= 1e-4      # tiny rows -> subnormal E4M3 block scales
    w[64:72, :1024] = 0  # all-zero blocks -> zero block scales
    s = (w.abs().amax() / 448.0).float()
    return (w / s).to(torch.float8_e4m3fn), s

def make_lin(w8, s, flag=True):
    lin = Linear(K, N, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = flag
    lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                       "input_scale": torch.tensor(IN_SCALE, device=dev)}])
    lin.post_load_weights()
    return lin

def fp8_path(lin, x):  # original main path, explicit
    q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, lin.input_scale)
    if x.shape[0] <= 8:
        return torch.ops.trtllm.cuda_scaled_mm(q, lin.weight.t(), scale_a=lin.input_scale,
                                               scale_b=lin.weight_scale, bias=None,
                                               out_dtype=torch.bfloat16)
    return torch.ops.trtllm.cublas_scaled_mm(q, lin.weight.t(), scale_a=lin.input_scale,
                                             scale_b=lin.weight_scale, bias=None,
                                             out_dtype=torch.bfloat16)

def h(t):
    v = t.contiguous().view(torch.uint8).view(-1)
    v = v[: v.numel() // 8 * 8].view(torch.int64)
    return int((v * 1000003).sum().item()), int(v.sum().item())

LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                   dtype=torch.float32, device=dev)

# ---------------- A
w8, s = make_w(1)
lin = make_lin(w8, s)
check(getattr(lin, "_nvfp4_decode_weight", None) is not None, "A: copy built for flagged module")
wv = w8.to(torch.bfloat16)
amax = wv.abs().amax().float() * s
g = torch.exp2(torch.ceil(torch.log2(amax / 2688.0)))
fp4, sf = torch.ops.trtllm.fp4_quantize(wv, (s / g).reshape(1), 16, False, True)
mw, ms_, mg = L._pack_nvfp4_weight_for_marlin(fp4, sf, g.reshape(1), N, K, 16)
check(torch.equal(mw, lin._nvfp4_decode_weight) and torch.equal(ms_.view(torch.uint8), lin._nvfp4_decode_scale.view(torch.uint8))
      and torch.equal(mg, lin._nvfp4_decode_global_scale), "A: module copy == independent re-quantization + pack")
sf2 = torch.ops.trtllm.block_scale_interleave_reverse(sf.view(N, -1))[:, :K // 16].contiguous()
sb = sf2.view(torch.uint8)
n_zero = int((sb == 0).sum()); n_sub = int((((sb & 0x78) == 0) & ((sb & 7) != 0)).sum())
sub_rows = (((sb & 0x78) == 0) & ((sb & 7) != 0)).any(1)
log(f"A: g=2^{int(torch.log2(g))} zero_sf={n_zero} subnormal_sf={n_sub} rows_with_subnormal={int(sub_rows.sum())}")
check(n_sub > 0, "A: test weight exercises subnormal block scales")
pk = fp4.view(torch.uint8)
vals = torch.stack([LUT[(pk & 0xF).long()], LUT[(pk >> 4).long()]], -1).reshape(N, K)
w_deq = (vals.view(N, K // 16, 16) * sf2.view(torch.float8_e4m3fn).float().unsqueeze(-1)).reshape(N, K) * g
w_orig = w8.float() * s
rel_w = ((w_deq - w_orig).norm() / w_orig.norm()).item()
log(f"A: weight rel RMS error NVFP4 vs FP8 = {rel_w:.5f}")
# Marlin's S0E5M3 dequant (fp4 * SF * 2^-119 in BF16, 2^119 folded into the
# global scale) flushes blocks whose E4M3 scale is subnormal (< 2^-6). Model
# that explicitly: w_marlin = w_deq with subnormal-scale blocks zeroed.
sub_blk = (((sb & 0x78) == 0) & ((sb & 7) != 0))
w_marlin = (w_deq.view(N, K // 16, 16) * (~sub_blk).unsqueeze(-1)).reshape(N, K)
flush_rel = ((w_marlin - w_deq).norm() / w_orig.norm()).item()
log(f"A: weight energy in subnormal-scale blocks (rel to ||W||) = {flush_rel:.3e}")

# ---------------- B
errs = {}
for m in MS:
    x = (torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 2.0).to(torch.bfloat16)
    y = lin(x).float()
    ref_q = x.float() @ w_deq.t()          # same quantized operands, exact dequant
    ref_m = x.float() @ w_marlin.t()       # ... with subnormal-scale blocks flushed
    ref_o = x.float() @ w_orig.t()         # original math
    y_main = fp8_path(lin, x).float()
    bf16_ulp = (ref_q.abs().max() * 2 ** -8).item()
    e_m = (y - ref_m).abs()
    e_q = (y - ref_q).abs()
    norm_rows = ~sub_rows
    errs[m] = {"vs_flushed_ref_max": e_m.max().item(), "vs_flushed_ref_mean": e_m.mean().item(),
               "normal_rows_vs_exact_ref_max": e_q[:, norm_rows].max().item(),
               "subnormal_rows_vs_exact_ref_max": e_q[:, sub_rows].max().item(),
               "subnormal_rows_exact_ref_absmax": ref_q[:, sub_rows].abs().max().item(),
               "subnormal_rows_vs_flushed_ref_max": e_m[:, sub_rows].max().item(),
               "cand_vs_orig_rel_rms": ((y - ref_o).norm() / ref_o.norm()).item(),
               "main_vs_orig_rel_rms": ((y_main - ref_o).norm() / ref_o.norm()).item()}
    log(f"B: M={m} {errs[m]}")
    check(torch.isfinite(y).all().item(), f"B: M={m} finite")
    # BF16 output rounding + BF16 split-K partial reduce: allow a few BF16 ulps
    # of the output magnitude.
    check(e_m.max().item() <= 4 * bf16_ulp, f"B: M={m} Marlin vs FP32 ref of the dequantized operands (subnormal-scale blocks flushed) max {e_m.max().item():.4g} <= 4 ulp {4*bf16_ulp:.4g}")
    check(e_q[:, norm_rows].max().item() <= 4 * bf16_ulp, f"B: M={m} rows with only normal block scales vs exact-dequant FP32 ref max {e_q[:, norm_rows].max().item():.4g} <= 4 ulp")

# ---------------- C
lins = [lin] + [make_lin(*make_w(100 + i)) for i in range(C - 1)]
hashes0 = [(h(l.weight), h(l._nvfp4_decode_weight), h(l._nvfp4_decode_scale), h(l._nvfp4_decode_global_scale)) for l in lins]
for m in MS:
    x = (torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 2.0).to(torch.bfloat16)
    eager = [l(x).clone() for l in lins]
    s0 = torch.cuda.Stream(); s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            outs = [l(x) for l in lins]
    torch.cuda.current_stream().wait_stream(s0); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs = [l(x) for l in lins]
    first, ndiff, maxdrift = None, 0, 0.0
    for it in range(WARM + ITERS):
        gr.replay(); torch.cuda.synchronize()
        cur = [o.clone() for o in outs]
        if first is None:
            first = cur
        for a, b, e in zip(cur, first, eager):
            if not (torch.equal(a.view(torch.int16), b.view(torch.int16)) and torch.equal(a.view(torch.int16), e.view(torch.int16))):
                ndiff += 1
                maxdrift = max(maxdrift, (a.float() - e.float()).abs().max().item())
    check(ndiff == 0, f"C: M={m} {WARM+ITERS} graph replays x {C} copies bitwise == replay 1 == eager (diffs {ndiff}, max drift {maxdrift})")
    del gr
hashes1 = [(h(l.weight), h(l._nvfp4_decode_weight), h(l._nvfp4_decode_scale), h(l._nvfp4_decode_global_scale)) for l in lins]
check(hashes0 == hashes1, "C: FP8 weights and Marlin copies unchanged across the loop")

# ---------------- D
for m in [17, 2048]:
    x = (torch.randn(m, K, device=dev) * 2.0).to(torch.bfloat16)
    check(torch.equal(lin(x).view(torch.int16), fp8_path(lin, x).view(torch.int16)), f"D: M={m} uses the original FP8 path bitwise")
os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE"] = "0"
lin_off = make_lin(w8, s)
del os.environ["TRTLLM_GDN_INPROJ_NVFP4_DECODE"]
check(getattr(lin_off, "_nvfp4_decode_weight", None) is None, "D: kill-switch env=0 builds no copy")
lin_unflagged = make_lin(w8, s, flag=False)
check(getattr(lin_unflagged, "_nvfp4_decode_weight", None) is None, "D: unflagged FP8 Linear builds no copy")
for m in [1, 4, 8, 16]:
    x = (torch.randn(m, K, device=dev) * 2.0).to(torch.bfloat16)
    check(torch.equal(lin_off(x).view(torch.int16), fp8_path(lin_off, x).view(torch.int16)), f"D: env=0 M={m} bitwise == original FP8 path")
x3 = (torch.randn(1, 4, K, device=dev)).to(torch.bfloat16)
y3 = lin(x3)
check(tuple(y3.shape) == (1, 4, N) and torch.equal(y3.view(4, N).view(torch.int16), lin(x3.view(4, K)).view(torch.int16)), "D: 3-D input reshape")
log(f"D: memory allocated {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---------------- E: refactor equivalence vs the pre-change body
def old_pack(weight, weight_scale, weight_scale_2, size_n, size_k, group_size):
    size_k_pad = fp4_utils.pad_up(size_k, 64); size_n_pad = fp4_utils.pad_up(size_n, 128)
    num_groups = size_k // group_size
    scale_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(weight_scale.view(size_n_pad, -1))
    scale_2d = scale_unswizzled[:size_n, :num_groups]
    if size_k_pad != size_k or size_n_pad != size_n:
        num_groups_pad = size_k_pad // group_size
        weight = F.pad(weight, (0, (size_k_pad - size_k) // 2, 0, size_n_pad - size_n))
        scale_2d = F.pad(scale_2d, (0, num_groups_pad - num_groups), value=0x08)
        scale_2d = F.pad(scale_2d, (0, 0, 0, size_n_pad - size_n), value=0)
    qweight_int32 = weight.view(torch.int32).T.contiguous()
    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
    marlin_weight = torch.ops.trtllm.gptq_marlin_repack(b_q_weight=qweight_int32, perm=perm, size_k=size_k_pad,
                                                        size_n=size_n_pad, num_bits=4, is_a_8bit=False)
    scale_2d = scale_2d.view(torch.float8_e4m3fn).T.contiguous()
    marlin_scale = marlin_utils.marlin_permute_scales(scale_2d.to(torch.half), size_k_pad, size_n_pad, group_size=group_size)
    marlin_scale = marlin_utils.nvfp4_marlin_process_scales(marlin_scale)
    if weight_scale_2.numel() == 0 or not torch.isfinite(weight_scale_2).all() or weight_scale_2.item() == 0:
        weight_scale_2 = torch.tensor([1.0], dtype=torch.float32, device=weight.device)
    return marlin_weight, marlin_scale, marlin_utils.nvfp4_marlin_process_global_scale(weight_scale_2.to(torch.bfloat16))
for (n, k) in [(16384, 5120), (200, 96), (5120, 17408)]:
    wb = (torch.randn(n, k, device=dev)).to(torch.bfloat16)
    f4, sfx = torch.ops.trtllm.fp4_quantize(wb, torch.tensor([448 * 6 / wb.abs().max().float().item()], device=dev), 16, False, True)
    gs2 = torch.tensor([wb.abs().max().float().item() / (448 * 6)], device=dev)
    f4 = f4.view(torch.uint8)
    a = old_pack(f4, sfx, gs2, n, k, 16); b = L._pack_nvfp4_weight_for_marlin(f4, sfx, gs2, n, k, 16)
    check(all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)), f"E: helper == pre-change Marlin pack bitwise for ({n},{k})")

print(json.dumps({"passed": not fails, "failures": fails, "weight_rel_rms_err": rel_w, "zero_sf": n_zero,
                  "subnormal_sf": n_sub, "errors": {str(k): v for k, v in errs.items()},
                  "compared_against": "independent FP32 reference from dequantized NVFP4 operands; original FP8 math; main's explicit FP8 path (bitwise for M>16 and env=0); pre-change Marlin pack body",
                  "repeated_execution": f"{C} copies in one CUDA graph, {WARM} warmup + {ITERS} replays per M, bitwise vs replay 1 and eager"}))
