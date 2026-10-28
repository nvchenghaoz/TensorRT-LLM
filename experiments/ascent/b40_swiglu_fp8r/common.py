"""Shared inputs and op chains for branch 40 (fused SwiGLU + fp8r per-token E4M3 quant).

FP8R MLP of the SM120 checkpoint-excluded blocks (content_mode_blocks.image.0 M=4096,
single_blocks.24-27 M=4176 / 4256 with the longer text):
  gate_up: torch._scaled_mm(e4m3 [M,3072], W[18432,3072]) -> BF16 [M, 18432] ([gate | up])
  unfused: torch.ops.trtllm.silu_and_mul -> BF16 [M,9216] -> quantize_fp8r (warp_fp8_per_token_quant)
  fused:   torch.ops.trtllm.flux3_swiglu_fp8r_quant -> (e4m3 [M,9216], fp32 [M])
  down:    torch._scaled_mm(e4m3 [M,9216], W[3072,9216]) -> BF16 [M, 3072]
"""
import torch

HIDDEN = 3072
INTER = 9216


def log(msg):
    print(msg, flush=True)


def has_fused():
    from tensorrt_llm._torch.visual_gen.models.flux3 import fp8

    return hasattr(fp8, "swiglu_fp8r_quant")


def unfused(x):
    from tensorrt_llm._torch.modules.swiglu import swiglu
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

    h = swiglu(x)
    return quantize_fp8r(h)


def fused(x):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import swiglu_fp8r_quant

    return swiglu_fp8r_quant(x)


def quant_weight(w):
    w = w.float()
    s = torch.clamp(w.abs().amax(dim=1) / 448.0, min=1e-12)
    return (w / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous(), s.contiguous()


def weights(seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wu = torch.randn(2 * INTER, HIDDEN, device="cuda", generator=g, dtype=torch.float32) * 0.02
    wd = torch.randn(HIDDEN, INTER, device="cuda", generator=g, dtype=torch.float32) * 0.02
    return quant_weight(wu), quant_weight(wd)


def gemm(data, scales, w):
    wq, ws = w
    return torch._scaled_mm(data, wq.t(), scale_a=scales.reshape(-1, 1),
                            scale_b=ws.reshape(1, -1), out_dtype=torch.bfloat16)


def act_quant(x):
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

    return quantize_fp8r(x)


def gate_up_input(m, w_up, seed):
    """Realistic gate_up output: FP8R GEMM of a random BF16 activation."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 2.0).to(torch.bfloat16)
    d, s = act_quant(a)
    return gemm(d, s, w_up).contiguous(), (d, s)


def special_input(m, seed):
    """Random BF16 [M, 2*INTER] with zero / negative-zero / all-zero-row / tail-max cases."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, 2 * INTER, device="cuda", generator=g) * 3.0).to(torch.bfloat16)
    nz = torch.tensor(-0.0, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()                        # all-zero row
    x[1].fill_(nz)                      # all negative-zero row
    x[2, :INTER] = nz                   # gate -0, up random
    x[3, INTER:] = nz                   # up -0, gate random
    x[4, :INTER] = 0.0                  # gate +0
    x[5, INTER:] = 0.0                  # up +0
    x[6, : INTER // 2] = nz             # half -0 gate
    x[6, INTER: INTER + INTER // 2] = nz
    x[7] = x[7] * 1e-30                 # tiny row (scale clamps to 1e-12)
    x[8] = x[8] * 1e3                   # large row
    x[9, :8192] = x[9, :8192] * 1e-2    # amax in the tail part (cols 8192..9215)
    x[10, 8192:INTER] = x[10, 8192:INTER] * 1e-2  # amax in the head part
    x[11, 9000] = 60.0                  # isolated max in tail
    x[11, INTER + 9000] = 50.0
    # sprinkle zeros / -0 across rows
    mask = torch.rand(m, 2 * INTER, device="cuda", generator=g) < 0.01
    x[mask] = 0.0
    mask = torch.rand(m, 2 * INTER, device="cuda", generator=g) < 0.01
    x[mask] = nz
    return x.contiguous()
