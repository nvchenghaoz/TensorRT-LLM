# Shared builders for the GDN out_proj decode NVFP4 (W4A16 Marlin) bench/check.
# The chain under test is the real Qwen3NextGatedDeltaNet._postprocess_gdn_output
# (gated RMSNorm token-major + out_proj Linear), called unbound on a small
# stand-in object that carries only the attributes it reads. Shapes are
# Qwen3.8-27B's: 48 v-heads x 128 (value_dim 6144) -> hidden 5120; z is a
# column slice of the (16384)-wide in_proj_qkvz output, as in the model.
import os

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402
from tensorrt_llm._torch.modules.mamba import gdn_mixer  # noqa: E402
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm as RMSNormGated  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

HEADS, HD = 48, 128
VD = HEADS * HD  # 6144
HID = 5120
QKVZ = 16384
CONV = QKVZ - VD  # 10240
EPS = 1e-6
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


def candidate_flag() -> bool:
    """What this tree's GDN __init__ sets on out_proj (False on main)."""
    fn = getattr(gdn_mixer, "_gdn_out_proj_nvfp4_decode_requested", None)
    return bool(fn()) if fn is not None else False


class FakeGDN:
    """Carries exactly the attributes _postprocess_gdn_output reads."""

    def __init__(self, norm, out_proj):
        self.norm = norm
        self.out_proj = out_proj
        self.head_v_dim = HD
        self.value_dim_per_tp = VD

    def __call__(self, attn_out, z):
        return gdn_mixer.Qwen3NextGatedDeltaNet._postprocess_gdn_output(self, attn_out, z)


def make_weight(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(HID, VD, generator=g, device=dev) * 0.02
    w = w * (torch.rand(HID, 1, generator=g, device=dev) * 1.5 + 0.25)
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    nw = (1.0 + 0.1 * torch.randn(HD, generator=g, device=dev)).to(torch.bfloat16)
    return w8, s, nw


# Static input scale of out_proj: amax of the gated-norm output on typical
# inputs is ~10-20, so 0.04 keeps it well inside E4M3 range.
INPUT_SCALE = 0.04


def make_gdn(seed, flag):
    w8, s, nw = make_weight(seed)
    lin = Linear(VD, HID, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
    lin._nvfp4_decode_copy = flag
    lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                       "input_scale": torch.tensor(INPUT_SCALE, device=dev)}])
    lin.post_load_weights()
    norm = RMSNormGated(HD, eps=EPS, group_size=None, norm_before_gate=True,
                        device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(nw)
    # As GDN cache_derived_state does.
    sc = lin.quant_method.get_static_input_scale(lin)
    norm.fp8_scale = sc.detach() if sc is not None else None
    return FakeGDN(norm, lin), (w8, s, nw)


def make_inputs(m, seed=7):
    g = torch.Generator(device=dev).manual_seed(seed + m)
    attn_out = (torch.randn(m, HEADS, HD, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    qkvz = (torch.randn(m, QKVZ, generator=g, device=dev) * 1.5).to(torch.bfloat16)
    z = qkvz[:, CONV:].view(m, HEADS, HD)
    return attn_out, z, qkvz


def fp32_reference(attn_out, z, ref):
    """FP32 gated norm + FP8 weight * scale, from the same BF16 inputs."""
    w8, s, nw = ref
    x = attn_out.reshape(-1, HD).float()
    zz = z.reshape(-1, HD).float()
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * nw.float()
    y = y * zz * torch.sigmoid(zz)
    y = y.view(-1, VD)
    return y @ (w8.float() * s).t()


def err(out, ref):
    d = (out.float() - ref).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "rel_rms": ((out.float() - ref).norm() / ref.norm()).item(),
            "finite": bool(torch.isfinite(out.float()).all().item())}
