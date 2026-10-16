"""Prefill-time NVFP4 W4A4 copies of static per-tensor FP8 linears (SM12x).

A flagged FP8 linear keeps its FP8 weight for decode (rows <= 16, unchanged)
and gets a load-time NVFP4 copy of the same weight for prefill (rows > 16):

* weight copy: W = (fp8 weight * weight_scale) in BF16, quantized once with
  ``fp4_quantize`` (group 16, E4M3 block scales, power-of-two global scale);
* activations: ``fp4_quantize`` with a global scale of 6 / s, where s is the
  checkpoint's static FP8 input_scale, so NVFP4 saturates at the same amax as
  the FP8 static quantization it replaces;
* GEMM: ``trtllm::nvfp4_gemm`` (AutoTuner over cutlass/cublaslt/cuda_core)
  with alpha = (s / 6) * g.

Output class: precision (NVFP4 W4A4 on prefill-time copies of FP8 weights).
Per-group switches (=0 disables that group's copies, =1 enables):
  TRTLLM_FP8_PREFILL_NVFP4_GDN       GDN in_proj_qkvz (default on)
  TRTLLM_FP8_PREFILL_NVFP4_ATTN_O    gated-attention o_proj (default on)
  TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT   GDN out_proj (default on; its gated norm
                                     then emits BF16 for rows > 16)
  TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV  gated-attention qkv_proj (default OFF,
                                     opt-in: it produces the K/V written to
                                     the KV cache; kept out of the default
                                     for the shared precision budget)
"""

import math
import os
from typing import Optional

import torch

from tensorrt_llm.logger import logger

ENV_GDN_IN_PROJ_QKVZ = "TRTLLM_FP8_PREFILL_NVFP4_GDN"
ENV_ATTN_QKV = "TRTLLM_FP8_PREFILL_NVFP4_ATTN_QKV"
ENV_ATTN_O = "TRTLLM_FP8_PREFILL_NVFP4_ATTN_O"
ENV_GDN_OUT = "TRTLLM_FP8_PREFILL_NVFP4_GDN_OUT"
# Rows <= MAX_DECODE_ROWS keep the FP8 path (decode, CUDA graphs).
MAX_DECODE_ROWS = 16
_SF_VEC = 16
_FP8_MAX = 448.0
_E2M1_MAX = 6.0

_ATTR = "_fp8_prefill_nvfp4"
_stats = {"built": 0, "bytes": 0, "calls": 0}
# Per-group copy count / bytes / prefill calls (group = kill-switch env).
_group_stats = {}


_ENV_DEFAULT = {ENV_ATTN_QKV: "0"}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, _ENV_DEFAULT.get(name, "1")) != "0"


def prefill_nvfp4_stats() -> dict:
    out = dict(_stats)
    out["groups"] = {k: dict(v) for k, v in _group_stats.items()}
    return out


def maybe_build_prefill_nvfp4_copy(module: torch.nn.Module,
                                   env_name: str = ENV_GDN_IN_PROJ_QKVZ
                                   ) -> bool:
    """Build (once) the NVFP4 prefill copy for a static FP8 linear."""
    from .linear import FP8QDQLinearMethod

    if not _env_enabled(env_name):
        return False
    if module.__dict__.get(_ATTR) is not None:
        return True
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    qm = getattr(module, "quant_method", None)
    if not isinstance(qm, FP8QDQLinearMethod):
        return False
    if getattr(module, "force_dynamic_quantization", False):
        return False
    s = qm.get_static_input_scale(module)
    if s is None or s.numel() != 1:
        return False
    w = getattr(module, "weight", None)
    if (w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2
            or not w.is_cuda or w.shape[1] % _SF_VEC != 0
            or w.shape[0] % 8 != 0):
        return False
    ws = module.weight_scale
    if ws is None or ws.numel() != 1:
        return False

    with torch.no_grad():
        w_bf16 = (w.float() * ws.float().reshape(())).to(torch.bfloat16)
        amax = float(w_bf16.abs().amax().float().item())
        if not math.isfinite(amax) or amax <= 0.0:
            return False
        g = 2.0**math.ceil(math.log2(amax / (_FP8_MAX * _E2M1_MAX)))
        w_global = torch.tensor([1.0 / g], dtype=torch.float32, device=w.device)
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w_bf16, w_global, _SF_VEC,
                                                    False)
        del w_bf16
        s_val = float(s.float().reshape(()).item())
        act_global = torch.tensor([_E2M1_MAX / s_val],
                                  dtype=torch.float32,
                                  device=w.device)
        alpha = torch.tensor([(s_val / _E2M1_MAX) * g],
                             dtype=torch.float32,
                             device=w.device)
    # Plain attribute (not a Parameter/buffer): not part of the state dict.
    module.__dict__[_ATTR] = _PrefillNVFP4Copy(w_fp4, w_sf, act_global, alpha,
                                               g, env_name)
    nbytes = (w_fp4.numel() * w_fp4.element_size() +
              w_sf.numel() * w_sf.element_size())
    _stats["built"] += 1
    _stats["bytes"] += nbytes
    gs = _group_stats.setdefault(env_name, {"built": 0, "bytes": 0, "calls": 0})
    gs["built"] += 1
    gs["bytes"] += nbytes
    if _stats["built"] == 1:
        logger.info(
            f"[fp8_prefill_nvfp4] built NVFP4 prefill copy for "
            f"{tuple(w.shape)} FP8 linear (rows > {MAX_DECODE_ROWS}); "
            f"kill switch {env_name}=0")
    return True


def get_prefill_nvfp4_copy(module: torch.nn.Module):
    return module.__dict__.get(_ATTR)


def _main_decoder_layers(model: torch.nn.Module) -> list:
    """The main decoder stack (model.model.layers[:num_hidden_layers])."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        return []
    n = getattr(inner, "num_hidden_layers", None)
    layers = list(layers)
    return layers[:n] if isinstance(n, int) else layers


def build_fp8_prefill_nvfp4_copies(model: torch.nn.Module) -> int:
    """Model walk after post-load: copies for every GDN in_proj_qkvz, plus
    gated-attention qkv_proj / o_proj and GDN out_proj of the main decoder
    layers (matched by module class, not by name)."""
    n = 0
    for mod in model.modules():
        qkvz = mod.__dict__.get("_modules", {}).get("in_proj_qkvz")
        if qkvz is not None and maybe_build_prefill_nvfp4_copy(qkvz):
            n += 1
    counts = {ENV_GDN_IN_PROJ_QKVZ: n, ENV_ATTN_QKV: 0, ENV_ATTN_O: 0,
              ENV_GDN_OUT: 0}
    from ..attention.attention import Attention
    from .mamba.gdn_mixer import Qwen3NextGatedDeltaNet
    for layer in _main_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        # attn_output_gate: qkv_proj / o_proj inputs are always BF16.
        if isinstance(attn, Attention) and getattr(attn, "attn_output_gate",
                                                   False):
            for proj, env in ((attn.qkv_proj, ENV_ATTN_QKV),
                              (attn.o_proj, ENV_ATTN_O)):
                if maybe_build_prefill_nvfp4_copy(proj, env):
                    counts[env] += 1
        gdn = getattr(layer, "linear_attn", None)
        if isinstance(gdn, Qwen3NextGatedDeltaNet) and \
                maybe_build_prefill_nvfp4_copy(gdn.out_proj, ENV_GDN_OUT):
            counts[ENV_GDN_OUT] += 1
    total = sum(counts.values())
    if total:
        logger.info(f"[fp8_prefill_nvfp4] NVFP4 W4A4 prefill copies per group "
                    f"(rows > {MAX_DECODE_ROWS}): {counts}; "
                    f"{_stats['bytes'] / 2**30:.2f} GiB")
    return total


class _PrefillNVFP4Copy:
    """Load-time NVFP4 copy; called from FP8QDQLinearMethod.apply."""

    __slots__ = ("w_fp4", "w_sf", "act_global", "alpha", "g", "group")

    def __init__(self, w_fp4, w_sf, act_global, alpha, g,
                 group=ENV_GDN_IN_PROJ_QKVZ):
        self.w_fp4, self.w_sf = w_fp4, w_sf
        self.act_global, self.alpha, self.g = act_global, alpha, g
        self.group = group

    def __getitem__(self, i):
        return (self.w_fp4, self.w_sf, self.act_global, self.alpha, self.g)[i]

    def __call__(self, module, input, original_shape, bias):
        return prefill_nvfp4_apply(self, module, input, original_shape, bias)


def prefill_nvfp4_apply(copy, module: torch.nn.Module, input: torch.Tensor,
                        original_shape: torch.Size,
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
    """NVFP4 W4A4 prefill GEMM on a 2D BF16 input with rows > 16."""
    w_fp4, w_sf, act_global, alpha = (copy.w_fp4, copy.w_sf, copy.act_global,
                                      copy.alpha)
    _stats["calls"] += 1
    gs = _group_stats.get(copy.group)
    if gs is not None:
        gs["calls"] += 1
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(input, act_global, _SF_VEC,
                                                    False)
    output = torch.ops.trtllm.nvfp4_gemm(
        act_fp4,
        w_fp4,
        act_sf,
        w_sf,
        alpha,
        module.dtype or input.dtype,
        allowed_backends="cutlass,cublaslt,cuda_core")
    if len(original_shape) > 2:
        output = output.reshape(*original_shape[:-1], output.shape[-1])
    if bias is not None:
        output = output + bias
    return output
