# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer attention backend for visual generation models."""

import math
import os
from typing import Any, Literal

import flashinfer
import torch
import torch.nn.functional as F

from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ....attention.backends.interface import PredefinedAttentionMask
from .fp8_sm12x import visual_gen_fp8_attention_sm120
from .interface import AttentionBackend, AttentionTensorLayout

_WORKSPACE_BYTES = 128 * 1024 * 1024
_BlockScaledQKMode = Literal["mxfp8", "nvfp4"]
# Below this sequence length the per-tensor quantize passes cost more than the FP8
# kernel saves (S=512: BF16 SDPA 0.053 ms vs FP8 0.161 ms with quantize), so short
# calls run unquantized BF16 SDPA.
_FP8_MIN_SEQ_LEN = 1024
# head_dim 128 E4M3 calls run the FLUX.3 fmha_v2 variant (fmha_v2_flux3.py: same math,
# bitwise equal to the stock kernel); set TRTLLM_FMHA_V2_FLUX3=0 to use the stock kernel.
_USE_FMHA_V2_FLUX3 = os.environ.get("TRTLLM_FMHA_V2_FLUX3", "1") != "0"


def run_fmha_v2_fp8_sm12x(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    *,
    scale_bmm1_d: torch.Tensor,
    scale_bmm2_d: torch.Tensor,
    causal: bool,
    return_lse: bool = False,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """The one entry to FlashInfer's SM120 FP8 fmha_v2 (contiguous E4M3 BSHD Q/K/V, BF16 out)."""
    if _USE_FMHA_V2_FLUX3 and head_dim == 128:
        from .fmha_v2_flux3 import fmha_v2_flux3_d128

        return fmha_v2_flux3_d128(
            q_fp8,
            k_fp8,
            v_fp8,
            output,
            num_heads,
            head_dim,
            seq_len,
            scale_bmm1_d=scale_bmm1_d,
            scale_bmm2_d=scale_bmm2_d,
            causal=causal,
            lse=lse if return_lse else None,
        )
    from flashinfer.prefill import fmha_v2_prefill_sm120

    fmha_v2_prefill_sm120(
        q_fp8,
        k_fp8,
        v_fp8,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale_softmax=1.0,
        scale_bmm1=1.0,
        scale_bmm2=1.0,
        scale_bmm1_d=scale_bmm1_d,
        scale_bmm2_d=scale_bmm2_d,
        causal=causal,
        return_lse=return_lse,
        lse=lse,
    )
    return output


class FlashInferAttention(AttentionBackend):
    """Dense FlashInfer attention without an LLM KV cache.

    The backend accepts the VisualGen NHD layout ``[B, S, H, D]``. The FP16/BF16
    path uses FlashInfer's single-request or batched ragged prefill kernel. Block-scaled
    MXFP8 or NVFP4 Q/K dispatches to FlashInfer's SM100/SM103 FMHA with FP8 V. The
    SM120/SM121 path supports dense NVFP4 Q/K/V, according to the validated YAML recipe.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD
        self._metadata_state = (
            attention_metadata_state if attention_metadata_state is not None else {}
        )

        self._single_prefill = flashinfer.single_prefill_with_kv_cache
        self._batch_prefill_cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
        self._fp8_sm12x = (
            quant_attention_config is not None
            and quant_attention_config.qk_dtype == "fp8"
            and quant_attention_config.v_dtype == "fp8"
        )
        if self._fp8_sm12x:
            capability = torch.cuda.get_device_capability()
            if capability not in ((12, 0), (12, 1)):
                raise ValueError(
                    "FlashInfer per-tensor FP8 attention (qk_dtype='fp8', v_dtype='fp8') "
                    f"runs on SM120/SM121 only, got SM{capability[0]}{capability[1]}."
                )
            if self.num_heads != self.num_kv_heads or head_dim not in (64, 128):
                raise ValueError(
                    "FlashInfer SM120/SM121 FP8 attention requires equal Q/K/V head counts "
                    "and head_dim=64 or 128."
                )
            self._fp8_min_seq_len = int(
                os.environ.get("TRTLLM_VISUAL_GEN_FP8_ATTN_MIN_SEQ", _FP8_MIN_SEQ_LEN)
            )
        elif quant_attention_config is not None and quant_attention_config.qk_dtype not in (
            "mxfp8",
            "nvfp4",
        ):
            raise ValueError(
                "FlashInfer quantized attention supports qk_dtype='mxfp8' or 'nvfp4', "
                f"or 'fp8' with v_dtype='fp8' on SM120/SM121, got {quant_attention_config.qk_dtype!r}."
            )
        # The SM12x FP8 path reads strided [B, S, H, D] views (packed/planar QKV rows) in its
        # fused quantize; every other path makes its own contiguous copies.
        self.accepts_strided_qkv = self._fp8_sm12x

    def _validate_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("FlashInfer attention expects 4D NHD Q/K/V tensors.")
        if q.device.type != "cuda":
            raise RuntimeError("FlashInfer attention requires CUDA tensors.")
        if q.device != k.device or q.device != v.device:
            raise ValueError("FlashInfer attention requires Q/K/V on the same device.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("FlashInfer attention requires Q/K/V with the same dtype.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashInfer attention expects FP16 or BF16 Q/K/V tensors.")
        if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
            raise ValueError("FlashInfer attention requires aligned Q/K/V batch dimensions.")
        if q.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "Invalid query shape for FlashInfer attention: "
                f"expected [..., {self.num_heads}, {self.head_dim}], got {tuple(q.shape)}."
            )
        expected_kv = (self.num_kv_heads, self.head_dim)
        if k.shape[2:] != expected_kv or v.shape[2:] != expected_kv:
            raise ValueError(
                "Invalid key/value shape for FlashInfer attention: "
                f"expected [..., {expected_kv[0]}, {expected_kv[1]}], "
                f"got K={tuple(k.shape)}, V={tuple(v.shape)}."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("FlashInfer attention requires num_heads divisible by num_kv_heads.")
        if key_padding_mask is not None and key_padding_mask.shape != (
            q.shape[0],
            k.shape[1],
        ):
            raise ValueError(
                "key_padding_mask must have shape "
                f"{(q.shape[0], k.shape[1])}, got {tuple(key_padding_mask.shape)}."
            )

    def _run_single_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        custom_mask = None
        if key_padding_mask is not None:
            custom_mask = (
                key_padding_mask[0]
                .to(device=q.device, dtype=torch.bool)
                .unsqueeze(0)
                .expand(q.shape[1], -1)
                .contiguous()
            )
        output, lse = self._single_prefill(
            q[0].contiguous(),
            k[0].contiguous(),
            v[0].contiguous(),
            o_dtype=q.dtype,
            custom_mask=custom_mask,
            causal=is_causal,
            sm_scale=self.scale,
            return_lse=True,
        )
        # Single prefill reports log2 LSE; VisualGen exposes natural-log LSE.
        return output.unsqueeze(0), lse.transpose(0, 1).unsqueeze(0) * math.log(2.0)

    @torch.compiler.disable
    def _run_batch_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        stream = torch.cuda.current_stream(q.device)
        stream_key = (q.device, stream.cuda_stream)

        workspace_cache = self._metadata_state.setdefault("flashinfer_workspace_cache", {})
        workspace = workspace_cache.get(stream_key)
        if workspace is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=q.device,
            )
            workspace_cache[stream_key] = workspace

        plan_key = (
            *stream_key,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            key_padding_mask is not None,
        )
        plan_cache = self._metadata_state.setdefault("flashinfer_prefill_plan_cache", {})
        plan_entry = plan_cache.get(plan_key)
        if plan_entry is None:
            plan_entry = {
                "wrapper": self._batch_prefill_cls(workspace, "NHD"),
                "mask_signature": None,
                "mask_reference": None,
                "is_planned": False,
            }
            plan_cache[plan_key] = plan_entry

        mask_signature = None
        if key_padding_mask is not None:
            mask_signature = (
                id(key_padding_mask),
                key_padding_mask.data_ptr(),
                key_padding_mask._version,
            )
        if not plan_entry["is_planned"] or plan_entry["mask_signature"] != mask_signature:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "Cannot plan FlashInfer batched prefill while the CUDA stream is "
                    "capturing. Run a warmup forward pass before capture."
                )

            query_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * query_length
            )
            key_value_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * key_value_length
            )
            custom_mask = None
            if key_padding_mask is not None:
                custom_mask = (
                    key_padding_mask.to(device=q.device, dtype=torch.bool)
                    .unsqueeze(1)
                    .expand(-1, query_length, -1)
                    .reshape(-1)
                    .contiguous()
                )

            stream.synchronize()
            plan_entry["wrapper"].plan(
                query_indptr,
                key_value_indptr,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                custom_mask=custom_mask,
                causal=is_causal,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            plan_entry["mask_signature"] = mask_signature
            plan_entry["mask_reference"] = key_padding_mask
            plan_entry["is_planned"] = True

        output, lse = plan_entry["wrapper"].run(
            q.contiguous().view(-1, self.num_heads, self.head_dim),
            k.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            v.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            return_lse=True,
        )
        output = output.view(batch_size, query_length, self.num_heads, self.head_dim)
        lse = lse.view(batch_size, query_length, self.num_heads).transpose(1, 2)
        # Batch prefill reports log2 LSE; VisualGen exposes natural-log LSE.
        return output, lse * math.log(2.0)

    @staticmethod
    def _quantize_fp8_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_dtype = torch.float8_e4m3fn
        fp8_info = torch.finfo(fp8_dtype)
        scale = (tensor.abs().amax().float() / fp8_info.max).clamp(min=1e-12)
        quantized = (tensor / scale).clamp(min=fp8_info.min, max=fp8_info.max)
        return quantized.to(fp8_dtype).contiguous(), scale

    def _run_blockscaled_sm10x(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        qk_mode: _BlockScaledQKMode,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_dim != 128:
            raise ValueError(
                "FlashInfer SM100/SM103 block-scaled attention requires head_dim=128, "
                f"got {self.head_dim}."
            )
        from flashinfer.attention.cute_dsl.fmha_blockscaled import cute_dsl_fmha_blockscaled_prefill
        from flashinfer.cute_dsl.attention.fmha.quantize import quantize_blockscaled_qk

        q_store, k_store, q_sf, k_sf, q_scale, k_scale = quantize_blockscaled_qk(q, k, qk_mode)
        v_store, v_scale = self._quantize_fp8_per_tensor(v)
        output = torch.empty_like(q)
        lse = torch.empty(
            (q.shape[0], q.shape[1], self.num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        cute_dsl_fmha_blockscaled_prefill(
            q_store,
            k_store,
            q_sf,
            k_sf,
            v_store,
            output,
            qk_mode=qk_mode,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse,
            scale_q=q_scale,
            scale_k=k_scale,
            scale_v=v_scale,
        )
        # CuTe DSL reports log2 LSE; VisualGen exposes natural-log LSE.
        return output, lse.transpose(1, 2).contiguous() * math.log(2.0)

    def _run_nvfp4_sm12x(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention supports dense self-attention only."
            )
        if self.num_heads != self.num_kv_heads or self.head_dim not in (64, 128):
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention requires equal Q/K/V head counts "
                "and head_dim=64 or 128."
            )
        sequence_length = q.shape[1]
        # FlashInfer 0.6.16 pads internally but cannot mask the padded K/V tokens.
        if sequence_length % 128 != 0:
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention requires sequence length "
                f"to be a multiple of 128, got {sequence_length}."
            )
        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()
        quantized_qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
            query,
            key,
            value,
            per_block_mean=False,
        )
        output, lse = flashinfer.nvfp4_attention_sm120_fwd(
            *quantized_qkv,
            sm_scale=self.scale,
            causal=is_causal,
            per_block_mean=False,
            out_dtype=q.dtype,
        )
        output = output[:, :, :sequence_length].transpose(1, 2).contiguous()
        return output, lse[:, :, :sequence_length]

    @staticmethod
    def _quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-tensor E4M3 quantize (fused amax + cast); descale stays on device."""
        x_q, descale = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
        return x_q, descale.float().reshape(1)

    def _run_fp8_sm12x(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-tensor FP8 E4M3 Q/K/V attention with BF16 output (fmha_v2 SM120 kernel).

        Q/K/V may be strided views. The no-LSE FP8 call is one in-graph custom op (fused
        strided quantize + fmha_v2); short sequences, where quantizing costs more than FP8
        saves, run unquantized BF16 attention instead.
        """
        if return_lse:
            return self._run_fp8_sm12x_with_lse(q, k, v, is_causal=is_causal)
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("FlashInfer SM120/SM121 FP8 attention supports self-attention only.")
        if q.shape[1] < self._fp8_min_seq_len or q.dtype != torch.bfloat16:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=is_causal,
                scale=self.scale,
            )
            return output.transpose(1, 2), None
        return visual_gen_fp8_attention_sm120(q, k, v, self.scale, is_causal), None

    @torch.compiler.disable
    def _run_fp8_sm12x_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        return_lse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The FP8 path that also returns LSE; runs eagerly on contiguous Q/K/V copies."""
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("FlashInfer SM120/SM121 FP8 attention supports self-attention only.")
        batch_size, sequence_length = q.shape[:2]
        if sequence_length < self._fp8_min_seq_len or q.dtype != torch.bfloat16:
            if return_lse:
                if batch_size == 1:
                    return self._run_single_prefill(
                        q, k, v, is_causal=is_causal, key_padding_mask=None
                    )
                return self._run_batch_prefill(q, k, v, is_causal=is_causal, key_padding_mask=None)
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=is_causal,
                scale=self.scale,
            )
            return output.transpose(1, 2), None

        q_fp8, q_descale = self._quantize_e4m3(q)
        k_fp8, k_descale = self._quantize_e4m3(k)
        v_fp8, v_descale = self._quantize_e4m3(v)
        # scale_bmm1 folds both descales and the softmax scale; scale_bmm2 is V's descale.
        scale_bmm1 = (q_descale * k_descale).mul_(self.scale)
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        stats = None
        if return_lse:
            stats = torch.empty(
                (batch_size, sequence_length, self.num_heads, 2),
                dtype=torch.float32,
                device=q.device,
            )
        run_fmha_v2_fp8_sm12x(
            q_fp8,
            k_fp8,
            v_fp8,
            output,
            self.num_heads,
            self.head_dim,
            sequence_length,
            scale_bmm1_d=scale_bmm1,
            scale_bmm2_d=v_descale,
            causal=is_causal,
            return_lse=return_lse,
            lse=stats,
        )
        lse = None
        if stats is not None:
            # fmha_v2 stores (row max, row sum of P scaled by 256 for E4M3).
            lse = (stats[..., 0] + torch.log(stats[..., 1] / 256.0)).transpose(1, 2).contiguous()
        return output, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if self._fp8_sm12x:
            self._validate_inputs(q, k, v, key_padding_mask)
            if key_padding_mask is not None:
                raise ValueError(
                    "FlashInfer quantized attention does not support key padding masks."
                )
            output, _ = self._run_fp8_sm12x(
                q,
                k,
                v,
                is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
                return_lse=False,
            )
            return output
        output, _ = self.forward_with_lse(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and FP32 LSE in ``[B, H, S]`` layout."""
        self._validate_inputs(q, k, v, key_padding_mask)
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        if is_causal and q.shape[1] != k.shape[1]:
            raise ValueError("FlashInfer causal attention requires equal Q and K/V lengths.")
        if is_causal and key_padding_mask is not None:
            raise ValueError("FlashInfer attention does not combine causal and padding masks.")

        quant_config = self.quant_attention_config
        if quant_config is None:
            # FlashInfer's single-prefill FA2/FA3 heuristics regress on SM10x.
            if q.shape[0] == 1 and torch.cuda.get_device_capability(q.device)[0] != 10:
                return self._run_single_prefill(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    key_padding_mask=key_padding_mask,
                )
            return self._run_batch_prefill(
                q,
                k,
                v,
                is_causal=is_causal,
                key_padding_mask=key_padding_mask,
            )
        if key_padding_mask is not None:
            raise ValueError("FlashInfer quantized attention does not support key padding masks.")

        capability = torch.cuda.get_device_capability(q.device)
        qk_dtype = quant_config.qk_dtype
        if (
            capability in ((10, 0), (10, 3))
            and qk_dtype in ("mxfp8", "nvfp4")
            and quant_config.v_dtype == "fp8"
        ):
            return self._run_blockscaled_sm10x(
                q,
                k,
                v,
                qk_mode=qk_dtype,
                is_causal=is_causal,
            )
        if (
            capability in ((12, 0), (12, 1))
            and qk_dtype == "nvfp4"
            and quant_config.v_dtype == "nvfp4"
        ):
            return self._run_nvfp4_sm12x(q, k, v, is_causal=is_causal)
        if self._fp8_sm12x:
            return self._run_fp8_sm12x(q, k, v, is_causal=is_causal, return_lse=True)
        raise RuntimeError(
            "Unsupported FlashInfer quantized attention recipe for "
            f"SM{capability[0]}{capability[1]}: qk_dtype={qk_dtype!r}, "
            f"v_dtype={quant_config.v_dtype!r}. Use qk_dtype='mxfp8' or 'nvfp4' with "
            "v_dtype='fp8' on SM100/SM103, or qk_dtype='nvfp4' with "
            "v_dtype='nvfp4' on SM120/SM121."
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_lse(cls) -> bool:
        return True
