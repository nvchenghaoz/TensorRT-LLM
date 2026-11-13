# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer attention backend for visual generation models."""

import hashlib
import math
from typing import Any, Literal

import flashinfer
import torch

from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ....attention.backends.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_WORKSPACE_BYTES = 128 * 1024 * 1024
_BlockScaledQKMode = Literal["mxfp8", "nvfp4"]


class FlashInferAttention(AttentionBackend):
    """Dense FlashInfer attention without an LLM KV cache.

    The backend accepts the VisualGen NHD layout ``[B, S, H, D]``. The FP16/BF16
    path uses FlashInfer's single-request or batched ragged prefill kernel. Block-scaled
    MXFP8 or NVFP4 Q/K dispatches to FlashInfer's SM100/SM103 FMHA with FP8 V. The
    SM120/SM121 path supports dense NVFP4 Q/K/V, according to the validated YAML recipe,
    and per-tensor FP8 Q/K/V on SM120, optionally BF16 for the first ``dense_steps``
    denoising steps.
    """

    # Set after the first serving call of the SM120 FP8 recipe logs its quantize tactic.
    _fp8_tactic_logged = False

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
        self._static_global_scale = self._metadata_state.get("_ltx23_static_global_scale") == 1.0

        self._single_prefill = flashinfer.single_prefill_with_kv_cache
        self._batch_prefill_cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
        if quant_attention_config is not None and quant_attention_config.qk_dtype not in (
            "fp8",
            "mxfp8",
            "nvfp4",
        ):
            raise ValueError(
                "FlashInfer quantized attention supports qk_dtype='fp8', 'mxfp8' or 'nvfp4', "
                f"got {quant_attention_config.qk_dtype!r}."
            )
        if quant_attention_config is not None and quant_attention_config.qk_dtype == "fp8":
            if quant_attention_config.v_dtype != "fp8":
                raise ValueError(
                    "FlashInfer per-tensor FP8 attention requires v_dtype='fp8', "
                    f"got {quant_attention_config.v_dtype!r}."
                )
            if torch.cuda.is_available() and torch.cuda.get_device_capability() != (12, 0):
                raise RuntimeError(
                    "FlashInfer per-tensor FP8 attention (qk_dtype='fp8', v_dtype='fp8') "
                    "runs the SM120 CuTe DSL FP8 FMHA and requires an SM120 GPU."
                )


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
        if self._static_global_scale:
            return self._run_capture_safe_batch_prefill(
                q, k, v, is_causal=is_causal, key_padding_mask=key_padding_mask
            )
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

    def _prefill_mask_digest(self, mask: torch.Tensor | None, *, capturing: bool) -> str | None:
        if mask is None:
            return None
        version = None if torch.is_inference(mask) else mask._version
        identity = (id(mask), mask.data_ptr(), version)
        identities = self._metadata_state.setdefault("flashinfer_mask_identities", {})
        if capturing:
            if identity not in identities:
                raise RuntimeError("FlashInfer padding mask must be warmed before CUDA capture.")
            return identities[identity][1]
        digest = hashlib.sha256(mask.detach().to("cpu", torch.bool).numpy().tobytes()).hexdigest()
        # Retain the tensor so its identity cannot be recycled while a plan uses it.
        identities[identity] = (mask, digest)
        return digest

    def _run_capture_safe_batch_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse immutable mask-content plans across warmup and capture streams.

        The component's shared metadata state requires serialized forwards.
        Planning and mask readback occur only outside capture; a changed mask
        owns a new wrapper rather than mutating a plan retained by an old graph.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        mask_digest = self._prefill_mask_digest(key_padding_mask, capturing=capturing)
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        plan_key = (
            q.device,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            mask_digest,
        )
        plans = self._metadata_state.setdefault("flashinfer_immutable_prefill_plans", {})
        entry = plans.get(plan_key)
        if entry is None:
            if capturing:
                raise RuntimeError("FlashInfer batched prefill must be warmed before CUDA capture.")
            workspaces = self._metadata_state.setdefault("flashinfer_immutable_workspaces", {})
            workspace = workspaces.get(q.device)
            if workspace is None:
                workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
                workspaces[q.device] = workspace
            wrapper = self._batch_prefill_cls(workspace, "NHD")
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
                    .clone()
                )
            torch.cuda.current_stream(q.device).synchronize()
            wrapper.plan(
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
            entry = (wrapper, query_indptr, key_value_indptr, custom_mask)
            plans[plan_key] = entry
        output, lse = entry[0].run(
            q.contiguous().view(-1, self.num_heads, self.head_dim),
            k.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            v.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            return_lse=True,
        )
        return (
            output.view(batch_size, query_length, self.num_heads, self.head_dim),
            lse.view(batch_size, query_length, self.num_heads).transpose(1, 2) * math.log(2.0),
        )

    @staticmethod
    def _quantize_qk_static_global(
        tensor: torch.Tensor, mode: _BlockScaledQKMode
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode NHD Q/K with global one and native, input-dependent block scales."""
        from flashinfer.quantization import fp4_quantize, mxfp8_quantize

        batch, sequence, heads, dimension = tensor.shape
        padded_sequence = ((sequence + 127) // 128) * 128
        data = tensor.transpose(1, 2).contiguous()
        if padded_sequence != sequence:
            data = torch.nn.functional.pad(data, (0, 0, 0, padded_sequence - sequence))
        matrix = data.reshape(batch * heads * padded_sequence, dimension)
        if mode == "mxfp8":
            encoded, factors = mxfp8_quantize(
                matrix, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
            )
            encoded = encoded.view(batch, heads, padded_sequence, dimension)[:, :, :sequence]
            factors = factors.view(torch.float8_e8m0fnu)
        elif mode == "nvfp4":
            # fp4_quantize takes the inverse global dequantization scale. Both
            # encode and FMHA decode use one; only native block factors vary.
            inverse_global = torch.ones(1, device=tensor.device, dtype=torch.float32)
            encoded, factors = fp4_quantize(
                matrix,
                inverse_global,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cute-dsl",
            )
            encoded = encoded.view(batch, heads, padded_sequence, dimension // 2)[:, :, :sequence]
            factors = factors.view(torch.float8_e4m3fn)
        else:
            raise ValueError(f"Unsupported static-global Q/K mode: {mode!r}.")
        encoded = encoded.transpose(1, 2).contiguous()
        if mode == "nvfp4":
            # Copy packed bytes before viewing Float4, whose deterministic fill is unsupported.
            encoded = encoded.view(torch.float4_e2m1fn_x2)
        return encoded, factors

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

        if self._static_global_scale:
            q_store, q_sf = self._quantize_qk_static_global(q, qk_mode)
            k_store, k_sf = self._quantize_qk_static_global(k, qk_mode)
            v_store = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            q_scale = k_scale = v_scale = 1.0
        else:
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
            scale_o=1.0,
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
    def _quantize_e4m3_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused amax + E4M3 cast with one device-side descale (x ~= x_q * descale)."""
        encoded, descale = torch.ops.trtllm.quantize_e4m3_per_tensor(tensor.contiguous())
        return encoded, descale.float().reshape(1)

    @torch.compiler.disable
    def _run_fp8_sm120(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        with_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-tensor E4M3 Q/K/V attention on SM120 with BF16/FP16 output.

        Q, K and V are each quantized with one tensor-wide scale; the Q and K descales are
        folded into the softmax scale and the V descale scales the output. The CuTe DSL
        kernel takes host scalars, so the three descales are read back once per call.
        """
        from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill

        from .fp8_attn_prep import quantize_qkv_e4m3_per_tensor

        batch, q_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        # Two Triton launches for Q, K and V, reading V's strided view in place; bitwise equal
        # to _quantize_e4m3_per_tensor on each (which it falls back to for other layouts).
        q_fp8, k_fp8, v_fp8, descales = quantize_qkv_e4m3_per_tensor(q, k, v)
        descales = descales.tolist()
        output = torch.empty(batch * q_len, num_heads, head_dim, dtype=q.dtype, device=q.device)
        lse = (
            torch.empty(batch * q_len, num_heads, dtype=torch.float32, device=q.device)
            if with_lse
            else None
        )
        cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q_len
        cu_seqlens_k = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * kv_len
        sm120_fmha_fp8_ragged_prefill(
            q_fp8.view(batch * q_len, num_heads, head_dim),
            k_fp8.view(batch * kv_len, self.num_kv_heads, head_dim),
            v_fp8.view(batch * kv_len, self.num_kv_heads, head_dim),
            output,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=q_len,
            is_causal=is_causal,
            sm_scale=descales[0] * descales[1] * self.scale,
            lse=lse,
            v_scale=descales[2],
        )
        output = output.view(batch, q_len, num_heads, head_dim)
        if lse is None:
            return output, None
        # The kernel reports log2 LSE; VisualGen exposes natural-log LSE in [B, H, S].
        lse = lse.view(batch, q_len, num_heads).transpose(1, 2).contiguous() * math.log(2.0)
        return output, lse

    def _is_fp8_sm120(self) -> bool:
        quant_config = self.quant_attention_config
        return (
            quant_config is not None
            and quant_config.qk_dtype == "fp8"
            and quant_config.v_dtype == "fp8"
        )

    def _step_controller(self) -> Any:
        # Published by the model (MiniMax-H3: models/minimax_h3/step_precision.py).
        return self._metadata_state.get("step_precision_controller")

    def _bf16_forwards(self) -> int | None:
        # visual_gen_args.step_precision.attention_bf16_forwards, published by the model.
        return self._metadata_state.get("attention_bf16_forwards")

    def _runs_bf16_forward(self) -> bool:
        """Whether this call runs BF16 attention under the step gate.

        The transformer's step-precision controller (set by the pipeline before every
        denoising forward) holds the forward index; forwards below
        ``attention_bf16_forwards`` run main's BF16 path (VANILLA torch SDPA).
        """
        bf16_forwards = self._bf16_forwards()
        if not bf16_forwards:
            return False
        controller = self._step_controller()
        if controller is None:
            raise RuntimeError(
                "step_precision.attention_bf16_forwards needs a model that publishes a "
                "step_precision_controller (MiniMax-H3); none was found."
            )
        return controller.runs_bf16(bf16_forwards)

    def _log_forward_path(self, bf16: bool) -> None:
        """One host log line per denoising forward (layer 0): which attention path ran."""
        controller = self._step_controller()
        if controller is None or controller.step_index is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = (
            "bf16 via torch SDPA (VANILLA path)"
            if bf16
            else "fp8 via FlashInfer sm120_fmha_fp8_ragged_prefill"
        )
        print(
            f"[step_precision] rank={rank} forward {controller.step_index}/"
            f"{controller.num_steps} warmup={controller.warmup} attention={path} "
            f"(attention_bf16_forwards={self._bf16_forwards()})",
            flush=True,
        )

    def _run_bf16_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        """The BF16 phase of the step gate: main's BF16 attention, bitwise.

        Main runs the VANILLA backend (torch SDPA, FA2). This is VanillaAttention's call on
        the same HND views of the NHD buffers; the output goes back to NHD as a view.
        """
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return output.transpose(1, 2)

    def _log_fp8_tactic_once(self, tensor: torch.Tensor) -> None:
        """Log the quantize tactic serving dispatches to (layer 0, first non-warmup call)."""
        controller = self._step_controller()
        if FlashInferAttention._fp8_tactic_logged or (
            controller is not None and (controller.warmup or controller.step_index is None)
        ):
            return
        FlashInferAttention._fp8_tactic_logged = True
        from . import fp8_attn_prep

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        prep = (
            "C++ quantize_e4m3_per_tensor x3 (TLLM_FP8_ATTN_PREP_DISABLE=1)"
            if fp8_attn_prep._DISABLED
            else "triton fp8_attn_prep (multi-tensor amax + E4M3 quantize, strided V read in "
            "place; C++ quantize_e4m3_per_tensor only for unsupported layouts)"
        )
        print(f"[FlashInfer FP8 attention] rank={rank} prep={prep}", flush=True)
        try:
            from tensorrt_llm._torch.autotuner import AutoTuner
            from tensorrt_llm._torch.custom_ops.torch_custom_ops import QuantizeE4M3PerTensorRunner
            from tensorrt_llm.logger import logger

            tuner = AutoTuner.get()
            runner = QuantizeE4M3PerTensorRunner()
            hit, runner_id, tactic, min_time = tuner.profiling_cache.search_cache(
                "trtllm::quantize_e4m3_per_tensor",
                [runner],
                tuple(tuner._get_input_sizes([tensor])),
                QuantizeE4M3PerTensorRunner.tuning_config,
                apply_map_to_tuning_buckets=True,
            )
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            logger.warning(
                f"[FlashInfer FP8 attention] rank={rank} quantize_e4m3_per_tensor dispatch: "
                f"tactic={tactic!r} cache_hit={hit} runner_id={runner_id} "
                f"profiled_time={min_time} shape={tuple(tensor.shape)}"
            )
            print(
                f"[FlashInfer FP8 attention] rank={rank} quantize_e4m3_per_tensor tactic="
                f"{tactic!r} cache_hit={hit}",
                flush=True,
            )
        except Exception as exc:  # logging only; never fail the forward
            print(f"[FlashInfer FP8 attention] tactic lookup failed: {exc!r}", flush=True)

    @torch.compiler.disable
    def _run_fp8_gated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        """Per-tensor FP8 attention, or BF16 SDPA on the gate's leading forwards."""
        bf16 = self._runs_bf16_forward()
        controller = self._step_controller()
        if controller is not None and controller.step_index is not None:
            controller.attention_calls += 1
        if self.layer_idx == 0:
            self._log_forward_path(bf16)
        if bf16:
            return self._run_bf16_dense(q, k, v, is_causal)
        if controller is not None and controller.warmup and self._bf16_forwards():
            # Startup warmup: run the BF16 phase too, then FP8 (CuTe DSL JIT, quantize
            # autotune), so no timed request compiles or tunes either phase.
            self._run_bf16_dense(q, k, v, is_causal)
        if self.layer_idx == 0:
            self._log_fp8_tactic_once(q)
        output, _ = self._run_fp8_sm120(q, k, v, is_causal=is_causal, with_lse=False)
        return output

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
        if self._is_fp8_sm120():
            # No LSE consumer: skip the kernel's LSE write.
            self._validate_inputs(q, k, v, key_padding_mask)
            if key_padding_mask is not None:
                raise ValueError(
                    "FlashInfer quantized attention does not support key padding masks."
                )
            is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
            if is_causal and q.shape[1] != k.shape[1]:
                raise ValueError("FlashInfer causal attention requires equal Q and K/V lengths.")
            return self._run_fp8_gated(q, k, v, is_causal=is_causal)
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
        if capability[0] == 12 and qk_dtype == "nvfp4" and quant_config.v_dtype == "nvfp4":
            return self._run_nvfp4_sm12x(q, k, v, is_causal=is_causal)
        if capability == (12, 0) and qk_dtype == "fp8" and quant_config.v_dtype == "fp8":
            return self._run_fp8_sm120(q, k, v, is_causal=is_causal, with_lse=True)
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
