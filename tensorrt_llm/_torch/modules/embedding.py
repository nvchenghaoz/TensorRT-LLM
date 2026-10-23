import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from ..distributed import allgather
from .linear import Linear, TensorParallelMode


class LMHead(Linear):
    """LM head layer.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        use_custom_cublas_mm: bool = False,
        quant_config=None,
    ):
        local_in_features = embedding_dim
        local_out_features = num_embeddings
        mapping = mapping or Mapping()
        self.enable_lm_head_tp_in_adp = mapping.enable_attention_dp and \
            getattr(mapping, 'enable_lm_head_tp_in_adp', False)

        tp_size = mapping.tp_size

        # Attention DP doesn't work with embedding parallelization.
        if mapping.enable_attention_dp:
            tensor_parallel_mode = None

        if tensor_parallel_mode == TensorParallelMode.ROW:
            local_in_features = math.ceil(embedding_dim / tp_size)
            self.padding_size = tp_size * local_in_features - embedding_dim
        elif tensor_parallel_mode == TensorParallelMode.COLUMN:
            local_out_features = math.ceil(num_embeddings / tp_size)
            self.padding_size = tp_size * local_out_features - num_embeddings
        else:
            self.padding_size = 0

        # Linear re-shards only the dim selected by tensor_parallel_mode, so
        # only that dim may be passed as local*tp (to carry the ceil-div
        # padding); the other dim must be the true full size — Linear keeps it
        # as-is, and quant_method.create_weights sizes the (packed) quantized
        # weight and scales from it.
        in_features_full = (local_in_features * tp_size if tensor_parallel_mode
                            == TensorParallelMode.ROW else embedding_dim)
        out_features_full = (local_out_features *
                             tp_size if tensor_parallel_mode
                             == TensorParallelMode.COLUMN else num_embeddings)
        super().__init__(
            in_features_full,
            out_features_full,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
            quant_config=quant_config,
        )

        if tensor_parallel_mode == TensorParallelMode.ROW:
            if self.tp_rank == self.tp_size - 1:
                local_in_features -= self.padding_size
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if self.has_any_quant:
            # Keep the quantized weights created by Linear (e.g. NVFP4 packed
            # weight + scales). The plain-Parameter override below only exists
            # to (a) drop bias and (b) shrink the ROW-mode padded shard, both
            # of which assume a dense high-precision weight.
            if self.padding_size > 0 or tensor_parallel_mode == TensorParallelMode.ROW:
                raise NotImplementedError(
                    "Quantized LMHead does not support vocab/hidden padding or "
                    "ROW tensor-parallel mode")
            if self.enable_lm_head_tp_in_adp:
                logger.error(
                    "Quantized LMHead constructed with lm_head TP in ADP: "
                    "the spec-decoding head slices the raw weight, which is "
                    "incompatible with quantized (packed) weights. The "
                    "lm_head quant entry should have been dropped upstream "
                    f"(quant_algo={quant_config.quant_algo}).")
                raise NotImplementedError(
                    "Quantized LMHead does not support lm_head TP in ADP "
                    "(spec-decoding head slices the raw weight)")
            # lm_head has historically always been bf16; make the switch to a
            # quantized head visible so unexpected accuracy/perf behavior is
            # attributable without a profiler.
            logger.info(f"LMHead is quantized: quant_algo="
                        f"{quant_config.quant_algo}, weight shape "
                        f"{tuple(self.weight.shape)} dtype {self.weight.dtype}")
        else:
            weight_shape = (self.out_features, self.in_features)
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype))
            self.register_parameter("bias", None)

    def get_quant_method(self, quant_config=None):
        from .linear import (MarlinNVFP4LinearMethod,
                             MarlinW4A4NVFP4LMHeadMethod, lmhead_w4a4_min_rows)
        method = super().get_quant_method(quant_config)
        # SM120 Marlin lm_head: keep the checkpoint's NVFP4 W4A4 mode for
        # large row counts (the copy is built only when the checkpoint has a
        # static input_scale; see MarlinW4A4NVFP4LMHeadMethod).
        if (type(method) is MarlinNVFP4LinearMethod
                and get_sm_version() in (120, 121)
                and lmhead_w4a4_min_rows() > 0):
            return MarlinW4A4NVFP4LMHeadMethod()
        return method

    @property
    def vocab_size_padded(self) -> int:
        if self.tp_mode == TensorParallelMode.COLUMN and self.gather_output:
            return self.out_features * self.tp_size
        else:
            return self.out_features

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        mapping_lm_head_tp: Optional[Mapping] = None,
        is_spec_decoding_head: bool = False,
    ) -> torch.Tensor:
        if is_spec_decoding_head and self.enable_lm_head_tp_in_adp:
            # For LM head TP in ADP, we need to slice the weight for the LM head
            tp_rank = mapping_lm_head_tp.tp_rank
            tp_size = mapping_lm_head_tp.tp_size
            slice_width = ceil_div(self.out_features, tp_size)
            slice_start = tp_rank * slice_width
            slice_end = min((tp_rank + 1) * slice_width, self.out_features)
            output = F.linear(input, self.weight[slice_start:slice_end, :],
                              None)
        else:
            output = super().forward(input, all_reduce_params=all_reduce_params)
        if (self.tp_mode == TensorParallelMode.COLUMN and self.gather_output
                and self.padding_size > 0):
            output = output[..., :-self.padding_size]

        return output

    def skip_forward(
            self,
            input: torch.Tensor,
            *,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        output_shape = input.shape[:-1] + (self.num_embeddings, )
        output = input.new_empty(output_shape)
        return output

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        original_weight = None
        if self.tp_mode == TensorParallelMode.COLUMN:
            if self.tp_rank == self.tp_size - 1 and self.padding_size > 0:
                original_weight = self.weight.data.zero_()
                self.weight.data = self.weight[:-self.padding_size, :]

        super().load_weights(weights,
                             allow_partial_loading=allow_partial_loading)

        if original_weight is not None:
            self.weight.data = original_weight


def get_masked_input_and_mask(
    input_: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_mask = (input_ >= vocab_start_index) & (input_ < vocab_end_index)
    valid_offset = vocab_start_index * vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask.unsqueeze(-1)


# Kill switch: TRTLLM_EMBED_FUSED_GATHER=0 restores the masked-input +
# F.embedding + masked_fill_ path (9 kernels) at TP1.
_EMBED_FUSED_GATHER = os.environ.get("TRTLLM_EMBED_FUSED_GATHER", "1") != "0"


@triton.jit
def _masked_embedding_gather_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    lo,
    hi,
    weight_stride_row,
    out_stride_row,
    embedding_dim,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    idx = tl.load(ids_ptr + token).to(tl.int64)
    valid = (idx >= lo) & (idx < hi)
    row = tl.where(valid, idx - lo, 0)
    offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    values = tl.load(weight_ptr + row * weight_stride_row + offsets,
                     mask=mask & valid,
                     other=0.0)
    tl.store(out_ptr + token * out_stride_row + offsets, values, mask=mask)


def _fused_masked_embedding(input_: torch.Tensor, weight: torch.Tensor,
                            vocab_start_index: int,
                            vocab_end_index: int) -> torch.Tensor:
    """out[t] = W[id - lo] if lo <= id < hi else 0, in one kernel.

    Bitwise equal to get_masked_input_and_mask + F.embedding + masked_fill_:
    valid rows are copied unchanged and invalid ids (e.g. the -1 padding of
    rejected speculative tokens) give +0.0 rows.
    """
    ids = input_.reshape(-1)
    num_tokens = ids.numel()
    embedding_dim = weight.shape[1]
    output = torch.empty((num_tokens, embedding_dim),
                         dtype=weight.dtype,
                         device=weight.device)
    if num_tokens > 0:
        block_d = min(1024, triton.next_power_of_2(embedding_dim))
        grid = (num_tokens, triton.cdiv(embedding_dim, block_d))
        _masked_embedding_gather_kernel[grid](
            ids,
            weight,
            output,
            vocab_start_index,
            vocab_end_index,
            weight.stride(0),
            output.stride(0),
            embedding_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return output.view(*input_.shape, embedding_dim)


def pre_comm_embedding_ops(
    input_: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_mode: TensorParallelMode,
    vocab_start_index: int,
    vocab_end_index: int,
    gather_output: bool,
    padding_size: int,
):
    if (tp_size == 1 and _EMBED_FUSED_GATHER and input_.is_cuda
            and weight.dim() == 2 and weight.stride(-1) == 1
            and not input_.dtype.is_floating_point
            and not torch.compiler.is_compiling()):
        if tp_mode == TensorParallelMode.COLUMN:
            lo, hi = vocab_start_index, vocab_end_index
        else:
            lo, hi = 0, weight.shape[0]
        output = _fused_masked_embedding(input_, weight, lo, hi)
        if tp_mode == TensorParallelMode.ROW and gather_output:
            if tp_rank == tp_size - 1 and padding_size > 0:
                output = F.pad(output, (0, padding_size))
        return output

    if tp_mode == TensorParallelMode.COLUMN:
        input_, input_mask = get_masked_input_and_mask(
            input_,
            vocab_start_index,
            vocab_end_index,
        )
    else:
        # flashinfer's rejection kernel (chain_speculative_sampling) pads non-accepted
        # tokens with -1. When the full vocab is local (non-TP or ROW TP), mask
        # out-of-range ids (e.g. -1) over [0, weight.shape[0]) to avoid an OOB
        # embedding lookup.
        input_, input_mask = get_masked_input_and_mask(
            input_,
            0,
            weight.shape[0],
        )

    # Get the embeddings.
    output = F.embedding(input_, weight)

    output.masked_fill_(input_mask, 0)

    if tp_mode == TensorParallelMode.ROW and gather_output:
        if tp_rank == tp_size - 1 and padding_size > 0:
            output = F.pad(output, (0, padding_size))

    return output


def embedding_skip_forward_impl(input: torch.Tensor, embedding_dim: int,
                                dtype: torch.dtype) -> torch.Tensor:
    output_shape = input.shape[:] + (embedding_dim, )
    output = input.new_empty(output_shape, dtype=dtype)
    return output


@torch.library.custom_op("trtllm::embedding_skip_forward", mutates_args=())
def embedding_skip_forward(input: torch.Tensor, embedding_dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


@embedding_skip_forward.register_fake
def _(input, embedding_dim, dtype):
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


class Embedding(LMHead):
    """Embedding layer.

    Adapted from torch.nn.Embedding.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        enable_torch_compile_for_embedding: Optional[bool] = False,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        self.enable_torch_compile_for_embedding = enable_torch_compile_for_embedding

        if self.tp_size > 1:
            slice_width = math.ceil(num_embeddings / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width,
                                       num_embeddings)
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings

    def forward(self, input):
        if self.tp_size > 1:
            # Run the ops before all_reduce/all_gather.
            # We use torch.compile() to fuse the tiny pointwise ops before all_reduce/all_gather for Embedding module.
            embedding_ops_func = torch.compile(
                pre_comm_embedding_ops,
                options={"max-autotune": True},
                disable=not self.enable_torch_compile_for_embedding)
        else:
            # Skip torch.compile when TP size is 1 to avoid unnecessary host overhead
            embedding_ops_func = pre_comm_embedding_ops
        output = embedding_ops_func(input, self.weight, self.tp_size,
                                    self.tp_rank, self.tp_mode,
                                    self.vocab_start_index,
                                    self.vocab_end_index, self.gather_output,
                                    self.padding_size)

        # Run the all_reduce/all_gather.
        if self.tp_size > 1:
            if self.tp_mode == TensorParallelMode.COLUMN:
                # Reduce across all the model parallel GPUs.
                output = self.all_reduce(output)
            elif self.tp_mode == TensorParallelMode.ROW:
                if self.gather_output:
                    # Run allgather.
                    output = allgather(output, self.mapping)
                    # Remove the padding.
                    if self.padding_size > 0:
                        output = output[..., :-self.padding_size]

        return output

    def skip_forward(self, input):
        return embedding_skip_forward(input, self.embedding_dim, self.dtype)
