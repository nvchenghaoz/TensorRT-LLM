# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact storage, batched arithmetic, selection and reload tests for packed BF16 projections."""

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.lossless_bf16_gemm import lossless_bf16_gemm
from tensorrt_llm._torch.modules.gemma4.packed_bf16 import (
    _PackedBF16LinearMethod,
    configure_packed_bf16,
    pack_lossless_bf16_weights,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode, UnquantizedLinearMethod
from tensorrt_llm.mapping import Mapping


def _unpack_bits(
    data: torch.Tensor, metadata: torch.Tensor, exceptions: torch.Tensor, *, reassociated: bool
) -> torch.Tensor:
    packed = data[:, 128:].long().reshape(-1, 16, 3)
    words = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    if reassociated:
        sm = data[:, :128].long()
        codes = torch.stack([(words >> (j * 3)) & 7 for j in range(8)], dim=-1).reshape(-1, 128)
        offsets = (metadata >> 8) & 0xFFFFFFFF
    else:
        sm = data[:, :128].reshape(-1, 32, 4).transpose(1, 2).reshape(-1, 128).long()
        words = torch.stack((words & 4095, words >> 12), dim=-1).reshape(-1, 32)
        codes = torch.stack([(words >> (j * 3)) & 7 for j in range(4)], dim=-1)
        codes = codes.transpose(1, 2).reshape(-1, 128)
        offsets = (metadata >> 8) - 1
    exponent = (metadata[:, None] & 255) - codes
    escaped = codes == 7
    indices = offsets[:, None] + escaped.long().cumsum(1) - 1
    exponent[escaped] = exceptions[indices[escaped]].long()
    return (sm & 127) | ((sm & 128) << 8) | (exponent << 7)


@pytest.mark.parametrize("reassociated", [False, True])
def test_every_encoding_and_chunk_offsets(reassociated: bool) -> None:
    bits = (
        torch.randperm(65536, generator=torch.Generator().manual_seed(123))
        .repeat(3)
        .reshape(-1, 128)
    )
    weight = bits.to(torch.int16).view(torch.bfloat16)
    torch.testing.assert_close(
        _unpack_bits(
            *pack_lossless_bf16_weights(weight, reassociated=reassociated),
            reassociated=reassociated,
        ),
        bits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weight.view(torch.int16).long() & 65535, bits, rtol=0, atol=0)


@pytest.mark.parametrize("reassociated", [False, True])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 256, 257, 1024])
@pytest.mark.parametrize("n,k", [(19, 128), (132, 384), (257, 5376)])
@torch.inference_mode()
def test_batched_gemm_and_graph_replay(m: int, n: int, k: int, reassociated: bool) -> None:
    torch.manual_seed(419)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Force exponent exceptions, including small values in otherwise large blocks.
    weight[:, ::7] *= 0.0001
    packed = pack_lossless_bf16_weights(weight, reassociated=reassociated)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    lossless_bf16_gemm(x, *packed, n, reassociated)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lossless_bf16_gemm(x, *packed, n, reassociated)
    for scale in (0.01, 1.0, 10.0):
        x.normal_().mul_(scale)
        graph.replay()
        expected = torch.nn.functional.linear(x.double(), weight.double())
        # FP32 accumulation with one final BF16 round, including cancellation.
        error = (output.double() - expected).abs()
        allowance = expected.abs() * 0.004 + 2e-5 * x.double().norm(dim=1)[
            :, None
        ] * weight.double().norm(dim=1)
        assert torch.all(error <= allowance)


@pytest.mark.parametrize("m", [0, 1, 2, 129, 256])
def test_fake_output(m: int) -> None:
    with FakeTensorMode():
        x = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
        data = torch.empty(19, 176, device="cuda", dtype=torch.uint8)
        metadata = torch.empty(19, device="cuda", dtype=torch.int64)
        exceptions = torch.empty(0, device="cuda", dtype=torch.uint8)
        output = lossless_bf16_gemm(x, data, metadata, exceptions, 19)
    assert output.shape == (m, 19)
    assert output.dtype == x.dtype


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_selection_reload_and_fallbacks(reassociated: bool) -> None:
    linear = Linear(256, 128, bias=False, dtype=torch.bfloat16).cuda()
    linear.weight.normal_()
    configure_packed_bf16(linear, reassociated=reassociated)
    method = linear.quant_method
    assert isinstance(method, _PackedBF16LinearMethod)
    assert not any("packed_bf16" in key for key in linear.state_dict())
    for _ in range(2):
        for m in (1, 2, 7, 64, 256):
            x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            for runner in method.runners:
                with patch.object(AutoTuner.get(), "choose_one", return_value=(runner, 0)):
                    result = linear(x)
                if runner.packed:
                    expected = lossless_bf16_gemm(
                        x,
                        linear._packed_bf16_data,
                        linear._packed_bf16_metadata,
                        linear._packed_bf16_exceptions,
                        128,
                        reassociated,
                    )
                else:
                    expected = torch.nn.functional.linear(x, linear.weight)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        linear.weight.normal_()
        linear.cache_derived_state()
        torch.testing.assert_close(
            _unpack_bits(
                linear._packed_bf16_data,
                linear._packed_bf16_metadata,
                linear._packed_bf16_exceptions,
                reassociated=reassociated,
            ),
            linear.weight.view(torch.int16).long().reshape(-1, 128) & 65535,
            rtol=0,
            atol=0,
        )
    with (
        torch.inference_mode(False),
        torch.enable_grad(),
        patch.object(method.original, "apply") as original,
    ):
        assert method.apply(linear, x, None) is original.return_value
    with patch.object(method.original, "apply") as original:
        assert (
            method.apply(linear, x, torch.ones(128, device="cuda", dtype=x.dtype))
            is original.return_value
        )
    for flag in ("use_custom_cublas_mm", "use_cute_dsl_bf16_gemm"):
        with patch.object(linear, flag, True), patch.object(method.original, "apply") as original:
            assert method.apply(linear, x, None) is original.return_value
    with (
        patch(
            "tensorrt_llm._torch.modules.gemma4.packed_bf16.is_torch_compiling", return_value=True
        ),
        patch.object(method.original, "apply") as original,
    ):
        assert method.apply(linear, x, None) is original.return_value


@pytest.mark.parametrize("failure", ["input", "weight", "metadata", "exceptions", "alignment", "k"])
def test_buffer_contract(failure: str) -> None:
    packed = list(
        pack_lossless_bf16_weights(torch.ones(4, 128, device="cuda", dtype=torch.bfloat16))
    )
    x = torch.ones(3, 128, device="cuda", dtype=torch.bfloat16)
    if failure == "input":
        x = x.float()
    elif failure == "weight":
        packed[0] = packed[0][:-1]
    elif failure == "metadata":
        packed[1] = packed[1].int()
    elif failure == "exceptions":
        packed[2] = packed[2].float()
    elif failure == "alignment":
        packed[0] = torch.cat((packed[0].flatten()[:1], packed[0].flatten()))[1:].reshape_as(
            packed[0]
        )
    else:
        x = x[:, :64].contiguous()
    with pytest.raises(ValueError):
        lossless_bf16_gemm(x, *packed, 4)


@torch.inference_mode()
def test_empty_batch() -> None:
    packed = pack_lossless_bf16_weights(torch.ones(4, 128, device="cuda", dtype=torch.bfloat16))
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    assert lossless_bf16_gemm(x, *packed, 4).shape == (0, 4)


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_current_stream_and_graph_replay(reassociated: bool) -> None:
    weight = torch.randn(29, 384, device="cuda", dtype=torch.bfloat16)
    packed = pack_lossless_bf16_weights(weight, reassociated=reassociated)
    x = torch.randn(7, 384, device="cuda", dtype=torch.bfloat16)
    lossless_bf16_gemm(x, *packed, 29, reassociated)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = lossless_bf16_gemm(x, *packed, 29, reassociated)
        x.normal_()
        graph.replay()
        expected = lossless_bf16_gemm(x, *packed, 29, reassociated)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.double(),
        torch.nn.functional.linear(x.double(), weight.double()),
        rtol=0.004,
        atol=0.001,
    )


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_compile_cache_accepts_new_exception_buffers(reassociated: bool) -> None:
    from tensorrt_llm._torch.custom_ops.lossless_bf16_gemm import _compile

    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    first_weight = torch.ones(17, 256, device="cuda", dtype=torch.bfloat16)
    first = pack_lossless_bf16_weights(first_weight, reassociated=reassociated)
    lossless_bf16_gemm(x, *first, 17, reassociated)
    cached = _compile.cache_info()
    second_weight = first_weight.clone()
    second_weight[:, ::7] = 1e-6
    second = pack_lossless_bf16_weights(second_weight, reassociated=reassociated)
    assert first[2].numel() != second[2].numel()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lossless_bf16_gemm(x, *second, 17, reassociated)
    graph.replay()
    assert _compile.cache_info().misses == cached.misses
    torch.testing.assert_close(
        output.double(),
        torch.nn.functional.linear(x.double(), second_weight.double()),
        rtol=0.004,
        atol=0.001,
    )


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_chunk_boundary_and_batch_independent_arithmetic(reassociated: bool) -> None:
    torch.manual_seed(714)
    weight = torch.randn(1025, 384, device="cuda", dtype=torch.bfloat16)
    weight[:, ::3] *= 1e-5
    packed = pack_lossless_bf16_weights(weight, reassociated=reassociated)
    x = torch.randn(9, 384, device="cuda", dtype=torch.bfloat16)
    individual = torch.cat(
        [lossless_bf16_gemm(row[None], *packed, 1025, reassociated) for row in x]
    )
    for m in (2, 3, 4, 7, 9):
        output = lossless_bf16_gemm(x[:m], *packed, 1025, reassociated)
        torch.testing.assert_close(output, individual[:m], rtol=0, atol=0)
    torch.testing.assert_close(
        _unpack_bits(*packed, reassociated=reassociated),
        weight.view(torch.int16).long().reshape(-1, 128) & 65535,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("reassociated", [False, True])
@pytest.mark.parametrize("tp_mode", list(TensorParallelMode))
@torch.inference_mode()
def test_leading_dimensions_noncontiguous_and_loading(reassociated: bool, tp_mode) -> None:
    module = Linear(256, 128, bias=False, dtype=torch.bfloat16, tensor_parallel_mode=tp_mode).cuda()
    module.weight.normal_()
    module.weight.data = module.weight.t().contiguous().t()
    configure_packed_bf16(module, reassociated=reassociated)
    method = module.quant_method
    assert isinstance(method, _PackedBF16LinearMethod)
    for _ in range(2):
        replacement = torch.randn_like(module.weight)
        module.load_weights([{"weight": replacement}])
        module.post_load_weights()
        x = torch.randn(3, 2, 256, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        with patch.object(AutoTuner.get(), "choose_one", return_value=(method.runners[1], 0)):
            result = module(x)
        expected = torch.nn.functional.linear(x.double(), replacement.double())
        torch.testing.assert_close(result.double(), expected, rtol=0.004, atol=0.001)
        assert result.shape == (2, 3, 128)
        configure_packed_bf16(module, reassociated=reassociated)
        assert module.quant_method is method


@pytest.mark.parametrize("reason", ["cpu", "dtype", "k", "bias", "custom", "subclass"])
@torch.inference_mode()
def test_selection_preserves_unsupported_methods(reason: str) -> None:
    module = Linear(
        64 if reason == "k" else 128,
        19,
        bias=reason == "bias",
        dtype=torch.float32 if reason == "dtype" else torch.bfloat16,
    )
    if reason != "cpu":
        module = module.cuda()
    if reason == "custom":
        module.use_custom_cublas_mm = True
    if reason == "subclass":

        class OtherMethod(UnquantizedLinearMethod):
            pass

        module.quant_method = OtherMethod()
    original = module.quant_method
    configure_packed_bf16(module, reassociated=True)
    assert module.quant_method is original
    assert not any("packed_bf16" in key for key, _ in module.named_buffers())


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_actual_autotuning_and_replayed_dispatch(reassociated: bool) -> None:
    module = Linear(384, 132, bias=False, dtype=torch.bfloat16).cuda()
    module.weight.normal_()
    configure_packed_bf16(module, reassociated=reassociated)
    x = torch.randn(7, 384, device="cuda", dtype=torch.bfloat16)
    with autotune():
        module(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = module(x)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(
        result.double(),
        torch.nn.functional.linear(x.double(), module.weight.double()),
        rtol=0.004,
        atol=0.001,
    )


@pytest.mark.parametrize("reassociated", [False, True])
@torch.inference_mode()
def test_compile_fake_and_eager_backend(reassociated: bool) -> None:
    weight = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    packed = pack_lossless_bf16_weights(weight, reassociated=reassociated)

    def project(x):
        return lossless_bf16_gemm(x, *packed, 19, reassociated)

    compiled = torch.compile(project, backend="eager", fullgraph=True, dynamic=True)
    for m in (1, 2, 9):
        x = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(x), project(x), rtol=0, atol=0)


@pytest.mark.parametrize("tp_mode", list(TensorParallelMode))
@pytest.mark.parametrize("rank", [0, 1])
@torch.inference_mode()
def test_tp_shards(tp_mode, rank: int) -> None:
    # Test each local shard without launching a collective or another GPU.
    mapping = Mapping(world_size=2, tp_size=2, rank=rank)
    module = Linear(
        512,
        256,
        bias=False,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=tp_mode,
        reduce_output=False,
    ).cuda()
    weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    module.load_weights([{"weight": weight}])
    configure_packed_bf16(module, reassociated=True)
    method = module.quant_method
    assert isinstance(method, _PackedBF16LinearMethod)
    dim = TensorParallelMode.split_dim(tp_mode)
    expected_weight = weight.chunk(2, dim=dim)[rank]
    torch.testing.assert_close(module.weight, expected_weight, rtol=0, atol=0)
    x = torch.randn(4, expected_weight.shape[1], device="cuda", dtype=torch.bfloat16)
    with patch.object(AutoTuner.get(), "choose_one", return_value=(method.runners[1], 0)):
        result = module(x)
    torch.testing.assert_close(
        result.double(),
        torch.nn.functional.linear(x.double(), expected_weight.double()),
        rtol=0.004,
        atol=0.001,
    )


@torch.inference_mode()
def test_lora_remains_separate_from_packed_base() -> None:
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
            self.b = torch.randn(19, 8, device="cuda", dtype=torch.bfloat16)

        def forward(self, x, params, layer_idx):
            assert layer_idx == 3
            return torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)

    adapter = Adapter()
    module = Linear(128, 19, bias=False, dtype=torch.bfloat16, lora=adapter).cuda()
    module.weight.normal_()
    configure_packed_bf16(module, reassociated=True)
    method = module.quant_method
    for m in (1, 4, 128):
        x = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)
        expected = lossless_bf16_gemm(
            x,
            module._packed_bf16_data,
            module._packed_bf16_metadata,
            module._packed_bf16_exceptions,
            19,
            True,
        ) + adapter(x, {}, 3)
        with patch.object(AutoTuner.get(), "choose_one", return_value=(method.runners[1], 0)):
            result = module(x, lora_params={"enabled": True}, layer_idx=3)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["0", "attention", "head", "all"])
@torch.inference_mode()
def test_model_cache_hook(mode: str, monkeypatch) -> None:
    from transformers import Gemma4TextConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    monkeypatch.setenv("TRTLLM_GEMMA4_PACKED_BF16", mode)
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=128,
        num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        dtype="bfloat16",
    )
    model = Gemma4ForCausalLM(ModelConfig(pretrained_config=config)).cuda().bfloat16()
    for parameter in model.parameters():
        parameter.normal_()
    ModelLoader._walk_cache_state(model)
    for layer in model.model.layers:
        for module in (layer.self_attn.qkv_proj, layer.self_attn.o_proj):
            assert isinstance(module.quant_method, _PackedBF16LinearMethod) == (
                mode in ("attention", "all")
            )
    assert isinstance(model.lm_head.quant_method, _PackedBF16LinearMethod) == (
        mode in ("head", "all")
    )
    ModelLoader._walk_cache_state(model)
    assert not any("packed_bf16" in key for key in model.state_dict())
