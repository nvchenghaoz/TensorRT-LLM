# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FLUX.3 text encoder: Qwen3-VL stacking hidden states across selected layers.

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from tensorrt_llm.logger import logger


@torch.library.custom_op("trtllm::flux3_text_rms_norm", mutates_args=())
def _text_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Preserve Qwen's eager RMSNorm reduction and BF16 rounding under compile."""
    normalized = x.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)


@_text_rms_norm.register_fake
def _text_rms_norm_fake(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return torch.empty_like(x)


class _CompileRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        return _text_rms_norm(x, self.weight, self.variance_epsilon)


@dataclass
class Qwen3VLEmbedderParams:
    # Layer indices (1-indexed into hidden_states) to stack along the channel dim.
    # 8 layers × 2560 (4B hidden) = 20480 = context_in_dim for the small model.
    output_layer: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32])
    torch_dtype: str = "bfloat16"
    max_length: int = 8192
    pad_multiple: int = 80
    use_compile: bool = False
    # Compiled path only: replay one CUDA graph per pre-warmed padded-length
    # bucket (kill switch TRTLLM_FLUX3_TEXT_GRAPH=0). Must be off when the
    # encoder is offloaded, since graphs bake in device weight pointers.
    use_cuda_graph: bool = True
    # Compiled path only: buckets k * pad_multiple for k = 1..warmup_buckets are
    # compiled and captured at warmup (T=80 covers the served prompts). Other
    # buckets compile on first use as before and never capture a graph.
    warmup_buckets: int = 1


class Qwen3VLEmbedder(torch.nn.Module):
    """FLUX.3 text encoder: stacks hidden states from selected Qwen3-VL layers.

    Small model (Qwen3-VL-4B, hidden=2560): 8 layers × 2560 → context_in_dim 20480.
    """

    def __init__(
        self,
        model_spec: str,
        params: Qwen3VLEmbedderParams | None = None,
        tokenizer_spec: Optional[str] = None,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if params is None:
            params = Qwen3VLEmbedderParams()
        self.params = params
        self.dtype = getattr(torch, params.torch_dtype)
        self.output_layer = list(params.output_layer)
        self.max_length = params.max_length

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_spec, torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(tokenizer_spec or model_spec)
        self.processor.tokenizer.padding_side = "right"

        lm = self.model.model.language_model
        if not self.output_layer or any(k < 0 or k > len(lm.layers) for k in self.output_layer):
            raise ValueError("output_layer must select existing Qwen hidden states")
        last_layer = max(self.output_layer)
        if last_layer < len(lm.layers):
            # Truncated outputs must retain the intermediate, pre-final-norm states.
            lm.layers = lm.layers[:last_layer]
            lm.norm = torch.nn.Identity()

        if params.use_compile:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

            for name, module in list(lm.named_modules()):
                if isinstance(module, Qwen3VLTextRMSNorm):
                    parent, _, child = name.rpartition(".")
                    lm.get_submodule(parent).add_module(
                        child, _CompileRMSNorm(module.weight, module.variance_epsilon)
                    )
            self._forward_tensors = torch.compile(
                self._forward_tensors,
                dynamic=False,
                options={"emulate_precision_casts": True, "triton.cudagraphs": False},
            )
        # Compiled path state: per-bucket graphs (T -> (graph, ids, mask, out)),
        # set once at warmup and never extended afterwards.
        self._graphs: dict[int, tuple] = {}
        self._graph_pool = None
        self._warm_lengths: set[int] = set()
        self._sealed = False

    def _tokenize(self, prompt: str) -> tuple[Tensor, Tensor]:
        """One tokenizer call, right-padded to the next pad_multiple bucket.

        Equal to tokenizing once for the real length and again with
        padding="max_length" to the bucket (right padding, pad_token_id, mask 0).
        """
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        ids = tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
        real = len(ids)
        length = min(
            math.ceil(real / self.params.pad_multiple) * self.params.pad_multiple,
            self.max_length,
        )
        pad = length - real
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * pad], dtype=torch.long)
        attention_mask = torch.tensor([[1] * real + [0] * pad], dtype=torch.long)
        return input_ids, attention_mask

    def _graphs_enabled(self) -> bool:
        return (
            self.params.use_compile
            and self.params.use_cuda_graph
            and os.environ.get("TRTLLM_FLUX3_TEXT_GRAPH", "1") != "0"
            and self.model.device.type == "cuda"
        )

    def _capture(self, input_ids: Tensor, attention_mask: Tensor) -> None:
        """Capture the compiled forward for one bucket on static device inputs.

        The compiled region has no host reads (text-only: positions are an
        arange, the 4D mask is built on device), so the graph is data-independent.
        All buckets share one pool: replays never overlap, and each replay's
        output is cloned before any other replay is enqueued.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._forward_tensors(input_ids, attention_mask)
        torch.cuda.current_stream().wait_stream(stream)
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool, capture_error_mode="thread_local"):
            out = self._forward_tensors(input_ids, attention_mask)
        torch.cuda.synchronize()
        self._graphs[input_ids.shape[1]] = (graph, input_ids, attention_mask, out)

    @torch.inference_mode()
    def warmup_buckets(self) -> None:
        """Compile and capture every bucket k * pad_multiple, k = 1..warmup_buckets.

        Graphs are captured only here; a later bucket outside this set compiles on
        first use (as without graphs) and runs the compiled code without a graph.
        """
        if not self.params.use_compile or self._sealed:
            return
        device = self.model.device
        pm = self.params.pad_multiple
        lengths = sorted(
            {min(k * pm, self.max_length) for k in range(1, self.params.warmup_buckets + 1)}
        )
        ids, mask = self._tokenize("warmup")
        real = int(mask.sum())
        use_graph = self._graphs_enabled()
        for length in lengths:
            t0 = time.time()
            n = min(real, length)
            input_ids = torch.full((1, length), self.processor.tokenizer.pad_token_id)
            input_ids[:, :n] = ids[:, :n]
            attention_mask = torch.zeros((1, length), dtype=torch.long)
            attention_mask[:, :n] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            for _ in range(2):
                self._forward_tensors(input_ids, attention_mask)
            if use_graph:
                self._capture(input_ids, attention_mask)
            torch.cuda.synchronize()
            self._warm_lengths.add(length)
            logger.info(
                f"FLUX.3 text encoder: warmed bucket T={length} "
                f"({'graph' if use_graph else 'compiled'}) in {time.time() - t0:.1f}s"
            )
        self._sealed = True

    def _run(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        length = input_ids.shape[1]
        entry = self._graphs.get(length)
        if entry is not None:
            graph, static_ids, static_mask, out = entry
            static_ids.copy_(input_ids, non_blocking=True)
            static_mask.copy_(attention_mask, non_blocking=True)
            graph.replay()
            return out.clone()
        if self._sealed and length not in self._warm_lengths:
            self._warm_lengths.add(length)
            logger.info(f"FLUX.3 text encoder: bucket T={length} not pre-warmed; compiles on use")
        device = self.model.device
        return self._forward_tensors(input_ids.to(device), attention_mask.to(device)).clone()

    @torch.no_grad()
    def _forward_hidden(self, prompts: list[str]) -> list[Tensor]:
        contexts = []
        for prompt in prompts:
            input_ids, attention_mask = self._tokenize(prompt)
            if self.params.use_compile:
                contexts.append(self._run(input_ids, attention_mask))
            else:
                contexts.append(
                    self._forward_tensors(
                        input_ids.to(self.model.device), attention_mask.to(self.model.device)
                    )
                )
        return contexts

    def _forward_tensors(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return torch.cat([out.hidden_states[k] for k in self.output_layer], dim=-1).to(self.dtype)

    @torch.inference_mode()
    def encode(self, prompts: List[str]) -> list[Tensor]:
        return self._forward_hidden(prompts)
