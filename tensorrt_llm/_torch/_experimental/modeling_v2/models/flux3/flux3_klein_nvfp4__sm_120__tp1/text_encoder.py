# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 klein text encoder as this target serves it: Qwen3-VL hidden states of selected
layers, stacked along the channel dim.

HF ``Qwen3VLForConditionalGeneration``'s language model, truncated after the deepest
selected layer (final norm dropped: the selected states are pre-norm), bf16, its
forward torch.compile'd (dynamic=False, emulate_precision_casts) with every
Qwen3-VL RMSNorm replaced by the bitwise-exact ``text_rms_norm`` op. A prompt is
tokenized once and right-padded to a multiple of ``pad_multiple``; the first bucket
(T = pad_multiple, which covers the served prompts) is compiled and captured as a CUDA
graph at warmup and replayed per prompt. A longer prompt runs its bucket's compiled
code without a graph (compiled on first use).

The weights must stay on one CUDA device: the graph bakes in their addresses, so this
encoder cannot be offloaded.
"""

from __future__ import annotations

import math
import time
from typing import List

import torch
from torch import Tensor

from tensorrt_llm.logger import logger

from .ops.norm.text_rms_norm import text_rms_norm

_COMPILE_OPTIONS = {"emulate_precision_casts": True, "triton.cudagraphs": False}


class _TextRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        return text_rms_norm(x, self.weight, self.variance_epsilon)


class Flux3KleinTextEncoder(torch.nn.Module):
    """Prompts -> per-prompt bf16 contexts [1, T, len(output_layer) * hidden].

    output_layer, max_length, pad_multiple and dtype are the package's text-encoder
    config (``layers``, ``max_tokens``, ``pad_multiple``, ``dtype``); output_layer
    indexes hidden_states (1-based: 0 is the embeddings).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        output_layer: List[int],
        max_length: int,
        pad_multiple: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

        self.dtype = dtype
        self.output_layer = list(output_layer)
        self.max_length = max_length
        self.pad_multiple = pad_multiple

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(tokenizer_path)
        self.processor.tokenizer.padding_side = "right"

        lm = self.model.model.language_model
        if not self.output_layer or any(k < 0 or k > len(lm.layers) for k in self.output_layer):
            raise ValueError("output_layer must select existing Qwen hidden states")
        last_layer = max(self.output_layer)
        if last_layer < len(lm.layers):
            # Truncated outputs must retain the intermediate, pre-final-norm states.
            lm.layers = lm.layers[:last_layer]
            lm.norm = torch.nn.Identity()

        for name, module in list(lm.named_modules()):
            if isinstance(module, Qwen3VLTextRMSNorm):
                parent, _, child = name.rpartition(".")
                lm.get_submodule(parent).add_module(
                    child, _TextRMSNorm(module.weight, module.variance_epsilon)
                )
        self._forward_tensors = torch.compile(
            self._forward_tensors, dynamic=False, options=_COMPILE_OPTIONS
        )
        # The warmed bucket's graph: (graph, static ids, static mask, static output).
        self._graph: tuple | None = None
        self._warm_lengths: set[int] = set()

    def _tokenize(self, prompt: str) -> tuple[Tensor, Tensor]:
        """One tokenizer call, right-padded to the next pad_multiple bucket (pad_token_id,
        mask 0): the same ids and mask as tokenizing for the real length and again with
        padding="max_length" to the bucket."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        ids = tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
        real = len(ids)
        length = min(math.ceil(real / self.pad_multiple) * self.pad_multiple, self.max_length)
        pad = length - real
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * pad], dtype=torch.long)
        attention_mask = torch.tensor([[1] * real + [0] * pad], dtype=torch.long)
        return input_ids, attention_mask

    @torch.inference_mode()
    def warmup(self) -> None:
        """Compile the T = pad_multiple bucket and capture it as a CUDA graph (once).

        The compiled region has no host reads (text only: positions are an arange, the
        4D mask is built on device), so the graph is data-independent; its output is
        cloned per replay.
        """
        if self._graph is not None:
            return
        device = self.model.device
        if device.type != "cuda":
            raise RuntimeError("the FLUX.3 klein text encoder must be on a CUDA device")
        t0 = time.time()
        length = min(self.pad_multiple, self.max_length)
        ids, mask = self._tokenize("warmup")
        n = min(int(mask.sum()), length)
        input_ids = torch.full((1, length), self.processor.tokenizer.pad_token_id)
        input_ids[:, :n] = ids[:, :n]
        attention_mask = torch.zeros((1, length), dtype=torch.long)
        attention_mask[:, :n] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        for _ in range(2):
            self._forward_tensors(input_ids, attention_mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._forward_tensors(input_ids, attention_mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = self._forward_tensors(input_ids, attention_mask)
        torch.cuda.synchronize()
        self._graph = (graph, input_ids, attention_mask, out)
        self._warm_lengths.add(length)
        logger.info(f"FLUX.3 text encoder: captured bucket T={length} in {time.time() - t0:.1f}s")

    def _run(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self._graph is not None and input_ids.shape[1] == self._graph[1].shape[1]:
            graph, static_ids, static_mask, out = self._graph
            static_ids.copy_(input_ids, non_blocking=True)
            static_mask.copy_(attention_mask, non_blocking=True)
            graph.replay()
            return out.clone()
        # A prompt past the captured bucket: its bucket's compiled code, no graph.
        length = input_ids.shape[1]
        if length not in self._warm_lengths:
            self._warm_lengths.add(length)
            logger.info(f"FLUX.3 text encoder: bucket T={length} not captured; compiles on use")
        device = self.model.device
        return self._forward_tensors(input_ids.to(device), attention_mask.to(device)).clone()

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
        """One context per prompt, in order (the pipeline passes [negative, positive], or
        one prompt for a flow-map checkpoint)."""
        return [self._run(*self._tokenize(prompt)) for prompt in prompts]
