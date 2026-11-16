# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Gemma-4-31B-IT-NVFP4 / sm_120 / tp1.

One `W` per weight role (shape, checkpoint source, dtype, load-time transform);
`Gemma431BTp1Weights` unrolls the table into declaration, manifest and load. tp1 means no
per-rank slicing. Storage is HF `[out, in]`.

The checkpoint is ModelOpt NVFP4 with every layer's `self_attn` excluded: the attention
projections, norms and the (tied) embedding are bf16; the dense MLP is NVFP4 (group 16, one
global weight scale and one static input scale per projection). The vision tower and its
embedder are not loaded (the target serves text).

Load-time transforms reproduce the built-in loader bit for bit:

* Sliding-window layers' `qkv_proj` is `[q ; k ; v]`. Full-attention layers have K = V
  (`attention_k_eq_v`, no `v_proj` in the checkpoint) and keep `[q ; k]` only: V is taken
  from the raw K block.
* MLP `gate_up` is `[gate ; up]` with the two scale-factor blocks concatenated and then
  128x4-swizzled; gate and up share the input scale and global scale. Activation scales are
  stored the way the NVFP4 quantize and GEMMs read them: `input_scale = 1 / ckpt
  input_scale`, `alpha = ckpt input_scale * weight_scale_2`.
* `layer_scalar` is fp32, as the built-in model's buffer is.

`load(model, weights)` fills every declared parameter once and asserts coverage both ways:
every checkpoint key is consumed or belongs to the vision half.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import nn

ALL_LAYERS = object()

_LM = "model.language_model"
_NOT_LOADED = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")

#: Elements per NVFP4 scale factor.
SF_VEC = 16

f32 = torch.float32
u8 = torch.uint8


@dataclass(frozen=True)
class W:
    """One weight role. `shape` takes the dimension bundle; `src` is a checkpoint key template
    (`{p}` is the layer prefix) or a tuple of templates handed to `transform` together;
    `layers` is `ALL_LAYERS`, `None` (not per-layer) or `d -> iterable[int]`."""

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[str, ...]
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def keys(self, prefix: str) -> tuple[str, ...]:
        templates = self.src if isinstance(self.src, tuple) else (self.src,)
        return tuple(t.format(p=prefix) for t in templates)


def _materialize(entry) -> torch.Tensor:
    """`[:]` on a lazy safetensors slice; a 0-dim scale rejects that index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


def _sliding_layers(d):
    return [i for i in range(d.num_layers) if d.is_sliding[i]]


def _full_layers(d):
    return [i for i in range(d.num_layers) if not d.is_sliding[i]]


# ------------------------------------------------------------- transforms ---


def _cat_rows(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat(t, 0)


def _swizzled_sf(t) -> torch.Tensor:
    """`[rows, K/16]` e4m3 scale blocks, concatenated along rows, as the 128x4-swizzled 1-D
    buffer the NVFP4 GEMMs read."""
    blocks = t if isinstance(t, tuple) else (t,)
    sf = torch.cat([b.view(u8) for b in blocks], 0)
    return torch.ops.trtllm.block_scale_interleave(sf)


def _shared_scalar(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """One scalar every fused shard carries; shards that disagree are refused."""
    first = t[0].to(f32)
    for other in t[1:]:
        assert torch.allclose(first, other.to(f32)), f"fused shards disagree: {t}"
    return first.reshape([])


def _act_scale(t) -> torch.Tensor:
    """`1 / input_scale` (the checkpoint stores amax / (448 * 6)), one element."""
    s = _shared_scalar(t if isinstance(t, tuple) else (t,))
    return (1.0 / s).reshape([1])


def _alpha(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`input_scale * weight_scale_2` for `(input scales..., weight_scale_2s...)`."""
    half = len(t) // 2
    input_scale = _shared_scalar(t[:half])
    weight_scale_2 = _shared_scalar(t[half:])
    return (input_scale.float() * weight_scale_2.float()).reshape([1])


def _to_f32_one(w: torch.Tensor) -> torch.Tensor:
    return w.to(f32).reshape([1])


_ONE = lambda d: (1,)  # noqa: E731 - a one-element parameter


def _sf_bytes(rows: int, cols: int) -> int:
    """Bytes of a 128x4-swizzled scale-factor buffer for an `[rows, cols]` NVFP4 weight."""
    pad = lambda x, y: (x + y - 1) // y * y  # noqa: E731
    return pad(rows, 128) * pad(cols // SF_VEC, 4)


def _mlp(tag: str, names: tuple[str, ...], rows, cols) -> tuple[W, ...]:
    """An NVFP4 projection (or the gate|up pair): packed weight, swizzled scale factors,
    activation scale and alpha, as the W4A4 GEMMs read them."""
    mods = tuple(f"{{p}}.mlp.{n}" for n in names)
    return (
        W(
            f"{tag}_w",
            shape=lambda d: (rows(d), cols(d) // 2),
            dtype=u8,
            src=tuple(f"{m}.weight" for m in mods),
            transform=_cat_rows if len(mods) > 1 else None,
        ),
        W(
            f"{tag}_sf",
            shape=lambda d: (_sf_bytes(rows(d), cols(d)),),
            dtype=u8,
            src=tuple(f"{m}.weight_scale" for m in mods),
            transform=_swizzled_sf,
        ),
        W(
            f"{tag}_is",
            shape=_ONE,
            dtype=f32,
            src=tuple(f"{m}.input_scale" for m in mods),
            transform=_act_scale,
        ),
        W(
            f"{tag}_alpha",
            shape=_ONE,
            dtype=f32,
            src=tuple(f"{m}.input_scale" for m in mods)
            + tuple(f"{m}.weight_scale_2" for m in mods),
            transform=_alpha,
        ),
    )


def _norm(name: str, src: str, width) -> W:
    return W(name, shape=lambda d: (width(d),), src=src)


class Gemma431BTp1Weights:
    """`declare` allocates every weight the table names, `manifest` maps checkpoint keys onto
    them, `load` fills them. `declare` and `manifest` share one "which parameters exist"
    decision (`_targets`)."""

    WEIGHTS: tuple[W, ...] = (
        # --- every layer: four norms, the layer scalar and the NVFP4 MLP ---
        _norm("in_norm", "{p}.input_layernorm.weight", lambda d: d.hidden),
        _norm("post_attn_norm", "{p}.post_attention_layernorm.weight", lambda d: d.hidden),
        _norm("pre_ffn_norm", "{p}.pre_feedforward_layernorm.weight", lambda d: d.hidden),
        _norm("post_ffn_norm", "{p}.post_feedforward_layernorm.weight", lambda d: d.hidden),
        W(
            "layer_scalar",
            shape=_ONE,
            dtype=f32,
            src="{p}.layer_scalar",
            transform=_to_f32_one,
        ),
        *_mlp("gate_up", ("gate_proj", "up_proj"), lambda d: 2 * d.inter, lambda d: d.hidden),
        *_mlp("down", ("down_proj",), lambda d: d.hidden, lambda d: d.inter),
        # --- sliding-window layers: bf16 [q ; k ; v], head 256 ---
        W(
            "qkv_w",
            shape=lambda d: (d.sliding_qkv_rows, d.hidden),
            src=tuple(f"{{p}}.self_attn.{n}_proj.weight" for n in ("q", "k", "v")),
            transform=_cat_rows,
            layers=_sliding_layers,
        ),
        W(
            "o_w",
            shape=lambda d: (d.hidden, d.sliding_q_width),
            src="{p}.self_attn.o_proj.weight",
            layers=_sliding_layers,
        ),
        W(
            "q_norm",
            shape=lambda d: (d.sliding_head_dim,),
            src="{p}.self_attn.q_norm.weight",
            layers=_sliding_layers,
        ),
        W(
            "k_norm",
            shape=lambda d: (d.sliding_head_dim,),
            src="{p}.self_attn.k_norm.weight",
            layers=_sliding_layers,
        ),
        # --- full-attention layers: bf16 [q ; k] (K = V), head 512 ---
        W(
            "qkv_w",
            shape=lambda d: (d.full_qk_rows, d.hidden),
            src=tuple(f"{{p}}.self_attn.{n}_proj.weight" for n in ("q", "k")),
            transform=_cat_rows,
            layers=_full_layers,
        ),
        W(
            "o_w",
            shape=lambda d: (d.hidden, d.full_q_width),
            src="{p}.self_attn.o_proj.weight",
            layers=_full_layers,
        ),
        W(
            "q_norm",
            shape=lambda d: (d.full_head_dim,),
            src="{p}.self_attn.q_norm.weight",
            layers=_full_layers,
        ),
        W(
            "k_norm",
            shape=lambda d: (d.full_head_dim,),
            src="{p}.self_attn.k_norm.weight",
            layers=_full_layers,
        ),
        # --- not per-layer: the tied embedding / lm_head and the final norm ---
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src=f"{_LM}.embed_tokens.weight",
            layers=None,
        ),
        W("norm", shape=lambda d: (d.hidden,), src=f"{_LM}.norm.weight", layers=None),
    )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            is_sliding=[t == "sliding_attention" for t in cfg.layer_types],
            hidden=cfg.hidden_size,
            inter=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            sliding_head_dim=core.sliding.head_dim,
            sliding_q_width=core.sliding.q_width,
            sliding_qkv_rows=core.sliding.q_width + 2 * core.sliding.kv_width,
            full_head_dim=core.full.head_dim,
            full_q_width=core.full.q_width,
            full_qk_rows=core.full.q_width + core.full.kv_width,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, weights) -> set:
        """The vision (and any audio) half of the composite checkpoint."""
        return {k for k in weights if k.startswith(_NOT_LOADED)}

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares (meta-init intercepts `torch.empty`)."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(tuple(entry.shape(d)), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"{_LM}.layers.{i}") for i in layers]

    def manifest(self, core) -> dict[str, tuple[tuple[str, ...], Callable | None, bool]]:
        """param key -> (checkpoint keys, transform, whether the transform takes a tuple)."""
        d = self.dims(core)
        rows = {}
        for entry in self.WEIGHTS:
            for key, _, prefix in self._targets(entry, d):
                assert key not in rows, f"two table entries declare {key}"
                rows[key] = (entry.keys(prefix), entry.transform, isinstance(entry.src, tuple))
        return rows

    def load(self, model, weights) -> None:
        """Fill every declared parameter and assert coverage in both directions."""
        core = model.model
        manifest = self.manifest(core)
        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )
        consumed: set = set()
        for param_key, (keys, transform, multi) in manifest.items():
            dst = core.w[param_key].data
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            src_all = tuple(
                _materialize(weights[key]).to(dst.device, non_blocking=True) for key in keys
            )
            if transform is not None:
                src = transform(src_all if multi else src_all[0])
            else:
                assert len(src_all) == 1, "a multi-key weight needs a transform"
                src = src_all[0]
            assert tuple(dst.shape) == tuple(src.shape), (param_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (param_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)
        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed - self.expected_unconsumed(weights)
        assert not leftover, f"checkpoint keys this target does not load: {sorted(leftover)[:8]}"


MODEL_WEIGHTS = Gemma431BTp1Weights()
