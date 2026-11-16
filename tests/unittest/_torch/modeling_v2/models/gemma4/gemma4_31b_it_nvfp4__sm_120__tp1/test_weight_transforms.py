# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gemma-4-31B-IT-NVFP4 target's weight table, on the CPU with synthetic tensors.

* **manifest**: at the checkpoint's own dimensions the table reads exactly the text-tower
  names of the ModelOpt export (1372 of its 1728 keys; the other 356 are the vision tower and
  its embedder), sliding layers `[q ; k ; v]` and full-attention layers (index % 6 == 5,
  K = V) `[q ; k]`; on a small config, one manifest row per declared parameter;
* **transforms**: `[gate ; up]`, `1 / input_scale`, `alpha = input_scale * weight_scale_2`
  and the refusal of fused shards that disagree, `_sf_bytes`, and the scale blocks
  concatenated before the 128x4 swizzle (`block_scale_interleave` has a CPU kernel);
* **load**: a whole small synthetic checkpoint plus vision keys fills every parameter with
  its declared shape and dtype and leaves only the vision keys unread.

The key generator below is a compact restatement of the export's `weight_map` names.
"""

from __future__ import annotations

import re
import zlib
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.gemma4_31b_it_nvfp4__sm_120__tp1 import (  # noqa: E501
    weights as WT,
)

pytestmark = pytest.mark.cpu_only

e4m3 = torch.float8_e4m3fn
_LM = "model.language_model"
_NVFP4 = ("weight", "weight_scale", "weight_scale_2", "input_scale")
#: A sample of the composite checkpoint's other half (vision tower, its embedder, audio).
_VISION_KEYS = (
    "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
    "model.vision_tower.encoder.layers.26.mlp.down_proj.linear.weight",
    "model.vision_tower.patch_embedder.input_proj.weight",
    "model.vision_tower.patch_embedder.position_embedding_table",
    "model.vision_tower.std_scale",
    "model.embed_vision.embedding_projection.weight",
    "model.audio_tower.layers.0.weight",
)

#: The checkpoint's text_config.
_REAL = dict(
    num_hidden_layers=60,
    hidden_size=5376,
    intermediate_size=21504,
    vocab_size=262144,
    num_attention_heads=32,
    num_key_value_heads=16,
    head_dim=256,
    num_global_key_value_heads=4,
    global_head_dim=512,
)
#: Twelve layers with the checkpoint's pattern (two full-attention layers) and widths small
#: enough to build every tensor; hidden and intermediate are multiples of 64 (NVFP4 blocks).
_SMALL = dict(
    num_hidden_layers=12,
    hidden_size=64,
    intermediate_size=128,
    vocab_size=32,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    num_global_key_value_heads=1,
    global_head_dim=32,
)


def _full(i: int) -> bool:
    return i % 6 == 5


def _core(dims: dict) -> SimpleNamespace:
    """What `dims()` reads off the core: the text config (as the shell swaps it in) and the
    two attention geometries."""
    n = dims["num_hidden_layers"]
    cfg = PretrainedConfig(
        **dims,
        layer_types=["full_attention" if _full(i) else "sliding_attention" for i in range(n)],
    )
    hq = dims["num_attention_heads"]
    sliding = SimpleNamespace(
        head_dim=dims["head_dim"],
        q_width=hq * dims["head_dim"],
        kv_width=dims["num_key_value_heads"] * dims["head_dim"],
    )
    full = SimpleNamespace(
        head_dim=dims["global_head_dim"],
        q_width=hq * dims["global_head_dim"],
        kv_width=dims["num_global_key_value_heads"] * dims["global_head_dim"],
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=cfg, torch_dtype=torch.bfloat16),
        sliding=sliding,
        full=full,
    )


def _text_keys(num_layers: int) -> set:
    """The text-tower names of the export (`weight_map` minus the vision half)."""
    keys = {f"{_LM}.embed_tokens.weight", f"{_LM}.norm.weight"}
    for i in range(num_layers):
        p = f"{_LM}.layers.{i}"
        keys |= {
            f"{p}.{n}.weight"
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            )
        }
        keys.add(f"{p}.layer_scalar")
        keys |= {f"{p}.mlp.{m}.{s}" for m in ("gate_proj", "up_proj", "down_proj") for s in _NVFP4}
        projections = "qko" if _full(i) else "qkvo"  # K = V layers carry no v_proj
        keys |= {f"{p}.self_attn.{m}_proj.weight" for m in projections}
        keys |= {f"{p}.self_attn.{n}_norm.weight" for n in "qk"}
    return keys


def _read(manifest) -> set:
    return {k for keys, _, _ in manifest.values() for k in keys}


# ------------------------------------------------------------------ manifest ---


def test_manifest_reads_exactly_the_checkpoints_text_tower():
    manifest = WT.Gemma431BTp1Weights().manifest(_core(_REAL))
    want = _text_keys(60)
    assert len(want) == 1372  # the export's 1728 keys less 355 vision-tower + 1 embed_vision
    assert _read(manifest) == want, sorted(_read(manifest) ^ want)[:8]


def test_attention_sources_per_layer_kind():
    """Sliding layers concatenate q, k, v; the full-attention layers (every sixth) q and k."""
    manifest = WT.Gemma431BTp1Weights().manifest(_core(_REAL))
    for i in range(60):
        keys, transform, multi = manifest[f"l{i}_qkv_w"]
        names = tuple(k.rsplit(".", 2)[-2] for k in keys)
        assert names == (("q_proj", "k_proj") if _full(i) else ("q_proj", "k_proj", "v_proj")), i
        assert transform is WT._cat_rows and multi
    assert [i for i in range(60) if _full(i)] == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]


def test_manifest_rows_are_the_declared_parameters():
    table = WT.Gemma431BTp1Weights()
    core = _core(_SMALL)
    d = table.dims(core)
    declared = table.declare(d)
    assert set(table.manifest(core)) == set(declared.keys())
    s, hidden = _SMALL, _SMALL["hidden_size"]
    q_s, kv_s = 2 * s["head_dim"], s["num_key_value_heads"] * s["head_dim"]
    q_f, kv_f = 2 * s["global_head_dim"], s["num_global_key_value_heads"] * s["global_head_dim"]
    for i in range(12):
        qkv, o = declared[f"l{i}_qkv_w"], declared[f"l{i}_o_w"]
        if _full(i):
            assert qkv.shape == (q_f + kv_f, hidden) and o.shape == (hidden, q_f)
            assert declared[f"l{i}_q_norm"].shape == (s["global_head_dim"],)
        else:
            assert qkv.shape == (q_s + 2 * kv_s, hidden) and o.shape == (hidden, q_s)
            assert declared[f"l{i}_k_norm"].shape == (s["head_dim"],)
        assert declared[f"l{i}_gate_up_w"].shape == (2 * s["intermediate_size"], hidden // 2)
        assert declared[f"l{i}_gate_up_w"].dtype == torch.uint8
        assert declared[f"l{i}_down_sf"].shape == (WT._sf_bytes(hidden, s["intermediate_size"]),)
        assert declared[f"l{i}_layer_scalar"].dtype == torch.float32
    assert declared["embed"].shape == (s["vocab_size"], hidden)


def test_the_vision_keys_are_the_only_unconsumed_ones():
    weights = dict.fromkeys([*_text_keys(12), *_VISION_KEYS])
    assert WT.Gemma431BTp1Weights().expected_unconsumed(weights) == set(_VISION_KEYS)


# ---------------------------------------------------------------- transforms ---


def test_gate_up_is_gate_then_up():
    gate = torch.full((3, 4), 7, dtype=torch.uint8)
    up = torch.full((3, 4), 9, dtype=torch.uint8)
    assert torch.equal(WT._cat_rows((gate, up)), torch.cat([gate, up], 0))


def test_act_scale_is_the_reciprocal_and_alpha_the_product():
    is_, ws2 = torch.tensor(0.004), torch.tensor(0.125)
    act = WT._act_scale((is_, is_.clone()))
    assert act.shape == (1,) and act.dtype == torch.float32
    assert act.item() == pytest.approx(1.0 / 0.004, rel=1e-7)
    alpha = WT._alpha((is_, is_.clone(), ws2, ws2.clone()))
    assert alpha.shape == (1,) and alpha.dtype == torch.float32
    assert alpha.item() == pytest.approx(0.004 * 0.125, rel=1e-7)
    # down_proj's single shard: a bare tensor or a one-tuple.
    assert WT._act_scale(is_).item() == act.item()
    assert WT._act_scale((is_,)).item() == act.item()
    assert WT._alpha((is_, ws2)).item() == alpha.item()


@pytest.mark.parametrize("which", ["input_scale", "weight_scale_2"])
def test_fused_shards_that_disagree_are_refused(which):
    a, b = torch.tensor(0.004), torch.tensor(0.008)
    if which == "input_scale":
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._act_scale((a, b))
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._alpha((a, b, a, a))
    else:
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._alpha((a, a, a, b))


def test_layer_scalar_and_scalars_keep_one_element():
    out = WT._to_f32_one(torch.tensor([0.75], dtype=torch.bfloat16))
    assert out.shape == (1,) and out.dtype == torch.float32 and out.item() == 0.75
    scalar = torch.tensor(0.5)
    assert WT._materialize(scalar) is scalar  # a 0-dim scale rejects `[:]`


@pytest.mark.parametrize(
    "rows, cols, want",
    [
        (2 * 21504, 5376, 43008 * 336),  # gate_up: rows and K/16 already aligned
        (5376, 21504, 5376 * 1344),  # down_proj
        (1, 16, 128 * 4),  # one row, one block: padded to 128 x 4
        (129, 80, 256 * 8),  # 5 blocks pad to 8
    ],
)
def test_sf_bytes_pads_rows_to_128_and_blocks_to_4(rows, cols, want):
    assert WT._sf_bytes(rows, cols) == want


def test_gate_up_scale_blocks_concatenate_before_the_swizzle():
    """Two 64-row blocks swizzle as one 128-row buffer (no padding between them), the
    length the fused weight's `_sf_bytes` declares."""
    rows, cols = 64, 64
    g = torch.Generator().manual_seed(8)
    gate_sf = torch.randint(0, 127, (rows, cols // WT.SF_VEC), generator=g, dtype=torch.uint8)
    up_sf = torch.randint(0, 127, (rows, cols // WT.SF_VEC), generator=g, dtype=torch.uint8)
    out = WT._swizzled_sf((gate_sf.view(e4m3), up_sf.view(e4m3)))
    assert out.dtype == torch.uint8 and out.dim() == 1
    assert out.numel() == WT._sf_bytes(2 * rows, cols) == 128 * 4
    assert torch.equal(out, torch.ops.trtllm.block_scale_interleave(torch.cat([gate_sf, up_sf])))


# ---------------------------------------------------------------------- load ---


def _source(key: str, d: SimpleNamespace) -> torch.Tensor:
    """A synthetic checkpoint tensor of the stored shape and dtype for `key`. gate_proj and
    up_proj share their scalars, as the export's fused pair does."""
    role = re.sub(r"(gate|up)_proj", "gate_up", key)
    g = torch.Generator().manual_seed(zlib.crc32(role.encode()))
    if key.endswith(("weight_scale_2", "input_scale")):
        return torch.tensor(0.01 + torch.rand(1, generator=g).item())
    if key.endswith("layer_scalar"):
        return torch.rand(1, generator=g).to(torch.bfloat16)
    mlp = re.search(r"\.mlp\.(gate|up|down)_proj\.(weight|weight_scale)$", key)
    if mlp:
        rows, cols = (d.hidden, d.inter) if mlp.group(1) == "down" else (d.inter, d.hidden)
        if mlp.group(2) == "weight":
            return torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
        return torch.rand(rows, cols // WT.SF_VEC, generator=g).to(e4m3)
    layer = re.search(r"\.layers\.(\d+)\.", key)
    full = layer is not None and not d.is_sliding[int(layer.group(1))]
    head = d.full_head_dim if full else d.sliding_head_dim
    q_width = d.full_q_width if full else d.sliding_q_width
    kv_width = d.full_qk_rows - q_width if full else (d.sliding_qkv_rows - q_width) // 2
    shapes = {
        "q_proj.weight": (q_width, d.hidden),
        "k_proj.weight": (kv_width, d.hidden),
        "v_proj.weight": (kv_width, d.hidden),
        "o_proj.weight": (d.hidden, q_width),
        "_norm.weight": (head,),
        "layernorm.weight": (d.hidden,),
        "embed_tokens.weight": (d.vocab, d.hidden),
        "language_model.norm.weight": (d.hidden,),
    }
    for suffix, shape in shapes.items():
        if key.endswith(suffix):
            return torch.randn(*shape, generator=g).to(torch.bfloat16)
    raise AssertionError(f"no synthetic source for {key}")


def test_load_fills_every_parameter_and_covers_the_checkpoint(monkeypatch):
    """`load` asserts each transformed source has its parameter's shape and dtype and that
    only the vision keys stay unread; then spot-check the stored values."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    table = WT.Gemma431BTp1Weights()
    core = _core(_SMALL)
    d = table.dims(core)
    core.w = table.declare(d)
    weights = {k: _source(k, d) for k in _text_keys(12)}
    weights.update({k: torch.zeros(2, 2) for k in _VISION_KEYS})

    table.load(SimpleNamespace(model=core), weights)

    p = f"{_LM}.layers"
    attn = lambda i, n: weights[f"{p}.{i}.self_attn.{n}_proj.weight"]  # noqa: E731
    assert torch.equal(core.w["l0_qkv_w"], torch.cat([attn(0, "q"), attn(0, "k"), attn(0, "v")]))
    assert torch.equal(core.w["l5_qkv_w"], torch.cat([attn(5, "q"), attn(5, "k")]))
    assert torch.equal(core.w["l11_o_w"], attn(11, "o"))
    mlp = lambda i, n, s: weights[f"{p}.{i}.mlp.{n}_proj.{s}"]  # noqa: E731
    assert torch.equal(
        core.w["l3_gate_up_w"], torch.cat([mlp(3, "gate", "weight"), mlp(3, "up", "weight")])
    )
    gate_is, gate_ws2 = mlp(0, "gate", "input_scale"), mlp(0, "gate", "weight_scale_2")
    assert core.w["l0_gate_up_is"].item() == pytest.approx(1.0 / gate_is.item(), rel=1e-6)
    assert core.w["l0_gate_up_alpha"].item() == pytest.approx(
        gate_is.item() * gate_ws2.item(), rel=1e-6
    )
    down_is, down_ws2 = mlp(0, "down", "input_scale"), mlp(0, "down", "weight_scale_2")
    assert core.w["l0_down_is"].item() == pytest.approx(1.0 / down_is.item(), rel=1e-6)
    assert core.w["l0_down_alpha"].item() == pytest.approx(
        down_is.item() * down_ws2.item(), rel=1e-6
    )
    assert core.w["l7_layer_scalar"].item() == weights[f"{p}.7.layer_scalar"].float().item()
    assert torch.equal(core.w["embed"], weights[f"{_LM}.embed_tokens.weight"])


def test_load_refuses_an_unknown_text_key(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    table = WT.Gemma431BTp1Weights()
    core = _core(_SMALL)
    d = table.dims(core)
    core.w = table.declare(d)
    weights = {k: _source(k, d) for k in _text_keys(12)}
    weights[f"{_LM}.layers.5.self_attn.v_proj.weight"] = torch.zeros(2, 2)
    with pytest.raises(AssertionError, match="does not load"):
        table.load(SimpleNamespace(model=core), weights)
