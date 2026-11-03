"""Real-checkpoint load check for the CSA2 shared-input projections (no full model load).

Builds DeepseekV41Attention for one FULL/REINDEX layer and one SWA-only layer from
the checkpoint config, loads ONLY that layer's attention tensors from the
safetensors shards through ``load_hf_weights`` (the CSA2 loader, which reads every
parameter's checkpoint key), checks every checkpoint key of the layer's attention is
consumed (the census rule of DeepseekV41 attention ``load_weights``), and compares
the fused path (_project_inputs) bitwise against the separate Linears on the real
weights at M = 1, 2, 4, 8, 2048.
"""

import json
import os
import sys

import torch


def log(*a):
    print(*a, flush=True)


def main():
    ckpt = sys.argv[1]
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from safetensors import safe_open

    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Mode
    from tensorrt_llm.mapping import Mapping

    with open(os.path.join(ckpt, "config.json")) as f:
        raw_cfg = json.load(f)
    cfg_dict = raw_cfg.get("text_config", raw_cfg)

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    if not hasattr(cfg, "rope_scaling"):
        cfg.rope_scaling = None
    from transformers import PretrainedConfig

    pcfg = PretrainedConfig(**cfg_dict)
    layout = CSA2Layout(
        tuple(cfg.compress_ratios), tuple(cfg.kv_source_layer_ids),
        tuple(cfg.index_source_layer_ids), cfg.candidate_source_layer_id,
        cfg.candidate_topk_blocks, cfg.candidate_block_size, cfg.index_topk, cfg.sliding_window)
    n = cfg.num_hidden_layers
    modes = [layout.layer(i).mode for i in range(n)]
    full = next(i for i in range(n) if modes[i] in (CSA2Mode.FULL, CSA2Mode.REINDEX))
    swa = next(i for i in range(n) if modes[i] not in (CSA2Mode.FULL, CSA2Mode.REINDEX))
    log("modes", [str(m) for m in modes])
    log("test layers", full, modes[full], swa, modes[swa])

    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    failures = []
    report = {}
    for layer_idx in (full, swa):
        cands = [k for k in weight_map if f"layers.{layer_idx}.attn." in k
                 or f"layers.{layer_idx}.self_attn." in k]
        prefix = None
        for k in cands:
            for marker in (f"layers.{layer_idx}.attn.", f"layers.{layer_idx}.self_attn."):
                if marker in k:
                    prefix = k[: k.index(marker) + len(marker)]
                    break
            if prefix:
                break
        log("layer", layer_idx, "prefix", prefix, "keys", len(cands))
        # Only keys of this attention (not nested compressor/indexer of other kinds are included too).
        keys = [k for k in cands if k.startswith(prefix)]
        tensors = {}
        by_file = {}
        for k in keys:
            by_file.setdefault(weight_map[k], []).append(k)
        for fname, ks in by_file.items():
            with safe_open(os.path.join(ckpt, fname), framework="pt", device="cpu") as f:
                for k in ks:
                    tensors[k] = f.get_tensor(k)
        sample = sorted(k[len(prefix):] for k in keys)
        log("checkpoint attention keys:", sample)
        built = None
        for tp in (4, 1):
            try:
                mapping = Mapping(world_size=tp, tp_size=tp, rank=0, gpus_per_node=8)
                with torch.device("cuda"):
                    built = DeepseekV41Attention.from_hf_config(
                        pcfg, layer_idx, mapping=mapping, projection_quantization="mxfp8")
                built = built.cuda()
                log("constructed with tp", tp)
                break
            except Exception as e:
                log("construct tp", tp, "failed:", type(e).__name__, str(e)[:300])
        if built is None:
            failures.append(f"construct layer {layer_idx}")
            continue
        attn = built
        # Census as in DeepseekV41 CSA2 attention load_weights: every key has a consumer.
        aliases = {
            "o_a_proj": "wo_a.weight", "o_b_proj.weight": "wo_b.weight",
            "index_wq_b.weight": "indexer.wq_b.weight",
            "index_weights_proj.weight": "indexer.weights_proj.weight",
            "index_wk.weight": "indexer.wk.weight", "index_k_norm.weight": "indexer.k_norm.weight",
        }
        expected = {aliases.get(nm, nm) for nm, _ in attn.named_parameters()
                    if not nm.endswith("weight_scale")}
        local = {k[len(prefix):]: v for k, v in tensors.items()}
        scales = {k.removesuffix("weight") + "weight_scale_inv" for k in expected
                  if k.endswith("weight") and k in local and local[k].dtype == torch.float8_e4m3fn}
        extras = sorted(set(local) - expected - scales)
        missing = sorted(expected - set(local))
        log("census extras", extras, "missing", missing)
        try:
            attn.load_hf_weights({prefix + k: v for k, v in local.items()}, prefix)
        except Exception as e:
            log("load failed", type(e).__name__, str(e)[:500])
            failures.append(f"load layer {layer_idx}")
            continue
        torch.cuda.synchronize()
        fused_built = attn._wq_a_wkv is not None and (
            (attn._wq_b_index is not None) == hasattr(attn, "index_wq_b"))
        active = attn._shared_mxfp8_active(torch.empty((1, cfg.hidden_size), device="cuda",
                                                       dtype=torch.bfloat16))
        rec = {"census_extras": extras, "census_missing": missing, "fused_built": fused_built,
               "path_active": active, "has_index": hasattr(attn, "index_wq_b"),
               "wq_b_rows": attn.wq_b.weight.shape[0]}
        if extras or missing or not fused_built or not active:
            failures.append(f"census/fusion layer {layer_idx}")
        gen = torch.Generator(device="cuda").manual_seed(7 + layer_idx)
        for m in (1, 2, 4, 8, 2048):
            h = torch.randn((m, cfg.hidden_size), generator=gen, device="cuda").to(torch.bfloat16)
            qr, q, swa_rows, idx = attn._project_inputs(h)
            ref_qa = attn.wq_a(h)
            ref_qr = attn.q_norm(ref_qa)
            ref_q = attn.wq_b(ref_qr)
            ref_swa = attn.wkv(h)
            ok = torch.equal(qr, ref_qr) and torch.equal(q, ref_q) and torch.equal(swa_rows, ref_swa)
            if hasattr(attn, "index_wq_b"):
                ref_idx = attn.index_wq_b(ref_qr)
                ok = ok and idx is not None and torch.equal(idx, ref_idx)
            ok = ok and bool(torch.isfinite(q).all()) and bool(torch.isfinite(swa_rows).all())
            rec[f"M{m}_bitwise"] = ok
            log(f"layer {layer_idx} M={m} bitwise={ok}")
            if not ok:
                failures.append(f"layer {layer_idx} M={m}")
        report[str(layer_idx)] = rec
        del attn, built
        torch.cuda.empty_cache()
    print(json.dumps({"passed": not failures, "failures": failures, "report": report,
                      "compared": "real-checkpoint fused _project_inputs vs separate MXFP8 Linears; "
                                  "per-layer key census"}, default=str))


if __name__ == "__main__":
    main()
