"""Branch 70 (idea 128): shared-expert row split in DeepseekV4MoE.forward.

Builds one rank's shared GatedMLP exactly as DeepseekV4MoE.__init__ does
(quant configs from _get_shared_expert_projection_quant_configs on the
checkpoint's config, overridden_tp_size = shared_tp_size under a TP4 mapping,
swiglu_limit), with seeded synthetic weights (no model load), and drives the
REAL DeepseekV4MoE.forward of whichever tree runs it. The routed path is a
stub returning a preallocated buffer (restored before every call, the reset
point), and the all-reduce is an identity, so forward = shared chain + add.
"""

import copy
import json
import math
import os

import torch
from torch import nn

CKPT = os.environ.get("CKPT", "")
TP = 4
LAYER = 0


def log(*a):
    print(*a, flush=True)


def read_dims():
    cfg = {}
    if CKPT and os.path.exists(os.path.join(CKPT, "config.json")):
        with open(os.path.join(CKPT, "config.json")) as f:
            cfg = json.load(f)
    hidden = int(cfg.get("hidden_size", 5120))
    inter = int(cfg.get("moe_intermediate_size", 2304)) * int(cfg.get("n_shared_experts", 1))
    limit = cfg.get("swiglu_limit", None)
    return hidden, inter, limit, cfg


def make_mapping(rank, tp=TP):
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=tp, rank=rank, gpus_per_node=8, tp_size=tp, moe_ep_size=tp,
                   moe_tp_size=1)


def model_config_for(rank):
    """ModelConfig with the checkpoint's quant config and a TP4 mapping."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    mapping = make_mapping(rank)
    src = "fallback"
    mc = None
    if CKPT:
        try:
            mc = ModelConfig.from_pretrained(CKPT, trust_remote_code=True, mapping=mapping)
            src = "checkpoint"
        except Exception as e:  # noqa: BLE001
            log("ModelConfig.from_pretrained failed:", repr(e)[:400])
    if mc is None:
        mc = ModelConfig(mapping=mapping,
                         quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                                                  group_size=128))
    return mc, src


def build_shared_mlp(rank, seed=1234):
    """One rank's shared GatedMLP as DeepseekV4MoE builds it, seeded weights."""
    from tensorrt_llm._torch.models import modeling_deepseekv4 as m4
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    hidden, inter, limit, _ = read_dims()
    mc, src = model_config_for(rank)
    gu, dn = m4.DeepseekV4MoE._get_shared_expert_projection_quant_configs(mc, LAYER)
    shared_mc = mc
    if gu == dn and gu is not mc.quant_config:
        shared_mc = copy.copy(mc)
        shared_mc.quant_config = gu
    blocks = [q.group_size for q in (gu, dn)
              if q is not None and q.quant_algo is not None and q.group_size is not None]
    block = max(blocks, default=1)
    shared_tp = math.gcd(inter // block, TP)
    scale = shared_tp / TP if shared_tp != TP else None
    mlp = GatedMLP(hidden_size=hidden, intermediate_size=inter, bias=False, dtype=torch.bfloat16,
                   config=shared_mc, overridden_tp_size=shared_tp, reduce_output=False,
                   use_cute_dsl_blockscaling_mm=mc.use_cute_dsl_blockscaling_mm,
                   swiglu_limit=limit).cuda()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in mlp.named_parameters():
        if p.numel() == 0:
            continue
        if p.dtype == torch.float8_e4m3fn:
            v = torch.randn(p.shape, generator=gen).clamp_(-8, 8).to(torch.float8_e4m3fn)
        elif p.dtype == torch.float32 and "weight_scale" in name:
            base = 0.01 if "gate_up" in name else 0.02
            v = base * torch.exp2(torch.randint(-1, 2, p.shape, generator=gen).float())
        elif p.dtype == torch.float32:
            v = torch.ones(p.shape)
        elif p.dtype in (torch.bfloat16, torch.float16):
            v = (torch.randn(p.shape, generator=gen) * 0.02).to(p.dtype)
        elif p.dtype == torch.uint8:
            v = torch.randint(0, 256, p.shape, generator=gen, dtype=torch.uint8)
        else:
            raise RuntimeError(f"unhandled param {name} {p.dtype}")
        p.data.copy_(v.to(p.device))
    for lin in (getattr(mlp, "gate_up_proj", None), mlp.down_proj):
        if lin is not None and hasattr(lin, "post_load_weights"):
            lin.post_load_weights()
    mlp.post_load_weights()
    info = {"quant_src": src, "gate_up_quant": str(getattr(gu, "quant_algo", None)),
            "down_quant": str(getattr(dn, "quant_algo", None)), "block": block,
            "shared_tp": shared_tp, "scale": scale, "hidden": hidden, "inter": inter,
            "swiglu_limit": limit,
            "gate_up_tp_rank": mlp.gate_up_proj.tp_rank, "down_tp_rank": mlp.down_proj.tp_rank,
            "down_tp_size": mlp.down_proj.tp_size}
    return mlp, shared_tp, scale, info


class RoutedStub:
    def __init__(self):
        self.buf = None

    def __call__(self, *a, **k):
        return self.buf


def build_moe(rank, mlp, shared_tp, scale, top_k=6):
    """A DeepseekV4MoE without routed experts; forward is the tree's real code."""
    from tensorrt_llm._torch.models import modeling_deepseekv4 as m4

    moe = m4.DeepseekV4MoE.__new__(m4.DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.top_k = top_k
    moe.use_dp = False
    moe.mapping = make_mapping(rank)
    moe.shared_experts = mlp
    moe.shared_output_scale = scale
    moe._per_layer_dynamic_eplb_enabled = False
    moe.event_dict = {m4.EventType.Main: torch.cuda.Event(),
                      m4.EventType.MoeShared: torch.cuda.Event()}
    moe.aux_stream = None
    stub = RoutedStub()
    moe.compute_routed_output = stub
    moe.allreduce = lambda x, all_reduce_params=None: x
    has_split = hasattr(moe, "_setup_shared_row_split")
    if has_split:
        moe._setup_shared_row_split(shared_tp, LAYER)
    return moe, stub, has_split


def sm_clock_mhz():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:  # noqa: BLE001
        return None
