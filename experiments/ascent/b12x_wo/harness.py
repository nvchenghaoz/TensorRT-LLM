"""Run the CSA2 decode output projection exactly as the model calls it.

Builds a stand-in for the CSA2 attention module (per-rank TP4 shapes, synthetic
weights stored the way csa2/weights.py stores them on SM120) with a real MXFP8
``Linear`` as o_b_proj, and calls ``DeepseekV41Attention._project_output`` from
whichever tree runs it: main's chain in the baseline tree, the b12x path in the
candidate tree.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention  # noqa: E402
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

try:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import b12x_wo as B  # noqa: F401
    HAS_B12X_WO = True
except ImportError:
    B = None
    HAS_B12X_WO = False


def make_o_b_proj(dims, wb_q, wb_sf):
    with torch.device("cuda"):
        lin = Linear(dims["groups_local"] * dims["rank"], dims["hidden"], bias=False,
                     dtype=torch.bfloat16, mapping=Mapping(world_size=1, tp_size=1, rank=0),
                     tensor_parallel_mode=TensorParallelMode.ROW,
                     quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    with torch.no_grad():
        lin.weight.copy_(wb_q)
        lin.weight_scale.copy_(wb_sf.view(torch.uint8).reshape(lin.weight_scale.shape))
    return lin


def make_module(dims, w, cos_sin):
    """Stand-in module with the attributes _project_output and its helpers read."""
    m = SimpleNamespace()
    m.num_heads_tp = dims["heads_tp"]
    m.qk_head_dim = m.v_head_dim = dims["head_dim"]
    m.qk_rope_head_dim = dims["rope"]
    m.qk_nope_head_dim = dims["nope"]
    m.n_local_groups = dims["groups_local"]
    m.o_lora_rank = dims["rank"]
    m.projection_quantization = "mxfp8"
    m.dtype = torch.bfloat16
    m.o_a_proj = w.wa_main
    m.o_a_proj_scale = w.sa_main
    m.o_a_proj_dequant = None
    m._o_a_proj_block_scale = w.sa_pow2
    m._o_a_fp8_block128 = True
    m._b12x_wo = None
    m._b12x_wo_built = False
    m.use_cute_dsl_blockscaling_bmm = False
    m.use_cute_dsl_bf16_bmm = False
    m.inverse_rotary_emb = SimpleNamespace(rotary_cos_sin=cos_sin, is_neox=False)
    m.o_b_proj = make_o_b_proj(dims, w.wb_q, w.wb_sf)
    return m


def project(m, output, positions):
    return DeepseekV41Attention._project_output(m, output, positions)


def path_name(m):
    b = getattr(m, "_b12x_wo", None)
    if b is None:
        return "main"
    return "b12x_wo_a+fused_wo_b" if b._use_direct_wo_b() else "b12x_wo_a+o_b_proj"
