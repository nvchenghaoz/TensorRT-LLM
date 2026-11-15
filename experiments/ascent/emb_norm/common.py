"""Shared setup for the Gemma4 embedding-chain / final-norm microbenchmarks.

Builds main's exact modules (Gemma4TextScaledWordEmbedding, the layer-0
input RMSNorm and the model-final RMSNorm) at the checkpoint's shapes with
synthetic weights, plus the token-id and batch layouts of each case.
"""
import json
import os

import torch

# label -> per-request token counts packed into one forward
CASES = {
    "300": [300],
    "1000": [1000],
    "2000": [2000],
    "4000": [4000],
    "8000": [8000],
    "8192": [8192],
    "16x300": [300] * 16,
    "27x300": [300] * 27,
}


def read_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    dims = dict(hidden=tc["hidden_size"], vocab=tc["vocab_size"], eps=tc["rms_norm_eps"],
                dtype=str(tc.get("torch_dtype", cfg.get("torch_dtype", "bfloat16"))))
    print("dims", dims, flush=True)
    return dims


def build_modules(dims, seed=0):
    import math

    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    H, V = dims["hidden"], dims["vocab"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.device("cuda"):
        # As Gemma4TextModel builds it without attention DP (TP1 here).
        emb = Gemma4TextScaledWordEmbedding(V, H, dtype=torch.bfloat16, mapping=Mapping(),
                                            tensor_parallel_mode=TensorParallelMode.COLUMN,
                                            gather_output=True, embed_scale=math.sqrt(H))
        norm0 = RMSNorm(hidden_size=H, eps=dims["eps"], dtype=torch.bfloat16)
        normf = RMSNorm(hidden_size=H, eps=dims["eps"], dtype=torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_((torch.randn(V, H, device="cuda", generator=g) * 0.03).bfloat16())
        norm0.weight.copy_((torch.randn(H, device="cuda", generator=g) * 0.3 + 1.0).bfloat16())
        normf.weight.copy_((torch.randn(H, device="cuda", generator=g) * 0.3 + 1.0).bfloat16())
    print("emb tp_size", emb.tp_size, "tp_mode", emb.tp_mode, "scale", emb.embed_scale,
          flush=True)
    return emb, norm0, normf


def make_inputs(label, dims, seed):
    lens = CASES[label]
    M = sum(lens)
    V, H = dims["vocab"], dims["hidden"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, V, (M, ), device="cuda", generator=g, dtype=torch.int32)
    ids[0] = 0
    ids[-1] = V - 1
    if M > 2:
        ids[M // 2] = V - 1
        ids[1] = 0
    x_last = torch.randn(M, H, device="cuda", generator=g).bfloat16()  # last layer's output
    seq_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return ids, x_last, seq_lens


def main_chain(emb, norm0, normf, ids, x_last, seq_lens):
    """main 8070e5d7: embed_tokens(ids) (masked path * embed_scale), layer-0
    input_layernorm, self.norm on all rows, logits-processor gather."""
    h = emb(ids)
    n0 = norm0(h)
    fin_all = normf(x_last)
    last = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1
    return h, n0, fin_all[last]


def fused_chain(emb, norm0, normf, ids, x_last, seq_lens):
    from tensorrt_llm._torch.kernels.fused_ops.embed_scale_rmsnorm import embed_scale_rmsnorm
    # As Gemma4TextModel does: the host copy of the bf16 scale (no device read).
    h, n0 = embed_scale_rmsnorm(ids, emb.weight, emb.embed_scale_value,
                                norm_weight=norm0.weight, eps=norm0.variance_epsilon)
    last = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1
    return h, n0, normf(x_last[last])


def have_fused():
    try:
        import tensorrt_llm._torch.kernels.fused_ops.embed_scale_rmsnorm  # noqa: F401
        return True
    except ImportError:
        return False
