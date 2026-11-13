"""Confirm cuDNN sdpa_fp8 has no SM120 engine regardless of shape, heuristic mode, out dtype or LSE."""

import json

import cudnn
import torch

from tensorrt_llm._torch.visual_gen.attention.backends import cudnn as cb

cb.CuDNNAttention.check_hardware_compatibility = classmethod(lambda cls, d, q=None: None)

orig_build = cudnn.pygraph.build


def build_all(self, modes):
    return orig_build(self, [cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])


cudnn.pygraph.build = build_all

res = {}
for quant in ("fp8", "mxfp8"):
    for (S, H) in ((128, 1), (1024, 1), (4096, 28)):
        for out_dtype in (torch.bfloat16, torch.float16):
            for causal in (False, True):
                for lse in (False, True):
                    st = (S * H * 128, 128, H * 128, 1)
                    shape = cb._CuDNNProblemShape(b=1, h_q=H, h_kv=H, s_q=S, s_kv=S, d_qk=128, d_v=128,
                                                  q_strides=st, k_strides=st, v_strides=st, o_strides=st)
                    key = f"{quant} S={S} H={H} {str(out_dtype)[6:]} causal={causal} lse={lse}"
                    try:
                        cb.CuDNNAttention._build_graph(quant, shape, causal, 0.088, out_dtype, lse)
                        res[key] = "BUILT"
                    except Exception as e:  # noqa
                        res[key] = f"{type(e).__name__}: {str(e)[:160]}"
                    print(key, "->", res[key][:200], flush=True)
built = [k for k, v in res.items() if v == "BUILT"]
print(json.dumps({"cudnn": cudnn.backend_version(), "cap": list(torch.cuda.get_device_capability()),
                  "n": len(res), "built": built}))
