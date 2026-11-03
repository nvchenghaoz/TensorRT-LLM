"""Stage 0 probe for idea 73: b12x wo_projection import, versions, model dims, scale geometry."""
import glob
import json
import os
import struct
import sys

out = {}
ckpt = sys.argv[1]
cfg = json.load(open(os.path.join(ckpt, "config.json")))
tc = cfg.get("text_config", cfg)
for k in ("hidden_size", "num_attention_heads", "head_dim", "qk_rope_head_dim", "o_lora_rank",
          "o_groups", "q_lora_rank", "num_hidden_layers", "compress_ratios"):
    out[k] = tc.get(k)
out["quantization_config"] = tc.get("quantization_config", cfg.get("quantization_config"))
print(json.dumps(out, default=str)[:3000], flush=True)

# safetensors header scan for wo_a / wo_b scale shapes in layer 0
shapes = {}
for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if (".0." in k or "layers.0." in k) and ("wo_a" in k or "wo_b" in k or "o_a_proj" in k
                                                   or "o_b_proj" in k):
            shapes[k] = (v.get("dtype"), v.get("shape"))
    if len(shapes) >= 4:
        break
print("SHAPES", json.dumps(shapes), flush=True)

import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
try:
    import cutlass  # noqa: F401
    import importlib.metadata as md
    print("cutlass-dsl", md.version("nvidia-cutlass-dsl"), flush=True)
except Exception as e:  # noqa
    print("cutlass-dsl missing", repr(e), flush=True)
import flashinfer  # noqa: E402
print("flashinfer", flashinfer.__version__, flush=True)
ok = True
try:
    import b12x  # noqa: F401
    import b12x.gemm as bg
    print("b12x.gemm attrs", [a for a in dir(bg) if not a.startswith("_")], flush=True)
    from b12x.gemm import wo_projection as wp
    print("wo_projection attrs", wp.__all__ if hasattr(wp, "__all__") else dir(wp), flush=True)
    print("is_supported", wp.is_supported(torch.device("cuda")), flush=True)
except Exception as e:  # noqa
    import traceback
    traceback.print_exc()
    ok = False
print(json.dumps({"passed": ok, **{k: out[k] for k in out if k != "quantization_config"},
                  "shapes": shapes}, default=str))
