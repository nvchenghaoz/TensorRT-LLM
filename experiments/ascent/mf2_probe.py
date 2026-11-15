import json, sys, os, glob
ck = sys.argv[1]
cfg = json.load(open(os.path.join(ck, "config.json")))
tc = cfg.get("text_config", cfg)
print("hidden", tc.get("hidden_size"), "layers", tc.get("num_hidden_layers"), "eps", tc.get("rms_norm_eps"))
for f in ["hf_quant_config.json"]:
    p = os.path.join(ck, f)
    if os.path.exists(p):
        q = json.load(open(p))
        s = json.dumps(q)
        print(f, s[:3000])
qc = cfg.get("quantization_config")
if qc:
    print("quantization_config", json.dumps(qc)[:3000])
# check safetensors index for qkv weights' dtypes
idx = os.path.join(ck, "model.safetensors.index.json")
if os.path.exists(idx):
    wm = json.load(open(idx))["weight_map"]
    ks = [k for k in wm if "layers.0." in k and ("self_attn" in k or "mlp" in k)]
    print("\n".join(sorted(ks)))
import torch
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
from tensorrt_llm._torch.kernels.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant, rmsnorm_fp4_quant_available
print("rmsnorm_fp4_quant_available", rmsnorm_fp4_quant_available())
try:
    x = torch.randn(300, 5376, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(5376, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor([10.0], device="cuda")
    fp4, sf = rmsnorm_fp4_quant(x, w, 1e-6, gs)
    torch.cuda.synchronize()
    print("rmsnorm_fp4_quant ran", fp4.shape, sf.shape)
except Exception as e:
    print("rmsnorm_fp4_quant FAILED", type(e).__name__, str(e)[:500])
print(json.dumps({"passed": True}))
