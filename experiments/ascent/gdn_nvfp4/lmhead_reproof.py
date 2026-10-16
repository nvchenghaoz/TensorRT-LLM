import hashlib, json, os, torch
from tensorrt_llm._torch.modules import linear as L
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
# Bitwise re-proof after sync to main 87bcb219 (run as a pair: main tree vs branch tree).
# (a) lm_head-shaped W4A16 NVFP4 Linear (Marlin, SM120): packed tensors and outputs at
#     M=1/2/4/16 hashed; both trees must print identical hashes.
# (b) GDN in_proj_qkvz-shaped FP8 Linear flagged for the decode copy (bench.py seed 1000):
#     hashes of the Marlin copy and of M=1/2/4 outputs (branch only) + M=2048 output (both).
dev = torch.device("cuda")
def log(*a): print(*a, flush=True)
def hb(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).view(-1).cpu().numpy().tobytes()).hexdigest()[:16]
res = {"lm_head": {}, "in_proj_qkvz": {}}
timing = {}
for (N, K) in [(248320, 5120), (200, 96)]:
    g = torch.Generator(device=dev).manual_seed(N + K)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    amax = w.abs().amax().float()
    fp4, sf = torch.ops.trtllm.fp4_quantize(w, (448.0 * 6.0 / amax).reshape(1), 16, False, True)
    sf_lin = torch.ops.trtllm.block_scale_interleave_reverse(
        sf.view(((N + 127) // 128) * 128, -1))[:N, :K // 16].contiguous().view(torch.float8_e4m3fn)
    lin = Linear(K, N, bias=False, dtype=torch.bfloat16,
                 quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)).to(dev)
    method = type(lin.quant_method).__name__
    lin.load_weights([{"weight": fp4.view(torch.uint8), "weight_scale": sf_lin,
                       "weight_scale_2": (amax / (448.0 * 6.0)).reshape(())}])
    lin.post_load_weights()
    key = f"{N}x{K}"
    r = {"method": method, "weight": hb(lin.weight), "weight_scale": hb(lin.weight_scale),
         "weight_global_scale": hb(lin.weight_global_scale)}
    for m in [1, 2, 4, 16]:
        x = (torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 2.0).to(torch.bfloat16)
        y = lin(x)
        y2 = lin(x)
        r[f"out_M{m}"] = hb(y)
        r[f"repeat_equal_M{m}"] = bool(torch.equal(y.view(torch.int16), y2.view(torch.int16)))
        if N == 248320:
            for _ in range(5):
                lin(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(30):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); lin(x); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            ts.sort()
            timing[str(m)] = {"kernel_ms": ts[15], "noise_floor": (ts[27] - ts[3]) / ts[15]}
    res["lm_head"][key] = r
    log(key, r)
    del lin, w, fp4, sf
torch.cuda.empty_cache()
# (b) in_proj_qkvz copy
N, K = 16384, 5120
g = torch.Generator(device=dev).manual_seed(1000)
w = torch.randn(N, K, generator=g, device=dev) * 0.02
w = w * (torch.rand(N, 1, generator=g, device=dev) * 1.5 + 0.25)
s = (w.abs().amax() / 448.0).float()
w8 = (w / s).to(torch.float8_e4m3fn)
lin = Linear(K, N, bias=False, dtype=torch.bfloat16,
             quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)).to(dev)
lin._nvfp4_decode_copy = True
lin.load_weights([{"weight": w8, "weight_scale": s.reshape(()),
                   "input_scale": torch.tensor(0.05, device=dev)}])
lin.post_load_weights()
cp = getattr(lin, "_nvfp4_decode_weight", None)
r = {"copy_built": cp is not None, "fp8_weight": hb(lin.weight)}
if cp is not None:
    r.update({"copy_weight": hb(cp), "copy_scale": hb(lin._nvfp4_decode_scale),
              "copy_global_scale": hb(lin._nvfp4_decode_global_scale)})
for m in [1, 2, 4, 2048]:
    x = (torch.randn(m, K, device=dev, generator=torch.Generator(device=dev).manual_seed(m)) * 2.0).to(torch.bfloat16)
    r[f"out_M{m}"] = hb(lin(x))
res["in_proj_qkvz"] = r
log("in_proj_qkvz", r)
print(json.dumps({"batch": timing, "output": {"class": "exact"}, "hashes": res}))
