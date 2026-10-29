"""STEP 0(a): checkpoint quant types of content_mode_blocks.image_cond.* vs image.* (headers only)."""
import glob
import json
import os
import struct
import sys

ckpt = sys.argv[1]
print("checkpoint", ckpt, flush=True)
for root, dirs, files in os.walk(ckpt):
    depth = root[len(ckpt):].count(os.sep)
    if depth <= 2:
        print("DIR", root, sorted(f for f in files if not f.endswith(".safetensors"))[:20],
              len([f for f in files if f.endswith(".safetensors")]), "safetensors", flush=True)

for cfg in glob.glob(os.path.join(ckpt, "**", "*.json"), recursive=True):
    if any(k in cfg for k in ("quant", "config.json")) and "tokenizer" not in cfg:
        try:
            text = open(cfg).read()
        except Exception:
            continue
        if "quant" in text.lower():
            print("=== ", cfg, flush=True)
            print(text[:4000], flush=True)

keys = {}
for path in glob.glob(os.path.join(ckpt, "**", "*.safetensors"), recursive=True):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for k, v in header.items():
        if k == "__metadata__":
            continue
        keys[k] = (v["dtype"], v["shape"], os.path.relpath(path, ckpt))

print("total keys", len(keys), flush=True)
for pref in ("image_cond.0.", "image_cond.1.", "image.0.", "image.1.", "single_blocks.0.", "single_blocks.24."):
    sel = sorted(k for k in keys if pref in k and ("mode_blocks" in k or pref.startswith("single")))
    print(f"--- {pref}: {len(sel)} keys", flush=True)
    for k in sel:
        print("  ", k, keys[k][0], keys[k][1], keys[k][2])
# input_scale presence summary
for pref in ("image_cond", "image.", "single_blocks", "txt"):
    n_in = sum(1 for k in keys if pref in k and k.endswith("input_scale"))
    n_w2 = sum(1 for k in keys if pref in k and k.endswith("weight_scale_2"))
    print(f"{pref}: input_scale {n_in}, weight_scale_2 {n_w2}")
print(json.dumps({"passed": True}))
