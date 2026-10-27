"""List excluded (BF16) modules of the nvfp4 variant and check fp8r variant availability."""
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
print("root entries:", sorted(os.listdir(root)), flush=True)
vdir = root / "variants"
if vdir.is_dir():
    for v in sorted(os.listdir(vdir)):
        print("variant", v, sorted(os.listdir(vdir / v)), flush=True)

cfg = json.loads((root / "variants" / "nvfp4" / "config.json").read_text())
print("dit cfg:", json.dumps(cfg["dit"]), flush=True)
with safe_open(str(root / "variants" / "nvfp4" / "model.safetensors"), "pt", device="cpu") as f:
    keys = set(f.keys())
excl = sorted(k[:-7] for k in keys if k.endswith(".weight") and k[:-7] + ".weight_scale_2" not in keys)
print("N excluded:", len(excl), flush=True)
for e in excl:
    with safe_open(str(root / "variants" / "nvfp4" / "model.safetensors"), "pt", device="cpu") as f:
        s = f.get_slice(e + ".weight")
        print("EXCL", e, s.get_shape(), s.get_dtype())
fp8 = root / "variants" / "fp8r" / "model.safetensors"
res = {"excluded": excl, "fp8r_exists": fp8.is_file()}
if fp8.is_file():
    with safe_open(str(fp8), "pt", device="cpu") as f:
        k8 = set(f.keys())
        have = {}
        for e in excl:
            ok = (e + ".weight_scale") in k8
            dt = f.get_slice(e + ".weight").get_dtype() if (e + ".weight") in k8 else None
            have[e] = (ok, dt)
            print("FP8R", e, ok, dt)
        res["fp8r_has"] = {k: v[0] for k, v in have.items()}
print(json.dumps({"passed": True, "n_excluded": len(excl), "fp8r": fp8.is_file()}))
