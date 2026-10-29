"""List the nvfp4 checkpoint keys of single_blocks.0 and content_mode_blocks.image.{0,1} (shapes)."""
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
f = root / "variants" / "nvfp4" / "model.safetensors"
out = {}
with safe_open(str(f), "pt", device="cpu") as h:
    for k in sorted(h.keys()):
        if k.startswith(("single_blocks.0.", "content_mode_blocks.image.0.",
                         "content_mode_blocks.image.1.")):
            s = h.get_slice(k)
            out[k] = [list(s.get_shape()), str(s.get_dtype())]
for k, v in out.items():
    print(k, v, flush=True)
print(json.dumps({"passed": True, "n": len(out)}))
