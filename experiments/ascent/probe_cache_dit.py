"""Probe the installed cache_dit package: dump selected sources to artifacts, steps_mask for 49 steps."""
import json
import os
import shutil
import sys

import cache_dit

root = os.path.dirname(cache_dit.__file__)
out = os.environ.get("AUTO_DEV_ARTIFACTS") or "artifacts"
print("artifacts dir", out, {k: v for k, v in os.environ.items() if "ARTIFACT" in k}, flush=True)
os.makedirs(out, exist_ok=True)
for rel in sys.argv[1:]:
    if rel.startswith("CKPT:"):
        p = rel[5:]
        print("=" * 20, p)
        print(open(p).read())
        continue
    dst = os.path.join(out, "cache_dit_src", rel.replace("/", "__"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(root, rel), dst)
    print("copied", rel, flush=True)
print(json.dumps({"passed": True}))
