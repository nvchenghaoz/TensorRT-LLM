"""Dump the installed cache-dit package source (and look for branch-2 job artifacts).

No GPU work. Copies cache_dit/*.py into $AUTO_DEV_ARTIFACTS/cache_dit_src and prints
a short index so the toy check can be written against the real 1.5.3 API.
"""
import glob
import json
import os
import shutil
import sys

import cache_dit

root = os.path.dirname(cache_dit.__file__)
out = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/artifacts")
dst = os.path.join(out, "cache_dit_src")
os.makedirs(dst, exist_ok=True)
files = []
for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
    rel = os.path.relpath(p, root)
    d = os.path.join(dst, rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy(p, d)
    files.append((rel, os.path.getsize(p)))
print("cache_dit", getattr(cache_dit, "__version__", "?"), root, flush=True)
for rel, sz in sorted(files):
    print(f"{sz:8d} {rel}")
# grep key symbols
keys = ["def refresh_context", "def reset", "def steps_mask", "max_continuous_cached_steps",
        "steps_computation_mask", "steps_computation_policy", "class DBCacheConfig",
        "Pattern_3", "def enable_cache", "calibrator_config"]
for rel, _ in sorted(files):
    with open(os.path.join(root, rel)) as f:
        for i, line in enumerate(f, 1):
            if any(k in line for k in keys):
                print(f"{rel}:{i}: {line.rstrip()}")
# look for branch-2 artifacts
print("AUTO_DEV_ARTIFACTS=", out)
for cand in [os.path.dirname(out), os.path.dirname(os.path.dirname(out))]:
    try:
        ents = sorted(os.listdir(cand))[:200]
        print("ls", cand, ents)
    except Exception as e:  # noqa
        print("ls fail", cand, e)
hits = []
for pat in ["**/*302*", "**/*292*", "**/*261*", "**/*260*"]:
    try:
        hits += glob.glob(os.path.join(os.path.dirname(os.path.dirname(out)), pat), recursive=False)
    except Exception:
        pass
print("hits", hits[:50])
print(json.dumps({"passed": True, "n_files": len(files)}))
