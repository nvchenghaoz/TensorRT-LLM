"""Branch 19 probe: git availability in the GPU worktree and the SM120 permutation helper source."""

import inspect
import subprocess
import sys

print("python", sys.version, flush=True)
for cmd in (["git", "log", "-3", "--format=%H %s"], ["git", "branch", "-a"], ["git", "status", "--short"]):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print("$", " ".join(cmd), "rc", out.returncode, flush=True)
        print(out.stdout[:3000], out.stderr[:1000], flush=True)
    except Exception as e:  # noqa: BLE001
        print("ERR", cmd, e, flush=True)

import cutlass.cute as cute  # noqa: E402
import cutlass.utils.blackwell_helpers as h  # noqa: E402

print(inspect.getsource(h.get_permutation_mnk), flush=True)
print("recast_ptr" in dir(cute), "autovec_copy" in dir(cute), flush=True)
print([n for n in dir(cute) if "recast" in n], flush=True)
