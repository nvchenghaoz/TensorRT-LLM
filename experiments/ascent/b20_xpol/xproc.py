"""Branch 20: cross-process determinism of the served pipeline, X_POLICY 2 vs 0 (fresh processes).

  xproc.py <ckpt> [rounds]
Runs pair.py <ckpt> <rounds> in 4 fresh processes in the order X2, X0, X2, X0 (by env
TRTLLM_FLUX3_ADALN_WARP_X_POLICY; on this branch 0 is the default and 2 is main's), collecting
each process's latent/image sha256 per case. Reports which processes agree. If the two X2
processes (main's behaviour, A/A) differ from each other the same way X0 differs from X2, the
cross-process difference is pre-existing process-level nondeterminism (load-time autotune /
algorithm selection), not the cache hint. Within-process bitwise across arms: census.py.
"""

import json
import os
import subprocess
import sys

CKPT = sys.argv[1]
ROUNDS = sys.argv[2] if len(sys.argv) > 2 else "2"
HERE = os.path.dirname(os.path.abspath(__file__))
runs = []
for i, xp in enumerate(("2", "0", "2", "0")):
    env = dict(os.environ, TRTLLM_FLUX3_ADALN_WARP_X_POLICY=xp)
    print(f"=== process {i} X_POLICY={xp}", flush=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "pair.py"), CKPT, ROUNDS], env=env,
                       capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
        print(json.dumps({"passed": False, "error": f"process {i} rc={p.returncode}"}))
        sys.exit(1)
    out = json.loads(lines[-1])
    rec = {"i": i, "x_policy": xp, "seen_x_policy": out["info"]["x_policy"],
           "hashes": out["hashes"], "deterministic": out["deterministic"],
           "step_ms": {k: v["step_ms"] for k, v in out["batch"].items()}}
    print("PROC", json.dumps(rec), flush=True)
    runs.append(rec)


def key(r, c):
    return tuple(tuple(h) for h in r["hashes"][c])


res = {}
for c in ("t2i", "i2i"):
    ks = [key(r, c) for r in runs]
    uniq = {k: n for n, k in enumerate(dict.fromkeys(ks))}
    res[c] = {"process_class": [uniq[k] for k in ks],
              "aa_x2_equal": ks[0] == ks[2], "aa_x0_equal": ks[1] == ks[3],
              "x0_vs_x2_first": ks[0] == ks[1],
              "latent_equal_all": len({k[0][0] for k in ks}) == 1,
              "image_equal_all": len({k[0][1] for k in ks}) == 1}
print(json.dumps({"passed": all(r["deterministic"][c] for r in runs for c in ("t2i", "i2i")),
                  "summary": res, "runs": runs}))
