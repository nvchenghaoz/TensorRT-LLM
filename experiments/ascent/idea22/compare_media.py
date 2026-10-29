"""Idea 22: compare the served outputs of a run_serving_pair (main vs branch), byte for byte.

usage: python compare_media.py <baseline proxy dir> <candidate proxy dir>
Every media_proxy_<case>_c1/*.png (the b64-decoded b64_json of each timed request, same
prompts/seeds/references in both arms) must have identical sha256 in both arms.
"""
import glob
import hashlib
import json
import os
import sys


def digests(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "media_proxy_*", "*.png"))):
        rel = os.path.relpath(p, root)
        with open(p, "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def main():
    a, b = digests(sys.argv[1]), digests(sys.argv[2])
    rows = []
    for k in sorted(set(a) | set(b)):
        rows.append({"file": k, "equal": a.get(k) == b.get(k) and a.get(k) is not None,
                     "base": (a.get(k) or "")[:16], "cand": (b.get(k) or "")[:16]})
        print(rows[-1], flush=True)
    cases = {}
    for r in rows:
        c = r["file"].split("/")[0]
        cases.setdefault(c, [0, 0])
        cases[c][0] += r["equal"]
        cases[c][1] += 1
    passed = bool(rows) and all(r["equal"] for r in rows) and len(cases) == 2
    print(json.dumps({"passed": passed,
                      "compared": "served PNG bytes (b64decode of b64_json) main vs branch, same requests",
                      "per_case_equal_over_total": cases, "n_files": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
