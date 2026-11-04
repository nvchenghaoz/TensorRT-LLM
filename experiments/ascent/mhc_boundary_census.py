"""Step 0(a'): per call site census of every mHC boundary in the profile-373 prefill rank-0
trace (main 1d696e72, job 466). For each mhcBigFuse launch, report the GEMM kernel that
precedes it on the same stream (fused_pmap_gemm_fma_ksplit = MhcFusedHcRunner path,
anything else = MhcPreMappingRunner / other path), grouped by (kernel name, M). CPU only."""
import collections
import gzip
import json
import re
import sys

PATH = (
    "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots/4/jobs/"
    "c5aacea4876d4c6a9d12df22025c8113/artifacts/profile/traces/prefill/"
    "trtllm-trace-1793733283-ea9eb619-rank-0.json.gz"
)
if len(sys.argv) > 1:
    PATH = sys.argv[1]
print("loading", PATH, flush=True)
with gzip.open(PATH, "rt") as f:
    data = json.load(f)
evs = data["traceEvents"] if isinstance(data, dict) else data
k = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")]
k.sort(key=lambda e: e["ts"])
print("kernels", len(k), flush=True)

mhc_names = collections.Counter()
for e in k:
    n = e.get("name", "")
    if re.search(r"mhc|Mhc|pmap|sqrsum|gemm_rms|BigFuse|bigFuse|hc_head|postMapping|post_mapping", n):
        mhc_names[n[:140]] += 1
print("ALL mHC-like kernel names (count over 3 requests):")
for n, c in mhc_names.most_common():
    print(f"  {c:5d}  {n}")

bystream = collections.defaultdict(list)
for e in k:
    bystream[(e.get("pid"), e.get("tid"))].append(e)

groups = collections.defaultdict(list)
for st, ks in bystream.items():
    for i, e in enumerate(ks):
        n = e.get("name", "")
        if "BigFuse" not in n and "bigFuse" not in n:
            continue
        prev = ks[i - 1] if i > 0 else None
        pn = prev.get("name", "")[:110] if prev else "NONE"
        pgrid = prev.get("args", {}).get("grid") if prev else None
        bgrid = e.get("args", {}).get("grid")
        bn = re.sub(r"\(.*", "", n)[:110]
        groups[(pn, str(pgrid), bn, str(bgrid))].append((prev["dur"] if prev else 0.0, e["dur"]))

out = []
for (pn, pg, bn, bg), xs in sorted(groups.items(), key=lambda kv: -sum(a + b for a, b in kv[1])):
    g = sum(a for a, _ in xs)
    b = sum(b for _, b in xs)
    row = dict(prev_kernel=pn, prev_grid=pg, bigfuse=bn, bigfuse_grid=bg, calls_per_req=len(xs) / 3.0,
               prev_mean_us=g / len(xs), bigfuse_mean_us=b / len(xs), ms_per_req=(g + b) / 3000.0)
    out.append(row)
    print(json.dumps(row), flush=True)
print(json.dumps({"boundaries": out}))
