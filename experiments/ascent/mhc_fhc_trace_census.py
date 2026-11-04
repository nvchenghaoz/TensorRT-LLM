"""Step 0(a): census of fused_pmap_gemm_fma_ksplit + mhcBigFuseKernel launches in the
profile-373 prefill rank-0 Kineto trace (main 1d696e72, job 466). CPU only, one process."""
import collections
import gzip
import json
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
print("events", len(evs), flush=True)
k = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")]
k.sort(key=lambda e: e["ts"])
print("kernels", len(k), flush=True)
fma = []
for i, e in enumerate(k):
    n = e.get("name", "")
    if "fused_pmap_gemm_fma_ksplit" in n:
        a = e.get("args", {})
        nxt = None
        for j in range(i + 1, min(i + 6, len(k))):
            if "BigFuse" in k[j].get("name", "") or "bigFuse" in k[j].get("name", "") or "mhcBigFuse" in k[j].get("name", ""):
                nxt = k[j]
                break
        fma.append(
            dict(
                ts=e["ts"],
                dur=e["dur"],
                grid=a.get("grid"),
                block=a.get("block"),
                regs=a.get("registers per thread"),
                name=n[:160],
                bf_dur=(nxt["dur"] if nxt else None),
                bf_grid=(nxt.get("args", {}).get("grid") if nxt else None),
            )
        )
print("fma launches", len(fma), flush=True)
names = collections.Counter(x["name"] for x in fma)
for n, c in names.items():
    print("NAME", c, n)
byM = collections.defaultdict(list)
for x in fma:
    M = x["grid"][0] if x["grid"] else -1
    byM[M].append(x)
summary = {}
for M, xs in sorted(byM.items()):
    d = [x["dur"] for x in xs]
    bf = [x["bf_dur"] for x in xs if x["bf_dur"] is not None]
    summary[str(M)] = dict(
        calls=len(xs),
        calls_per_req=len(xs) / 3.0,
        mean_us=sum(d) / len(d),
        min_us=min(d),
        max_us=max(d),
        total_ms_per_req=sum(d) / 3000.0,
        bigfuse_mean_us=(sum(bf) / len(bf) if bf else None),
        bigfuse_total_ms_per_req=(sum(bf) / 3000.0 if bf else None),
        grid=xs[0]["grid"],
        block=xs[0]["block"],
        regs=xs[0]["regs"],
        bf_grid=xs[0]["bf_grid"],
    )
# per-request sequence (first request): M per call index
if fma:
    t0 = fma[0]["ts"]
    seq = []
    for x in fma:
        seq.append((round((x["ts"] - t0) / 1000.0, 2), x["grid"][0] if x["grid"] else -1, x["dur"]))
    print("SEQ first 90:", seq[:90])
print(json.dumps({"summary": summary, "total_calls": len(fma)}))
