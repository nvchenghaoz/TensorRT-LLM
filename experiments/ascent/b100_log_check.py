"""Read-only check of the pair 1033/1034 server logs, plus the stall search in old kept logs."""
import glob, json, os, re
R = "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots"
ARMS = {
    "base_1033": R + "/4/jobs/09ed7681120541ccb6e0c63a58b80e0a/artifacts/proxy/server/server.log",
    "cand_1034": R + "/4/jobs/c5ecb6873d104b0c9aad0f2f49a2c85d/artifacts/proxy/server/server.log",
}
PATS = {
    "moe_tp": r"moe_tensor_parallel_size\W+\d+|moe_tp_size\W+\d+",
    "moe_ep": r"moe_expert_parallel_size\W+\d+|moe_ep_size\W+\d+",
    "fi_warm": r"FlashInfer decode chain warmed up[^\n]*",
    "fi_on": r"decode chain on[^\n]*",
    "fi_off": r"FlashInfer decode chain off[^\n]*",
    "kv_alloc": r"final_kv_cache_allocation[^\n]*",
    "group0": r"group0[^\n]*",
    "jit_capture": r"(JIT|jit_compile|Compiling)[^\n]*(captur|graph)[^\n]*|captur[^\n]*(JIT|compil)[^\n]*",
    "full_tok": r"full tokenizations[^\n]*",
    "error": r"(Traceback|CUDA error|OOM|out of memory)[^\n]*",
}
out = {}
ok = True
for arm, p in ARMS.items():
    res = {}
    try:
        t = open(p, errors="replace").read()
    except Exception as e:
        res["read_error"] = str(e); out[arm] = res; ok = False; continue
    for k, pat in PATS.items():
        ms = [m.group(0)[:300] for m in re.finditer(pat, t)]
        res[k] = {"n": len(ms), "first": ms[:4]}
    out[arm] = res
    print(arm, json.dumps(res, indent=1), flush=True)
c = out.get("cand_1034", {})
cand_ok = (c.get("fi_warm", {}).get("n", 0) > 0 and c.get("fi_off", {}).get("n", 1) == 0
           and c.get("jit_capture", {}).get("n", 1) == 0 and c.get("error", {}).get("n", 1) == 0)
warm = " ".join(c.get("fi_warm", {}).get("first", []))
cand_ok = cand_ok and ("1152" in warm) and ("192" in warm)
# stall search: old kept server logs containing request 3af3337b
hits = []
for p in glob.glob(R + "/*/jobs/*/artifacts/proxy/server/server.log"):
    try:
        t = open(p, errors="replace").read()
    except Exception:
        continue
    if "3af3337b" in t:
        hits.append(p)
        print("STALL_LOG", p, flush=True)
        for i, line in enumerate(t.splitlines()):
            if re.search(r"3af3337b|autotun|captur|JIT|evict|paused|waiting|gc|NCCL|timeout|retry|warn|WARN", line):
                print("  L%d %s" % (i, line[:300]))
print("old server logs scanned", len(glob.glob(R + "/*/jobs/*/artifacts/proxy/server/server.log")), flush=True)
print(json.dumps({"passed": bool(ok and cand_ok), "compared": "pair 1033/1034 server logs: moe layout, FI decode chain warm-up inter=1152 experts=192, no chain off, no JIT in capture, no errors; kv allocation lines", "stall_logs_found": hits,
                  "kv_base": out.get("base_1033", {}).get("kv_alloc", {}).get("first"), "kv_cand": c.get("kv_alloc", {}).get("first"),
                  "fi_warm": c.get("fi_warm", {}).get("first")}))
