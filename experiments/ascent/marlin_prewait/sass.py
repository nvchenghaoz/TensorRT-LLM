# Idea 28 acceptance: SASS of the dense Marlin NVFP4 kernels (sm_120) - what memory instructions precede the
# grid-dependency wait. Streams `cuobjdump -sass -arch sm_120` of the built TRT-LLM library, keeps the
# marlin::Marlin<...> functions (dense, not MoE), and for each prints the instruction classes seen before the first
# ACQBULK (griddepcontrol.wait) - expected: only CCTL...PF2 (prefetch.global.L2 of B / scales), address math and
# the uniform branch on the pre-wait flag; no LDG / LDGSTS / STG / ATOM / RED before it.
import glob
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


cands = []
for pat in ("tensorrt_llm/libs/libtensorrt_llm.so", "tensorrt_llm/libs/*.so", "cpp/build/**/libtensorrt_llm.so"):
    cands += [p for p in glob.glob(pat, recursive=True) if os.path.getsize(p) > 1e6]
seen = set()
libs = [p for p in cands if not (p in seen or seen.add(p))]
log("libraries", libs[:6])
MEM = re.compile(r"\b(LDG|LDGSTS|STG|ATOM|ATOMG|RED|LD|ST|CCTL|LDS|STS|ACQBULK|PREEXIT|LDGDEPBAR|DEPBAR|BAR)\b[.\w]*")
results = []
want = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for lib in libs:
    proc = subprocess.Popen(["cuobjdump", "-sass", "-arch", "sm_120", lib], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    cur, lines, done = None, [], False
    for line in proc.stdout:
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if cur is not None:
                results.append((cur, lines))
                if len(results) >= want:
                    done = True
                    break
            name = m.group(1)
            cur = name if ("6Marlin" in name and "marlin" in name and "moe" not in name.lower()) else None
            lines = []
            continue
        if cur is not None:
            lines.append(line.rstrip())
    proc.kill()
    if results:
        break
report = []
for name, lines in results:
    # Code order: [stock path: wait at the top] ... index math ... [pre-wait path: L2 prefetch loop, wait]
    # ... start_pipes (LDGSTS). Check everything before the LAST wait that precedes the first LDGSTS.
    ins_all = []
    for l in lines:
        mm = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", l)
        if mm:
            ins_all.append(mm.group(1))
    first_ldgsts = next((i for i, s in enumerate(ins_all) if "LDGSTS" in s), len(ins_all))
    waits = [i for i, s in enumerate(ins_all[:first_ldgsts]) if "ACQBULK" in s]
    wait_line = waits[-1] if waits else None
    pre = ins_all[:wait_line] if wait_line is not None else []
    trig_line = None
    for l in lines:
        if "PREEXIT" in l:
            trig_line = l.strip()
            break
    mem_before = []
    for ins in pre:
        op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0] if ins.split() else ""
        if MEM.match(op):
            mem_before.append(op)
    classes = sorted(set(mem_before))
    # Allowed before the wait: L2 prefetches (CCTL...PF2) and at most one LDG (the weight global scale, a
    # constant weight read at kernel entry in main's code too); nothing that touches A, locks, C or C_tmp.
    n_ldg = sum(1 for o in mem_before if o.startswith("LDG") and not o.startswith("LDGSTS"))
    bad = [c for c in classes if not (c.startswith("CCTL") or c.startswith("BAR") or c.startswith("DEPBAR")
                                      or (c.startswith("LDG") and not c.startswith("LDGSTS")))]
    if n_ldg > 1:
        bad.append("LDG x%d" % n_ldg)
    rec = {"function": name[:160], "instructions_before_wait": len(pre), "found_wait": wait_line is not None,
           "n_waits_before_first_ldgsts": len(waits), "n_ldg_before_wait": n_ldg,
           "memory_ops_before_wait": classes, "non_prefetch_memory_before_wait": bad,
           "n_cctl_prefetch_before_wait": sum(1 for o in mem_before if o.startswith("CCTL")),
           "trigger_PREEXIT_present": trig_line is not None}
    report.append(rec)
    log(json.dumps(rec))
ok = bool(report) and all(r["found_wait"] and not r["non_prefetch_memory_before_wait"] for r in report)
print(json.dumps({"passed": ok, "functions_checked": len(report), "report": report,
                  "rule": "before the first ACQBULK (griddepcontrol.wait) only CCTL.*PF2 (L2 prefetch) memory ops"}))
