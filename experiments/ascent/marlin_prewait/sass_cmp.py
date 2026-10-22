import glob, hashlib, json, os, re, subprocess, sys, time
# Idea 67 rework proof (brace-free: passed verbatim via python -c to both trees of a run_gpu_pair).
# For every dense marlin::Marlin<...> instance (not MoE) in the built library (sm_120):
#   - sha of the normalized SASS body (addresses / encodings stripped; branch targets and constant-bank offsets kept)
#   - register count and stack/local use (cuobjdump -res-usage)
#   - number of CCTL (prefetch) instructions and of ACQBULK (griddepcontrol.wait)
# The pair compares base vs candidate: every instance except the three pre-wait ones (128 threads, n tile 8x16,
# k tile 4x16, 6 stages, not full_k) must have identical SASS sha and registers.
T0 = time.time()
def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
log("TREE_COMMIT=" + head)
cands = []
for pat in ("tensorrt_llm/libs/libtensorrt_llm.so", "tensorrt_llm/libs/*.so"):
    cands += [p for p in glob.glob(pat) if os.path.getsize(p) > 1e6]
lib = cands[0]
log("lib", lib)
WANT = 26
def short(name):
    d = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
    m = re.search(r"Marlin<(.*?)>\(", d)
    return (m.group(1) if m else d)[:120]
def is_dense(name):
    return "6Marlin" in name and "marlin" in name and "moe" not in name.lower()
# Pass 1: SASS
sass = dict()
proc = subprocess.Popen(["cuobjdump", "-sass", "-arch", "sm_120", lib], stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
cur, lines = None, []
for line in proc.stdout:
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
        if cur is not None:
            sass[cur] = lines
            if len(sass) >= WANT:
                break
        cur = m.group(1) if is_dense(m.group(1)) else None
        lines = []
        continue
    if cur is not None:
        mm = re.search(r"/\*[0-9a-f][0-9a-f][0-9a-f][0-9a-f]+\*/\s+(.*?);", line)
        if mm:
            lines.append(re.sub(r"\s+", " ", mm.group(1)).strip())
if cur is not None and cur not in sass and len(sass) < WANT:
    sass[cur] = lines
proc.kill()
log("sass functions", len(sass))
# Pass 2: resource usage
res = dict()
proc = subprocess.Popen(["cuobjdump", "-res-usage", "-arch", "sm_120", lib], stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
pending = None
for line in proc.stdout:
    m = re.search(r"Function (\S+?):?\s*$", line.strip())
    if m and "REG:" not in line:
        pending = m.group(1).rstrip(":") if is_dense(m.group(1)) else None
        continue
    if pending is not None and "REG:" in line:
        res[pending] = line.strip()
        pending = None
        if len(res) >= WANT:
            break
proc.kill()
log("res-usage functions", len(res))
out = dict()
for name, ins in sass.items():
    body = "\n".join(ins)
    r = res.get(name, "")
    reg = re.search(r"REG:(\d+)", r)
    stk = re.search(r"STACK:(\d+)", r)
    loc = re.search(r"LOCAL:(\d+)", r)
    key = short(name)
    out[key] = dict(sha=hashlib.sha256(body.encode()).hexdigest()[:16], n_ins=len(ins),
                    reg=int(reg.group(1)) if reg else -1, stack=int(stk.group(1)) if stk else -1,
                    local=int(loc.group(1)) if loc else -1,
                    n_cctl=sum(1 for s in ins if "CCTL" in s), n_acqbulk=sum(1 for s in ins if "ACQBULK" in s))
    log("%-70s sha %s ins %5d reg %3d stack %d local %d cctl %d acqbulk %d" % (key, out[key]["sha"], len(ins),
        out[key]["reg"], out[key]["stack"], out[key]["local"], out[key]["n_cctl"], out[key]["n_acqbulk"]))
print(json.dumps(dict(passed=len(out) >= WANT, tree=head, functions=out)))
