import glob, json, os, re, sys
R = "/home/scratch.chengzhang/trtllm-dev-dsv41-1017-c124-20261102-145209/runner/slots"
mans = glob.glob(R + "/*/jobs/*/manifest.json")
print("manifests", len(mans), flush=True)
if mans:
    try:
        m = json.load(open(mans[0]))
        print("sample keys", list(m.keys())[:40] if isinstance(m, dict) else type(m))
        print(json.dumps(m)[:1500])
    except Exception as e:
        print("err", e)
want = {"875", "876", "907", "908", "938", "939", "957"}
for p in mans:
    try:
        txt = open(p).read()
    except Exception:
        continue
    for w in want:
        if re.search(r'"(job_id|id)"\s*:\s*%s\b' % w, txt):
            print("MATCH", w, p, flush=True)
n = 0
for p in glob.glob(R + "/*/jobs/*/artifacts/**/*", recursive=True):
    if not os.path.isfile(p):
        continue
    try:
        sz = os.path.getsize(p)
    except Exception:
        continue
    if sz > 2e9:
        continue
    n += 1
    try:
        with open(p, "rb") as f:
            data = f.read()
    except Exception:
        continue
    if b"3af3337b" in data:
        print("HAS3af3337b", p, sz, flush=True)
print("files scanned", n)
print("done")
