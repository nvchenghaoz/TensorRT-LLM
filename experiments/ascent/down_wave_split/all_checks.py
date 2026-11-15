"""One worker job for idea 64: runs check.py (synthetic every-M int16, kill
switch, FP32 reference, repeated A,B,A from 0xA5 outputs) and then llm_ab.py
(in-process own-prompt A/X/A2, same-process re-send, op-level int16 on real
layer 0/30/59 down_proj inputs) as subprocesses; prints one combined JSON.
usage: all_checks.py <checkpoint>"""
import json
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
ck = sys.argv[1]
env = dict(os.environ, CKPT=ck)
res = {}
for name, argv in (("check", [sys.executable, os.path.join(here, "check.py")]),
                   ("llm_ab", [sys.executable, os.path.join(here, "llm_ab.py"), ck])):
    print(f"=== running {name}", flush=True)
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.startswith("{"):
            last = line
    p.wait()
    try:
        res[name] = json.loads(last) if last else {"passed": False, "error": "no json"}
    except Exception as e:  # noqa
        res[name] = {"passed": False, "error": str(e)}
    res[name]["returncode"] = p.returncode
print(json.dumps({"passed": all(r.get("passed") and r["returncode"] == 0 for r in res.values()),
                  "compared": "check.py + llm_ab.py (see each)", **res}, default=str))
