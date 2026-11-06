"""Idea 329 gate (diagnostic, not submission evidence): a quick correctness smoke of every
knob config (check_329.py CHECK_QUICK=1, slot 0), then pair_329.py with several arms in one
process (GATE_ARMS).  Runs both as child processes, streams their output, and prints one
merged JSON line."""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, extra):
    env = dict(os.environ, **extra)
    p = subprocess.Popen([sys.executable, os.path.join(HERE, script)], env=env,
                         stdout=subprocess.PIPE, text=True)
    last = None
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("{"):
            last = line
    rc = p.wait()
    return rc, (json.loads(last) if last else None)


def main():
    res = {}
    if os.environ.get("GATE_SKIP_CHECK", "0") != "1":
        rc, out = run("check_329.py", {"CHECK_QUICK": "1", "CHECK_SLOTS": "0",
                                       "CHECK_REPLAYS": os.environ.get("CHECK_REPLAYS", "8")})
        res["check_rc"] = rc
        res["check"] = out and out.get("summary")
        if rc != 0 or not out or not out.get("passed"):
            print(json.dumps({"passed": False, "stage": "check", **res}))
            return
    arms = os.environ.get("GATE_ARMS", "BASE:;A:KACQ;C:ACQREL;AC:KACQ,ACQREL;ABC:KACQ,EW2,ACQREL")
    rc, out = run("pair_329.py", {"PAIR_ARMS": arms})
    res["pair_rc"] = rc
    res["pair"] = out
    print(json.dumps({"passed": rc == 0, **res}))


if __name__ == "__main__":
    main()
