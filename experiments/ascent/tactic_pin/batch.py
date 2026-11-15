"""Idea 116 (branch 50): every owed check in one GPU job (queue rule: one job per branch).

usage: batch.py <checkpoint> <captures_dir> [steps]   steps: comma list of check,ab,bcg,axa (default all)

  check : tactic_pin/check.py   real-capture + synthetic numerics of the pins (untimed)
  repeat: tactic_pin/repeat_check.py, twice: pins on, then TRTLLM_NVFP4_DOWN_PIN=0 +
          TRTLLM_NVFP4_GATEUP_PIN=0; bench_pair.py's timed sequence call for call (untimed)
  ab    : tactic_pin/lifetimes.py drive_ab 3   3 fresh lifetimes per arm, interleaved
          (main = both pin kill switches 0, pin = branch default): kernels per served M and TTFT
  bcg   : bcg_prefill/bcg_ab.py inproc   BCG replay vs eager bitwise in one process (pins on)
  axa   : tactic_pin/axa.py   own-prompt A/X/A'/Y in one eager process
Each step's last JSON line is collected; passed = all steps passed.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, cmd, env=None):
    print(f"[batch] === {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, **(env or {})))
    last = None
    for line in p.stdout:
        sys.stdout.write(f"[{name}] {line}")
        sys.stdout.flush()
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            last = s
    p.wait()
    try:
        res = json.loads(last) if last else {"passed": False, "error": "no JSON"}
    except json.JSONDecodeError as e:
        res = {"passed": False, "error": repr(e)}
    res["rc"] = p.returncode
    res["seconds"] = round(time.time() - t0, 1)
    print(f"[batch] === {name} done rc={p.returncode} passed={res.get('passed')} in {res['seconds']}s", flush=True)
    return res


def main():
    ck, cap = sys.argv[1], sys.argv[2]
    steps = (sys.argv[3] if len(sys.argv) > 3 else "check,ab,bcg").split(",")
    py = sys.executable
    out = {}
    if "check" in steps:
        out["check"] = run("check", [py, os.path.join(HERE, "check.py"), "--checkpoint", ck, "--captures", cap])
    if "repeat" in steps:
        rc = [py, os.path.join(HERE, "repeat_check.py"), "--checkpoint", ck, "--captures", cap]
        out["repeat_pins_on"] = run("repeat_pins_on", rc)
        out["repeat_pins_off"] = run("repeat_pins_off", rc, {"TRTLLM_NVFP4_DOWN_PIN": "0",
                                                             "TRTLLM_NVFP4_GATEUP_PIN": "0"})
    if "ab" in steps:
        out["ab"] = run("ab", [py, os.path.join(HERE, "lifetimes.py"), "drive_ab", "3"], {"CKPT": ck})
    if "bcg" in steps:
        out["bcg"] = run("bcg", [py, os.path.join(HERE, "..", "bcg_prefill", "bcg_ab.py"), "inproc"],
                         {"CKPT": ck, "BCG_OVERLAY": "1"})
    if "axa" in steps:
        out["axa"] = run("axa", [py, os.path.join(HERE, "axa.py"), ck], {"CKPT": ck})
    passed = all(v.get("passed") for v in out.values())
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    with open(os.path.join(art, "batch.json"), "w") as f:
        json.dump(out, f, indent=1)
    short = {k: {kk: vv for kk, vv in v.items() if kk not in ("down_pin_vs_main_tactics", "comparisons",
                                                              "dispatches")} for k, v in out.items()}
    print(json.dumps({"passed": bool(passed), "steps": short}))


if __name__ == "__main__":
    main()
