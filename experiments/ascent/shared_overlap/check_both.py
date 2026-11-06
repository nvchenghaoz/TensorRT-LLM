"""Run bench.py --mode check twice, with TRTLLM_DSV4_ROUTER_K1_FOLD at its default (unset) and at 0
(area-lead rule after landing 55), and print one combined {"passed": ...} JSON line.

Usage: python check_both.py --ckpt <ckpt> [--ms 512,2048,6144]
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(env_over, args):
    env = dict(os.environ)
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    label = ",".join(f"{k}={v}" for k, v in env_over.items())
    print(f"=== check with {label} ===", flush=True)
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "bench.py"), "--mode", "check", *args],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if line.strip():
            last = line.strip()
    rc = p.wait()
    try:
        res = json.loads(last) if last else {}
    except Exception:  # noqa: BLE001
        res = {"passed": False, "error": f"unparsable last line: {last[:200] if last else None}"}
    res["returncode"] = rc
    if rc != 0:
        res["passed"] = False
    return label, res


def main():
    args = sys.argv[1:]
    out = {}
    for over in ({"TRTLLM_DSV4_ROUTER_K1_FOLD": None}, {"TRTLLM_DSV4_ROUTER_K1_FOLD": "0"}):
        label, res = run(over, args)
        out[label] = res
    passed = all(r.get("passed") is True for r in out.values())
    print(json.dumps({"passed": passed, "runs": out}))


if __name__ == "__main__":
    main()
