"""Idea 16 correctness: executor lifecycle tests + b64 splice byte-identity."""
import json
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [os.path.join(os.getcwd(), "tests", "unittest"), env.get("PYTHONPATH", "")]
)
print("== pytest test_executor_lifecycle.py", flush=True)
p = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
     "tests/unittest/_torch/visual_gen/test_executor_lifecycle.py"],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print(p.stdout[-6000:], flush=True)
summary = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
print("== b64 splice test", flush=True)
s = subprocess.run([sys.executable, os.path.join(here, "b64_splice_test.py")],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(s.stdout[-8000:], flush=True)
splice = {}
for line in reversed(s.stdout.splitlines()):
    if line.startswith("{"):
        splice = json.loads(line)
        break
passed = p.returncode == 0 and s.returncode == 0 and bool(splice.get("passed"))
print(json.dumps({
    "passed": passed,
    "lifecycle_pytest_rc": p.returncode,
    "lifecycle_summary": summary[-1] if summary else None,
    "splice": splice,
}), flush=True)
