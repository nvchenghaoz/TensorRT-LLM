"""Branch 185 correctness: kill-switch digest equality + existing csa2 unit tests.

1. bench.py digest mode (>= 10 alternating c1/c4 EXTEND steps, metadata reuse) with
   TRTLLM_CSA2_FAST_PREPARE=0 (today's path) and =1 (fast path), each in a fresh process;
   every per-step digest after prepare and after the replay entry must match.
2. pytest of the csa2 metadata unit tests, unchanged.
Last stdout line: {"passed": ..., ...}.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]


def log(*a):
    print(*a, flush=True)


def digest(flag):
    env = dict(os.environ, TRTLLM_CSA2_FAST_PREPARE=flag, B185_DIGEST_STEPS="12")
    proc = subprocess.run([sys.executable, os.path.join(HERE, "bench.py"), CKPT, "digest"],
                          env=env, capture_output=True, text=True, timeout=1200)
    for line in proc.stdout.splitlines():
        if line.startswith("DIGEST") or line.startswith("host"):
            log(f"[{flag}] {line}")
    if proc.returncode:
        log(proc.stdout[-4000:])
        log(proc.stderr[-4000:])
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    result = {}
    slow, fast = digest("0"), digest("1")
    ok = slow is not None and fast is not None
    if ok:
        pairs = [((a["after_prepare"], a["after_replay_entry"]), (b["after_prepare"],
                  b["after_replay_entry"])) for a, b in zip(slow["steps"], fast["steps"])]
        same = [x == y for x, y in pairs]
        result["digest_steps"] = len(pairs)
        result["digest_equal_steps"] = sum(same)
        result["digest_off"] = slow["digest"]
        result["digest_on"] = fast["digest"]
        ok = all(same) and len(pairs) >= 8 and slow["digest"] == fast["digest"]
    result["killswitch_digest_equal"] = ok
    tests = [
        "tests/unittest/_torch/attention/sparse/csa2/test_metadata.py",
        "tests/unittest/_torch/attention/sparse/csa2/test_metadata_uploads.py",
        "tests/unittest/_torch/attention/sparse/csa2/test_context_only_metadata.py",
    ]
    log("running pytest", tests)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                          capture_output=True, text=True, timeout=2400)
    tail = proc.stdout.strip().splitlines()[-30:]
    for line in tail:
        log("  " + line)
    result["pytest_rc"] = proc.returncode
    result["pytest_summary"] = tail[-1] if tail else ""
    result["passed"] = bool(ok and proc.returncode == 0)
    result["compared"] = ("sha256 digests of every csa2_* metadata tensor/host tuple, "
                          "TRTLLM_CSA2_FAST_PREPARE=0 vs 1; csa2 metadata unit tests")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
