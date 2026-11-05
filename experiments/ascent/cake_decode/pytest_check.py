"""CSA2 unit tests on this tree, judged against the failures already present on main.

Runs tests/unittest/_torch/attention/sparse/csa2 (no -x) and passes iff every failure
is one that also fails on the unmodified base:
- the 4 ruled pre-existing on main 545d0d40/2fa8e034 (top lead rulings 19:5x / 20:10 PST
  11-04; deselected with -k), and
- test_backend.py::test_flashinfer_shared_plans_bound_eager_context_and_keep_protected,
  which fails on the unmodified base 2fa8e034 (job 1253: 1 failed, 413 passed) because its
  fake Paged wrapper has no reset_workspace_buffer (backend.py _plan_with_int_workspace).
Last stdout line: {"passed": ..., "failed": [...], "summary": "..."}.
"""
import json
import re
import subprocess
import sys

KNOWN_ON_BASE = {
    "tests/unittest/_torch/attention/sparse/csa2/test_backend.py::"
    "test_flashinfer_shared_plans_bound_eager_context_and_keep_protected",
}
DESELECT = (
    "not test_module_sends_whole_context_and_generation_phases"
    " and not test_gathered_decode_graph_grows_past_warmup_prefix"
    " and not test_constant_rows_retained_only_for_captured_graphs"
)


def main():
    cmd = [sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
           "tests/unittest/_torch/attention/sparse/csa2", "-k", DESELECT]
    print("running", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    print(out[-6000:], flush=True)
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, flags=re.M)))
    summary = ""
    for line in out.splitlines()[::-1]:
        if re.search(r"\d+ (passed|failed)", line):
            summary = line.strip("= ")
            break
    unexpected = [f for f in failed if f not in KNOWN_ON_BASE]
    passed = bool(summary) and not unexpected and " passed" in summary
    print(json.dumps({
        "passed": passed,
        "compared": "csa2 unit tests on this tree vs the failure set of unmodified base 2fa8e034 "
                    "(job 1253) plus the 4 ruled pre-existing failures (deselected)",
        "failed": failed,
        "unexpected_failures": unexpected,
        "known_on_base": sorted(KNOWN_ON_BASE),
        "deselected_preexisting": DESELECT,
        "summary": summary,
        "returncode": proc.returncode,
    }))


if __name__ == "__main__":
    main()
