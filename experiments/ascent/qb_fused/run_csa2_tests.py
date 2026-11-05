"""Run the csa2 unit suites (test_module.py, test_kernel.py, test_indexer.py) and compare against main.

Main 2fa8e034 (job 1144) has exactly these 4 pre-existing failures (237 passed, 154 skipped):
  test_module.py::test_module_sends_whole_context_and_generation_phases[trtllm|flashinfer]
      (main's _shared_mxfp8_active reads self.wq_a.quant_method on an nn.Identity stub)
  test_indexer.py::test_gathered_decode_graph_grows_past_warmup_prefix (pitfall 1042)
  test_indexer.py::test_constant_rows_retained_only_for_captured_graphs
passed = no failure outside that set and at least as many passes as main. Prints one JSON line last.
"""

import json
import re
import subprocess
import sys

TESTS = [
    "tests/unittest/_torch/attention/sparse/csa2/test_module.py",
    "tests/unittest/_torch/attention/sparse/csa2/test_kernel.py",
    "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py",
]
MAIN_FAILURES = {
    "tests/unittest/_torch/attention/sparse/csa2/test_module.py::test_module_sends_whole_context_and_generation_phases[trtllm]",
    "tests/unittest/_torch/attention/sparse/csa2/test_module.py::test_module_sends_whole_context_and_generation_phases[flashinfer]",
    "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py::test_gathered_decode_graph_grows_past_warmup_prefix",
    "tests/unittest/_torch/attention/sparse/csa2/test_indexer.py::test_constant_rows_retained_only_for_captured_graphs",
}
MAIN_PASSED = 237


def main():
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", *TESTS],
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out[-6000:], flush=True)
    failed = set(re.findall(r"^FAILED (\S+)", out, re.M))
    m = re.search(r"(\d+) passed", out)
    passed_n = int(m.group(1)) if m else 0
    sk = re.search(r"(\d+) skipped", out)
    new_failures = sorted(failed - MAIN_FAILURES)
    ok = not new_failures and passed_n >= MAIN_PASSED
    print(json.dumps({"passed": ok, "pytest_returncode": proc.returncode, "passed_tests": passed_n,
                      "skipped": int(sk.group(1)) if sk else 0, "failed": sorted(failed),
                      "new_failures_vs_main": new_failures,
                      "compared_against": "main 2fa8e034 csa2 suites (job 1144): same 4 pre-existing failures"}))


if __name__ == "__main__":
    main()
