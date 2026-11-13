"""Frozen check adapter plus a per-request reset sequence in the SAME server (idea 113).

Imports experiments/ascent-benchmark/run_check.py unchanged and wraps its run_bench: right after the
frozen "repeat" pass (park_portrait seed 42 again), it sends park_portrait at seed 43 and then
park_portrait at seed 42 once more. Afterwards the last seed-42 output is compared with the first-pass
park output by the frozen tolerance (ab_metrics.passes with the check's own tolerance), and the seed-43
output is reported against the first pass (it must differ: a different seed) and against the reference.
Order in one server: park42, perfume42, robot42 (pass 1), park42 (repeat), park43, park42 (reset).
Result: the frozen check's JSON plus "reset_sequence"; emitted the same way (adapter_result.json + last
stdout line).

usage: python run_check_reset.py '<harness_config JSON>'
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "ascent-benchmark"
sys.path.insert(0, str(BENCH))

import run_check as rc  # noqa: E402  (frozen adapter, imported unchanged)

EXTRA: dict = {}
_orig_run_bench = rc.run_bench


def _run_bench(srv, requests, tag, common, concurrency=1, hb=None):
    out = _orig_run_bench(srv, requests, tag, common, concurrency=concurrency, hb=hb)
    if tag == "repeat" and requests:
        prompts = rc.load_json("prompts.json")
        key = requests[0]["key"]
        for t, seed in (("reset43", 43), ("reset42", 42)):
            try:
                EXTRA[t] = _orig_run_bench(srv, [{"key": key, "prompt": prompts["prompts"][key], "seed": seed}],
                                           t, common, hb=hb)
            except Exception:  # keep the frozen check's result even if the extra requests fail
                EXTRA[t] = {"error": traceback.format_exc()[-2000:]}
    return out


rc.run_bench = _run_bench


def reset_report(out: dict, harness: dict) -> dict:
    from ab_metrics import VideoMetrics, passes

    manifest = rc.load_json("reference_manifest.json")
    store = Path(manifest["store"])
    common = rc.workload_params(harness)
    tol = out.get("tolerance")
    first = {}
    for r in out.get("requests") or []:
        first.setdefault(r["key"], r["path"])
    rep: dict = {"order": ["pass1 park42", "perfume42", "robot42", "repeat park42", "park43", "park42"]}
    vm = VideoMetrics()
    ok = True
    for t in ("reset43", "reset42"):
        b = EXTRA.get(t)
        if b is None or "error" in b:
            rep[t] = {"error": (b or {}).get("error", "not run")}
            ok = False
            continue
        obs, errs, reqs = rc.observe(b, common)
        if errs or not reqs:
            rep[t] = {"error": errs or "no output"}
            ok = False
            continue
        rec = reqs[0]
        base = first.get(rec["key"])
        m_first = vm.compare_files(Path(base), Path(rec["path"])) if base else None
        m_ref = vm.compare_files(store / manifest["reference"][rec["key"]], Path(rec["path"]))
        row = {"key": rec["key"], "vs_first_pass": m_first, "vs_reference": m_ref,
               "server_denoise_s": rec.get("server_denoise_s"), "client_e2e_s": rec.get("client_e2e_s")}
        if t == "reset42":
            row["passed"] = bool(m_first is not None and tol and passes(m_first, tol))
            ok = ok and row["passed"]
        else:
            row["differs_from_seed42"] = bool(m_first is not None and not m_first.get("bitwise_equal"))
            ok = ok and row["differs_from_seed42"]
        rep[t] = row
    rep["passed"] = bool(ok and out.get("repeated_execution_passed"))
    return rep


def main() -> dict:
    harness = rc.parse_harness(sys.argv)
    out = rc.main()
    try:
        out["reset_sequence"] = reset_report(out, harness)
    except Exception:
        out["reset_sequence"] = {"passed": False, "error": traceback.format_exc()[-3000:]}
    return out


if __name__ == "__main__":
    try:
        res = main()
    except Exception:
        res = {"adapter": "check", "valid": False, "reference_independent": True,
               "errors": [traceback.format_exc()[-4000:]]}
    rc.emit(rc.finite_json(res))
