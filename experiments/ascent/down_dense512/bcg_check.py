"""Idea 112 / branch 51: BCG R/E/R' evidence for the down_proj CUTLASS-only dense band.

A copy of experiments/ascent/nvfp4_dense/bcg_inproc.py (not edited) that labels the new band's
dispatches ("dp", runner.dense_cutlass_only) apart from the first dense band ("dense") and main
("main"). It runs experiments/ascent/bcg_prefill/bcg_ab.py mode `inproc` unchanged: ONE LLM with
this tree's engine.json (breakable prefill graphs at 300..2000), passes A eager / B replay /
A' eager / B' replay plus 25 repeated replays; token ids + top-20 logprobs bitwise. Groups reach
600/900/1000 (2x300, 3x300, 1x1000) and other captured sizes.

Also logs, for this lifetime, every nvfp4 autotuner entry for down (N=5376) and gate_up after
startup (the picks R and E share, since both run in this process), and requires:
  - dp dispatches during BCG capture at M 600, 900 and 1000 (> 0 each);
  - no lazy init / tuning under capture;
  - dp dispatches only at 513 <= M <= 1023.

usage: bcg_check.py <checkpoint>
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Tee(io.TextIOBase):
    def __init__(self, inner):
        self.inner = inner
        self.last_json = None
        self._buf = ""

    def write(self, s):
        self.inner.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.startswith("{"):
                self.last_json = line
        return len(s)

    def flush(self):
        self.inner.flush()


def main():
    os.environ["CKPT"] = sys.argv[1]
    os.environ["TRTLLM_NVFP4_DENSE_TUNING"] = "1"
    os.environ["TRTLLM_FI_HOST_PREFILL_PLAN"] = os.environ.get("TRTLLM_FI_HOST_PREFILL_PLAN", "1")
    print(f"[env] TLLM_AUTOTUNER_CACHE_PATH={os.environ.get('TLLM_AUTOTUNER_CACHE_PATH')} "
          f"TRTLLM_CACHE_DIR={os.environ.get('TRTLLM_CACHE_DIR')}", flush=True)
    sys.path.insert(0, os.path.join(HERE, "..", "bcg_prefill"))
    import bcg_ab  # noqa: E402  (sets TLLM_WORKER_USE_SINGLE_PROCESS=1)
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner

    counts = {}
    seen = set()
    bad = {"lazy_init_under_bcg_capture": 0, "tuning_under_bcg_capture": 0, "dp_outside_band": 0}
    in_bcg = [False]
    picks_logged = [False]
    picks = {}
    orig_capture = BreakableCUDAGraphRunner.capture

    def log_picks():
        if picks_logged[0]:
            return
        picks_logged[0] = True
        for key, val in AutoTuner.get().profiling_cache.cache.items():
            if key[0].startswith("trtllm::nvfp4_gemm::"):
                sh = key[3]
                picks[f"{key[0].split('::')[-1]}|N{sh[1][0]}|M{sh[0][0]}"] = str(val[1])
        for k in sorted(picks, key=lambda s: (s.split("|")[0], s.split("|")[1], int(s.split("|")[2][1:]))):
            if "N5376" in k:
                print(f"[pick] {k} -> {picks[k]}", flush=True)

    def bcg_capture(self, *a, **k):
        log_picks()  # tuning is done before the first capture
        in_bcg[0] = True
        try:
            return orig_capture(self, *a, **k)
        finally:
            in_bcg[0] = False

    BreakableCUDAGraphRunner.capture = bcg_capture
    orig = tco.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        cap = torch.cuda.is_current_stream_capturing()
        tuning = AutoTuner.get().is_tuning_mode
        n, m = int(inputs[1].shape[0]), int(inputs[0].shape[0])
        path = "dp" if getattr(self, "dense_cutlass_only", False) else \
            ("dense" if self.dense_min_m is not None else "main")
        if in_bcg[0]:
            phase = "bcg-capture" if cap else "bcg-warmup"
        else:
            phase = "other-capture" if cap else "eager"
        key = f"{path}|{phase}|{'tuning' if tuning else 'infer'}|N{n}|M{m}"
        counts[key] = counts.get(key, 0) + 1
        sig = (n, m, str(tactic), path)
        if phase == "bcg-capture" and sig not in seen:
            bad["lazy_init_under_bcg_capture"] += 1
        if phase == "bcg-capture" and tuning:
            bad["tuning_under_bcg_capture"] += 1
        if path == "dp" and not tuning and not (513 <= m <= 1023):
            bad["dp_outside_band"] += 1
        seen.add(sig)
        return orig(self, inputs, tactic=tactic, bias=bias, **kw)

    tco.NVFP4GemmUnifiedRunner.forward = fwd
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        bcg_ab.mode_inproc()
    finally:
        sys.stdout = tee.inner
    log_picks()
    res = json.loads(tee.last_json) if tee.last_json else {"passed": False, "error": "no inproc JSON"}
    cap_dp = {k: v for k, v in counts.items() if k.startswith("dp|bcg-capture")}
    need = {m: sum(v for k, v in cap_dp.items() if k.endswith(f"|N5376|M{m}")) for m in (600, 900, 1000)}
    eager_dp = {k: v for k, v in counts.items() if k.startswith("dp|eager|infer")}
    passed = bool(res.get("passed")) and all(v == 0 for v in bad.values()) and all(v > 0 for v in need.values())
    print(json.dumps({"passed": passed,
                      "compared": "BCG replay (R=B,B2) vs eager (E=A,A2), one process, band ON (bcg_ab.py inproc), "
                                  "token ids + top-20 logprobs bitwise; dispatch census during capture",
                      "inproc": res, "dp_dispatches_during_capture_by_M": need, "dp_capture_counts": cap_dp,
                      "dp_eager_counts": eager_dp, "capture_hazards": bad,
                      "down_picks_this_lifetime": {k: v for k, v in picks.items() if "N5376" in k},
                      "all_dispatch_counts": counts}))


if __name__ == "__main__":
    main()
