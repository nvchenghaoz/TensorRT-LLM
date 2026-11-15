"""Idea 9 / branch 30: BCG (landing 16) evidence for the dense NVFP4 path, candidate ON.

Runs experiments/ascent/bcg_prefill/bcg_ab.py mode `inproc` unchanged (imported, not
edited): ONE LLM with this tree's engine.json (prefill_cuda_graph_backend=breakable,
disable_mm_encoder, buckets 300..2000), replay gated by BreakableCUDAGraphRunner.has_graph;
passes A eager / B replay / A' eager / B' replay plus 25 repeated replays; token ids + top-20
logprobs bitwise. Groups reach every bucket (1x300, 2..6x300 = 600..1800, 1x1000, 2x1000,
1x2000; 1x8000 and 4x1000 run eager).

Added here: every trtllm::nvfp4_gemm dispatch is counted by (path dense/main, phase
capture/eager, tuning or inference, N, M); a dispatch whose (N, M, tactic) is first seen
while the stream is capturing counts as lazy init under capture; a dispatch under capture
in tuning mode counts as tuning under capture. Both must be 0.

usage: bcg_inproc.py <checkpoint>
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
    sys.path.insert(0, os.path.join(HERE, "..", "bcg_prefill"))
    import bcg_ab  # noqa: E402  (sets TLLM_WORKER_USE_SINGLE_PROCESS=1)
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco

    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner

    counts = {}
    seen = set()
    bad = {"lazy_init_under_bcg_capture": 0, "tuning_under_bcg_capture": 0}
    in_bcg = [False]
    orig_capture = BreakableCUDAGraphRunner.capture

    def bcg_capture(self, *a, **k):
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
        path = "dense" if self.dense_min_m is not None else "main"
        # phase: bcg-capture (inside BreakableCUDAGraphRunner.capture, stream capturing),
        # bcg-warmup (inside capture(), eager warm-up forwards), other-capture (decode CUDA
        # graphs, or the AutoTuner's own timing graphs in tuning mode), eager
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
        seen.add(sig)
        return orig(self, inputs, tactic=tactic, bias=bias, **kw)

    tco.NVFP4GemmUnifiedRunner.forward = fwd
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        bcg_ab.mode_inproc()
    finally:
        sys.stdout = tee.inner
    res = json.loads(tee.last_json) if tee.last_json else {"passed": False, "error": "no inproc JSON"}
    cap_dense = {k: v for k, v in counts.items() if k.startswith("dense|bcg-capture")}
    cap_main = {k: v for k, v in counts.items() if k.startswith("main|bcg-capture")}
    passed = bool(res.get("passed")) and bad["lazy_init_under_bcg_capture"] == 0 \
        and bad["tuning_under_bcg_capture"] == 0 and sum(cap_dense.values()) > 0
    print(json.dumps({"passed": passed, "compared": "BCG replay vs eager, one process, dense NVFP4 tuning ON "
                      "(bcg_ab.py inproc) + dispatch census during capture",
                      "inproc": res, "dense_dispatches_during_capture": cap_dense,
                      "main_dispatches_during_capture": cap_main, "capture_hazards": bad,
                      "all_dispatch_counts": counts}))


if __name__ == "__main__":
    main()
