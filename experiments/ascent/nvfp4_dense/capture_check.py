"""Idea 9 / branch 30: CUDA-graph capture evidence for the dense NVFP4 path (BCG, landing 16).

Synthetic operands of the real MLP shapes (config.json). After the engine-style
tuning (one M=8192 call per shape under autotune()), for gate_up and down at
M = 300, 301 (first call at that M happens under capture), 1200 and 2000:
  - capture trtllm::nvfp4_gemm in torch.cuda.graph with
    torch.cuda.set_sync_debug_mode("error") (a host sync raises);
  - count allocations made during capture and compare with main's path
    (switch OFF) captured the same way;
  - record which path ran (dense / main) and the tactic;
  - replay 25 times; every replay's output int16-equal to eager (dense) and to
    main's eager output.
The decision is a pure function of the shape and the process-start switch
(TRTLLM_NVFP4_DENSE_TUNING); nothing is read per call that capture would freeze.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

log = bench.log


def main():
    ck = sys.argv[1]
    import tensorrt_llm  # noqa: F401
    t = bench.tco()
    dims = bench.read_dims(ck)
    weights = {name: bench.make_weight(*dims[name], 100 + i) for i, name in enumerate(("gate_up", "down"))}
    bench.tune(weights, dims)
    paths = []
    _fwd = t.NVFP4GemmUnifiedRunner.forward

    def fwd(self, inputs, tactic=-1, bias=None, **kw):
        paths.append(("dense" if self.dense_min_m is not None else "main", str(tactic)))
        return _fwd(self, inputs, tactic=tactic, bias=bias, **kw)

    t.NVFP4GemmUnifiedRunner.forward = fwd
    results = {}
    passed = True
    for name in ("gate_up", "down"):
        n, k = dims[name]
        wq, wsf, wgs = weights[name]
        for m in (300, 301, 1200, 2000):
            xq, xsf, xgs = bench.make_act(m, k, 31 + m, heavy=(m == 1200))
            alpha = (1.0 / (xgs * wgs)).float()
            row = {}
            for arm, flag in (("dense", True), ("main", False)):
                bench.set_dense(flag)
                if not (m == 301):
                    eager = bench.gemm(xq, xsf, wq, wsf, alpha)  # warm call (BCG warms before capture)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
                paths.clear()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    with torch.cuda.graph(g):
                        out = bench.gemm(xq, xsf, wq, wsf, alpha)
                    sync_ok = True
                except Exception as e:  # noqa: BLE001
                    sync_ok = False
                    row[f"{arm}_error"] = str(e)[:300]
                finally:
                    torch.cuda.set_sync_debug_mode(0)
                s1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
                row[f"{arm}_path_on_capture"] = list(paths)
                row[f"{arm}_allocs_during_capture"] = s1 - s0
                row[f"{arm}_no_host_sync"] = sync_ok
                if not sync_ok:
                    passed = False
                    continue
                eager = bench.gemm(xq, xsf, wq, wsf, alpha)
                torch.cuda.synchronize()
                bad = 0
                for _ in range(25):
                    g.replay()
                    torch.cuda.synchronize()
                    bad += bench.int16_mismatch(out, eager)[0]
                row[f"{arm}_replay_vs_eager_mismatch"] = bad
                row[f"{arm}_out"] = out.clone()
                del g
            bench.set_dense(True)
            vs_main = bench.int16_mismatch(row.pop("dense_out"), row.pop("main_out"))[0] \
                if "dense_out" in row and "main_out" in row else -1
            row["dense_replay_vs_main_replay_mismatch"] = vs_main
            ok = (row.get("dense_no_host_sync") and row.get("main_no_host_sync")
                  and row.get("dense_replay_vs_eager_mismatch") == 0 and row.get("main_replay_vs_eager_mismatch") == 0
                  and vs_main == 0 and row["dense_allocs_during_capture"] == row["main_allocs_during_capture"])
            row["ok"] = bool(ok)
            passed &= bool(ok)
            results[f"{name}@{m}"] = row
            log(f"[capture] {name}@{m} {json.dumps(row)}")
    print(json.dumps({"passed": bool(passed), "compared": "torch.cuda.graph capture of trtllm::nvfp4_gemm (dense vs "
                      "main path) under sync-debug error mode; allocations during capture equal to main's; 25 "
                      "replays int16 vs eager and dense replay vs main replay", "results": results}))


if __name__ == "__main__":
    main()
