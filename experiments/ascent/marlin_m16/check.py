# Correctness check for the dense Marlin NVFP4 rows 9-16 launch-shape table (kM16Table).
#  A. Accuracy, rows 1-16 at every tabled shape: table output (default env) vs an independent FP32
#     reference of the dequantized NVFP4 operands; main's selection (TRTLLM_MARLIN_M16_TABLE=0) on the
#     same inputs sets the bar: max-abs and mean-abs <= 1.25 x main's + 1 BF16 ulp of the output
#     magnitude (protocol class "reordered"). Rows 1-8 (outside the table) must be bitwise equal to
#     main's selection. No NaN/Inf.
#  B. Repeated execution with the timed loop's structure: C distinct copies in one CUDA graph (PDL on),
#     10 warmup + 50 replays; before every replay the outputs are NaN-poisoned; after every replay all
#     outputs must be bitwise equal to replay 1 and to the eager call of that copy; an eager call
#     (rotating copy) between replays must equal that copy's first eager call.
#  C. Persistent lock workspace: every bound slot reads 0 after all calls complete.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from common import SHAPES, eq, err, gemm, log, make_copy, set_cfg  # noqa: E402

WARM, ITERS = 10, 50
fails = []
summary = dict()


def check(cond, msg):
    log(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


for (n, k), C in SHAPES.items():
    C = min(C, 12)
    copies, refs = [], None
    for i in range(C):
        c, r = make_copy(n, k, 3000 + i, want_ref=(i == 0))
        copies.append(c)
        if r is not None:
            refs = r
    key = "%d,%d" % (n, k)
    summary[key] = dict(subnormal_blocks=refs["n_sub"], rows=dict(), repeated=dict())
    # ---------------- A
    for m in range(1, 17):
        g = torch.Generator(device="cuda").manual_seed(100 + m)
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        ref = x.float() @ refs["w_ref"].t()
        set_cfg(None)
        y_t = gemm(x, copies[0], n, k)
        set_cfg("main")
        y_m = gemm(x, copies[0], n, k)
        set_cfg(None)
        et, em = err(y_t, ref), err(y_m, ref)
        ulp = et["bf16_ulp"]
        same = eq(y_t, y_m)
        summary[key]["rows"][str(m)] = dict(table_max_abs=et["max_abs"], table_mean_abs=et["mean_abs"],
                                            main_max_abs=em["max_abs"], main_mean_abs=em["mean_abs"],
                                            bf16_ulp=ulp, bitwise_vs_main=same)
        log("A (%d,%d) rows=%2d table max %.5f mean %.6f | main max %.5f mean %.6f | ulp %.4f | bitwise_vs_main %s" % (
            n, k, m, et["max_abs"], et["mean_abs"], em["max_abs"], em["mean_abs"], ulp, same))
        check(et["finite"], "A (%d,%d) rows=%d finite" % (n, k, m))
        check(et["max_abs"] <= 1.25 * em["max_abs"] + ulp and et["mean_abs"] <= 1.25 * em["mean_abs"] + ulp,
              "A (%d,%d) rows=%d table error <= 1.25 x main's + 1 BF16 ulp" % (n, k, m))
        if m <= 8:
            check(same, "A (%d,%d) rows=%d (outside the table) bitwise == main selection" % (n, k, m))
    # ---------------- B
    for m in [8, 9, 12, 16]:
        set_cfg(None)
        g = torch.Generator(device="cuda").manual_seed(200 + m)
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        eager0 = [gemm(x, c, n, k).clone() for c in copies]
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            for _ in range(2):
                [gemm(x, c, n, k) for c in copies]
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = [gemm(x, c, n, k) for c in copies]
        first, ndiff, n_eager_diff, nan_seen = None, 0, 0, 0
        max_err_vs_first = 0.0
        for it in range(WARM + ITERS):
            for t in outs:
                t.fill_(float("nan"))
            gr.replay()
            torch.cuda.synchronize()
            cur = [t.clone() for t in outs]
            nan_seen += sum(int(torch.isnan(t.float()).any().item()) for t in cur)
            if first is None:
                first = cur
            for a, b, e in zip(cur, first, eager0):
                if not (eq(a, b) and eq(a, e)):
                    ndiff += 1
                    max_err_vs_first = max(max_err_vs_first, (a.float() - b.float()).abs().max().item())
            j = it % C
            ye = gemm(x, copies[j], n, k)
            if not eq(ye, eager0[j]):
                n_eager_diff += 1
        summary[key]["repeated"][str(m)] = dict(replay_diffs=ndiff, eager_diffs=n_eager_diff, nan_outputs=nan_seen,
                                                max_abs_drift=max_err_vs_first)
        check(ndiff == 0 and n_eager_diff == 0 and nan_seen == 0,
              "B (%d,%d) M=%d %d NaN-poisoned replays x %d copies bitwise == replay 1 == eager; interleaved eager "
              "bitwise stable (replay diffs %d, eager diffs %d, NaN outputs %d)" % (
                  n, k, m, WARM + ITERS, C, ndiff, n_eager_diff, nan_seen))
        del gr, outs
    del copies, refs
    torch.cuda.empty_cache()

torch.cuda.synchronize()
snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device="cuda"))
check(snap.numel() == 0 or int(snap.abs().sum().item()) == 0,
      "C persistent lock workspace all zero after all calls (%d entries)" % snap.numel())

print(json.dumps(dict(passed=not fails, failures=fails, summary=summary,
                      compared_against="independent FP32 reference from the dequantized NVFP4 operands "
                                       "(E2M1 LUT x E4M3 SF x global, subnormal-scale blocks flushed as Marlin does); "
                                       "main's selection (TRTLLM_MARLIN_M16_TABLE=0) on the same inputs",
                      repeated_execution="<=12 distinct copies per shape in one CUDA graph (PDL), %d warmup + %d "
                                         "NaN-poisoned replays per M in 8,9,12,16, interleaved eager calls" % (
                                             WARM, ITERS))))
