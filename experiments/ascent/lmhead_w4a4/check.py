# Correctness for the SM120 lm_head W4A4 rows (branch only; run_gpu_job).
# Builds this tree's lm_head (Marlin < TRTLLM_LMHEAD_W4A4_MIN_ROWS <= W4A4 FP32) and a
# Marlin-only lm_head (kill switch 0 = main's path) from the checkpoint's lm_head tensors.
# Same warmup/iteration/flush points as pair.py: autotune emulation (1 row under autotune()),
# then per M: 10 + ITERS graph replays (graph M) or eager calls (eager M) with a 512 MiB read
# flush before each; every iteration compared bitwise with iteration 1 and with the eager
# output (graph M). Errors of both heads vs an FP64 exact-dequant reference (same quantized
# weight; synthetic activations at the checkpoint's input_scale range), argmax agreement,
# and cross-process determinism (two fresh processes, each autotuning, hash the outputs).
# usage: python check.py <checkpoint> [hash]
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

import lmhead_bench as lb  # noqa: E402

K = 5120
GRAPH_MS = [1, 16, 32, 64, 128]
EAGER_MS = [48, 51, 52, 56, 63, 65, 96, 104, 112, 132]
HASH_MS = [56, 64, 96, 112, 128, 132]
WARM, ITERS = 10, int(os.environ.get("ITERS", "100"))


def log(*a):
    print(*a, flush=True)


def hb(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).view(-1).cpu().numpy()
                          .tobytes()).hexdigest()[:20]


def build(t, min_rows=None):
    old = os.environ.get("TRTLLM_LMHEAD_W4A4_MIN_ROWS")
    if min_rows is not None:
        os.environ["TRTLLM_LMHEAD_W4A4_MIN_ROWS"] = str(min_rows)
    try:
        head, arm = lb.build_head(t)
    finally:
        if min_rows is not None:
            if old is None:
                os.environ.pop("TRTLLM_LMHEAD_W4A4_MIN_ROWS")
            else:
                os.environ["TRTLLM_LMHEAD_W4A4_MIN_ROWS"] = old
    return head, arm


def inputs(t):
    sigma = float(t["input_scale"].float()) * 2688.0 / 6.0
    g = torch.Generator(device="cuda").manual_seed(4242)
    return {m: (torch.randn(m, K, generator=g, device="cuda") * sigma).to(torch.bfloat16)
            for m in sorted(set(GRAPH_MS + EAGER_MS + HASH_MS))}


def tune(head, x):
    from tensorrt_llm._torch.autotuner import autotune
    with torch.inference_mode(), autotune():
        head(x[:1])
    torch.cuda.synchronize()


def hash_mode(ckpt):
    t = lb.load_lm_head(ckpt)
    head, arm = build(t)
    xs = inputs(t)
    tune(head, xs[min(xs)])
    out = {}
    with torch.inference_mode():
        for m in HASH_MS:
            out[str(m)] = hb(head(xs[m]).float())
    print(json.dumps({"arm": arm, "hashes": out}))


def main():
    ckpt = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "hash":
        return hash_mode(ckpt)
    t = lb.load_lm_head(ckpt)
    head, arm = build(t)
    ref_head, ref_arm = build(t, min_rows=0)
    min_rows = head.__dict__.get("_w4a4_min_rows", 0)
    log("tree arm %s min_rows %s; marlin-only arm %s" % (arm, min_rows, ref_arm))
    xs = inputs(t)
    tune(head, xs[min(xs)])
    tune(ref_head, xs[min(xs)])
    flush = torch.empty(512 << 18, dtype=torch.float32, device="cuda")
    fsum = torch.empty((), dtype=torch.float32, device="cuda")
    ok = arm == "MarlinW4A4NVFP4LMHeadMethod" and min_rows > 0
    per_m = {}
    for m in GRAPH_MS + EAGER_MS:
        x = xs[m]
        graph = m in GRAPH_MS
        with torch.inference_mode():
            eager = head(x).float().clone()
            eager2 = head(x).float().clone()
            marlin = ref_head(x).float().clone()
        if graph:
            gr, gout = lb.capture(lambda y: head(y).float(), x)
        first = None
        mis_first = mis_eager = 0
        it_err = []
        ref = lb.ref_fp64(t, x)
        with torch.inference_mode():
            for i in range(WARM + ITERS):
                torch.sum(flush, dim=0, out=fsum)
                if graph:
                    gr.replay()
                    o = gout.clone()
                else:
                    o = head(x).float()
                torch.cuda.synchronize()
                if first is None:
                    first = o
                mis_first += int(not torch.equal(o, first))
                mis_eager += int(not torch.equal(o, eager))
                if i % 10 == 0:
                    it_err.append(float((o.double() - ref).abs().max()))
        w4 = m >= min_rows > 0
        e_tree = lb.err_stats(eager, ref)
        e_marlin = lb.err_stats(marlin, ref)
        d = {"path": "w4a4_fp32" if w4 else "marlin", "mode": "graph" if graph else "eager",
             "out_dtype": str(head(x).dtype) if True else None,
             "iters": WARM + ITERS, "mismatch_vs_first": mis_first,
             "mismatch_vs_eager": mis_eager,
             "eager_repeat_equal": bool(torch.equal(eager, eager2)),
             "per_iter_max_err_min": min(it_err), "per_iter_max_err_max": max(it_err),
             "finite": bool(torch.isfinite(eager).all()),
             "err_tree_vs_fp64": e_tree, "err_marlin_vs_fp64": e_marlin,
             "tree_vs_marlin_argmax_match": float(
                 (eager.argmax(-1) == marlin.argmax(-1)).float().mean()),
             "tree_equals_marlin_bitwise": bool(torch.equal(eager, marlin))}
        ok = ok and mis_first == 0 and mis_eager == 0 and d["eager_repeat_equal"] and d["finite"]
        if not w4:
            ok = ok and d["tree_equals_marlin_bitwise"]
        per_m[str(m)] = d
        log("M=%d %s" % (m, json.dumps(d)))
        if graph:
            del gr, gout
        del ref
        torch.cuda.empty_cache()
    # cross-process determinism (each process autotunes afresh)
    hashes = []
    for k in range(2):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), ckpt, "hash"],
                           capture_output=True, text=True, timeout=900)
        line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        hashes.append(json.loads(line[-1])["hashes"] if line else {"rc": r.returncode,
                                                                   "err": r.stderr[-2000:]})
        log("process %d hashes %s" % (k, hashes[-1]))
    with torch.inference_mode():
        here = {str(m): hb(head(xs[m]).float()) for m in HASH_MS}
    xproc = hashes[0] == hashes[1] and "rc" not in hashes[0]
    ok = ok and xproc
    print(json.dumps({"passed": bool(ok), "arm": arm, "w4a4_min_rows": min_rows,
                      "per_M": per_m, "cross_process_hashes": hashes,
                      "this_process_hashes": here, "cross_process_equal": xproc,
                      "compared": "per-iteration bitwise vs iteration 1 and vs eager "
                                  "(graph M), eager repeat, Marlin rows bitwise vs the "
                                  "Marlin-only head (main's path), cross-process hashes; "
                                  "errors vs FP64 exact-dequant reference (advisory, "
                                  "mode_switch)"}))


if __name__ == "__main__":
    main()
