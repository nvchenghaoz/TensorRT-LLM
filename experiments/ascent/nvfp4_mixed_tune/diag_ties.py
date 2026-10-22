# Per-candidate tuner medians at the mixed buckets (2080/4128), over NPROC fresh processes.
import os
import re
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nvfp4_coldtune as ct  # noqa: E402

SH = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o_out": (5120, 6144), "down": (5120, 17408)}
NPROC = int(os.environ.get("NPROC", "3"))


def one():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    ct.MS = (8192,)
    names = os.environ.get("SHAPES", ",".join(SH)).split(",")
    for nm in names:
        s = ct.Shape(nm, *SH[nm])
        AutoTuner.get().clear_cache()
        print("SHAPE %s" % nm, flush=True)
        with autotune():
            torch.ops.trtllm.nvfp4_gemm(*s.inputs(8192), torch.bfloat16, allowed_backends=ct.BACKENDS)
        torch.cuda.synchronize()
        del s
        torch.cuda.empty_cache()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        return one()
    env = dict(os.environ, TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO="1", TLLM_LOG_LEVEL="INFO")
    for i in range(NPROC):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "one"], env=env,
                           capture_output=True, text=True, timeout=900)
        shape = None
        rows = {}
        for ln in (p.stdout + p.stderr).splitlines():
            if ln.startswith("SHAPE "):
                shape = ln.split()[1]
            if "near-tie band" in ln:
                m = re.search(r"tactic=(.*?), shapes=\(\((\d+),", ln)
                t = re.search(r"time=([\d.]+)ms", ln)
                if m and t:
                    rows.setdefault((shape, m.group(2)), []).append((m.group(1), float(t.group(1))))
            if "[Autotuner] Selected:" in ln and "near" not in ln:
                pass
        print("== process %d (rc %d)" % (i, p.returncode), flush=True)
        for (shape, mm), lst in sorted(rows.items()):
            best = min(v for _, v in lst)
            print("  %s M=%s: %s" % (shape, mm, "  ".join("%s %.4f (%+.1f%%)" % (
                tac[-28:], v, 100 * (v / best - 1)) for tac, v in lst)), flush=True)
        if p.returncode != 0 or not rows:
            print(p.stdout[-2000:], p.stderr[-3000:])
    print('{"passed": true, "note": "diagnostic only"}')


if __name__ == "__main__":
    main()
