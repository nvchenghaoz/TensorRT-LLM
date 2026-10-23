"""Idea 50 step 0a, per-kernel view: Kineto trace of the step0a chain (arm A only).

Same chain as step0a.py (out_proj Marlin -> post norm -> fp4_quantize -> gate_up
-> silu_mul_nvfp4 -> down -> next norm -> qkvz). For each M, replays the
captured graph R times under torch.profiler and computes, per kernel instance,
  exposed = end - max(start, latest end of every earlier-started kernel)
  handoff = start - predecessor end (negative: launched early via PDL)
  tail    = end - predecessor end
for the post_attention norm (predecessor out_proj Marlin) and the next_layer
norm (predecessor the down GEMM's last kernel), plus the Marlin kernel names,
grids and smem. Medians over all layers and replays (first layer of each replay
dropped).
"""
import json
import os
import statistics
import sys
import tempfile

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0a as S  # noqa: E402

R = int(os.environ.get("TR_REPLAYS", "6"))


def med(v):
    return round(statistics.median(v), 3) if v else None


def main():
    ms = [int(a) for a in sys.argv[1:]] or [4, 8, 12, 16, 24, 32]
    W = S.build_weights()
    out = {}
    for m in ms:
        ch = S.Chain(m, W)
        g, _ = ch.capture(True, True)
        for _ in range(5):
            ch.reset()
            g.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(R):
                ch.reset()
                torch.sum(W["scrub"], dim=0, out=W["sink"])
                g.replay()
            torch.cuda.synchronize()
        fn = tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(fn)
        with open(fn) as f:
            tr = json.load(f)
        os.unlink(fn)
        ks = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
        ks.sort(key=lambda e: e["ts"])
        rows = []
        maxend = -1e30
        for e in ks:
            s, d = float(e["ts"]), float(e["dur"])
            en = s + d
            rows.append(dict(name=e["name"], s=s, e=en, exp=en - max(s, maxend), prev_end=maxend,
                             grid=e.get("args", {}).get("grid"),
                             smem=e.get("args", {}).get("shared memory"),
                             block=e.get("args", {}).get("block")))
            maxend = max(maxend, en)
        post, nxt, marlin_out, misc = [], [], [], {}
        names = [r["name"] for r in rows]
        for i, r in enumerate(rows):
            nm = r["name"]
            is_norm = "RMSNorm" in nm or "rmsnorm" in nm.lower() or "Norm" in nm
            if not is_norm or i == 0:
                continue
            p = rows[i - 1]
            rec = dict(exp=r["exp"], hand=r["s"] - p["e"], tail=r["e"] - p["e"], dur=r["e"] - r["s"],
                       pred=p["name"][:80], pred_dur=p["e"] - p["s"], pred_grid=p["grid"],
                       pred_smem=p["smem"], grid=r["grid"], smem=r["smem"])
            if "Marlin" in p["name"]:
                post.append(rec)
            else:
                nxt.append(rec)
        for r in rows:
            key = r["name"][:120]
            misc.setdefault(key, []).append(r["exp"])

        def summ(v):
            if not v:
                return None
            return dict(n=len(v), exposed_us=med([x["exp"] for x in v]),
                        handoff_us=med([x["hand"] for x in v]), tail_us=med([x["tail"] for x in v]),
                        dur_us=med([x["dur"] for x in v]), pred=v[0]["pred"],
                        pred_dur_us=med([x["pred_dur"] for x in v]), pred_grid=v[0]["pred_grid"],
                        pred_smem=v[0]["pred_smem"], grid=v[0]["grid"], smem=v[0]["smem"])

        row = dict(post=summ(post), next=summ(nxt))
        if row["post"] and row["next"]:
            row["excess_exposed_us"] = round(row["post"]["exposed_us"] - row["next"]["exposed_us"], 3)
        row["kernels_exposed_us"] = {k: [len(v), med(v)] for k, v in misc.items()}
        out[str(m)] = row
        print("M", m, json.dumps(row), flush=True)
        del g, ch
        torch.cuda.empty_cache()
    print(json.dumps({"trace": {k: dict(post=v["post"], next=v["next"],
                                        excess=v.get("excess_exposed_us")) for k, v in out.items()}}))


if __name__ == "__main__":
    main()
