"""Idea 50 step 0a cross-check: the same chain (arm A) under nsys --cuda-graph-trace=node.

Usage (inside one GPU job):
  nsys profile --cuda-graph-trace=node -t cuda -o OUT --export sqlite \
      python nsys_check.py run 8 16
  python nsys_check.py analyze OUT.sqlite
'analyze' computes, for each post_attention norm (predecessor Marlin) and each
next_layer norm (predecessor cuBLASLt splitKreduce), exposed = end - max(start,
latest end of earlier-started kernels), the same formula as trace.py (Kineto) and
the census (fact 279), to show whether node-level nsys reproduces the census excess.
"""
import json
import os
import sqlite3
import statistics
import sys


def run(ms):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import step0a as S
    W = S.build_weights()
    for m in ms:
        ch = S.Chain(m, W)
        g, _ = ch.capture(True, True)
        for _ in range(5):
            ch.reset()
            g.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"M{m}")
        for _ in range(6):
            ch.reset()
            torch.sum(W["scrub"], dim=0, out=W["sink"])
            g.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"M={m} done", flush=True)
        del g, ch
        torch.cuda.empty_cache()


def analyze(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    ranges = cur.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'M%'").fetchall()
    rows = cur.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                       "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start").fetchall()
    out = {}
    for text, rs, re_ in ranges:
        ks = [r for r in rows if rs <= r[0] <= re_]
        maxend = -1
        recs = []
        for i, (s, e, nm) in enumerate(ks):
            exp = e - max(s, maxend)
            recs.append((s, e, nm, exp))
            maxend = max(maxend, e)
        post, nxt = [], []
        for i in range(1, len(recs)):
            s, e, nm, exp = recs[i]
            if "RMSNorm" not in nm and "rmsnorm" not in nm.lower():
                continue
            ps, pe, pn, _ = recs[i - 1]
            item = (exp / 1e3, (s - pe) / 1e3)
            (post if "Marlin" in pn else nxt).append(item)
        md = lambda v: round(statistics.median(v), 3) if v else None  # noqa: E731
        out[text] = dict(post_exposed_us=md([x[0] for x in post]), post_n=len(post),
                         post_handoff_us=md([x[1] for x in post]),
                         next_exposed_us=md([x[0] for x in nxt]), next_n=len(nxt),
                         next_handoff_us=md([x[1] for x in nxt]))
        if post and nxt:
            out[text]["excess_us"] = round(out[text]["post_exposed_us"] - out[text]["next_exposed_us"], 3)
    print(json.dumps({"nsys_node": out}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run([int(a) for a in sys.argv[2:]] or [8, 16])
    else:
        analyze(sys.argv[2])
