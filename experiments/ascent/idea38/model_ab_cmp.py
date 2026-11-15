"""Comparison helpers for model_ab.py (A,B,A' arms, one pass each)."""
import json


def top1(lp_entry):
    # lp_entry: sorted [(token_id, logprob), ...] for the first generated position
    return max(v for _, v in lp_entry) if lp_entry else None


def cmp_runs(x, y, tol=0.125):
    n = eq = flips = over = 0
    maxd = maxd_top1 = 0.0
    first = []
    for name in x:
        for i, (a, b) in enumerate(zip(x[name], y[name])):
            n += 1
            if a == b:
                eq += 1
            elif len(first) < 5:
                first.append(f"{name}[{i}]")
            flip = a["ids"] != b["ids"]
            flips += flip
            d1 = 0.0
            if a["lp"] and b["lp"]:
                ta, tb = top1(a["lp"][0]), top1(b["lp"][0])
                d1 = abs(ta - tb)
                maxd_top1 = max(maxd_top1, d1)
                for la, lb in zip(a["lp"], b["lp"]):
                    da, db = dict(la), dict(lb)
                    for k in set(da) & set(db):
                        maxd = max(maxd, abs(da[k] - db[k]))
            over += bool(flip or d1 > tol)
    return dict(prompts=n, bitwise_equal=eq, token_flips=flips, flagged_flip_or_top1_gt_0125=over,
                max_abs_dtop1=maxd_top1, max_abs_dlogprob_top20=maxd, first_diffs=first)


def report(outs, arms):
    """outs: list of per-arm results (dict name -> list of summaries), arms: their names."""
    res = {}
    a0 = outs[0]
    if len(outs) >= 3:
        res[f"A_vs_A' ({arms[0]}#0 vs {arms[2]}#2)"] = cmp_runs(a0, outs[2])
    res[f"B_vs_A ({arms[1]}#1 vs {arms[0]}#0)"] = cmp_runs(a0, outs[1])
    if len(outs) >= 3:
        res[f"B_vs_A' ({arms[1]}#1 vs {arms[2]}#2)"] = cmp_runs(outs[2], outs[1])
    n = next(iter(res.values()))["prompts"]
    passed = all(v["bitwise_equal"] == v["prompts"] == n for v in res.values())
    return passed, res


if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    outs = [json.load(open(f))["pass0"] for f in files]
    print(json.dumps(report(outs, files)))
