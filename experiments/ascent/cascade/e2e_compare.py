#!/usr/bin/env python3
# Compares two e2e_predictor.py arms (base, branch): per concurrency group, each arm's pass2-vs-pass1 token-diverging
# prompts, and branch-vs-base diverging prompts per pass with the first divergent token index and the max |dlogprob|
# on the common token prefix (the frozen check's tolerance is 3.244; it teacher-forces past near-ties, so raw token
# divergence over-counts). Usage: python e2e_compare.py <base e2e_pred.json> <branch e2e_pred.json>
import json
import sys


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmp(pa, pb):
    out = {}
    for k in sorted(pa):
        ia, ib = pa[k]["ids"], pb[k]["ids"]
        fd = first_div(ia, ib)
        n = len(ia) if fd is None else fd
        dl = [abs(x - y) for x, y in zip(pa[k]["lp"][:n + 1], pb[k]["lp"][:n + 1]) if x is not None and y is not None]
        out[k] = dict(first_divergent_token=fd, max_abs_dlogprob_common=max(dl) if dl else 0.0)
    return out


def group_of(k):
    return k.split("-")[0]


def summarize(per):
    groups = {}
    for k, v in per.items():
        g = groups.setdefault(group_of(k), dict(prompts=0, diverging=[], max_abs_dlogprob=0.0))
        g["prompts"] += 1
        if v["first_divergent_token"] is not None:
            g["diverging"].append([k, v["first_divergent_token"]])
        g["max_abs_dlogprob"] = max(g["max_abs_dlogprob"], v["max_abs_dlogprob_common"])
    for g in groups.values():
        g["n_diverging"] = len(g["diverging"])
    return groups


def main():
    base = json.load(open(sys.argv[1]))
    br = json.load(open(sys.argv[2]))
    res = {}
    for name, arm in (("base", base), ("branch", br)):
        res[f"{name}_pass2_vs_pass1"] = summarize(cmp(arm["pass1"], arm["pass2"]))
    for p in ("pass1", "pass2"):
        res[f"branch_vs_base_{p}"] = summarize(cmp(base[p], br[p]))
    tot = lambda s: sum(g["n_diverging"] for g in s.values())  # noqa: E731
    res["totals"] = {k: tot(v) for k, v in res.items() if isinstance(v, dict)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
