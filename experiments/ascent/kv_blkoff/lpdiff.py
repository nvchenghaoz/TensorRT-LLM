"""Per-prompt id/logprob differences between two twopass_tokens.json files (arm A vs arm B).
Usage: python lpdiff.py <A.json> <B.json>"""
import json
import sys


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = dict()
    for p in ("pass1", "pass2"):
        rows = dict()
        for k in sorted(a[p]):
            ia, ib = a[p][k]["ids"], b[p][k]["ids"]
            la, lb = a[p][k]["lp"], b[p][k]["lp"]
            first_id = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
            d = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
            first_lp = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
            if first_id is not None or first_lp is not None:
                rows[k] = dict(first_id_div=first_id, first_lp_div=first_lp, max_abs_dlp=max(d) if d else None,
                               n_lp_diff=sum(1 for x, y in zip(la, lb) if x != y))
        out[p] = dict(n_prompts_differ=len(rows), rows=rows,
                      max_abs_dlp=max([r["max_abs_dlp"] or 0 for r in rows.values()] or [0]))
    print(json.dumps(dict(passed=True, **out)))


main()
