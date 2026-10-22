# Compare two twopass.py outputs (base arm vs candidate arm of one run_gpu_pair).
# Usage: python compare_twopass.py <base twopass_tokens.json> <candidate twopass_tokens.json>
# Per pass and concurrency: branch-vs-base diverging prompts, first divergent index, max |dlogprob| on the common
# prefix; plus pass2-vs-pass1 counts per arm. Last line: JSON.
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def cmp_runs(a, b):
    out = dict()
    for k in sorted(a):
        ia, ib = a[k]["ids"], b[k]["ids"]
        la, lb = a[k]["lp"], b[k]["lp"]
        first = None
        for t, (x, y) in enumerate(zip(ia, ib)):
            if x != y:
                first = t
                break
        n = len(ia) if first is None else first
        dl = [abs(la[t] - lb[t]) for t in range(n) if la[t] is not None and lb[t] is not None]
        out[k] = dict(diverge=first is not None, first=first, max_dlp=max(dl) if dl else 0.0,
                      lp_bitwise=(la[:n] == lb[:n]))
    return out


base, cand = load(sys.argv[1]), load(sys.argv[2])
res = dict()
for p in ("pass1", "pass2"):
    c = cmp_runs(base[p], cand[p])
    per_conc = dict()
    for conc in ("c8", "c16", "c32"):
        ks = [k for k in c if k.startswith(conc + "-")]
        dv = sorted(k for k in ks if c[k]["diverge"])
        per_conc[conc] = dict(n=len(ks), diverging=len(dv), prompts=[(k, c[k]["first"]) for k in dv],
                              max_dlp_common_prefix=max(c[k]["max_dlp"] for k in ks),
                              bitwise_prompts=sum(1 for k in ks if not c[k]["diverge"] and c[k]["lp_bitwise"]))
    res[p] = per_conc
for name, arm in (("base", base), ("cand", cand)):
    c = cmp_runs(arm["pass1"], arm["pass2"])
    res[name + "_pass2_vs_pass1_diverging"] = sorted(k for k in c if c[k]["diverge"])
for p in ("pass1", "pass2"):
    for conc, d in res[p].items():
        print(p, conc, "diverging %d/%d" % (d["diverging"], d["n"]), "bitwise", d["bitwise_prompts"],
              "max|dlp| common prefix %.4f" % d["max_dlp_common_prefix"], d["prompts"], flush=True)
print(json.dumps(dict(passed=True, compare=res)))
