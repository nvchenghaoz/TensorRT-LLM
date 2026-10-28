"""Per-call kernel list around one joint, one image and one txt attention call (from ab_check traces).

usage: python neighbors.py <dir with kernels_off.tsv and kernels_on.tsv>
Locates the first zeros_like fill of each size in the switch-off trace (joint [12528], image [12288],
txt [240] blocks), the attention kernel that follows it, and prints the same attention call
(same name, same ordinal) in both traces with the kernels from -10 to +8 around it.
"""
import json
import os
import re
import sys

ATT = re.compile(r"fmha|flash|attention|sdpa|fort_native", re.I)


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            cat, dur, grid, name = line.rstrip("\n").split("\t", 3)
            out.append((cat, float(dur), grid, name))
    return out


def window(ks, i, lo=10, hi=8):
    return [{"rel": j - i, "us": round(ks[j][1], 1), "grid": ks[j][2], "name": ks[j][3][:110]}
            for j in range(max(0, i - lo), min(len(ks), i + hi + 1))]


def main():
    d = sys.argv[1]
    off, on = load(os.path.join(d, "kernels_off.tsv")), load(os.path.join(d, "kernels_on.tsv"))
    res = {}
    for label, grid in (("joint_single_block", "[12528, 1, 1]"), ("image_mode_block", "[12288, 1, 1]"),
                        ("txt_mode_block", "[240, 1, 1]")):
        fills = [i for i, k in enumerate(off) if "FillFunctor<c10::BFloat16>" in k[3] and k[2] == grid]
        if not fills:
            res[label] = "no fill found"
            continue
        # use the 3rd occurrence to be past the first step's one-offs
        f = fills[min(2, len(fills) - 1)]
        a = next(i for i in range(f, len(off)) if ATT.search(off[i][3]))
        name = off[a][3]
        ordinal = sum(1 for i in range(a) if off[i][3] == name)
        same_on = [i for i, k in enumerate(on) if k[3] == name]
        b = same_on[ordinal] if ordinal < len(same_on) else None
        res[label] = {"attention_kernel": name[:110], "ordinal": ordinal,
                      "fill_rel_to_attention": f - a,
                      "off": window(off, a), "on": window(on, b) if b is not None else None}
        print(f"=== {label}: {name[:80]} (ordinal {ordinal})", flush=True)
        for tag, w in (("off", res[label]["off"]), ("on", res[label]["on"] or [])):
            for e in w:
                print(f"  {tag} {e['rel']:+3d} {e['us']:8.1f}us {e['grid']:>16} {e['name'][:95]}")
    print(json.dumps({"passed": True, "neighbors": {k: (v if isinstance(v, str) else
                                                        {"attention_kernel": v["attention_kernel"],
                                                         "fill_rel_to_attention": v["fill_rel_to_attention"]})
                                                    for k, v in res.items()}}))


if __name__ == "__main__":
    main()
