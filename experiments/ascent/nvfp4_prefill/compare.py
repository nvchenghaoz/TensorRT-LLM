#!/usr/bin/env python3
"""Compare arm A (kill switch, = main) and arm C (NVFP4 prefill copy)."""
import json
import sys


def main():
    a = json.load(open(sys.argv[1]))
    c = json.load(open(sys.argv[2]))
    ia = {i["id"]: i for i in a["items"]}
    ic = {i["id"]: i for i in c["items"]}
    lost, gained, flips, dl_all, divs = [], [], [], [], []
    n_ident = 0
    for k, x in ia.items():
        y = ic[k]
        if x["ok"] and not y["ok"]:
            lost.append(k)
        if y["ok"] and not x["ok"]:
            gained.append(k)
        ta, tc = x["tokens"], y["tokens"]
        n = min(len(ta), len(tc))
        fd = next((j for j in range(n) if ta[j] != tc[j]), None)
        if fd is None and len(ta) == len(tc):
            n_ident += 1
        else:
            flips.append({"id": k, "first_div": fd if fd is not None else n,
                          "a_ok": x["ok"], "c_ok": y["ok"]})
        upto = n if fd is None else fd
        la, lc = x["logprobs"], y["logprobs"]
        for j in range(min(upto, len(la), len(lc))):
            if la[j] is not None and lc[j] is not None:
                dl_all.append(abs(la[j] - lc[j]))
        if fd is not None:
            divs.append(fd)
    a_ret, a_sh = a["retrieval"], a["short"]
    c_ret, c_sh = c["retrieval"], c["short"]
    lost_ret = [k for k in lost if k.startswith("r")]
    valid = a_ret >= 62 and a_sh >= 15
    go = valid and not lost_ret and len(lost) <= 1
    out = {
        "A": {"retrieval": a_ret, "short": a_sh, "ready_s": a["llm_ready_s"],
              "nvfp4_copies": a["nvfp4_copies"], "nvfp4_calls": a["nvfp4_calls"]},
        "C": {"retrieval": c_ret, "short": c_sh, "ready_s": c["llm_ready_s"],
              "nvfp4_copies": c["nvfp4_copies"], "nvfp4_calls": c["nvfp4_calls"],
              "nvfp4_copy_bytes": c["nvfp4_copy_bytes"]},
        "baseline_valid": valid,
        "lost": lost, "lost_retrieval": lost_ret, "gained": gained,
        "n_items": len(ia), "n_identical_sequences": n_ident,
        "n_token_flips": len(flips),
        "flips": flips,
        "mean_abs_dlogprob_before_div": (sum(dl_all) / len(dl_all)
                                         if dl_all else 0.0),
        "max_abs_dlogprob_before_div": max(dl_all) if dl_all else 0.0,
        "n_tokens_compared": len(dl_all),
        "weight_relrms_C": c.get("weight_relrms"),
        "GO": go,
        "passed": go,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
