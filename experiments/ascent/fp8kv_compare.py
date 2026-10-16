#!/usr/bin/env python3
"""Compare greedy outputs of the BF16-KV arm and the FP8-KV arm of fp8kv_diag."""
import json
import sys


def main():
    a = json.load(open(sys.argv[1]))  # bf16 reference arm
    b = json.load(open(sys.argv[2]))  # fp8 arm
    per = []
    for i, (x, y) in enumerate(zip(a["compare"], b["compare"])):
        ta, tb = x["tokens"], y["tokens"]
        n = min(len(ta), len(tb))
        first_div = next((j for j in range(n) if ta[j] != tb[j]), None)
        upto = n if first_div is None else first_div
        la, lb = x["logprobs"], y["logprobs"]
        dl = [abs(la[j] - lb[j]) for j in range(upto)
              if la and lb and la[j] is not None and lb[j] is not None]
        # logprob of the BF16 token at the divergence point (near-tie check)
        div_info = None
        if first_div is not None and la and lb:
            div_info = {
                "pos": first_div,
                "bf16_tok_lp": la[first_div],
                "fp8_tok_lp": lb[first_div],
            }
        per.append({
            "prompt": i,
            "n": n,
            "first_divergence": first_div,
            "matched_prefix": upto,
            "max_abs_dlogprob_before_div": max(dl) if dl else 0.0,
            "mean_abs_dlogprob_before_div": sum(dl) / len(dl) if dl else 0.0,
            "div": div_info,
        })
    n_exact = sum(1 for p in per if p["first_divergence"] is None)
    out = {
        "n_prompts": len(per),
        "n_exact_match": n_exact,
        "per_prompt": per,
        "fp8_v_amax_max": b.get("v_amax_max"),
        "fp8_k_amax_max": b.get("k_amax_max"),
        "bf16_v_amax_max": a.get("v_amax_max"),
        "fp8_repeat_identical_tokens": b.get("repeat_identical_tokens"),
        "fp8_repeat_identical_logprobs": b.get("repeat_identical_logprobs"),
        "bf16_repeat_identical_tokens": a.get("repeat_identical_tokens"),
        "fp8_bench_request_cached": b.get("bench_request_cached"),
        "fp8_bench_request_graph_replays": b.get("bench_request_graph_replays"),
        "bf16_bench_request_graph_replays": a.get("bench_request_graph_replays"),
        "fp8_cuda_graphs": b.get("cuda_graphs"),
        "fp8_v_amax_per_layer": {k: v for k, v in (b.get("v_amax") or {}).items() if v},
        "fp8_k_amax_postnorm_per_layer": {
            k: v for k, v in (b.get("k_amax_postnorm") or {}).items() if v},
    }
    out["passed"] = bool(out["fp8_repeat_identical_tokens"]
                         and out["fp8_repeat_identical_logprobs"]
                         and (out["fp8_bench_request_graph_replays"] or 0) > 0
                         and (out["fp8_v_amax_max"] or 0) < 448.0
                         and (out["fp8_k_amax_max"] or 0) < 448.0)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
