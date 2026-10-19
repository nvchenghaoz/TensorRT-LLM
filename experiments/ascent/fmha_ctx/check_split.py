#!/usr/bin/env python3
"""Correctness for branch 31 (idea 76): the last q-tile KV split of the sm_120 e4m3 tiled context FMHA vs main.

Builds experiments/ascent/fmha_ctx/bench.cu against main's cpp/kernels/fmha_v2/src (git archive of BASE: no split
header, so the generated kernel alone) and against this tree's (with the split), runs both on the same deterministic
inputs with the timed loop's protocol (warmup 10 + 50 calls; before every call O is NaN-poisoned, the split workspace
too, and L2 is read-flushed; every call's output hash is compared with call 1; Q / KV pool hashes checked unchanged)
with --split -1 (the TRT-LLM dispatch), dumps the last output and compares:
  - cases the dispatch does not split (pieces 0): the whole output bitwise equal to main;
  - split cases: rows of q-tiles 0 .. Q-2 of every sequence bitwise equal to main; the rows of the last q-tile (all
    24 heads) within the reordered tolerance vs an FP32 torch reference (dequantized e4m3 Q/K/V, FP32 softmax):
    max-abs and mean-abs error <= 1.25 x main's + 1 BF16 ulp of the output magnitude, no NaN/Inf; plus the same
    tolerance on whole sequences for heads 0/11/23 (first and last sequence).
Last stdout line: {"passed": ..., ...}.

Branch 99 (idea 165, kMaxBatch 2 -> 3): BASE is main's explicit sha (66c3ff64, which carries the split at b <= 2).
Each arm's bench is built from its own tree with -DBENCH_MAX_BATCH = that tree's TRT-LLM dispatch constant
(last_tile_split_sm120/fmha_v2_e4m3_last_tile_split_sm120.cu kMaxBatch; override with FMHA_BASE_NVCC /
FMHA_CAND_NVCC), so the bench dispatch mirrors each tree. Cases where both arms dispatch the same pieces (c1/c2 split
in both, c4 / one unsplit in both) must be bitwise equal to main as a whole; a case only the candidate splits (c3)
keeps rows of q-tiles 0..Q-2 bitwise and the last q-tile within the reordered tolerance. FMHA_CHECK_SPLIT (default -1,
the dispatch), FMHA_CHECK_CASES, FMHA_CHECK_WARMUP, FMHA_CHECK_ITERS override the run.
"""
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import torch

BASE = sys.argv[1] if len(sys.argv) > 1 else "66c3ff64b97af2aafa395ed53e3d74ca40446262"
CASES = os.environ.get("FMHA_CHECK_CASES", "c1,c2,c3,c4,ragged,empty,one,c1_2000").split(",")
# Overridable (defaults unchanged): FMHA_CHECK_WARMUP=10 FMHA_CHECK_ITERS=50
WARMUP = int(os.environ.get("FMHA_CHECK_WARMUP", "10"))
ITERS = int(os.environ.get("FMHA_CHECK_ITERS", "50"))
SPLIT = os.environ.get("FMHA_CHECK_SPLIT", "-1")
DISPATCH_CU = ("cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/last_tile_split_sm120/"
               "fmha_v2_e4m3_last_tile_split_sm120.cu")
W = "/tmp/fmha_ctx_check_split"
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.getcwd()
H, H_KV, DIM, TPB, STEP = 24, 4, 256, 32, 64


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def parse(stdout):
    res, meta, split = {}, {}, {}
    for line in stdout.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "RESULT":
            res[t[1]] = dict(b=int(t[2]), ctas=int(t[3]), median_ms=float(t[4]), p90_ms=float(t[6]),
                             repeat_mismatches=int(t[9]), calls=int(t[10]), nonfinite=int(t[11]),
                             inputs_unchanged=bool(int(t[12])), out_hash=t[13])
        elif t[0] == "META":
            b = int(t[2])
            meta[t[1]] = dict(b=b, max_blocks=int(t[3]), q_lens=[int(x) for x in t[4:4 + b]],
                              past_lens=[int(x) for x in t[4 + b:4 + 2 * b]])
        elif t[0] == "SPLIT":
            split[t[1]] = dict(pieces=int(t[3]), q_loop_last=int(t[5]))
    return res, meta, split


def max_batch_of(cu_path):
    """The TRT-LLM dispatch's kMaxBatch in a tree (None if the split file is absent)."""
    if not os.path.exists(cu_path):
        return None
    m = re.search(r"constexpr int kMaxBatch = (\d+);", open(cu_path).read())
    return int(m.group(1)) if m else None


def load_o(d, case):
    o = torch.from_numpy(np.fromfile(f"{d}/{case}_o.bin", dtype=np.uint16).astype(np.int16)).cuda()
    return o.view(torch.bfloat16).float().view(-1, H, DIM)


def reference(d, case, m, i, heads, rows):
    """FP32 reference O[rows, heads] of sequence i (rows: local q positions)."""
    q = torch.from_numpy(np.fromfile(f"{d}/{case}_q.bin", dtype=np.uint8)).cuda()
    q = q.view(torch.float8_e4m3fn).float().view(-1, H, DIM)
    pool = torch.from_numpy(np.fromfile(f"{d}/{case}_pool.bin", dtype=np.uint8)).cuda().view(-1, H_KV, TPB, DIM)
    offs = torch.from_numpy(np.fromfile(f"{d}/{case}_offs.bin", dtype=np.int32)).view(m["b"], 2, m["max_blocks"])
    cu = np.concatenate([[0], np.cumsum(m["q_lens"])])
    ql, past = m["q_lens"][i], m["past_lens"][i]
    kvl = past + ql
    nblk = (kvl + TPB - 1) // TPB
    kb = offs[i, 0, :nblk].long().cuda()
    vb = offs[i, 1, :nblk].long().cuda()
    k = pool[kb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(H_KV, -1, DIM)[:, :kvl]
    v = pool[vb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(H_KV, -1, DIM)[:, :kvl]
    rows_t = torch.tensor(rows, device="cuda", dtype=torch.long)
    qi = q[cu[i]:cu[i + 1]][rows_t]
    qpos = rows_t[:, None] + past
    kpos = torch.arange(kvl, device="cuda")[None, :]
    mask = kpos > qpos
    out = torch.empty(len(rows), len(heads), DIM, device="cuda")
    for j, hq in enumerate(heads):
        g = hq // (H // H_KV)
        s = (qi[:, hq] @ k[g].T) * (1.0 / 16.0)
        s.masked_fill_(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, j] = p @ v[g]
    return out, cu


def errors(o, ref):
    e = (o - ref).abs()
    finite = bool(torch.isfinite(o).all())
    return dict(max_abs=float(e.max()), mean_abs=float(e.mean()), n=int(e.numel()), finite=finite,
                ref_max_abs=float(ref.abs().max()))


def bf16_ulp(x):
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 0.0


def within(err_c, err_m):
    ulp = bf16_ulp(err_m["ref_max_abs"])
    return (err_c["finite"] and err_c["max_abs"] <= 1.25 * err_m["max_abs"] + ulp
            and err_c["mean_abs"] <= 1.25 * err_m["mean_abs"] + ulp), ulp


def main():
    os.makedirs(f"{W}/base", exist_ok=True)
    sh(f"rm -rf {W}/base/cpp")
    sh(f"git archive {BASE} cpp/kernels/fmha_v2/src | tar -x -C {W}/base")
    sh(f"git archive {BASE} {DISPATCH_CU} | tar -x -C {W}/base || true")
    mb_base = max_batch_of(f"{W}/base/{DISPATCH_CU}")
    mb_cand = max_batch_of(f"{TREE}/{DISPATCH_CU}")
    nv_base = os.environ.get("FMHA_BASE_NVCC", f"-DBENCH_MAX_BATCH={mb_base}" if mb_base else "")
    nv_cand = os.environ.get("FMHA_CAND_NVCC", f"-DBENCH_MAX_BATCH={mb_cand}" if mb_cand else "")
    print(f"[check] base {BASE} kMaxBatch {mb_base} nvcc '{nv_base}'; cand {TREE} kMaxBatch {mb_cand} "
          f"nvcc '{nv_cand}'", flush=True)
    sh(f"bash {HERE}/build.sh {W}/base/cpp/kernels/fmha_v2/src {W}/bench_base {nv_base}")
    sh(f"bash {HERE}/build.sh {TREE}/cpp/kernels/fmha_v2/src {W}/bench_cand {nv_cand}")
    runs = {}
    for arm in ("base", "cand"):
        d = f"{W}/dump_{arm}"
        os.makedirs(d, exist_ok=True)
        r = subprocess.run([f"{W}/bench_{arm}", "--cases", ",".join(CASES), "--warmup", str(WARMUP), "--iters",
                            str(ITERS), "--dump", d, "--split", SPLIT], capture_output=True, text=True)
        print(f"===== {arm} (rc {r.returncode}) =====\n{r.stdout}{r.stderr}", flush=True)
        runs[arm] = (r.returncode,) + parse(r.stdout)

    summary = {"base_commit": BASE, "base_max_batch": mb_base, "cand_max_batch": mb_cand, "base_nvcc": nv_base,
               "cand_nvcc": nv_cand, "split_mode": SPLIT, "warmup": WARMUP, "iters": ITERS, "cases": {}}
    passed = runs["base"][0] == 0 and runs["cand"][0] == 0
    db, dc = f"{W}/dump_base", f"{W}/dump_cand"
    for case in CASES:
        rb, rc = runs["base"][1][case], runs["cand"][1][case]
        m = runs["cand"][2][case]
        sp = runs["cand"][3].get(case, dict(pieces=0, q_loop_last=-1))
        sp_base = runs["base"][3].get(case, dict(pieces=0, q_loop_last=-1))
        same_inputs = all(
            np.array_equal(np.fromfile(f"{db}/{case}_{n}.bin", dtype=np.uint8),
                           np.fromfile(f"{dc}/{case}_{n}.bin", dtype=np.uint8)) for n in ("q", "pool", "offs"))
        ob, oc = load_o(db, case), load_o(dc, case)
        raw_b = np.fromfile(f"{db}/{case}_o.bin", dtype=np.uint16).reshape(-1, H, DIM)
        raw_c = np.fromfile(f"{dc}/{case}_o.bin", dtype=np.uint16).reshape(-1, H, DIM)
        cu = np.concatenate([[0], np.cumsum(m["q_lens"])])
        entry = {"q_lens": m["q_lens"], "past_lens": m["past_lens"], "pieces": sp["pieces"],
                 "pieces_cand": sp["pieces"], "pieces_base": sp_base["pieces"],
                 "same_inputs": same_inputs, "calls": rc["calls"],
                 "cand_repeat_mismatches_vs_call1": rc["repeat_mismatches"],
                 "main_repeat_mismatches_vs_call1": rb["repeat_mismatches"], "cand_nonfinite": rc["nonfinite"],
                 "inputs_unchanged": rc["inputs_unchanged"], "out_hash_main": rb["out_hash"],
                 "out_hash_cand": rc["out_hash"]}
        ok = (same_inputs and rc["repeat_mismatches"] == 0 and rb["repeat_mismatches"] == 0 and rc["nonfinite"] == 0
              and rc["inputs_unchanged"])
        if sp["pieces"] == sp_base["pieces"]:
            # Same dispatch in both arms (unsplit in both, or the same piece count): the whole output bitwise.
            diff = int((raw_b != raw_c).sum())
            entry["bitwise_equal_to_main"] = diff == 0
            entry["diff_elems"] = diff
            ok = ok and diff == 0
        else:
            last0 = sp["q_loop_last"] * STEP
            # Rows of q-tiles 0 .. Q-2 (every sequence): bitwise.
            keep = np.ones(raw_b.shape[0], dtype=bool)
            for i in range(m["b"]):
                lo = cu[i] + min(last0, m["q_lens"][i])
                keep[lo:cu[i + 1]] = False
            diff_keep = int((raw_b[keep] != raw_c[keep]).sum())
            diff_last = int((raw_b[~keep] != raw_c[~keep]).sum())
            entry["rows_tiles_0_to_Q2"] = int(keep.sum())
            entry["rows_tiles_0_to_Q2_diff_elems"] = diff_keep
            entry["rows_last_tile"] = int((~keep).sum())
            entry["rows_last_tile_diff_elems_vs_main"] = diff_last
            ok = ok and diff_keep == 0
            # Last q-tile rows, all heads: tolerance vs the FP32 reference.
            heads = list(range(H))
            em_all, ec_all = [], []
            for i in range(m["b"]):
                rows = list(range(min(last0, m["q_lens"][i]), m["q_lens"][i]))
                if not rows:
                    continue
                ref, _ = reference(dc, case, m, i, heads, rows)
                sl = slice(cu[i] + rows[0], cu[i] + rows[-1] + 1)
                em_all.append((ob[sl] - ref).abs().flatten())
                ec_all.append((oc[sl] - ref).abs().flatten())
                ref_max = float(ref.abs().max())
                del ref
            em = torch.cat(em_all)
            ec = torch.cat(ec_all)
            err_m = dict(max_abs=float(em.max()), mean_abs=float(em.mean()), n=int(em.numel()), finite=True,
                         ref_max_abs=ref_max)
            err_c = dict(max_abs=float(ec.max()), mean_abs=float(ec.mean()), n=int(ec.numel()),
                         finite=bool(torch.isfinite(oc).all()), ref_max_abs=ref_max)
            tol_ok, ulp = within(err_c, err_m)
            entry["last_tile_fp32_ref_main"] = err_m
            entry["last_tile_fp32_ref_cand"] = err_c
            entry["last_tile_tolerance_ok"] = tol_ok
            entry["bf16_ulp"] = ulp
            ok = ok and tol_ok
            # Whole sequences (first and last), heads 0/11/23.
            wm, wc = [], []
            for i in sorted({0, m["b"] - 1}):
                rows = list(range(m["q_lens"][i]))
                ref, _ = reference(dc, case, m, i, [0, 11, 23], rows)
                sl = slice(cu[i], cu[i + 1])
                wm.append((ob[sl][:, [0, 11, 23]] - ref).abs().flatten())
                wc.append((oc[sl][:, [0, 11, 23]] - ref).abs().flatten())
                ref_max_w = float(ref.abs().max())
                del ref
            wm, wc = torch.cat(wm), torch.cat(wc)
            werr_m = dict(max_abs=float(wm.max()), mean_abs=float(wm.mean()), n=int(wm.numel()), finite=True,
                          ref_max_abs=ref_max_w)
            werr_c = dict(max_abs=float(wc.max()), mean_abs=float(wc.mean()), n=int(wc.numel()),
                          finite=bool(torch.isfinite(oc).all()), ref_max_abs=ref_max_w)
            wok, _ = within(werr_c, werr_m)
            entry["whole_fp32_ref_main_h0_11_23"] = werr_m
            entry["whole_fp32_ref_cand_h0_11_23"] = werr_c
            entry["whole_tolerance_ok"] = wok
            ok = ok and wok
        # The dispatch each arm's bench mirrors, as each tree's TRT-LLM constant: b <= base / cand kMaxBatch split.
        if SPLIT == "-1" and mb_base and mb_cand:
            exp_base = m["b"] <= mb_base
            exp_cand = m["b"] <= mb_cand
            entry["dispatch_matches_tree_constant"] = ((sp_base["pieces"] > 0) <= exp_base
                                                       and (sp["pieces"] > 0) <= exp_cand)
            ok = ok and entry["dispatch_matches_tree_constant"]
        # Idea 165's gate: c3 (3 x 2048 over 32768) unsplit on main (kMaxBatch 2), 5 pieces with kMaxBatch 3.
        if case == "c3" and SPLIT == "-1" and mb_base == 2 and mb_cand == 3:
            entry["c3_pieces_gate"] = sp["pieces"] == 5 and sp_base["pieces"] == 0
            ok = ok and entry["c3_pieces_gate"]
        entry["ok"] = bool(ok)
        passed = passed and ok
        summary["cases"][case] = entry
        print(case, json.dumps(entry), flush=True)
    summary["passed"] = bool(passed)
    summary["class"] = "reordered"
    summary["compared_against"] = (
        "main's own kernel (fmha_v2/src at the base commit, same template instantiation and inputs): bitwise for "
        "rows of q-tiles 0..Q-2 and for cases not split; an FP32 torch reference (dequantized e4m3 Q/K/V, FP32 "
        "softmax) for the last q-tile rows, tolerance 1.25 x main's max/mean-abs error + 1 BF16 ulp")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
