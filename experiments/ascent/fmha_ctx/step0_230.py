#!/usr/bin/env python3
"""STEP 0 of idea 230 (branch 132): base / arm E (FMHA_SM120_EXP2) / arm H (+FMHA_SM120_FP16_SOFTMAX) of the SM120
e4m3 tiled context FMHA (fmha_v2_flash_attention_e4m3_fp32_64_128_S_q_paged_kv_256_causal_output_bf16_sm120_kernel_nl_tiled
plus the last-q-tile split piece / merge kernels), all three built from THIS tree with -DBENCH_MAX_BATCH=3 (base = the
macros at their default 0), timed with experiments/ascent/fmha_ctx/bench.cu --split -1.

1. Builds: nvcc with build.sh's flags, full ptxas -v kept (regs / stack / spills per kernel); also the BASE tree's
   stock kernel (git archive BASE) for the error reference and the macro-0 identity check.
2. SASS census per build (cuobjdump): per kernel and per outermost backward-branch loop that holds QMMAs (static counts
   of FP32, MUFU, HFMA2/HADD2/HMUL2, F2FP, QMMA, total).
3. Timing: palindrome of positions (base E H H E base) x 2 = 12 processes, each --cases c1,c2,c3,c4 --warmup 10
   --iters 100 (each call: NaN-poisoned O and split workspace, 384 MiB L2 flush, own event pair). Clocks / temperature
   / throttle reasons before every position. Per arm: median of the position medians (and of the p10 / p90). A/A
   floor = (max - min) / median of the base positions' medians, per case.
4. Errors vs an FP32 torch reference (dequantized e4m3 Q/K/V, FP32 softmax, every row and head of every sequence):
   max-abs / mean-abs for the BASE tree's stock kernel (its own dump files), base (this tree, macros 0), E and H, on
   c1, c3, ragged, one, c1_out, c3_out (outliers: every 4th Q head x4, every 509th KV token row x8), split on.
   Bar: <= 1.5 x the stock kernel's error. Same inputs checked byte for byte across the dumps.
5. Repeated execution (the error runs use the timed loop's warmup 10 + 100 calls and reset points): every call's
   output hash equals call 1, nonfinite 0, Q / KV pool unchanged, per arm and case.
6. SASS identity: this tree's base build vs the BASE tree's stock build (main / piece / merge kernels), and
   experiments/ascent/fmha_ctx/sass_same.cu (BF16 / FP16 tiled, e4m3 non-tiled) with the macros on vs BASE.
7. Emulated P e4m3 code differences (torch, last q-tile of c1, against the row max over the whole row).
STOP rule: an arm passes if it beats base by more than 2 x the A/A floor at c1 AND c3 and is within the error bar;
H also needs >= 3% at c1. Last stdout line: a JSON summary; the full JSON goes to $AUTO_DEV_ARTIFACTS/step0_230.json.
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

BASE = sys.argv[1] if len(sys.argv) > 1 else "66cff73e70c6452eaa12f3261f8cddf0e976668a"
W = "/tmp/fmha_230"
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.getcwd()
ARMS = {"base": [], "E": ["-DFMHA_SM120_EXP2=1"], "H": ["-DFMHA_SM120_EXP2=1", "-DFMHA_SM120_FP16_SOFTMAX=1"]}
TIME_CASES = os.environ.get("S0_TIME_CASES", "c1,c2,c3,c4")
ERR_CASES = os.environ.get("S0_ERR_CASES", "c1,c3,ragged,one,c1_out,c3_out").split(",")
POSITIONS = ["base", "E", "H", "H", "E", "base"] * int(os.environ.get("S0_ROUNDS", "2"))
WARMUP, ITERS = 10, 100
H_, H_KV, DIM, TPB = 24, 4, 256, 32
ART = os.environ.get("AUTO_DEV_ARTIFACTS", W)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(*a):
    print(*a, flush=True)


def sh(cmd, **kw):
    log("+", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, capture_output=True, text=True, **kw)


def arch_flags():
    r = subprocess.run("nvcc --list-gpu-arch", shell=True, capture_output=True, text=True)
    if "compute_120f" in r.stdout:
        return ["-gencode", "arch=compute_120f,code=sm_120f"]
    return ["-gencode", "arch=compute_120a,code=sm_120a"]


def nvcc_build(src_dir, out, extra):
    cmd = (["nvcc", "-O3", "-std=c++17", "--expt-extended-lambda", "--expt-relaxed-constexpr",
            "-DUSE_DEMO_BERT_PARAMS=1", "-DGENERATE_CUBIN=1"] + arch_flags()
           + ["-Xptxas", "-v", f"-I{src_dir}", "-DBENCH_MAX_BATCH=3"] + extra
           + [f"{HERE}/bench.cu", "-o", out, "-lcuda"])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"build failed: {out}")
    log(f"[build] {out} in {time.time() - t0:.1f}s")
    return parse_ptxas(r.stderr + r.stdout)


def kname(fn):
    if "fmha_v2_flash_attention" in fn:
        return "main"
    if "last_tile_piece" in fn:
        return "piece"
    if "last_tile_merge" in fn:
        return "merge"
    return None


def parse_ptxas(text):
    res, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            cur = kname(m.group(1))
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            res.setdefault(cur, {}).update(stack=int(m.group(1)), spill_stores=int(m.group(2)),
                                           spill_loads=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            res.setdefault(cur, {})["regs"] = int(m.group(1))
            cur = None
    return res


def sass_functions(binary):
    out = subprocess.run(["cuobjdump", "-sass", binary], capture_output=True, text=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if m and cur is not None:
            funcs[cur].append((int(m.group(1), 16), m.group(2)))
    return funcs


FP32_OPS = {"FFMA", "FMUL", "FADD", "FMNMX", "FSETP", "FSEL", "FSET", "FFMA2", "FADD2", "FMUL2", "FMNMX3", "FSWZADD"}
HALF_OPS = {"HFMA2", "HADD2", "HMUL2", "HMNMX2", "HSETP2", "HSET2"}


def opcode(text):
    t = text.split()
    if t and t[0].startswith("@"):
        t = t[1:]
    return t[0] if t else ""


def census(instrs):
    c = dict(total=0, fp32=0, mufu=0, mufu_ex2=0, half=0, f2fp=0, qmma=0, nop=0, shfl=0)
    for _, txt in instrs:
        op = opcode(txt)
        base = op.split(".")[0]
        if base == "NOP":
            c["nop"] += 1
            continue
        c["total"] += 1
        if base in FP32_OPS:
            c["fp32"] += 1
        elif base == "MUFU":
            c["mufu"] += 1
            c["mufu_ex2"] += ".EX2" in op
        elif base in HALF_OPS:
            c["half"] += 1
        elif base in ("F2FP", "F2F"):
            c["f2fp"] += 1
        elif base in ("QMMA", "OMMA", "HMMA"):
            c["qmma"] += 1
        elif base == "SHFL":
            c["shfl"] += 1
    return c


def loops(instrs):
    """Outermost backward-branch loops holding QMMAs: [(start, end, census)]."""
    spans = []
    for addr, txt in instrs:
        if opcode(txt).split(".")[0] == "BRA":
            m = re.search(r"0x([0-9a-f]+)", txt)
            if m and int(m.group(1), 16) < addr:
                spans.append((int(m.group(1), 16), addr))
    out = []
    for s, e in spans:
        if any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for s2, e2 in spans):
            continue
        body = [(a, t) for a, t in instrs if s <= a <= e]
        c = census(body)
        if c["qmma"] > 0:
            out.append(dict(start=hex(s), end=hex(e), **c))
    return out


def sass_report(binary):
    rep = {}
    for fn, ins in sass_functions(binary).items():
        k = kname(fn)
        if k:
            rep[k] = dict(whole=census(ins), loops=loops(ins))
    return rep


def sass_text(binary):
    return {kname(fn): [t for _, t in ins] for fn, ins in sass_functions(binary).items() if kname(fn)}


def gpu_state():
    r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                        "clocks_throttle_reasons.active", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def parse_bench(stdout):
    res = {}
    meta = {}
    for line in stdout.splitlines():
        t = line.split()
        if t and t[0] == "RESULT":
            res[t[1]] = dict(b=int(t[2]), median=float(t[4]), p10=float(t[5]), p90=float(t[6]), min=float(t[7]),
                             mismatches=int(t[9]), calls=int(t[10]), nonfinite=int(t[11]), inputs_ok=bool(int(t[12])),
                             hash=t[13])
        elif t and t[0] == "META":
            b = int(t[2])
            meta[t[1]] = dict(b=b, max_blocks=int(t[3]), q_lens=[int(x) for x in t[4:4 + b]],
                              past_lens=[int(x) for x in t[4 + b:4 + 2 * b]])
    return res, meta


def run_bench(binary, cases, dump=None):
    cmd = [binary, "--cases", cases, "--warmup", str(WARMUP), "--iters", str(ITERS), "--split", "-1"]
    if dump:
        os.makedirs(dump, exist_ok=True)
        cmd += ["--dump", dump]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        log(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"bench failed rc {r.returncode}")
    return r.returncode, r.stdout


def load_o(d, case):
    o = torch.from_numpy(np.fromfile(f"{d}/{case}_o.bin", dtype=np.uint16).astype(np.int16)).cuda()
    return o.view(torch.bfloat16).float().view(-1, H_, DIM)


def ref_iter(d, case, m):
    """Yields (seq i, head, rows slice, FP32 ref O [q_len, DIM], raw scores s [q_len, kv] scaled) per head."""
    q = torch.from_numpy(np.fromfile(f"{d}/{case}_q.bin", dtype=np.uint8)).cuda()
    q = q.view(torch.float8_e4m3fn).float().view(-1, H_, DIM)
    pool = torch.from_numpy(np.fromfile(f"{d}/{case}_pool.bin", dtype=np.uint8)).cuda().view(-1, H_KV, TPB, DIM)
    offs = torch.from_numpy(np.fromfile(f"{d}/{case}_offs.bin", dtype=np.int32)).view(m["b"], 2, m["max_blocks"])
    cu = np.concatenate([[0], np.cumsum(m["q_lens"])])
    for i in range(m["b"]):
        ql, past = m["q_lens"][i], m["past_lens"][i]
        kvl = past + ql
        nblk = (kvl + TPB - 1) // TPB
        kb = offs[i, 0, :nblk].long().cuda()
        vb = offs[i, 1, :nblk].long().cuda()
        k = pool[kb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(H_KV, -1, DIM)[:, :kvl]
        v = pool[vb].view(torch.float8_e4m3fn).float().permute(1, 0, 2, 3).reshape(H_KV, -1, DIM)[:, :kvl]
        qi = q[cu[i]:cu[i + 1]]
        qpos = torch.arange(ql, device="cuda")[:, None] + past
        kpos = torch.arange(kvl, device="cuda")[None, :]
        mask = kpos > qpos
        for hq in range(H_):
            g = hq // (H_ // H_KV)
            s = qi[:, hq] @ k[g].T
            s.masked_fill_(mask, float("-inf"))
            p = torch.softmax(s * (1.0 / 16.0), dim=-1)
            yield i, hq, slice(cu[i], cu[i + 1]), p @ v[g], s, v[g]
            del p


def p_code_diff(s_raw, nrows=64):
    """Emulated P e4m3 codes for the last nrows rows: base expf vs E ex2 (FP32) vs H (FP16 arg and exp2)."""
    s_raw = s_raw[-nrows:]
    m = s_raw.max(dim=-1, keepdim=True).values
    valid = torch.isfinite(s_raw)
    sc = 1.0 / 16.0
    s2 = sc * 1.4426950408889634
    pb = torch.exp(s_raw * sc - (m * sc - math.log(256.0)))
    pe = torch.exp2(s_raw * s2 - (m * s2 - 8.0))
    arg_h = (s_raw * s2 - (m * s2 - 8.0)).half()
    ph = torch.exp2(arg_h).float()
    cb = pb.to(torch.float8_e4m3fn).view(torch.uint8)
    ce = pe.to(torch.float8_e4m3fn).view(torch.uint8)
    ch = ph.to(torch.float8_e4m3fn).view(torch.uint8)
    n = int(valid.sum())
    return int(((cb != ce) & valid).sum()), int(((cb != ch) & valid).sum()), n


def main():
    os.makedirs(W, exist_ok=True)
    log(f"BASE_MAIN={BASE[:8]} TREE_COMMIT=" + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                              text=True).stdout.strip())
    summary = {"base_commit": BASE, "tree": TREE, "warmup": WARMUP, "iters": ITERS, "positions": POSITIONS}
    # 1. builds
    sh(f"rm -rf {W}/stock && mkdir -p {W}/stock && git archive {BASE} cpp/kernels/fmha_v2/src | tar -x -C {W}/stock")
    builds = {"stock": nvcc_build(f"{W}/stock/cpp/kernels/fmha_v2/src", f"{W}/bench_stock", [])}
    for a, fl in ARMS.items():
        builds[a] = nvcc_build(f"{TREE}/cpp/kernels/fmha_v2/src", f"{W}/bench_{a}", fl)
    summary["ptxas"] = builds
    log("ptxas", json.dumps(builds))
    # 2. SASS census
    summary["sass_census"] = {a: sass_report(f"{W}/bench_{a}") for a in ["stock"] + list(ARMS)}
    for a, rep in summary["sass_census"].items():
        for k, r in rep.items():
            log(f"[sass] {a} {k} whole {r['whole']}")
            for lp in r["loops"]:
                log(f"[sass] {a} {k} loop {lp}")
    # 6a. SASS identity of this tree's base build vs the stock build
    st, bs = sass_text(f"{W}/bench_stock"), sass_text(f"{W}/bench_base")
    summary["sass_base_vs_stock_identical"] = {k: st.get(k) == bs.get(k) for k in st}
    log("[sass] base(this tree, macros 0) vs stock:", summary["sass_base_vs_stock_identical"])
    # 6b. other kernels with the macros on vs BASE
    other = {}
    flags = (["nvcc", "-O3", "-std=c++17", "--expt-extended-lambda", "--expt-relaxed-constexpr",
              "-DUSE_DEMO_BERT_PARAMS=1", "-DGENERATE_CUBIN=1", "-cubin", "-arch=sm_120a"])
    subprocess.run(flags + [f"-I{W}/stock/cpp/kernels/fmha_v2/src", f"{HERE}/sass_same.cu", "-o", f"{W}/ss_base.cubin"],
                   check=True)
    for tag, fl in (("macros0", []), ("E", ARMS["E"]), ("H", ARMS["H"])):
        subprocess.run(flags + fl + [f"-I{TREE}/cpp/kernels/fmha_v2/src", f"{HERE}/sass_same.cu", "-o",
                                     f"{W}/ss_{tag}.cubin"], check=True)
        res = {}
        for k in ("k_bf16_tiled", "k_fp16_tiled", "k_e4m3_nl"):
            a = subprocess.run(["cuobjdump", "-sass", "-fun", k, f"{W}/ss_base.cubin"], capture_output=True,
                               text=True).stdout
            b = subprocess.run(["cuobjdump", "-sass", "-fun", k, f"{W}/ss_{tag}.cubin"], capture_output=True,
                               text=True).stdout
            la = [l for l in a.splitlines() if re.match(r"\s+/\*[0-9a-f]{4}\*/", l)]
            lb = [l for l in b.splitlines() if re.match(r"\s+/\*[0-9a-f]{4}\*/", l)]
            res[k] = dict(lines=len(la), identical=(la == lb and len(la) > 100))
        other[tag] = res
        log(f"[sass_same] {tag}: {res}")
    summary["sass_other_kernels_vs_base"] = other

    # 3. timing palindrome
    pos_res = []
    for pi, arm in enumerate(POSITIONS):
        st0 = gpu_state()
        rc, out = run_bench(f"{W}/bench_{arm}", TIME_CASES)
        res, _ = parse_bench(out)
        st1 = gpu_state()
        pos_res.append(dict(pos=pi, arm=arm, rc=rc, gpu_before=st0, gpu_after=st1, res=res))
        log(f"[time] pos {pi} {arm} rc {rc} " + " ".join(f"{c}={r['median']:.4f}" for c, r in res.items())
            + f" | {st0} -> {st1}")
    summary["positions_raw"] = pos_res
    timing = {}
    for case in TIME_CASES.split(","):
        tc = {}
        for arm in ARMS:
            rs = [p["res"][case] for p in pos_res if p["arm"] == arm]
            meds = [r["median"] for r in rs]
            tc[arm] = dict(median=statistics.median(meds), p10=statistics.median([r["p10"] for r in rs]),
                           p90=statistics.median([r["p90"] for r in rs]), position_medians=meds,
                           mismatches=sum(r["mismatches"] for r in rs), nonfinite=max(r["nonfinite"] for r in rs),
                           inputs_ok=all(r["inputs_ok"] for r in rs))
        bm = tc["base"]["position_medians"]
        floor = (max(bm) - min(bm)) / statistics.median(bm)
        tc["aa_floor"] = floor
        for arm in ("E", "H"):
            tc[arm]["gain_vs_base"] = 1.0 - tc[arm]["median"] / tc["base"]["median"]
            tc[arm]["beats_2x_floor"] = tc[arm]["gain_vs_base"] > 2 * floor
        timing[case] = tc
        log(f"[timing] {case} floor {floor:.4f} " + " ".join(
            f"{a}: {tc[a]['median']:.4f} [{tc[a]['p10']:.4f},{tc[a]['p90']:.4f}]" for a in ARMS)
            + f" gainE {tc['E']['gain_vs_base']:.4f} gainH {tc['H']['gain_vs_base']:.4f}")
    summary["timing"] = timing

    # 4./5. error + repeated-call runs
    dumps, rep = {}, {}
    meta = None
    for arm in ["stock"] + list(ARMS):
        d = f"{W}/dump_{arm}"
        rc, out = run_bench(f"{W}/bench_{arm}", ",".join(ERR_CASES), dump=d)
        res, mt = parse_bench(out)
        meta = meta or mt
        dumps[arm] = d
        rep[arm] = {c: dict(calls=r["calls"], mismatches_vs_call1=r["mismatches"], nonfinite=r["nonfinite"],
                            inputs_unchanged=r["inputs_ok"], hash=r["hash"]) for c, r in res.items()}
        log(f"[repeat] {arm} rc {rc} {json.dumps(rep[arm])}")
    summary["repeated_calls"] = rep
    errs = {}
    pdiff = {}
    for case in ERR_CASES:
        m = meta[case]
        same_inputs = all(np.array_equal(np.fromfile(f"{dumps['stock']}/{case}_{n}.bin", dtype=np.uint8),
                                         np.fromfile(f"{dumps[a]}/{case}_{n}.bin", dtype=np.uint8))
                          for a in ARMS for n in ("q", "pool", "offs"))
        outs = {a: load_o(dumps[a], case) for a in dumps}
        acc = {a: dict(max_abs=0.0, sum_abs=0.0, n=0, finite=bool(torch.isfinite(outs[a]).all())) for a in dumps}
        raw_stock = np.fromfile(f"{dumps['stock']}/{case}_o.bin", dtype=np.uint16)
        raw_base = np.fromfile(f"{dumps['base']}/{case}_o.bin", dtype=np.uint16)
        ref_max = 0.0
        for i, hq, sl, ref, s, _v in ref_iter(dumps["stock"], case, m):
            ref_max = max(ref_max, float(ref.abs().max()))
            for a in dumps:
                e = (outs[a][sl, hq] - ref).abs()
                acc[a]["max_abs"] = max(acc[a]["max_abs"], float(e.max()))
                acc[a]["sum_abs"] += float(e.sum())
                acc[a]["n"] += e.numel()
            if case in ("c1", "c1_out") and i == 0:
                de, dh, n = p_code_diff(s)
                pd = pdiff.setdefault(case, dict(E_vs_base=0, H_vs_base=0, n=0))
                pd["E_vs_base"] += de
                pd["H_vs_base"] += dh
                pd["n"] += n
            del ref, s
        entry = {"same_inputs": same_inputs, "ref_max_abs": ref_max,
                 "base_bitwise_eq_stock": bool(np.array_equal(raw_stock, raw_base))}
        for a in dumps:
            entry[a] = dict(max_abs=acc[a]["max_abs"], mean_abs=acc[a]["sum_abs"] / max(acc[a]["n"], 1),
                            n=acc[a]["n"], finite=acc[a]["finite"])
        for a in ("E", "H"):
            entry[a]["max_ratio_vs_stock"] = entry[a]["max_abs"] / max(entry["stock"]["max_abs"], 1e-30)
            entry[a]["mean_ratio_vs_stock"] = entry[a]["mean_abs"] / max(entry["stock"]["mean_abs"], 1e-30)
            entry[a]["within_1p5x"] = (entry[a]["finite"] and entry[a]["max_ratio_vs_stock"] <= 1.5
                                       and entry[a]["mean_ratio_vs_stock"] <= 1.5)
        errs[case] = entry
        log(f"[err] {case} {json.dumps(entry)}")
    for c, pd in pdiff.items():
        pd["E_frac"] = pd["E_vs_base"] / max(pd["n"], 1)
        pd["H_frac"] = pd["H_vs_base"] / max(pd["n"], 1)
    summary["errors"] = errs
    summary["p_code_diff_emulated"] = pdiff
    log("[pdiff]", json.dumps(pdiff))

    # STOP rule
    verdict = {}
    for a in ("E", "H"):
        speed = all(timing[c][a]["beats_2x_floor"] for c in ("c1", "c3") if c in timing)
        err_ok = all(errs[c][a]["within_1p5x"] for c in errs)
        rep_ok = all(r["mismatches_vs_call1"] == 0 and r["nonfinite"] == 0 and r["inputs_unchanged"]
                     for r in rep[a].values())
        ok = speed and err_ok and rep_ok
        if a == "H":
            ok = ok and timing["c1"]["H"]["gain_vs_base"] >= 0.03
        verdict[a] = dict(speed_gate_c1_c3=speed, error_bar=err_ok, repeated_ok=rep_ok,
                          gain_c1=timing["c1"][a]["gain_vs_base"], gain_c3=timing.get("c3", {}).get(a, {}).get(
                              "gain_vs_base"), passes_step0=bool(ok))
    summary["verdict"] = verdict
    with open(f"{ART}/step0_230.json", "w") as f:
        json.dump(summary, f, indent=1)
    short = {"verdict": verdict,
             "timing": {c: {a: (round(t[a]["median"], 5) if a in ARMS else round(t[a], 5)) for a in list(ARMS) + [
                 "aa_floor"]} for c, t in timing.items()},
             "errors": {c: {a: [e[a]["max_abs"], e[a]["mean_abs"]] for a in ["stock"] + list(ARMS)} for c, e in
                        errs.items()},
             "ptxas": builds, "sass_base_vs_stock_identical": summary["sass_base_vs_stock_identical"],
             "sass_other_kernels_identical": {t: all(v["identical"] for v in r.values()) for t, r in other.items()},
             "passed": all(rep[a][c]["mismatches_vs_call1"] == 0 and rep[a][c]["nonfinite"] == 0
                           for a in rep for c in rep[a])}
    print(json.dumps(short))


if __name__ == "__main__":
    main()
