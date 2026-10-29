"""Idea 236 (branch 161): exact K/V load-path trims (FMHA_FLUX3_LOAD_TRIM, lever load_trim/ldt)
in the routed fmha_v2 flux3 d128 kernel, plus STEP 0 (census and P6a/P6b probes).

Modes (one process each; module builds print a heartbeat every 30 s):
  census  SASS census of the given arms: cuobjdump -res-usage (REG/STACK/LOCAL), whole-function
          and KV-loop-body (largest backward-branch range) opcode-group counts, the loop body's
          forward-branch regions; arms ending in 'li' are -lineinfo twins whose loop-body
          instructions are mapped to source file:line (load-path lines listed). With --ncu the
          script re-runs itself under ncu (SourceCounters) for one S=4176 call per arm and
          reports DYNAMIC executed warp-instructions per warp-step by opcode group.
  timing  33-call CUDA graph (5xS4096 + 28xS4176) plus 10-call S4176 and S4096 graphs per arm,
          10 s soak, --blocks x --block samples, arm order reversed every other block, NVML SM
          clock per block; per-call p10/p50/p90 and paired ratios vs each arm's reference.
          Exact arms then run the untimed correctness sequence (same soak count, same sample
          count, same reset points): every replay bitwise == replay 0 and == the reference arm.
  check   correctness: routed C0 vs load_trim at S = 4176/4096/4100/4160/4256/40, randn and
          peaky inputs, 100 repeated eager calls == call 0, graph == eager, inputs unchanged,
          the H=16 stride guard routes to the untrimmed module (bitwise) and the ldt module
          rejects stride 2048.
  pair    one arm for run_gpu_pair: the tree's own routing; prints the evidence JSON and the
          SASS sha256 of the module for main's lever set (ROUTED_LEVERS without load_trim).

Arms: C0 = main's routed levers; noUA = C0 without unit_alpha; P6a/P6b = noUA + probe;
LDT = C0 + load_trim; LDTnoUA = noUA + load_trim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import torch

H, D = 24, 128
SEQ = [4096] * 5 + [4176] * 28
K = len(SEQ)
NSUB = 10  # calls in the per-S graphs


def log(*a):
    print(*a, flush=True)


class Heartbeat:
    def __init__(self, what):
        self.what, self.stop = what, threading.Event()

    def __enter__(self):
        t0 = time.time()

        def run():
            while not self.stop.wait(30):
                log(f"[heartbeat] {self.what}: {time.time() - t0:.0f}s")

        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def lib_mod():
    from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

    lib._LEVER_FLAGS.setdefault("ldt_p6a", ("p6a", "-DFMHA_FLUX3_LDT_PROBE_P6A=1"))
    lib._LEVER_FLAGS.setdefault("ldt_p6b", ("p6b", "-DFMHA_FLUX3_LDT_PROBE_P6B=1"))
    lib._LEVER_FLAGS.setdefault("lineinfo", ("li", "-lineinfo"))
    if not hasattr(lib, "_B161_IMPORT_LEVERS"):
        # The arms below set lib.ROUTED_LEVERS; derive every arm from the import-time value.
        lib._B161_IMPORT_LEVERS = tuple(lib.ROUTED_LEVERS)
    return lib


def main_levers(lib):
    return tuple(x for x in lib._B161_IMPORT_LEVERS if x != "load_trim")


def arm_levers(lib, name):
    c0 = main_levers(lib)
    noua = tuple(x for x in c0 if x != "unit_alpha")
    table = {
        "C0": c0,
        "C0x": c0,  # A/A twin (same module, separate graphs)
        "noUA": noua,
        "P6a": noua + ("ldt_p6a",),
        "P6b": noua + ("ldt_p6b",),
        "LDT": c0 + ("load_trim",),
        "LDTnoUA": noua + ("load_trim",),
        "C0li": c0 + ("lineinfo",),
        "LDTli": c0 + ("load_trim", "lineinfo"),
    }
    return tuple(table[name])


def build(lib, levers):
    with Heartbeat(f"build {lib._tag(levers, 0)}"):
        t0 = time.time()
        mod = lib.get_flux3_d128_module(tuple(levers))
        log(f"[build] {lib._tag(levers, 0)} in {time.time() - t0:.1f}s")
    return mod


def so_path(lib, levers):
    return str(lib.gen_flux3_d128_spec(tuple(levers)).get_library_path())


def tool(name):
    for cand in (shutil.which(name), f"/usr/local/cuda/bin/{name}"):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError(f"{name} not found")


# ------------------------------------------------------------------------------------- SASS
INST_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")


def sass_text(so):
    return subprocess.run([tool("cuobjdump"), "-sass", so], capture_output=True, text=True,
                          check=True).stdout


def sass_sha(so):
    txt = sass_text(so)
    keep = [ln.strip() for ln in txt.splitlines() if "/*" in ln or "Function" in ln]
    return hashlib.sha256("\n".join(keep).encode()).hexdigest(), len(keep)


def res_usage(so):
    txt = subprocess.run([tool("cuobjdump"), "-res-usage", so], capture_output=True, text=True,
                         check=True).stdout
    out = {}
    for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
                         txt):
        out[m.group(1)[-40:]] = dict(REG=int(m.group(2)), STACK=int(m.group(3)),
                                     SHARED=int(m.group(4)), LOCAL=int(m.group(5)))
    if not out:
        out["raw"] = txt[-600:]
    return out


PAD_FN = "128x128_output_bf16_sm120_kernel_nl_tiled"  # the padding-mask kernel (not causal/custom)


def parse_kernel(txt):
    """Instructions of the padding-mask nl_tiled kernel: list of (addr, opcode, full text)."""
    insts, cur = [], None
    for ln in txt.splitlines():
        if "Function :" in ln:
            cur = PAD_FN in ln
            continue
        if not cur:
            continue
        m = INST_RE.search(ln)
        if not m:
            continue
        addr, body = int(m.group(1), 16), m.group(2)
        b = re.sub(r"^@!?U?P[T0-9]+\s+", "", body)
        op = b.split()[0] if b.split() else "?"
        insts.append((addr, op, body))
    return insts


GROUPS = [
    ("MMA", r"^(QMMA|HMMA|OMMA|IMMA)"),
    ("LDSM", r"^LDSM"),
    ("LDGSTS", r"^LDGSTS"),
    ("LDGDEPBAR/DEPBAR", r"^(LDGDEPBAR|DEPBAR)"),
    ("BAR", r"^BAR"),
    ("MUFU", r"^MUFU"),
    ("FFMA", r"^FFMA"),
    ("FMUL", r"^FMUL"),
    ("FADD", r"^FADD"),
    ("FMNMX", r"^FMNMX"),
    ("FSETP/FSEL", r"^(FSETP|FSEL|FCHK)"),
    ("F2FP/cvt", r"^(F2FP|F2F|I2F|F2I)"),
    ("PRMT", r"^PRMT"),
    ("SHFL", r"^SHFL"),
    ("ISETP", r"^ISETP"),
    ("IADD3/IMAD/LEA", r"^(IADD|IMAD|LEA|IABS|IMNMX|VIMNMX|IMUL)"),
    ("LOP3/SHF/SEL", r"^(LOP|SHF|SHL|SHR|SEL|BMSK|FLO|POPC|BREV)"),
    ("PLOP3/P2R/R2P", r"^(PLOP3|P2R|R2P|VOTE)"),
    ("MOV/S2R/LDC", r"^(MOV|S2R|CS2R|S2UR|R2UR|LDC)"),
    ("Uniform", r"^U"),
    ("LDS/STS/LDG/STG", r"^(STS|LDS|LDG|STG|LD|ST|ATOM|RED)"),
    ("control", r"^(BRA|BSSY|BSYNC|EXIT|WARPSYNC|NOP|BPT|CALL|RET|YIELD|JMP|BREAK|NANOSLEEP)"),
]


def group_of(op):
    for g, pat in GROUPS:
        if re.match(pat, op):
            return g
    return "other"


def hist(insts):
    h = {}
    for _, op, _ in insts:
        g = group_of(op)
        h[g] = h.get(g, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def bra_target(txt):
    if "BRA" not in txt:
        return None
    m = re.search(r"0x([0-9a-f]+)", txt.split("BRA", 1)[1])
    return int(m.group(1), 16) if m else None


def loop_body(insts):
    """The largest backward-branch range [target, branch]."""
    best = None
    for addr, op, body in insts:
        if op.startswith("BRA"):
            tgt = bra_target(body)
            if tgt is not None and tgt < addr and (best is None or addr - tgt > best[1] - best[0]):
                best = (tgt, addr)
    if best is None:
        return None, []
    return best, [i for i in insts if best[0] <= i[0] <= best[1]]


def regions(body):
    """Forward branches inside the loop body: (branch addr, target, predicate, size)."""
    out = []
    for addr, op, txt in body:
        if op.startswith("BRA"):
            tgt = bra_target(txt)
            if tgt is not None and tgt > addr:
                pred = txt.split()[0] if txt.startswith("@") else "always"
                n = sum(1 for a, _, _ in body if addr < a < tgt)
                out.append(dict(at=hex(addr), to=hex(tgt), pred=pred, skipped=n))
    return out


ADDR_SRC: dict = {}


def line_map(so, loop_range):
    """With a -lineinfo build: loop-body instruction counts per source file:line."""
    ADDR_SRC.clear()
    tmp = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "b161_cubin")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([tool("cuobjdump"), "-xelf", "all", so], cwd=tmp, capture_output=True,
                   check=True)
    cubins = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".cubin")]
    per_line, per_line_ops = {}, {}
    for cb in cubins:
        txt = subprocess.run([tool("nvdisasm"), "-g", "-c", cb], capture_output=True,
                             text=True).stdout
        infn, cur_src = False, None
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith(".section") or s.startswith(".text."):
                infn = PAD_FN in s
            m = re.search(r'//## File "([^"]+)", line (\d+)', ln)
            if m:
                cur_src = f"{os.path.basename(m.group(1))}:{m.group(2)}"
                continue
            m = INST_RE.search(ln)
            if m and infn and cur_src:
                a = int(m.group(1), 16)
                ADDR_SRC[a] = cur_src
                if loop_range and loop_range[0] <= a <= loop_range[1]:
                    b = re.sub(r"^@!?U?P[T0-9]+\s+", "", m.group(2))
                    op = b.split()[0]
                    per_line[cur_src] = per_line.get(cur_src, 0) + 1
                    per_line_ops.setdefault(cur_src, {})
                    g = group_of(op)
                    per_line_ops[cur_src][g] = per_line_ops[cur_src].get(g, 0) + 1
    return per_line, per_line_ops


LOAD_FILES = ("gmem_tile_qkv_packed.h", "utils.h", "smem_tile.h", "smem_tile_v.h")
# Load-path source lines (pinned FlashInfer 0.6.18 csrc/fmha_v2/fmha): all of the Q/K/V gmem tile;
# utils.h pack_predicates/ldgsts_ (1280-1450) and ldgdepbar/depbar/ldgsts128/Ldgsts_functor
# (1555-1615); smem_tile.h compute_store_pointers (148-180) and move_to_next_write_buffer
# (234-245). LDSM/PRMT/cvt helpers in utils.h are not load path.
LOAD_RANGES = {"gmem_tile_qkv_packed.h": [(1, 10 ** 6)], "utils.h": [(1280, 1450), (1555, 1615)],
               "smem_tile.h": [(148, 180), (234, 245)]}


def is_load(src):
    f, _, ln = src.partition(":")
    try:
        n = int(ln)
    except ValueError:
        return False
    return any(a <= n <= b for a, b in LOAD_RANGES.get(f, []))


def census_one(lib, name):
    levers = arm_levers(lib, name)
    build(lib, levers)
    so = so_path(lib, levers)
    txt = sass_text(so)
    insts = parse_kernel(txt)
    rng, body = loop_body(insts)
    res = dict(arm=name, tag=lib._tag(levers, 0), res_usage=res_usage(so),
               sass_sha256=sass_sha(so)[0], n_func=len(insts),
               stl_ldl=sum(1 for _, op, _ in insts if op.startswith(("STL", "LDL"))),
               func_hist=hist(insts),
               loop_range=[hex(rng[0]), hex(rng[1])] if rng else None, n_loop=len(body),
               loop_hist=hist(body), loop_regions=regions(body))
    if name.endswith("li"):
        pl, plo = line_map(so, rng)
        load = {k: v for k, v in pl.items() if is_load(k)}
        res["loop_lines_load_path"] = dict(sorted(load.items(), key=lambda kv: -kv[1]))
        res["loop_lines_load_path_ops"] = {k: plo[k] for k in load}
        res["loop_load_path_total"] = sum(load.values())
        files = {}
        for k, v in pl.items():
            f = k.split(":")[0]
            files[f] = files.get(f, 0) + v
        res["loop_lines_by_file"] = files
        grp = {}
        for k in load:
            for g, n in plo[k].items():
                grp[g] = grp.get(g, 0) + n
        res["loop_load_path_by_group"] = grp
        ARM_ADDR_SRC[name] = dict(ADDR_SRC)
        art = os.environ.get("AUTO_DEV_ARTIFACTS")
        if art:
            with open(os.path.join(art, f"loop_{name}.txt"), "w") as f:
                for a, op, t in body:
                    src = ADDR_SRC.get(a, "?")
                    mark = "L" if is_load(src) else " "
                    f.write(f"{mark} {a:05x} {src:32s} {t}\n")
    return res, txt


ARM_ADDR_SRC: dict = {}


def mode_census(args):
    lib = lib_mod()
    out = {}
    art = os.environ.get("AUTO_DEV_ARTIFACTS")
    for name in args.arms.split(","):
        r, txt = census_one(lib, name)
        if art:
            with open(os.path.join(art, f"sass_{name}.txt"), "w") as f:
                f.write(txt)
        log(f"[census {name}] {json.dumps(r)}")
        out[name] = r
    if args.ncu:
        out["ncu"] = ncu_census(args)
    print(json.dumps(dict(census=out)))


# -------------------------------------------------------------------------- ncu dynamic census
def ncu_census(args):
    """Per-SASS-instruction executed counts (ncu SourceCounters) for one S4176 call per arm."""
    ncu = shutil.which("ncu") or "/usr/local/bin/ncu"
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    res = {}
    lib = lib_mod()
    for name in args.ncu_arms.split(","):
        rep = os.path.join(art, f"ncu_{name}")
        cmd = [ncu, "--section", "SourceCounters", "-k", "regex:fmha_v2_flux3", "-c", "1",
               "-f", "-o", rep, sys.executable, os.path.abspath(__file__), "--mode", "one",
               "--arm", name]
        log(f"[ncu] {' '.join(cmd)}")
        try:
            with Heartbeat(f"ncu {name}"):
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001
            res[name] = dict(error=str(exc)[:300])
            continue
        log(p.stdout[-1500:], p.stderr[-1500:])
        reps = [os.path.join(art, f) for f in os.listdir(art)
                if f.startswith(f"ncu_{name}.ncu-rep")]
        if p.returncode != 0 or not reps:
            res[name] = dict(error=f"ncu rc {p.returncode}, files {os.listdir(art)}")
            continue
        q = subprocess.run([ncu, "--import", reps[0], "--page", "source", "--csv",
                            "--print-source", "sass"], capture_output=True, text=True)
        log(f"[ncu import] rc {q.returncode} {len(q.stdout)} bytes, err {q.stderr[-500:]}")
        with open(os.path.join(art, f"ncu_{name}_source.csv"), "w") as f:
            f.write(q.stdout)
        try:
            res[name] = summarize_ncu_source(q.stdout, ARM_ADDR_SRC.get(name))
        except Exception as exc:  # noqa: BLE001
            res[name] = dict(error=f"parse: {exc}", head=q.stdout[:800])
        log(f"[ncu {name}] {json.dumps(res[name])}")
    return res


def summarize_ncu_source(csv_text, addr_src=None):
    import csv
    import io

    rows = list(csv.reader(io.StringIO(csv_text)))
    hdr_i = next((i for i, r in enumerate(rows) if any("Address" in c for c in r)), None)
    if hdr_i is None:
        return dict(error="no header", head=csv_text[:800])
    hdr = [c.strip() for c in rows[hdr_i]]
    ia = next(i for i, c in enumerate(hdr) if "Address" in c)
    isrc = next(i for i, c in enumerate(hdr) if c == "Source")
    iex = next((i for i, c in enumerate(hdr) if c in ("Warp Instructions Executed",
                                                      "Instructions Executed")), None)
    if iex is None:
        iex = next(i for i, c in enumerate(hdr)
                   if "Instructions Executed" in c and "Thread" not in c)
    insts = []
    for r in rows[hdr_i + 1:]:
        if len(r) <= max(ia, isrc, iex):
            continue
        try:
            a = int(r[ia], 16)
            n = float(r[iex].replace(",", "") or 0)
        except ValueError:
            continue
        b = re.sub(r"^@!?U?P[T0-9]+\s+", "", r[isrc].strip())
        op = b.split()[0] if b.split() else "?"
        insts.append((a, op, r[isrc].strip(), n))
    S = 4176
    warp_steps = math.ceil(S / 64) * H * 4 * math.ceil(S / 64)
    tot = sum(n for *_, n in insts)
    rng, body = loop_body([(a, op, t) for a, op, t, _ in insts])
    bset = {a for a, _, _ in body}
    by_g, by_g_loop = {}, {}
    for a, op, _, n in insts:
        g = group_of(op)
        by_g[g] = by_g.get(g, 0) + n
        if a in bset:
            by_g_loop[g] = by_g_loop.get(g, 0) + n
    loop_tot = sum(by_g_loop.values())
    extra = {}
    if addr_src:
        # ncu reports absolute addresses; nvdisasm reports function-relative offsets.
        base = min(a for a, *_ in insts)
        per_file, per_line, load_g = {}, {}, {}
        for a, op, _, n in insts:
            if a not in bset:
                continue
            src = addr_src.get(a - base, "?:0")
            f = src.split(":")[0]
            per_file[f] = per_file.get(f, 0) + n
            if is_load(src):
                per_line[src] = per_line.get(src, 0) + n
                g = group_of(op)
                load_g[g] = load_g.get(g, 0) + n
        extra = dict(
            loop_per_warp_step_by_file={f: round(v / warp_steps, 2) for f, v in
                                        sorted(per_file.items(), key=lambda kv: -kv[1])},
            loop_load_files_per_warp_step_by_line={s: round(v / warp_steps, 2) for s, v in
                                                   sorted(per_line.items(),
                                                          key=lambda kv: -kv[1]) if v > 0},
            loop_load_files_per_warp_step_by_group={g: round(v / warp_steps, 2) for g, v in
                                                    sorted(load_g.items(),
                                                           key=lambda kv: -kv[1])})
    return dict(header=hdr, warp_steps=warp_steps, total_warp_inst=tot, **extra,
                total_per_warp_step=round(tot / warp_steps, 2),
                loop_per_warp_step=round(loop_tot / warp_steps, 2),
                loop_per_warp_step_by_group={g: round(v / warp_steps, 2) for g, v in
                                             sorted(by_g_loop.items(), key=lambda kv: -kv[1])},
                func_per_warp_step_by_group={g: round(v / warp_steps, 2) for g, v in
                                             sorted(by_g.items(), key=lambda kv: -kv[1])})


def mode_one(args):
    """One S4176 call of an arm (the ncu target)."""
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib

    lib = lib_mod()
    lib.ROUTED_LEVERS = arm_levers(lib, args.arm)
    build(lib, lib.ROUTED_LEVERS)
    inputs = make_inputs()
    out = torch.empty(1, 4176, H, D, device="cuda", dtype=torch.bfloat16)
    call(fib, inputs[4176], out)
    torch.cuda.synchronize()
    log("[one] done")


# ------------------------------------------------------------------------------------ inputs
class Clock:
    def __init__(self):
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            for cand in (uuid, "GPU-" + uuid):
                try:
                    self.h = pynvml.nvmlDeviceGetHandleByUUID(cand)
                    break
                except Exception:  # noqa: BLE001
                    continue
            self.nv = pynvml
        except Exception as exc:  # noqa: BLE001
            log(f"[clock] NVML unavailable: {exc}")

    def sm_mhz(self):
        if self.h is None:
            return None
        try:
            return self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
        except Exception:  # noqa: BLE001
            return None


def make_one(S, seed, peaky=False, heads=H):
    from tensorrt_llm._torch.visual_gen.attention.backends.fp8_sm12x import quantize_qkv_e4m3

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn((3, 1, S, heads, D), generator=g, device="cuda", dtype=torch.float32)
    if peaky:
        buf[0] *= 6.0
        idx = torch.randint(0, S, (max(1, S // 64),), generator=g, device="cuda")
        buf[1, :, idx] *= 8.0
    buf = buf.to(torch.bfloat16)
    q, k, v = buf.unbind(0)
    q8, k8, v8, scales = quantize_qkv_e4m3(q, k, v, 1.0 / math.sqrt(D))
    return dict(S=S, H=heads, q=q8, k=k8, v=v8, scales=scales)


def make_inputs(seed=1):
    out = {S: make_one(S, seed + S) for S in sorted(set(SEQ))}
    torch.cuda.synchronize()
    return out


NAMES = ("q", "k", "v", "scales")


def snapshot(inputs):
    return {S: {n: inputs[S][n].clone() for n in NAMES} for S in inputs}


def restore(inputs, snap):
    for S in inputs:
        for n in NAMES:
            inputs[S][n].copy_(snap[S][n])
    torch.cuda.synchronize()


def same(a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def unchanged(inputs, snap):
    return all(same(inputs[S][n], snap[S][n]) for S in inputs for n in NAMES)


def call(fib, inp, out):
    sc = inp["scales"]
    fib.run_fmha_v2_fp8_sm12x(inp["q"], inp["k"], inp["v"], out, inp.get("H", H), D, inp["S"],
                              scale_bmm1_d=sc[3:4], scale_bmm2_d=sc[2:3], causal=False,
                              return_lse=False)


def graph_kernels(g):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    return {e.key: e.count for e in prof.key_averages() if "fmha" in e.key}


def capture_seq(fib, inputs, seq):
    total = sum(seq) * H * D
    flat = torch.empty(total, device="cuda", dtype=torch.bfloat16)
    outs, off = [], 0
    for S in seq:
        outs.append(flat[off:off + S * H * D].view(1, S, H, D))
        off += S * H * D
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for S, o in zip(seq, outs):
                call(fib, inputs[S], o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for S, o in zip(seq, outs):
            call(fib, inputs[S], o)
    torch.cuda.synchronize()
    return g, flat


GTYPES = {"mix": SEQ, "s4176": [4176] * NSUB, "s4096": [4096] * NSUB}


def pct(xs, p):
    return torch.tensor(xs, dtype=torch.float64).quantile(p).item()


def summ(xs, ncalls):
    p10, p50, p90 = pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9)
    return dict(n=len(xs), p10_us=p10 / ncalls * 1e3, p50_us=p50 / ncalls * 1e3,
                p90_us=p90 / ncalls * 1e3, noise=(p90 - p10) / p50)


def mode_timing(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib

    lib = lib_mod()
    arms = args.arms.split(",")
    refs = dict(x.split(":") for x in args.refs.split(",")) if args.refs else {}
    exact = set(args.exact.split(",")) if args.exact else set()
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    graphs, kern, failures = {}, {}, []
    for a in arms:
        lib.ROUTED_LEVERS = arm_levers(lib, a)
        build(lib, lib.ROUTED_LEVERS)
        tag = "fmha_v2_flux3_" + lib._tag(lib.ROUTED_LEVERS, 0) + "_"
        for gt, seq in GTYPES.items():
            graphs[(a, gt)] = capture_seq(fib, inputs, seq)
        kern[a] = graph_kernels(graphs[(a, "mix")][0])
        n_ok = sum(c for n, c in kern[a].items() if tag in n)
        log(f"[graph {a}] tag {tag} {json.dumps(kern[a])}")
        if n_ok != K or len(kern[a]) != 1:
            failures.append(f"arm {a}: graph does not hold {K} launches of {tag}")
    restore(inputs, snap)
    allg = [graphs[k][0] for k in graphs]
    t0, n_soak = time.time(), 0
    while time.time() - t0 < args.soak_s:
        for g in allg:
            g.replay()
        n_soak += 1
        if n_soak % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    log(f"[soak] {n_soak} rounds, sm_mhz {clock.sm_mhz()}")
    samples = {(a, gt): [] for a in arms for gt in GTYPES}
    clocks = []
    for b in range(args.blocks):
        order = arms if b % 2 == 0 else arms[::-1]
        c0 = clock.sm_mhz()
        ev = []
        for _ in range(args.block):
            for gt in GTYPES:
                for a in order:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[(a, gt)][0].replay()
                    e1.record()
                    ev.append(((a, gt), e0, e1))
        torch.cuda.synchronize()
        for key, e0, e1 in ev:
            samples[key].append(e0.elapsed_time(e1))
        c1 = clock.sm_mhz()
        clocks.append(dict(block=b, order=order, sm_mhz=[c0, c1]))
        med = {a: round(sorted(samples[(a, "mix")][-args.block:])[args.block // 2] / K * 1e3, 2)
               for a in arms}
        log(f"[block {b + 1}/{args.blocks}] sm {c0}->{c1} mix per-call us {med}")
    table = {}
    for a in arms:
        table[a] = {gt: summ(samples[(a, gt)], len(seq)) for gt, seq in GTYPES.items()}
    ratios = {}
    for a, r in refs.items():
        ratios[f"{a}/{r}"] = {}
        for gt in GTYPES:
            pr = [x / y for x, y in zip(samples[(a, gt)], samples[(r, gt)])]
            ratios[f"{a}/{r}"][gt] = dict(p10=pct(pr, 0.1), p50=pct(pr, 0.5), p90=pct(pr, 0.9),
                                          frac_faster=sum(1 for x in pr if x < 1) / len(pr))
    for a in arms:
        log(f"[table {a}] " + " | ".join(
            f"{gt} p50 {v['p50_us']:.1f} [{v['p10_us']:.1f},{v['p90_us']:.1f}] us"
            for gt, v in table[a].items()))
    for k_, v in ratios.items():
        log(f"[ratio {k_}] " + " | ".join(f"{gt} {x['p50']:.5f} [{x['p10']:.5f},{x['p90']:.5f}] "
                                          f"faster {x['frac_faster']:.2f}" for gt, x in v.items()))
    if not unchanged(inputs, snap):
        failures.append("inputs changed during timing")
    # untimed correctness sequence for exact arms (same soak count / samples / reset points)
    check = {}
    if exact:
        restore(inputs, snap)
        for _ in range(n_soak):
            for g in allg:
                g.replay()
        torch.cuda.synchronize()
        ref0 = {}
        nrep = 0
        watch = exact | {refs[a] for a in exact}
        for b in range(args.blocks):
            order = arms if b % 2 == 0 else arms[::-1]
            for _ in range(args.block):
                for gt in GTYPES:
                    for a in order:
                        g, flat = graphs[(a, gt)]
                        g.replay()
                        if a not in watch:
                            continue
                        if (a, gt) not in ref0:
                            ref0[(a, gt)] = flat.clone()
                        elif not torch.equal(flat, ref0[(a, gt)]):
                            failures.append(f"{a} {gt} replay {nrep} != replay 0")
                nrep += 1
            torch.cuda.synchronize()
            if len(failures) > 20:
                break
        for a in exact:
            r = refs[a]
            for gt in GTYPES:
                eq = torch.equal(ref0[(a, gt)], ref0[(r, gt)])
                check[f"{a}=={r} {gt}"] = eq
                if not eq:
                    failures.append(f"{a} != {r} on {gt}")
        if not unchanged(inputs, snap):
            failures.append("inputs changed during the correctness sequence")
        log(f"[check] {nrep} replays per arm/graph, {json.dumps(check)}")
    print(json.dumps(dict(passed=not failures, failures=failures[:20], table=table,
                          ratios=ratios, kernels=kern, soak_rounds=n_soak, clocks=clocks,
                          bitwise=check)))


# ------------------------------------------------------------------------------------- check
def mode_check(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib

    lib = lib_mod()
    c0, ldt = arm_levers(lib, "C0"), arm_levers(lib, "LDT")
    for lv in (c0, ldt):
        build(lib, lv)
    failures, rows = [], []
    names = {}

    for S in (4176, 4096, 4100, 4160, 4256, 40):
        for peaky in (False, True):
            inp = make_one(S, 7 + S, peaky)
            snap = {n: inp[n].clone() for n in NAMES}
            outs = {}
            for an, lv in (("C0", c0), ("LDT", ldt)):
                lib.ROUTED_LEVERS = lv
                o0 = torch.empty(1, S, H, D, device="cuda", dtype=torch.bfloat16)
                call(fib, inp, o0)
                torch.cuda.synchronize()
                o = torch.empty_like(o0)
                rep_ok = True
                for _ in range(100):
                    call(fib, inp, o)
                    if not torch.equal(o, o0):
                        rep_ok = False
                torch.cuda.synchronize()
                og = torch.empty_like(o0)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    call(fib, inp, og)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    call(fib, inp, og)
                graph_ok = True
                for _ in range(3):
                    og.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    graph_ok &= torch.equal(og, o0)
                if S == 4176 and not peaky:
                    names[an] = graph_kernels(g)
                outs[an] = o0
                finite = bool(torch.isfinite(o0.float()).all())
                if not (rep_ok and graph_ok and finite):
                    failures.append(f"{an} S={S} peaky={peaky}: repeat {rep_ok} graph {graph_ok} "
                                    f"finite {finite}")
            inp_ok = all(same(inp[n], snap[n]) for n in NAMES)
            eq = torch.equal(outs["C0"], outs["LDT"])
            d = (outs["C0"].float() - outs["LDT"].float()).abs().max().item()
            rows.append(dict(S=S, peaky=peaky, ldt_eq_c0=eq, max_abs_diff=d, inputs_ok=inp_ok))
            log(f"[check] S={S} peaky={peaky} LDT==C0 {eq} maxdiff {d} inputs unchanged {inp_ok}")
            if not (eq and inp_ok):
                failures.append(f"S={S} peaky={peaky}: eq {eq} inputs {inp_ok}")
    tag_ldt = "fmha_v2_flux3_" + lib._tag(ldt, 0) + "_"
    tag_c0 = "fmha_v2_flux3_" + lib._tag(c0, 0) + "_"
    if not any(tag_ldt in n for n in names.get("LDT", {})):
        failures.append(f"LDT graph does not run {tag_ldt}: {names.get('LDT')}")
    if not any(tag_c0 in n for n in names.get("C0", {})):
        failures.append(f"C0 graph does not run {tag_c0}: {names.get('C0')}")
    # guard: H=16 (stride 2048) falls back to the untrimmed module, bitwise
    guard = {}
    inp = make_one(4176, 99, heads=16)
    oa = torch.empty(1, 4176, 16, D, device="cuda", dtype=torch.bfloat16)
    ob = torch.empty_like(oa)
    lib.ROUTED_LEVERS = c0
    call(fib, inp, oa)
    lib.ROUTED_LEVERS = ldt
    routed = lib.load_trim_levers(ldt, inp["k"], inp["v"])
    call(fib, inp, ob)
    torch.cuda.synchronize()
    guard["h16_routed_levers_drop_ldt"] = "load_trim" not in routed
    guard["h16_bitwise_vs_c0"] = torch.equal(oa, ob)
    try:
        lib.fmha_v2_flux3_d128(inp["q"], inp["k"], inp["v"], torch.empty_like(oa), 16, D, 4176,
                               scale_bmm1_d=inp["scales"][3:4], scale_bmm2_d=inp["scales"][2:3],
                               module=lib.get_flux3_d128_module(ldt))
        torch.cuda.synchronize()
        guard["ldt_module_rejects_stride_2048"] = False
    except Exception as exc:  # noqa: BLE001
        guard["ldt_module_rejects_stride_2048"] = "stride" in str(exc)
        guard["reject_msg"] = str(exc)[:200]
    log(f"[guard] {json.dumps(guard)}")
    for k_, v_ in guard.items():
        if v_ is False:
            failures.append(f"guard {k_}")
    print(json.dumps(dict(passed=not failures, failures=failures, rows=rows, guard=guard,
                          kernels=names,
                          compared="load_trim routed kernel vs main's routed kernel (C0, same "
                                   "process) bitwise at 6 S x 2 input kinds; 100 repeated eager "
                                   "calls == call 0; CUDA graph replays == eager; inputs "
                                   "unchanged; H=16 stride guard falls back bitwise")))


# -------------------------------------------------------------------------------------- pair
def mode_pair(args):
    from tensorrt_llm._torch.visual_gen.attention.backends import flashinfer as fib

    lib = lib_mod()
    routed = tuple(lib.ROUTED_LEVERS)
    log(f"[tree] ROUTED_LEVERS {routed} has load_trim_levers {hasattr(lib, 'load_trim_levers')}")
    mainset = main_levers(lib)
    build(lib, mainset)
    sha_main, nlines = sass_sha(so_path(lib, mainset))
    ru_main = res_usage(so_path(lib, mainset))
    build(lib, routed)
    ru = res_usage(so_path(lib, routed))
    log(f"[sass] main-lever-set module {lib._tag(mainset, 0)} sha256 {sha_main} ({nlines} lines) "
        f"res {json.dumps(ru_main)}; routed {lib._tag(routed, 0)} res {json.dumps(ru)}")
    clock = Clock()
    inputs = make_inputs()
    snap = snapshot(inputs)
    g, flat = capture_seq(fib, inputs, SEQ)
    names = graph_kernels(g)
    log(f"[graph] {json.dumps(names)}")
    restore(inputs, snap)
    t0, n = time.time(), 0
    while time.time() - t0 < args.soak_s:
        g.replay()
        n += 1
        if n % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    xs, clocks = [], []
    for b in range(args.blocks):
        c0 = clock.sm_mhz()
        ev = []
        for _ in range(args.block):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            ev.append((e0, e1))
        torch.cuda.synchronize()
        blk = [a.elapsed_time(z) for a, z in ev]
        xs += blk
        clocks.append([c0, clock.sm_mhz()])
        log(f"[block {b + 1}/{args.blocks}] sm {clocks[-1]} median per-call us "
            f"{sorted(blk)[len(blk) // 2] / K * 1e3:.2f}")
    s = summ(xs, K)
    ok_inputs = unchanged(inputs, snap)
    print(json.dumps({"batch": {"1": {"kernel_ms": s["p50_us"] / 1e3, "noise_floor": s["noise"]}},
                      "output": {"class": "exact"},
                      "sample": f"one CUDA-event-timed replay of a {K}-call graph "
                                f"(5xS4096 + 28xS4176); kernel_ms = per-call = graph/{K}",
                      "per_call": s, "kernels": names, "routed_levers": routed,
                      "main_lever_set_sass_sha256": sha_main, "res_usage_routed": ru,
                      "soak_replays": n, "clocks": clocks, "inputs_unchanged": ok_inputs}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["census", "timing", "check", "pair", "one"])
    p.add_argument("--arms", default="C0")
    p.add_argument("--arm", default="C0")
    p.add_argument("--refs", default="")
    p.add_argument("--exact", default="")
    p.add_argument("--ncu", action="store_true")
    p.add_argument("--ncu-arms", default="C0")
    p.add_argument("--soak-s", type=float, default=10.0)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--block", type=int, default=25)
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name()} mode {args.mode}")
    if args.mode != "one":
        r = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True)
        a = subprocess.run(["git", "merge-base", "--is-ancestor", "1185e8f9", "HEAD"],
                           capture_output=True, text=True)
        log(f"[ancestry] {r.stdout.strip()} | is-ancestor 1185e8f9 rc={a.returncode}")
    dict(census=mode_census, timing=mode_timing, check=mode_check, pair=mode_pair,
         one=mode_one)[args.mode](args)


if __name__ == "__main__":
    main()
