"""Idea 348 S0.P probe (no source edit, no rebuild): why serving profile 1783 shows id 20, not id 72.

1. Env / autotuner cache-file state: TRTLLM_CACHE_DIR, TLLM_AUTOTUNER_CACHE_PATH, whether the file
   exists, its 'mhc_fused_hc' entries; libth_common.so path / mtime / whether it carries the
   fused_pmap_gemm_fma_mtile_v3 symbol.
2. Direct launch of tactic ('fused_half_fma', 24, 1, 256, 72) at M=2048, V4.1-lagged + norm_weight;
   bitwise vs the ks=1 reference tactic (fused_half_fma, 24, 1, 256, 1).
3. Fresh AutoTuner (clear_cache, autotune(cache_path=None)), debug-to-info: per M the bucket, selected
   tactic and the candidate times of ids 20 and 72 (captured from the tuner's Candidate/Selected lines).
   (a) per-M tuning (skip_dynamic_tuning_buckets=True); (b) production-like: one autotune call at
   M=8192 that tunes every bucket, then mhc_fused_hc_tactic per M outside autotune.
Last stdout line: {"passed": ..., ...} JSON.
"""
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from guard import dirty_guard  # noqa: E402

dirty_guard()

env_pre = {k: os.environ.get(k) for k in ("TRTLLM_CACHE_DIR", "TLLM_AUTOTUNER_CACHE_PATH")}
print("[env before import]", env_pre, flush=True)

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402,F401
from tensorrt_llm._torch import autotuner as at_mod  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.modules.mhc import mhc_cuda  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from mhc_fhc_bench import make_inputs  # noqa: E402

report = {}
env_post = {k: os.environ.get(k) for k in ("TRTLLM_CACHE_DIR", "TLLM_AUTOTUNER_CACHE_PATH")}
print("[env after import]", env_post, flush=True)
report["env_before_import"] = env_pre
report["env_after_import"] = env_post
cp = env_post["TLLM_AUTOTUNER_CACHE_PATH"]
cache_info = {"path": cp, "exists": bool(cp and os.path.exists(cp))}
cands = []
if cp:
    d = os.path.dirname(cp)
    cands = sorted(glob.glob(os.path.join(d, "*")))[:20] if os.path.isdir(d) else []
cache_info["dir_files"] = cands
for f in ([cp] if cache_info["exists"] else []) + [c for c in cands if c != cp]:
    try:
        txt = open(f).read()
        hits = [m.start() for m in re.finditer("mhc_fused_hc", txt)]
        cache_info.setdefault("mhc_hits", {})[f] = {
            "n": len(hits), "mtime": time.ctime(os.path.getmtime(f)),
            "snippets": [txt[max(0, h - 40): h + 260] for h in hits[:6]]}
    except Exception as e:  # noqa: BLE001
        cache_info.setdefault("read_err", {})[f] = str(e)
report["cache"] = cache_info
print("[cache]", json.dumps(cache_info)[:4000], flush=True)

libs = {}
for p in sorted(set(glob.glob(os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libth_common*.so*")))):
    try:
        b = open(p, "rb").read()
        libs[p] = {"mtime": time.ctime(os.path.getmtime(p)), "size": len(b),
                   "has_mtile_v3": b.count(b"fused_pmap_gemm_fma_mtile_v3"),
                   "has_mtile_v2": b.count(b"fused_pmap_gemm_fma_mtile_v2")}
    except Exception as e:  # noqa: BLE001
        libs[p] = {"err": str(e)}
report["libth_common"] = libs
print("[lib]", json.dumps(libs), flush=True)

HC, HID = 4, 5120
ARGS = (HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)


def same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# ---- 2. direct id-72 launch at M=2048
M = 2048
inputs, nw = make_inputs(M, seed=3)
pre = torch.rand((M, HC), device="cuda").float() + 0.5
kw = dict(norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre)
rn = mhc_cuda.MhcFusedHcRunner(*ARGS)
offered = [tuple(t) for t in rn.get_valid_tactics(inputs, None)]
report["offered_2048"] = [list(t) for t in offered]
direct = {}
try:
    ref = [o.clone() for o in rn.forward(inputs, tactic=["fused_half_fma", 24, 1, 256, 1], **kw)]
    torch.cuda.synchronize()
    for tid in (20, 72):
        try:
            out = rn.forward(inputs, tactic=["fused_half_fma", 24, 1, 256, tid], **kw)
            torch.cuda.synchronize()
            direct[str(tid)] = {"ok": True, "bitwise_vs_ks1": all(same(a, b) for a, b in zip(out, ref)),
                                "n_out": len(out)}
        except Exception as e:  # noqa: BLE001
            direct[str(tid)] = {"ok": False, "error": repr(e)[:1500]}
except Exception as e:  # noqa: BLE001
    direct["ref_error"] = repr(e)[:1500]
report["direct_2048"] = direct
print("[direct M=2048]", json.dumps(direct), flush=True)
del inputs
torch.cuda.empty_cache()

# ---- 3. fresh tuner
lines = []
tuner = AutoTuner.get()
orig_dbg = tuner._debug_logger


def cap(msg, *a, **k):
    if "mhc_fused_hc" in msg:
        lines.append(msg)
    return orig_dbg(msg, *a, **k)


tuner._debug_logger = cap
TID_RE = re.compile(r"tactic=\('fused_half_fma', 24, 1, (\d+), (\d+)\)")


def parse(sel_lines):
    """Per tuned bucket M: selected tactic and the times of ids 20 / 72 / 1 / 4."""
    res = {}
    cur = {}
    for ln in sel_lines:
        ms = re.search(r"shapes=\(\(torch\.Size\(\[(\d+)", ln) or re.search(r"shapes=.*?\[(\d+), 5120\]", ln)
        if "Candidate:" in ln:
            m = TID_RE.search(ln)
            tm = re.search(r"time=([\d.inf]+)ms", ln)
            b = ms.group(1) if ms else "?"
            if m and tm:
                cur.setdefault(b, {})[f"id{m.group(2)}_bs{m.group(1)}"] = tm.group(1)
        elif "Selected:" in ln:
            t = re.search(r"tactic=(\(.*?\)|-1)", ln)
            tt = re.search(r"time=([\d.]+)ms", ln)
            # Selected line carries no shape: attach to the last candidate bucket seen.
            b = list(cur.keys())[-1] if cur else "?"
            res[b] = {"selected": t.group(1) if t else None, "time_ms": tt.group(1) if tt else None,
                      "cands": cur.pop(b, {})}
    return res


MS = [128, 512, 1024, 2048, 2176, 4096, 8192]
per_m = {}
for M in MS:
    inputs, nw = make_inputs(M, seed=5)
    pre = torch.rand((M, HC), device="cuda").float() + 0.5
    kw = dict(norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre)
    tuner.clear_cache()
    lines.clear()
    with autotune(cache_path=None, skip_dynamic_tuning_buckets=True):
        mhc_cuda.mhc_fused_hc(*inputs, *ARGS, **kw)
    torch.cuda.synchronize()
    tac = mhc_cuda.mhc_fused_hc_tactic(*inputs, *ARGS, **kw)
    p = parse(lines)
    per_m[str(M)] = {"bucket": mhc_cuda._mhc_map_to_tuning_bucket(M), "tactic_outside": list(tac), "tuned": p,
                     "n_lines": len(lines)}
    keyc = {k: v for b in p.values() for k, v in b["cands"].items() if k.startswith(("id20_", "id72_", "id4_", "id1_"))}
    print(f"[tune per-M] M={M} bucket={per_m[str(M)]['bucket']} pick={tuple(tac)} "
          f"cands(ms)={keyc} selected={[b['selected'] for b in p.values()]}", flush=True)
    if M == 2048:
        for ln in lines[:80]:
            print("   ", ln[:400], flush=True)
    del inputs
    torch.cuda.empty_cache()
report["per_m_tuning"] = per_m

# (b) production-like: one tune at M=8192 over every bucket, then the pick per M outside autotune.
tuner.clear_cache()
lines.clear()
M = 8192
inputs, nw = make_inputs(M, seed=6)
pre = torch.rand((M, HC), device="cuda").float() + 0.5
t0 = time.time()
with autotune(cache_path=None):
    mhc_cuda.mhc_fused_hc(*inputs, *ARGS, norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre)
torch.cuda.synchronize()
print(f"[tune all-buckets] {time.time() - t0:.1f}s, {len(lines)} lines", flush=True)
p_all = parse(lines)
prod = {}
for Mq in MS:
    sl = slice(0, Mq)
    tac = mhc_cuda.mhc_fused_hc_tactic(inputs[0][sl], inputs[1][sl], inputs[2][sl], inputs[3][sl], inputs[4],
                                       inputs[5], inputs[6], *ARGS, norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre[sl])
    b = str(mhc_cuda._mhc_map_to_tuning_bucket(Mq))
    prod[str(Mq)] = {"bucket": int(b), "tactic": list(tac),
                     "cands_20_72": {k: v for k, v in p_all.get(b, {}).get("cands", {}).items()
                                     if k.startswith(("id20_", "id72_"))}}
    print(f"[prod-like] M={Mq} bucket={b} pick={tuple(tac)} {prod[str(Mq)]['cands_20_72']}", flush=True)
report["prod_like"] = prod
report["prod_like_buckets_parsed"] = len(p_all)
ok72 = direct.get("72", {}).get("ok") and direct.get("72", {}).get("bitwise_vs_ks1")
print(json.dumps({"passed": bool(ok72), "report": report})[:60000])
