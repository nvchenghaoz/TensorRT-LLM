"""S1 correctness job for the spill-free fused_pmap_gemm_fma_mtile_v2 tactics (idea 212, branch 148).

One GPU, synthetic tensors (DeepSeek-V4.1-Flash: hc=4, hidden=5120, N=24, K=20480). Sections:
  A. bitwise: every fused_half_fma ks=1 tactic the runner offers (old mtile ids + new mtile_v2 ids
     tile_m = 16 + TM) vs the ks=1 ksplit reference tactic (fused_half_fma, 24, 1, bs, 1) on all
     outputs (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur, and pre_mix_out in V4.1-lagged
     mode), at M = 33..8192 incl. tail M, V4 and V4.1-lagged. Outputs and workspaces are NaN-poisoned
     before every call; 3 warmup + ITERS calls, each compared. Inputs are checked unchanged.
     residual_cur also vs an fp64 torch reference (rtol/atol 1.6e-2).
  B. M=128 torch CUDA-graph capture of each new tactic, >= 10 replays, outputs NaN-poisoned before
     each replay, each replay bitwise vs eager.
  C. step0.py (cuobjdump REG/STACK/LOCAL + SASS LDL/STL of every mtile/mtile_v2 instantiation, and the
     autotuner's pick at M=128/2048/4096/8192, V4 and V4.1-lagged).
Last stdout line: {"passed": ...} JSON.
"""
import json
import os
import subprocess
import sys

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modules.mhc import mhc_cuda

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mhc_fhc_bench import make_inputs, ref_residual  # noqa: E402

sys.path.insert(0, HERE)
from guard import dirty_guard  # noqa: E402

dirty_guard()
# Which tactic ids count as "new" (graph-replay section B) and which built kernels must be spill-free
# (section C). Defaults: idea 212's mtile_v2 (tile_m >= 16). Idea 292: NEW_ID_MIN=64 SPILL_KEY=mtile_v3.
NEW_ID_MIN = int(os.environ.get("NEW_ID_MIN", str(mhc_cuda._FUSED_HC_MT2_ID_BASE)))
SPILL_KEY = os.environ.get("SPILL_KEY", "mtile_v2")

HC, HID = 4, 5120
N = HC * (2 + HC)
EPS = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20)
ITERS = int(os.environ.get("ITERS", "5"))
MS = [int(x) for x in os.environ.get("MS", "33,64,127,128,129,255,512,1000,1024,2047,2048,4096,4099,8192").split(",")]
BACKEND = mhc_cuda._FUSED_HC_BACKEND_CODE["fused_half_fma"]


class Op:
    """Calls torch.ops.trtllm.mhc_fused_hc with caller-owned (poisonable) outputs."""

    def __init__(self, inputs, nw, pre_ext):
        self.inputs = inputs
        self.nw = nw
        self.pre_ext = pre_ext
        M = inputs[0].shape[0]
        dev = inputs[0].device
        self.res = torch.empty_like(inputs[1])
        self.pm = torch.empty((M, HC), dtype=torch.float32, device=dev)
        self.cm = torch.empty((M, HC * HC), dtype=torch.float32, device=dev)
        self.li = torch.empty_like(inputs[0])
        self.pmo = torch.empty((M, HC), dtype=torch.float32, device=dev) if pre_ext is not None else None
        self.yws = torch.empty((M, N), dtype=torch.float32, device=dev)
        self.rws = torch.empty((M,), dtype=torch.float32, device=dev)
        self.dws = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.M = M

    def outs(self):
        o = [self.res, self.pm, self.cm, self.li]
        if self.pmo is not None:
            o.append(self.pmo)
        return o

    def poison(self):
        for t in self.outs() + [self.yws, self.rws]:
            t.fill_(float("nan"))

    def __call__(self, tactic):
        backend, tn, ks, bs, tm = tactic
        x, r, pm, cm, w, sc, base = self.inputs
        mhc_cuda._fused_hc_call(BACKEND, tn, ks, bs, tm, x, r, pm, cm, w, sc, base, self.res, self.pm, self.cm,
                                self.li, self.yws, self.rws, self.dws, self.M, HID, HC, EPS["rms_eps"],
                                EPS["hc_pre_eps"], EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"],
                                EPS["sinkhorn_repeat"], norm_weight=self.nw, norm_eps=1e-6,
                                pre_mix_ext=self.pre_ext, pre_mix_out=self.pmo)


def same(a, b):
    return a.shape == b.shape and a.stride() == b.stride() and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


rn = mhc_cuda.MhcFusedHcRunner(HC, HID, 1e-6, 1e-6, 1e-6, 2.0, 20)
allpass = True
report = {"A": {}, "B": {}}
new_ids_seen = set()

# ------------------------------------------------------------------ A. bitwise
for M in MS:
    inputs, nw = make_inputs(M, seed=7)
    snap = [t.clone() for t in inputs] + [nw.clone()]
    g = torch.Generator(device="cuda").manual_seed(99 + M)
    pre_ext = torch.rand((M, HC), generator=g, device="cuda").float() + 0.5
    bs = 512 if M < 512 else 256
    ref_t = ("fused_half_fma", 24, 1, bs, 1)
    tacs = [t for t in rn.get_valid_tactics(inputs, None) if t[0] == "fused_half_fma" and t[2] == 1 and t != ref_t]
    new = [t for t in tacs if t[4] >= NEW_ID_MIN]
    new_ids_seen.update(new)
    rep = {"n_tactics": len(tacs), "new": [list(t) for t in new]}
    if M in (128, 2048):
        rep["all_offered"] = [list(t) for t in rn.get_valid_tactics(inputs, None)]
    for mode, pe in (("v4", None), ("v41_lagged", pre_ext)):
        op = Op(inputs, nw, pe)
        op.poison()
        op(ref_t)
        torch.cuda.synchronize()
        ref = [o.clone() for o in op.outs()]
        finite = all(torch.isfinite(o).all().item() for o in ref)
        nr = ref_residual(inputs)
        rerr = (ref[0].double() - nr).abs().max().item()
        tol_ok = torch.allclose(ref[0].double(), nr, rtol=1.6e-2, atol=1.6e-2)
        mism = {}
        for t in tacs:
            bad = 0
            for it in range(3 + ITERS):
                op.poison()
                op(t)
                torch.cuda.synchronize()
                if not all(same(a, b) for a, b in zip(op.outs(), ref)):
                    bad += 1
            if bad:
                mism[str(t)] = bad
        ok = finite and tol_ok and not mism
        allpass &= ok
        rep[mode] = {"ok": ok, "mismatch": mism, "finite": finite, "resid_fp64_maxabs": rerr, "resid_tol_ok": tol_ok,
                     "n_outputs": len(ref), "calls_per_tactic": 3 + ITERS}
    unchanged = all(torch.equal(a, b) for a, b in zip(inputs + [nw], snap))
    allpass &= unchanged
    rep["inputs_unchanged"] = unchanged
    report["A"][str(M)] = rep
    print(f"[A] M={M}: {json.dumps(rep)}", flush=True)
    del inputs, snap
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ B. M=128 graph replay
M = 128
inputs, nw = make_inputs(M, seed=11)
pre_ext = torch.rand((M, HC), device="cuda").float() + 0.5
new128 = [t for t in rn.get_valid_tactics(inputs, None) if t[0] == "fused_half_fma" and t[4] >= NEW_ID_MIN]
keep = []  # keep every captured graph's tensors alive (pitfall 2156)
for t in new128:
    for mode, pe in (("v4", None), ("v41_lagged", pre_ext)):
        op = Op(inputs, nw, pe)
        op.poison()
        op(t)
        torch.cuda.synchronize()
        eager = [o.clone() for o in op.outs()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            op(t)  # warm on the side stream
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(t)
        keep.append((graph, op))
        bad = 0
        n_rep = 12
        for _ in range(n_rep):
            op.poison()
            graph.replay()
            torch.cuda.synchronize()
            if not all(same(a, b) for a, b in zip(op.outs(), eager)):
                bad += 1
        ok = bad == 0
        allpass &= ok
        report["B"][f"{t}_{mode}"] = {"replays": n_rep, "bad": bad}
        print(f"[B] M=128 {t} {mode}: replays={n_rep} bad={bad}", flush=True)
report["B"]["n_new_tactics"] = len(new128)
allpass &= len(new128) > 0

# ------------------------------------------------------------------ C. step0 (regs, LDL/STL, tuner picks)
r = subprocess.run([sys.executable, os.path.join(HERE, "..", "mhc_mtile_step0", "step0.py")], capture_output=True,
                   text=True, timeout=2400)
print(r.stdout[-200000:], flush=True)
if r.returncode != 0:
    print(r.stderr[-6000:], flush=True)
s0 = {}
for line in reversed(r.stdout.splitlines()):
    if line.startswith("{"):
        try:
            s0 = json.loads(line)
            break
        except Exception:
            pass
v2rows = {k: v for k, v in s0.get("built", {}).items() if SPILL_KEY in k}
spill_free = bool(v2rows) and all(v["STACK"] == 0 and v["LOCAL"] == 0 and v["LDL"] == 0 and v["STL"] == 0
                                  for v in v2rows.values())
allpass &= spill_free
print(json.dumps({"passed": bool(allpass), "new_tactics": sorted(str(t) for t in new_ids_seen),
                  "v2_spill_free": spill_free, "v2_built": v2rows, "tuner": s0.get("chosen"),
                  "graph_m128": report["B"],
                  "bitwise_summary": {m: {k: v["ok"] for k, v in rep.items() if isinstance(v, dict) and "ok" in v}
                                      for m, rep in report["A"].items()},
                  "compared": "all fused_half_fma ks=1 tactics (old mtile + new mtile_v2) vs (fused_half_fma,24,1,bs,1)"
                              " ksplit reference, all outputs bitwise, NaN-poisoned before every call"}))
