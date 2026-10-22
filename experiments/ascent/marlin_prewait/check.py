# Idea 28 stage 1 untimed correctness check (candidate tree).
# 1. Bitwise, eager: shapes qkvz (16384,5120), qkv (14336,5120), out_proj (5120,6144) x rows 4/8/12/16 x
#    {default launch = TRTLLM_MARLIN_FULLK=0 + TRTLLM_MARLIN_INPROJ_FULLK=0, full_k launches = both 1}:
#    prewait (default) vs kill switch TRTLLM_MARLIN_PREWAIT_B=0 (main's wait-first prologue). Hashes are
#    printed so they can be compared with the base tree's bench hashes.
# 2. Repeated execution: the bench's graph chain (gemma add-RMSNorm -> Marlin -> gemma add-RMSNorm, C cold-L2
#    copies, one in-graph state restore) with the same warmup (10) and iteration (60) counts and the same reset
#    point (state copy at the start of every replay). Every output NaN-poisoned before each replay; each replay
#    compared bitwise with replay 1 and with the kill-switch eager reference; weights unchanged at the end.
# 3. PDL-live: a Triton stand-in producer launched with PDL triggers its dependents immediately, spins
#    ~DELAY_US, then writes Marlin's input A (which a plain fill kernel NaN-poisoned before it). Marlin with
#    pre-wait B must equal the reference on every replay. Negative control TRTLLM_MARLIN_DEBUG_SKIP_WAIT=1
#    (Marlin's griddepcontrol.wait removed) must FAIL (non-finite / mismatching output), proving the check
#    is live.
# 4. Advisory FP32 reference error (dequantized weights, BF16 activations) per shape/row.
# Rows 24/32 (qkv, out_proj only; workspace_mode 2 = FP32 split-K reduce through C_tmp) go through 1-3 as well.
# At the end the persistent lock workspace must read all zero; inputs are hashed before and after each case.
import hashlib
import json
import os
import subprocess
import time

os.environ["TRTLLM_ENABLE_PDL"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch.modules.linear import _pack_nvfp4_weight_for_marlin  # noqa: E402

dev = torch.device("cuda")
EPS = 1e-6
WARM, ITERS = 10, 60
SHAPES = [("qkvz", 16384, 5120, 6), ("qkv", 14336, 5120, 7), ("outproj", 5120, 6144, 16)]
MS = [4, 8, 12, 16]
# Rows 17-32 run Marlin only on qkv and out_proj/o_proj (main 7f09143a, per-(N,K) caps), with workspace_mode 2
# (split-K partials reduced in FP32 through a per-call C_tmp), as FP8QDQLinearMethod.apply dispatches them.
MS_FP32_REDUCE = [24, 32]


def rows_for(sname):
    return MS + (MS_FP32_REDUCE if sname in ("qkv", "outproj") else [])


def mode_for(m):
    return 2 if m > 16 else 0
DELAY_US = 30
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def tree():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           timeout=20).stdout.strip().splitlines()
        log(f"TREE_COMMIT={h} dirty_files={len(d)}")
        return h
    except Exception as e:  # noqa: BLE001
        return f"? {e!r}"


def sha(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def env(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def arm_env(prewait, fullk):
    # fullk=True: the served launches (full_k out_proj / in_proj at rows <= 8, kM16Table at rows 9-16).
    # fullk=False: the default split-K launch at every row count.
    env(TRTLLM_MARLIN_PREWAIT_B="1" if prewait else "0", TRTLLM_MARLIN_FULLK="1" if fullk else "0",
        TRTLLM_MARLIN_INPROJ_FULLK="1" if fullk else "0", TRTLLM_MARLIN_M16_TABLE="1" if fullk else "0",
        TRTLLM_MARLIN_DEBUG_SKIP_WAIT=None, TRTLLM_MARLIN_M16_CFG=None)


def build(n, k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    gs = (448.0 * 6.0 / w.float().abs().amax()).float().reshape(1)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, True)
    packed = _pack_nvfp4_weight_for_marlin(w_fp4, w_sf, (1.0 / gs).float(), n, k, 16)
    return packed, (w_fp4, w_sf, gs)


def marlin(x, wt, n, k):
    return torch.ops.trtllm.marlin_nvfp4_gemm(x, wt[0], scale_a=None, scale_b=wt[1], alpha=None,
                                              weight_global_scale=wt[2], bias=None, out_dtype=torch.bfloat16,
                                              size_n=n, size_k=k, workspace_mode=mode_for(x.shape[0]))


def norm(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm(x, r, w, EPS)


def dequant(qv, sf, gs, rows, k):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev,
                       dtype=torch.float32)
    qv = qv.view(torch.uint8)
    lo = lut[(qv & 0xF).long()]
    hi = lut[(qv >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(rows, k)
    ncol = k // 16
    ntk = (ncol + 3) // 4
    r = torch.arange(rows, device=dev).view(-1, 1)
    c = torch.arange(ncol, device=dev).view(1, -1)
    idx = ((r // 128) * ntk + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    s = sf.view(torch.uint8).view(-1)[idx].view(torch.float8_e4m3fn).float()
    return v * s.repeat_interleave(16, dim=1) / gs


def norm_early(x, r, w):
    torch.ops.trtllm.flashinfer_gemma_fused_add_rmsnorm_early_trigger(x, r, w, EPS)


class Chain:
    def __init__(self, n, k, m, ws, wn1, wn2, seed, early=False):
        self.n, self.k, self.m, self.ws, self.wn1, self.wn2 = n, k, m, ws, wn1, wn2
        self.early = early
        c = len(ws)
        per = m * k * 2 + m * n
        g = torch.Generator(device=dev).manual_seed(seed)
        st0 = torch.zeros(c * per, device=dev, dtype=torch.bfloat16)
        for i in range(c):
            st0[i * per: i * per + 2 * m * k] = (torch.randn(2 * m * k, generator=g, device=dev) * 0.5).to(
                torch.bfloat16)
        self.st0 = st0
        self.st = st0.clone()
        self.x, self.r1, self.r2 = [], [], []
        for i in range(c):
            b = i * per
            self.x.append(self.st[b: b + m * k].view(m, k))
            self.r1.append(self.st[b + m * k: b + 2 * m * k].view(m, k))
            self.r2.append(self.st[b + 2 * m * k: b + per].view(m, n))
        self.ys = None

    def run(self):
        self.st.copy_(self.st0)
        ys = []
        for i in range(len(self.ws)):
            (norm_early if self.early else norm)(self.x[i], self.r1[i], self.wn1)
            y = marlin(self.x[i], self.ws[i], self.n, self.k)
            norm(y, self.r2[i], self.wn2)
            ys.append(y)
        return ys

    def state(self, ys):
        return torch.cat([self.st.clone()] + [y.reshape(-1).clone() for y in ys])

    def poison(self, ys):
        self.st.fill_(float("nan"))
        for y in ys:
            y.fill_(float("nan"))


def capture(fn):
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = fn()
    torch.cuda.synchronize()
    return gr, o


@triton.jit
def _globaltimer(dummy):
    # Same form as triton.language.extra.cuda.globaltimer().
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                     pack=1)


@triton.jit
def _standin_producer(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    # Wait for the poisoning fill, then let dependents launch at once, then spin and only then write A.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    d = pid.to(tl.int64)
    t0 = _globaltimer(d)
    t = t0
    while t - t0 < delay_ns:
        t = _globaltimer(d)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, v, mask=mask)


def producer(src, dst):
    # Few CTAs (<= 24 at 32 x 6144) so most SMs stay free for Marlin's CTAs: with a grid that covers every SM
    # (e.g. 192 CTAs of 1024) a full-SM Marlin CTA cannot launch until the producer exits, and the negative
    # control could not race.
    n = src.numel()
    BLOCK = 8192
    _standin_producer[(triton.cdiv(n, BLOCK),)](src, dst, n, DELAY_US * 1000, BLOCK=BLOCK, num_warps=8,
                                                 launch_pdl=True)


def main():
    head = tree()
    failures = []
    report = {"bitwise": {}, "replays": {}, "pdl_live": {}, "ref": {}, "hashes": {}}
    for sname, n, k, c in SHAPES:
        base, raw = build(n, k, 11 + n + k)
        ws = [base] + [tuple(t.clone() for t in base) for _ in range(c - 1)]
        wsha = sha([t for w in ws for t in w])
        g = torch.Generator(device=dev).manual_seed(5)
        wn1 = (0.1 * torch.randn(k, generator=g, device=dev)).to(torch.bfloat16)
        wn2 = (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)
        wref = dequant(raw[0], raw[1], raw[2], n, k)
        for m in rows_for(sname):
            key = f"{sname}_M{m}"
            gx = torch.Generator(device=dev).manual_seed(7 + m)
            x = (torch.randn(m, k, generator=gx, device=dev) * 0.5).to(torch.bfloat16)
            xsha = sha([x])
            # 1. eager bitwise, both launch families
            rec = {}
            for fullk in (True, False):
                outs = {}
                for prewait in (True, False):
                    arm_env(prewait, fullk)
                    outs[prewait] = [marlin(x, w, n, k).clone() for w in ws[:2]] + [marlin(x, ws[0], n, k).clone()]
                torch.cuda.synchronize()
                eq = all(torch.equal(a, b) for a, b in zip(outs[True], outs[False]))
                det = all(torch.equal(o, outs[True][0]) for o in outs[True] + outs[False])
                fam = "served_fullk_or_m16table" if fullk else "default_splitk"
                rec[fam] = {"prewait_eq_kill": eq, "deterministic": det, "hash": sha([outs[True][0]]),
                            "hash_kill": sha([outs[False][0]])}
                report["hashes"][f"{key}_{fam}"] = sha([outs[True][0]])
                if not (eq and det):
                    failures.append(f"bitwise {key} {fam} eq={eq} det={det}")
                if fullk:
                    ref = x.float() @ wref.t()
                    err = (outs[True][0].float() - ref).abs().max().item()
                    report["ref"][key] = {"max_abs_vs_fp32": round(err, 5),
                                          "ref_mean_abs": round(ref.abs().mean().item(), 5)}
            report["bitwise"][key] = rec
            # 2. repeated execution (default config = full_k on, as served), prewait vs kill graphs
            arm_env(False, True)
            che = Chain(n, k, m, ws, wn1, wn2, 100 + m)
            eref = che.state(che.run())
            torch.cuda.synchronize()
            rr = {}
            for prewait, early in ((True, False), (False, False), (True, True)):
                arm_env(prewait, True)
                ch = Chain(n, k, m, ws, wn1, wn2, 100 + m, early=early)
                gr, ys = capture(ch.run)
                ch.ys = ys
                first, ok1, okr, worst = None, True, True, 0.0
                for it in range(WARM + ITERS):
                    ch.poison(ys)
                    gr.replay()
                    torch.cuda.synchronize()
                    out = ch.state(ys)
                    if first is None:
                        first = out
                    e1 = torch.equal(out, first)
                    er = torch.equal(out, eref)
                    ok1, okr = ok1 and e1, okr and er
                    if not er:
                        d = torch.nan_to_num((out.float() - eref.float()).abs(), nan=float("inf")).max().item()
                        worst = max(worst, d)
                arm = ("prewait" if prewait else "kill") + ("+early_trigger_norm" if early else "")
                rr[arm] = {"replays": WARM + ITERS, "eq_replay1": ok1, "eq_kill_eager": okr, "max_dev": worst}
                if not (ok1 and okr):
                    failures.append(f"replay {key} {arm} eq1={ok1} eqref={okr} dev={worst}")
                del gr, ys, ch
            report["replays"][key] = rr
            # 3. PDL-live stand-in producer, eager and graph
            arm_env(False, True)
            ref_out = marlin(x, ws[0], n, k).clone()
            a_buf = torch.empty_like(x)
            live = {}
            for label, skip in (("prewait", False), ("negative_control_skip_wait", True)):
                arm_env(True, True)
                if skip:
                    os.environ["TRTLLM_MARLIN_DEBUG_SKIP_WAIT"] = "1"

                def fn():
                    a_buf.fill_(float("nan"))
                    producer(x, a_buf)
                    return marlin(a_buf, ws[0], n, k)

                bad_e = 0
                for _ in range(5):
                    o = fn()
                    torch.cuda.synchronize()
                    bad_e += int(not torch.equal(o, ref_out))
                gr, o = capture(fn)
                bad_g = 0
                for _ in range(20):
                    o.fill_(float("nan"))
                    gr.replay()
                    torch.cuda.synchronize()
                    bad_g += int(not torch.equal(o, ref_out))
                nonfinite = int((~torch.isfinite(o.float())).sum().item())
                live[label] = {"eager_bad": bad_e, "graph_bad": bad_g, "last_nonfinite": nonfinite}
                del gr, o
                os.environ.pop("TRTLLM_MARLIN_DEBUG_SKIP_WAIT", None)
            live["live_check_passes"] = live["prewait"]["eager_bad"] == 0 and live["prewait"]["graph_bad"] == 0
            live["negative_control_fails"] = (live["negative_control_skip_wait"]["eager_bad"]
                                              + live["negative_control_skip_wait"]["graph_bad"]) > 0
            if not live["live_check_passes"]:
                failures.append(f"pdl_live {key} prewait")
            if not live["negative_control_fails"]:
                failures.append(f"pdl_live {key} negative control did not fail")
            report["pdl_live"][key] = live
            if sha([x]) != xsha:
                failures.append(f"input x changed {key}")
            log(f"{key}: bitwise {json.dumps(rec)} | replays {json.dumps(rr)} | live {json.dumps(live)} | "
                f"ref {json.dumps(report['ref'][key])}")
        if sha([t for w in ws for t in w]) != wsha:
            failures.append(f"weights changed {sname}")
        del ws, wref
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    snap = torch.ops.trtllm.marlin_nvfp4_persistent_workspace_snapshot(torch.empty(1, device=dev))
    locks_zero = bool(snap.numel() > 0 and (snap == 0).all().item())
    report["persistent_lock_workspace"] = {"ints": int(snap.numel()), "all_zero": locks_zero}
    if not locks_zero:
        failures.append("persistent lock workspace not all zero")
    report["passed"] = not failures
    report["failures"] = failures[:30]
    report["n_failures"] = len(failures)
    report["tree"] = head
    report["compared_against"] = ("kill switch TRTLLM_MARLIN_PREWAIT_B=0 (main's wait-first prologue, same "
                                  "launch config) bitwise, eager and per poisoned graph replay; hashes for "
                                  "comparison with the base tree; FP32 dequantized reference advisory")
    print(json.dumps(report))


main()
