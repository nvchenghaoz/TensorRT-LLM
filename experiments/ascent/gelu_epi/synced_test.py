"""Branch 23 on main with landing 11 (M-chunked chain): module-level checks (no model load).

Builds real _Gemma4GeluQuantMLP modules (NVFP4 QuantConfig) from synthetic checkpoint-format
weights (3 copies), loads them through Linear.load_weights and the loader's post_load_weights
walk, once with TRTLLM_GEMMA4_GATEUP_GELU_EPI=1 (gate_up rows permuted) and once with =0
(main's forward). Every forward's down_proj input (FP4 payload + full SF) and output are
captured. Every uint8 torch.empty in the chain modules (fused op, chunked chain, both act
kernels) is 0xA5-filled, so every SF padding byte must be written by the kernels.

References are main's own functions on the UNPERMUTED weights:
  ref_unchunked = trtllm::nvfp4_gemm (tuned) -> gelu_tanh_mul_fp4_quant -> down_proj
  ref_chunked   = gate_up_gelu_fp4_chunked (main's act) when main's chunk band applies, else
                  ref_unchunked  (= what main's forward runs)

Rebuild on main 4f541f15 (landing 22, TRTLLM_NVFP4_DENSE_TUNING): nvfp4_gemm runs the dense-bucket
pick (op key trtllm::nvfp4_gemm::gemm_dense) for gate_up at M in the dense range. A forced tactic
overwrites BOTH the standard and (when present) the dense entry for that M, so the forced tactic
is the one main runs on every path. Run once per TRTLLM_NVFP4_DENSE_TUNING value (env is read at
import), see checks.py.

Sections
  item5: path taken per call ('fused' / 'chunked' / 'unchunked'); calls of the UNPERMUTED act
         kernel made by EPI=1 forwards (must be 0) and of the permuted act by EPI=0 forwards (0).
  item6: forced gate_up tactic (tuner cache entry overwritten) ('cutlass',0), ('cutlass',1),
         ('cublaslt',0) and the natural-pick tactic ('cutlass',2) at M=2000 and 8000: EPI=1
         forward == ref_chunked bytes; expected paths chunked/chunked/unchunked/fused. A,B,A.
  item7: kill-switch matrix at M 300/1000/2000/8000 (natural picks, plus forced ('cutlass',0)):
         EPI=0 == ref_chunked; EPI=0 + chunk off (modeling_gemma4._MLP_CHUNK_ENABLED=False, the
         value TRTLLM_GEMMA4_MLP_CHUNK=0 sets at import) == ref_unchunked; EPI=1 likewise.
  item8: repeated execution in bench.py's M order over 1664-8192 (2000, 4000, 4096, 4800, 8000,
         8100, 8192), 5 warmup + 30 calls per M, call i on weight copy i % 3, passes A, B (reversed), A2
         = 105 calls per M, chunk band live; with natural picks and with ('cutlass',0) forced
         (path b). Every call == its (M, copy) call 1 and == ref_chunked; inputs unchanged.
usage: synced_test.py <checkpoint>
"""
import json
import os
import sys
import threading
import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(45)
        log("heartbeat")


GEMM_OP = "trtllm::nvfp4_gemm::gemm"
GEMM_DENSE_OP = "trtllm::nvfp4_gemm::gemm_dense"
MS_TUNE = [1, 16, 64, 300, 1000, 1200, 2000, 4000, 4800, 8000, 8100, 8192]
MS_KILL = [300, 1000, 2000, 8000]
MS_FORCE = [2000, 8000]
MS_REP = [2000, 4000, 4096, 4800, 8000, 8100, 8192]  # bench.py order, 1664 <= M <= 8192
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import FLUSH, ITERS, WARMUP  # noqa: E402  (the timed run's counts, not re-declared)
FORCED = [("cutlass", 0), ("cutlass", 1), ("cublaslt", 0), ("cutlass", 2)]
EXPECT_PATH = {("cutlass", 0): "chunked", ("cutlass", 1): "chunked", ("cublaslt", 0): "unchunked",
               ("cutlass", 2): "fused"}


def main():
    threading.Thread(target=_hb, daemon=True).start()
    ck = sys.argv[1]
    import torch
    cfg = json.load(open(os.path.join(ck, "config.json")))
    tc = cfg.get("text_config", cfg)
    H, I = int(tc["hidden_size"]), int(tc["intermediate_size"])
    log(f"hidden={H} intermediate={I} TRTLLM_NVFP4_DENSE_TUNING={os.environ.get('TRTLLM_NVFP4_DENSE_TUNING')}")
    print(f"[protocol] bench WARMUP={WARMUP} ITERS={ITERS} flush={FLUSH} | synced_test item8 calls/key/pass="
          f"{WARMUP + ITERS} passes=A,B,A2 flush={FLUSH} order(A)={MS_REP}", flush=True)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner
    from tensorrt_llm._torch.kernels.fused_ops import gate_up_gelu_fp4_chunked as gcm
    from tensorrt_llm._torch.kernels.fused_ops import gelu_tanh_mul_fp4_quant as gqm
    from tensorrt_llm._torch.kernels.fused_ops import nvfp4_gelu_gemm_sm120 as ge
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_gemma4 as mg
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    dev = torch.device("cuda")
    log(f"chunk cfg enabled={mg._MLP_CHUNK_ENABLED} rows={mg._MLP_CHUNK_ROWS} "
        f"min_m={mg._MLP_CHUNK_MIN_M} max_m={mg._MLP_CHUNK_MAX_M}")
    dense_range = tco._nvfp4_dense_range(2 * I, H)
    log(f"dense tuning {tco.NVFP4_DENSE_TUNING}, gate_up dense range {dense_range}")
    assert mg._MLP_CHUNK_ENABLED
    assert ge.get_gelu_gemm_module() is not None, ge._module_error
    ge._workspace(dev)  # the fused op's workspace, allocated as before (not 0xA5-filled)

    # ---- 0xA5 prefill of every uint8 torch.empty in the chain modules -------------------------
    class A5Torch:
        def __init__(self, t):
            self._t = t

        def __getattr__(self, n):
            return getattr(self._t, n)

        def empty(self, *a, **k):
            x = self._t.empty(*a, **k)
            if x.dtype == self._t.uint8:
                x.fill_(0xA5)
            return x

    for mod in (ge, gcm, gqm):
        mod.torch = A5Torch(torch)

    # ---- act-kernel call counters (every module global the forward resolves at call time) -------
    cnt = {"unperm_act": 0, "perm_act": 0, "main_chunked_fn": 0}
    _unperm, _perm = gqm.gelu_tanh_mul_fp4_quant, ge.gelu_tanh_mul_fp4_quant_permuted
    _chunked = gcm.gate_up_gelu_fp4_chunked

    def unperm_w(*a, **k):
        cnt["unperm_act"] += 1
        return _unperm(*a, **k)

    def perm_w(*a, **k):
        cnt["perm_act"] += 1
        return _perm(*a, **k)

    def chunked_w(*a, **k):
        cnt["main_chunked_fn"] += 1
        return _chunked(*a, **k)

    gcm.gelu_tanh_mul_fp4_quant = unperm_w  # chunk loop's default act
    mg.gelu_tanh_mul_fp4_quant = unperm_w  # _apply_activation
    ge.gelu_tanh_mul_fp4_quant_permuted = perm_w  # paths (b) and (c) on permuted rows
    gcm.gate_up_gelu_fp4_chunked = chunked_w  # resolved by the permuted dispatch
    mg.gate_up_gelu_fp4_chunked = chunked_w  # resolved by main's _chunked_gate_up_act

    # ---- synthetic checkpoint-format weights, 3 copies -----------------------------------------
    g = torch.Generator(device=dev)
    g.manual_seed(2323)

    def q_ckpt(w, amax):
        gs = (448 * 6 / amax).reshape(1).float()
        fp4, sf = torch.ops.trtllm.fp4_quantize(w, gs, 16, False, False)
        return (fp4.view(torch.uint8).reshape(w.shape[0], w.shape[1] // 2),
                sf.view(torch.uint8).reshape(-1)[: w.shape[0] * (w.shape[1] // 16)]
                .view(torch.float8_e4m3fn).reshape(w.shape[0], w.shape[1] // 16))

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    mc = ModelConfig(quant_config=qc)
    in_amax = torch.tensor(24.0, device=dev)
    copies = []
    for c in range(3):
        wg = torch.randn(I, H, device=dev, generator=g).to(torch.bfloat16)
        wu = torch.randn(I, H, device=dev, generator=g).to(torch.bfloat16)
        wd = (torch.randn(H, I, device=dev, generator=g) * 0.05).to(torch.bfloat16)
        a_gu = torch.maximum(wg.float().abs().max(), wu.float().abs().max())
        a_d = wd.float().abs().max()
        gf, gsf = q_ckpt(wg, a_gu)
        uf, usf = q_ckpt(wu, a_gu)
        df, dsf = q_ckpt(wd, a_d)
        del wg, wu, wd
        ws2 = (a_gu / (448 * 6)).float()
        copies.append(dict(gate=dict(weight=gf, weight_scale=gsf, weight_scale_2=ws2.clone(),
                                     input_scale=(in_amax / (448 * 6)).float()),
                           up=dict(weight=uf, weight_scale=usf, weight_scale_2=ws2.clone(),
                                   input_scale=(in_amax / (448 * 6)).float()),
                           down=dict(weight=df, weight_scale=dsf, weight_scale_2=(a_d / (448 * 6)).float())))
    torch.cuda.empty_cache()

    def build(c, epi, down_in):
        os.environ["TRTLLM_GEMMA4_GATEUP_GELU_EPI"] = "1" if epi else "0"
        with torch.device("cuda"):
            m = mg._Gemma4GeluQuantMLP(hidden_size=H, intermediate_size=I, bias=False,
                                       activation=mg.gelu_tanh, dtype=torch.bfloat16, config=mc,
                                       layer_idx=0)
        cc = copies[c]
        m.gate_up_proj.load_weights([cc["gate"], cc["up"]])
        d = dict(cc["down"])
        d["input_scale"] = down_in
        m.down_proj.load_weights([d])
        for sub in m.modules():  # ModelLoader's post_load_weights walk (parent first)
            f = getattr(sub, "post_load_weights", None)
            if f is not None and not getattr(sub, "_weights_removed", False):
                f()
        os.environ["TRTLLM_GEMMA4_GATEUP_GELU_EPI"] = "1"
        return m

    # down_proj static input scale from the act amax at M=512 on copy 0 (unpermuted chain)
    probe = build(0, False, torch.tensor(1.0 / 400.0, device=dev))
    a = torch.randn(512, H, device=dev, generator=g).to(torch.bfloat16)
    pf, psf = torch.ops.trtllm.fp4_quantize(a, probe.gate_up_proj.input_scale, 16, False)
    y = torch.ops.trtllm.nvfp4_gemm(pf, probe.gate_up_proj.weight, psf, probe.gate_up_proj.weight_scale,
                                    probe.gate_up_proj.alpha, torch.bfloat16)
    act_amax = (torch.nn.functional.gelu(y[:, :I].float(), approximate="tanh") * y[:, I:].float()).abs().max()
    down_in = (act_amax / (448 * 6)).float().reshape(())
    log(f"gate_up alpha={probe.gate_up_proj.alpha.item():.3e} act amax={act_amax.item():.3f}")
    del probe, y, a
    torch.cuda.empty_cache()

    on = [build(c, True, down_in.clone()) for c in range(3)]
    off = [build(c, False, down_in.clone()) for c in range(3)]
    perm_ok = all(m._gelu_epi_permuted for m in on) and not any(m._gelu_epi_permuted for m in off)
    w_ok = True
    for mo, mf in zip(on, off):
        wp, wsp = ge.permute_gate_up_rows(mf.gate_up_proj.weight.data, mf.gate_up_proj.weight_scale.data, H)
        w_ok &= torch.equal(wp.view(torch.uint8), mo.gate_up_proj.weight.data.view(torch.uint8))
        w_ok &= torch.equal(wsp.view(torch.uint8), mo.gate_up_proj.weight_scale.data.view(torch.uint8))
        w_ok &= torch.equal(mo.down_proj.weight.data.view(torch.uint8), mf.down_proj.weight.data.view(torch.uint8))
        del wp, wsp
    w_ok = bool(w_ok)
    backends = list(on[0].gate_up_proj.nvfp4_allowed_backends)
    log(f"modules built: EPI=1 permuted / EPI=0 not: {perm_ok}; on weights == permute(off) {w_ok}; "
        f"gate_up backends {backends}")

    # ---- capture down_proj inputs/outputs ---------------------------------------------------------
    CAP = []

    def wrap(m):
        dp = m.down_proj
        orig = dp.forward

        def fwd(inp, *a, **k):
            out = orig(inp, *a, **k)
            CAP.append((inp.fp4_tensor, inp.scaling_factor, out))
            return out
        dp.forward = fwd

    for m in on + off:
        wrap(m)

    # ---- inputs (gate_up input as the fused norm+quant producer hands it over) -----------------------
    acts = {}
    for M in sorted(set(MS_TUNE + MS_KILL + MS_REP)):
        a = torch.randn(M, H, device=dev, generator=g).to(torch.bfloat16)
        fp4, sf = torch.ops.trtllm.fp4_quantize(a, on[0].gate_up_proj.input_scale, 16, False)
        acts[M] = (fp4, sf)
    snap = {M: (v[0].clone(), v[1].clone()) for M, v in acts.items()}

    def X(M):
        return Fp4QuantizedTensor(acts[M][0], acts[M][1])

    # ---- engine-like in-process tuning (EPI=0 module, i.e. what main tunes) --------------------------
    log("autotune")
    with torch.inference_mode():
        with autotune():
            for M in MS_TUNE:
                off[0](X(M))
    torch.cuda.synchronize()
    tuner = AutoTuner.get()
    cache = tuner.profiling_cache
    runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, backends)
    tcfg = NVFP4GemmUnifiedRunner.tuning_config
    gp0 = on[0].gate_up_proj

    def shapes_of(M):
        fp4, sf = acts[M]
        return tuple(tuner._get_input_sizes([fp4, gp0.weight, sf, gp0.weight_scale, gp0.alpha]))

    def gu_key(M):
        return cache.get_cache_key(GEMM_OP, runner, shapes_of(M), tcfg, True)

    def gu_dense_key(M):
        if dense_range is None or not (dense_range[0] <= M <= dense_range[1]):
            return None
        dr = NVFP4GemmUnifiedRunner(0, torch.bfloat16, backends, dense_min_m=dense_range[0])
        k = cache.get_cache_key(GEMM_DENSE_OP, dr, shapes_of(M),
                                tco._nvfp4_dense_tuning_config(*dense_range), True)
        return k if k in cache.cache else None

    def pick(M):
        k = gu_key(M)
        return list(cache.cache[k][1]) if k in cache.cache else None

    def dense_pick(M):
        k = gu_dense_key(M)
        return list(cache.cache[k][1]) if k is not None else None

    picks = {str(M): pick(M) for M in MS_TUNE}
    dense_picks = {str(M): dense_pick(M) for M in MS_TUNE}
    n_keys = len(cache.cache)
    log(f"picks {picks}; dense picks {dense_picks}; cache keys {n_keys}")

    class Force:
        """Overwrite the standard and (if present) dense tuner entries of M with one tactic."""

        def __init__(self, M, tactic):
            self.keys = [k for k in (gu_key(M), gu_dense_key(M)) if k is not None]
            self.t = tactic

        def __enter__(self):
            if self.t is not None:
                self.old = {k: cache.cache[k] for k in self.keys}
                for k, v in self.old.items():
                    cache.cache[k] = (v[0], self.t, v[2])
            ge.clear_epi_decisions()

        def __exit__(self, *e):
            if self.t is not None:
                for k, v in self.old.items():
                    cache.cache[k] = v
            ge.clear_epi_decisions()

    def eq(a, b):
        return bool(a[0].shape == b[0].shape and a[1].shape == b[1].shape and torch.equal(a[0], b[0])
                    and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))

    def run_mod(m, M):
        """forward; returns ((fp4, sf, out), path, counter deltas)."""
        CAP.clear()
        c0, p0 = dict(cnt), dict(ge.path_calls)
        out = m(X(M))
        assert len(CAP) == 1
        fp4, sf, o = CAP[0]
        assert o is out
        CAP.clear()
        dp = {k: ge.path_calls[k] - p0[k] for k in p0}
        path = [k for k, v in dp.items() if v] or ["main"]
        assert len(path) == 1
        return (fp4, sf, out), path[0], {k: cnt[k] - c0[k] for k in cnt}

    def ref_unchunked(mf, M):
        fp4, sf = acts[M]
        gp, dpj = mf.gate_up_proj, mf.down_proj
        h1 = torch.ops.trtllm.nvfp4_gemm(fp4, gp.weight, sf, gp.weight_scale, gp.alpha, gp.dtype,
                                         allowed_backends=",".join(gp.nvfp4_allowed_backends))
        q, s = _unperm(h1, dpj.input_scale)
        del h1
        o = dpj(Fp4QuantizedTensor(q, s))
        CAP.clear()
        return q, s, o

    def ref_chunked(mf, M):
        fp4, sf = acts[M]
        gp, dpj = mf.gate_up_proj, mf.down_proj
        res = None
        if mg._MLP_CHUNK_MIN_M <= M <= mg._MLP_CHUNK_MAX_M:
            res = _chunked(fp4, sf, gp.weight, gp.weight_scale, gp.alpha, gp.dtype,
                           gp.nvfp4_allowed_backends, dpj.input_scale, mg._MLP_CHUNK_ROWS, act_fn=_unperm)
        if res is None:
            return ref_unchunked(mf, M)
        o = dpj(Fp4QuantizedTensor(res[0], res[1]))
        CAP.clear()
        return res[0], res[1], o

    report = {"modules": {"epi_on_permuted_epi_off_not": perm_ok, "on_weights_eq_permute_off": w_ok,
                          "backends": backends}, "picks": picks, "dense_picks": dense_picks,
              "dense_tuning": bool(tco.NVFP4_DENSE_TUNING), "dense_range": dense_range,
              "n_cache_keys": n_keys}
    ok_all = perm_ok and w_ok
    torch.cuda.synchronize()

    with torch.inference_mode():
        # ---------------- item 6: forced tactics at 2000 / 8000 -----------------------------------
        rows6 = []
        for M in MS_FORCE:
            for t in FORCED:
                with Force(M, t):
                    r_ch = ref_chunked(off[0], M)
                    r_un = ref_unchunked(off[0], M)
                    seq = []
                    for arm in ("on", "off", "on"):  # A, B, A
                        res, path, dc = run_mod(on[0] if arm == "on" else off[0], M)
                        seq.append((arm, res, path, dc))
                good = all(eq(res, r_ch) for _, res, _, _ in seq)
                good &= seq[0][2] == EXPECT_PATH[t] and seq[2][2] == EXPECT_PATH[t]
                good &= all(dc["unperm_act"] == 0 for arm, _, _, dc in seq if arm == "on")
                good &= all(dc["perm_act"] == 0 for arm, _, _, dc in seq if arm == "off")
                row = {"M": M, "forced": list(t), "forced_keys": len(Force(M, t).keys),
                       "epi1_path": seq[0][2], "expect": EXPECT_PATH[t],
                       "epi1_eq_main_chunked": [eq(seq[0][1], r_ch), eq(seq[2][1], r_ch)],
                       "epi0_eq_main_chunked": eq(seq[1][1], r_ch),
                       "main_chunked_eq_unchunked": eq(r_ch, r_un),
                       "epi1_unperm_act_calls": seq[0][3]["unperm_act"] + seq[2][3]["unperm_act"],
                       "epi1_perm_act_calls": seq[0][3]["perm_act"] + seq[2][3]["perm_act"],
                       "epi0_perm_act_calls": seq[1][3]["perm_act"],
                       "epi0_main_chunked_fn_calls": seq[1][3]["main_chunked_fn"], "ok": bool(good)}
                rows6.append(row)
                ok_all &= good
                log(f"item6 {json.dumps(row)}")
                del r_ch, r_un, seq
        report["item6_forced"] = rows6

        # ---------------- item 7: kill-switch matrix ----------------------------------------------
        rows7 = []
        for M in MS_KILL:
            for t in (None, ("cutlass", 0)):
                with Force(M, t):
                    for chunk_on in (True, False):
                        mg._MLP_CHUNK_ENABLED = chunk_on
                        r_un = ref_unchunked(off[0], M)
                        r_ch = ref_chunked(off[0], M) if chunk_on else r_un
                        r0, p0, d0 = run_mod(off[0], M)
                        r1, p1, d1 = run_mod(on[0], M)
                        good = eq(r0, r_ch) and eq(r1, r_ch) and d1["unperm_act"] == 0 and d0["perm_act"] == 0
                        if not chunk_on:
                            good &= p1 != "chunked" and d0["main_chunked_fn"] == 0
                        row = {"M": M, "forced": list(t) if t else None, "pick": pick(M),
                               "dense_pick": dense_pick(M),
                               "chunk": chunk_on, "ref": "main_chunked" if chunk_on else "unchunked",
                               "epi0_eq_ref": eq(r0, r_ch), "epi1_eq_ref": eq(r1, r_ch),
                               "epi1_path": p1, "epi0_main_chunked_fn_calls": d0["main_chunked_fn"],
                               "epi1_unperm_act_calls": d1["unperm_act"], "epi0_perm_act_calls": d0["perm_act"],
                               "main_chunked_eq_unchunked": eq(r_ch, r_un), "ok": bool(good)}
                        rows7.append(row)
                        ok_all &= good
                        log(f"item7 {json.dumps(row)}")
                        del r_ch, r_un, r0, r1
                mg._MLP_CHUNK_ENABLED = True
        report["item7_kill_switch"] = rows7

        # ---------------- item 8: repeated execution, bench.py structure ----------------------------
        rep = {}
        for mode, t in (("natural", None), ("forced_cutlass0", ("cutlass", 0))):
            refs = {}
            for M in MS_REP:
                with Force(M, t):
                    for c in range(3):
                        refs[(M, c)] = ref_chunked(off[c], M)
            firsts, rows = {}, {}
            for pname, order in (("A", MS_REP), ("B", MS_REP[::-1]), ("A2", MS_REP)):
                for M in order:
                    st = rows.setdefault(str(M), {"calls": 0, "bad_vs_first": 0, "bad_vs_main": 0,
                                                  "paths": {}, "unperm_act_calls": 0, "per_pass_bad": {}})
                    bad = 0
                    with Force(M, t):
                        for i in range(WARMUP + ITERS):
                            key = (M, i % 3)
                            res, path, dc = run_mod(on[i % 3], M)
                            if key not in firsts:
                                firsts[key] = res
                            st["calls"] += 1
                            st["paths"][path] = st["paths"].get(path, 0) + 1
                            st["unperm_act_calls"] += dc["unperm_act"]
                            e1, e2 = eq(res, firsts[key]), eq(res, refs[key])
                            st["bad_vs_first"] += int(not e1)
                            st["bad_vs_main"] += int(not e2)
                            bad += int(not (e1 and e2))
                    st["per_pass_bad"][pname] = bad
                torch.cuda.synchronize()
                log(f"item8 {mode} pass {pname}: bad {[(k, v['per_pass_bad'][pname]) for k, v in rows.items()]}")
            good = all(r["calls"] == 3 * (WARMUP + ITERS) and r["bad_vs_first"] == 0 and r["bad_vs_main"] == 0
                       and r["unperm_act_calls"] == 0 for r in rows.values())
            if t is not None:
                good &= all(r["paths"] == {"chunked": r["calls"]} for r in rows.values())
            rep[mode] = {"ok": bool(good), "rows": rows}
            ok_all &= good
            log(f"item8 {mode} ok={good} paths={ {k: v['paths'] for k, v in rows.items()} }")
            del refs, firsts
            torch.cuda.empty_cache()
        report["item8_repeated"] = rep

    # ---------------- CUDA-graph capture of the EPI=1 forward (BCG buckets 300 / 1200 / 2000) ----
    graph_rows = []
    for M in (300, 1200, 2000):
        ref = ref_chunked(off[0], M)
        with torch.inference_mode():
            eager, epath, _ = run_mod(on[0], M)
            eager = tuple(t.clone() for t in eager)
        xin = X(M)
        p0 = dict(ge.path_calls)
        g_ = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode(1)  # warn on any host sync under capture
        import warnings
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with torch.inference_mode(), torch.cuda.stream(s):
                st0 = torch.cuda.memory_stats()
                CAP.clear()
                with torch.cuda.graph(g_, stream=s):
                    gout = on[0](xin)
                st1 = torch.cuda.memory_stats()
        torch.cuda.set_sync_debug_mode(0)
        gfp4, gsf, _ = CAP[0]
        CAP.clear()
        cpath = [k for k in p0 if ge.path_calls[k] != p0[k]]
        sync_warn = [str(w.message)[:200] for w in wlist if "sync" in str(w.message).lower()]
        n_ok = 0
        for _ in range(25):
            g_.replay()
            torch.cuda.synchronize()
            r = (gfp4, gsf, gout)
            n_ok += int(eq(r, eager) and eq(r, ref))
        row = {"M": M, "capture_path": cpath, "eager_path": epath, "replays_equal_eager_and_main": n_ok,
               "host_sync_warnings_in_capture": sync_warn,
               "cudaMalloc_in_capture": st1.get("num_device_alloc", 0) - st0.get("num_device_alloc", 0),
               "ok": bool(n_ok == 25 and not sync_warn and cpath == [epath])}
        graph_rows.append(row)
        ok_all &= row["ok"]
        log(f"graph {json.dumps(row)}")
        del g_, gout
    report["graph_capture"] = graph_rows

    inputs_ok = all(torch.equal(acts[M][0], snap[M][0]) and torch.equal(acts[M][1], snap[M][1]) for M in acts)
    cache_ok = (len(cache.cache) == n_keys and {str(M): pick(M) for M in MS_TUNE} == picks
                and {str(M): dense_pick(M) for M in MS_TUNE} == dense_picks)
    report["inputs_unchanged"] = bool(inputs_ok)
    report["tuner_cache_unchanged"] = bool(cache_ok)
    ok_all &= inputs_ok and cache_ok
    report["path_calls_total"] = dict(ge.path_calls)
    print(json.dumps({"passed": bool(ok_all),
                      "compared": "real _Gemma4GeluQuantMLP forwards (EPI=1 permuted gate_up / EPI=0) vs "
                                  "main's chunked and unchunked chains on unpermuted weights: down_proj "
                                  "input FP4 payload + full SF (0xA5-prefilled) and down_proj output bytes",
                      **report}), flush=True)


if __name__ == "__main__":
    main()
