"""Idea 95 (branch 38): FA2 paged prefill longest-first CTA order (stage 1) and the
per-row split-KV merge (stage 2), kernel A/B and byte checks on synthetic Gemma4 shapes.

Arms (all in one process, same GPU, interleaved per round):
  base  the tree's default (run in the baseline = main tree; env switches do not exist there)
  off   FLASHINFER_FA2_LPT=0 FLASHINFER_FA2_MERGE_ROWS=0   (branch code paths == main's)
  s1    FLASHINFER_FA2_LPT=1 FLASHINFER_FA2_MERGE_ROWS=0   (stage 1)
  s12   FLASHINFER_FA2_LPT=1 FLASHINFER_FA2_MERGE_ROWS=1   (stage 1 + 2)

Calls are made exactly as TRT-LLM's FLASHINFER backend makes them: HND paged FP8 E4M3 KV,
page 32, BF16 Q/O, causal, window_left = sliding_window - 1 on sliding layers, plan() then
run(q, kv, out=out). Dims come from <checkpoint>/config.json.

--mode bench : digest pass (O + LSE sha256 per case per arm, all cases incl. extras), then
               CUDA-graph timing (R runs per replay, W warmup replays, ROUNDS replays per arm,
               arms interleaved per round). Last stdout line: evidence JSON.
--mode check : untimed; per case and arm the same W + ROUNDS replays as timing, every replay's
               O/LSE compared bitwise with the first; arms compared bitwise with 'off' (and
               with the base digests the baseline arm left in /tmp when present); FP32
               reference error; LPT 1,0,1 replans on one wrapper; A,B,A plan permutation;
               plan arrays of (1,1000) and (4,300) in both orders; one CUDA-graph plan.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch

PAGE = 32
ENV_KEYS = ("FLASHINFER_FA2_LPT", "FLASHINFER_FA2_MERGE_ROWS")
ARM_ENV = {
    "base": {},
    "off": {"FLASHINFER_FA2_LPT": "0", "FLASHINFER_FA2_MERGE_ROWS": "0"},
    "s1": {"FLASHINFER_FA2_LPT": "1", "FLASHINFER_FA2_MERGE_ROWS": "0"},
    "s12": {"FLASHINFER_FA2_LPT": "1", "FLASHINFER_FA2_MERGE_ROWS": "1"},
}
DIGEST_FILE = "/tmp/ascent_b38_lpt95_base_digests.json"


def log(*a):
    print(*a, flush=True)


def set_arm(arm):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARM_ENV[arm])


def load_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    nq = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    ghd = tc.get("global_head_dim") or hd
    gkv = nkv
    if tc.get("attention_k_eq_v", False):
        gkv = tc.get("num_global_key_value_heads") or nkv
    lt = tc.get("layer_types") or []
    d = dict(num_heads=nq, sliding_kv_heads=nkv, sliding_head_dim=hd, global_kv_heads=gkv,
             global_head_dim=ghd, sliding_window=tc.get("sliding_window"),
             num_layers=tc.get("num_hidden_layers"),
             n_sliding=sum(1 for t in lt if "sliding" in t),
             n_global=sum(1 for t in lt if "full" in t or "global" in t),
             hidden=tc.get("hidden_size"))
    qpath = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(qpath):
        q = json.load(open(qpath)).get("quantization", {})
        d["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo")
        d["quant_algo"] = q.get("quant_algo")
    return d


class Case:
    def __init__(self, name, layer, q_lens, kv_lens, dims, seed=1234, disable_split=False,
                 cuda_graph=False):
        self.name, self.layer = name, layer
        self.q_lens, self.kv_lens = list(q_lens), list(kv_lens)
        self.batch = len(q_lens)
        self.disable_split = disable_split
        self.cuda_graph = cuda_graph
        self.Hq = dims["num_heads"]
        if layer == "sliding":
            self.Hkv, self.D = dims["sliding_kv_heads"], dims["sliding_head_dim"]
            self.window_left = dims["sliding_window"] - 1
        else:
            self.Hkv, self.D = dims["global_kv_heads"], dims["global_head_dim"]
            self.window_left = -1
        self.sm_scale = 1.0 / math.sqrt(self.D)
        dev = "cuda"
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        npages = [(L + PAGE - 1) // PAGE for L in self.kv_lens]
        total_pages = sum(npages)
        self.total_q = sum(self.q_lens)
        qo, kvp = [0], [0]
        for q, n in zip(self.q_lens, npages):
            qo.append(qo[-1] + q)
            kvp.append(kvp[-1] + n)
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
        self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
        self.kv_indices = torch.randperm(total_pages, generator=g, device=dev).to(torch.int32)
        self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in self.kv_lens],
                                          dtype=torch.int32, device=dev)
        pool = torch.randn(total_pages, 2, self.Hkv, PAGE, self.D, generator=g, device=dev,
                           dtype=torch.float32)
        self.kv = pool.to(torch.float8_e4m3fn)
        del pool
        self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device=dev,
                             dtype=torch.float32).to(torch.bfloat16)

    def label(self):
        return f"{self.layer}_d{self.D}_{self.name}"

    @torch.no_grad()
    def reference(self):
        out = torch.empty(self.total_q, self.Hq, self.D, dtype=torch.float32, device="cuda")
        grp = self.Hq // self.Hkv
        for r in range(self.batch):
            s, e = int(self.qo_indptr[r]), int(self.qo_indptr[r + 1])
            ql, kl = e - s, self.kv_lens[r]
            ps, pe = int(self.kv_indptr[r]), int(self.kv_indptr[r + 1])
            pages = self.kv[self.kv_indices[ps:pe].long()].float()
            k = pages[:, 0].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:kl]
            v = pages[:, 1].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:kl]
            i = torch.arange(ql, device="cuda").view(-1, 1) + (kl - ql)
            j = torch.arange(kl, device="cuda").view(1, -1)
            mask = j <= i
            if self.window_left >= 0:
                mask &= (i - j) <= self.window_left
            for h in range(self.Hq):
                sc = (self.q[s:e, h].float() @ k[:, h // grp].T) * self.sm_scale
                sc.masked_fill_(~mask, float("-inf"))
                out[s:e, h] = torch.softmax(sc, dim=-1) @ v[:, h // grp]
        return out


def bench_cases(dims):
    shapes = [(1, 300), (1, 1000), (3, 1000), (1, 2000), (4, 2000), (16, 300), (1, 8000)]
    cs = []
    for layer in ("sliding", "global"):
        for b, s in shapes:
            cs.append(Case(f"{b}x{s}", layer, [s] * b, [s] * b, dims))
    return cs


def extra_case_specs():
    # (name, layer, q_lens, kv_lens, disable_split, cuda_graph)
    specs = []
    for layer in ("sliding", "global"):
        specs += [
            ("ragged", layer, [300, 1000, 37, 2000], [300, 1000, 37, 2000], False, False),
            ("prefix", layer, [500, 300], [1500, 2300], False, False),
            ("split4x300", layer, [300] * 4, [300] * 4, False, False),
            ("split2x700", layer, [700] * 2, [700] * 2, False, False),
            ("cg4x300", layer, [300] * 4, [300] * 4, False, True),
        ]
    specs += [
        ("rr1x8000", "global", [16], [8000], True, False),
        ("rr4x2000", "global", [16] * 4, [2000] * 4, True, False),
    ]
    return specs


def make_extra(spec, dims):
    name, layer, ql, kl, ds, cg = spec
    return Case(name, layer, ql, kl, dims, disable_split=ds, cuda_graph=cg)


WS_CACHE = {}


def workspace(key):
    if key not in WS_CACHE:
        WS_CACHE[key] = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return WS_CACHE[key]


class Arm:
    """One planned wrapper with its own workspace and outputs, planned under an arm's env."""

    def __init__(self, c, arm, ws_key=None):
        import flashinfer
        self.c, self.arm = c, arm
        set_arm(arm)
        ws = workspace(ws_key or arm)
        if c.cuda_graph:
            self.bufs = (c.qo_indptr.clone(), c.kv_indptr.clone(), c.kv_indices.clone(),
                         c.last_page_len.clone())
            self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws, "HND", backend="fa2", use_cuda_graph=True, qo_indptr_buf=self.bufs[0],
                paged_kv_indptr_buf=self.bufs[1], paged_kv_indices_buf=self.bufs[2],
                paged_kv_last_page_len_buf=self.bufs[3])
        else:
            self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        self.out = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        self.lse = torch.empty(c.total_q, c.Hq, dtype=torch.float32, device="cuda")
        self.plan()

    def plan(self, c=None):
        c = c or self.c
        set_arm(self.arm)
        kw = {}
        if c.disable_split:
            kw["disable_split_kv"] = True
        self.w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D,
                    PAGE, causal=True, sm_scale=c.sm_scale, window_left=c.window_left,
                    q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
                    o_data_type=None, **kw)
        pi = self.w._plan_info
        self.plan_info = [int(x) for x in pi]

    def run(self, with_lse=False, c=None):
        c = c or self.c
        set_arm(self.arm)
        if with_lse:
            self.w.run(c.q, c.kv, out=self.out, lse=self.lse)
        else:
            self.w.run(c.q, c.kv, out=self.out)

    def plan_arrays(self):
        pi = self.plan_info
        n = pi[0]
        buf = self.w._pin_memory_int_workspace_buffer
        res = {}
        for name, idx in (("request", 4), ("qo_tile", 5), ("kv_tile", 6)):
            off = pi[idx]
            res[name] = buf[off:off + 4 * n].view(torch.int32).tolist()
        res["cta_tile_q"] = pi[3]
        res["padded_batch_size"] = n
        res["split_kv"] = pi[14]
        return res

    def capture(self, R, with_lse=False):
        """Object with .replay() running R calls: a CUDA graph, or eager calls if capture fails."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.run(with_lse)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            set_arm(self.arm)
            with torch.cuda.graph(g):
                for _ in range(R):
                    self.run(with_lse)
            torch.cuda.synchronize()
            self.graphed = True
            return g
        except Exception as ex:
            log("graph capture failed, eager timing:", repr(ex)[:200])
            torch.cuda.synchronize()
            self.graphed = False
            arm = self

            class Eager:
                def replay(self_inner):
                    for _ in range(R):
                        arm.run(with_lse)
            return Eager()


def digest(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                          ).hexdigest()[:20]


def digest_pass(cases, arms):
    out = {}
    for c in cases:
        d = {}
        for arm in arms:
            try:
                a = Arm(c, arm)
                a.run(with_lse=True)
                torch.cuda.synchronize()
                d[arm] = dict(o=digest(a.out), lse=digest(a.lse), cta_tile_q=a.plan_info[3],
                              split_kv=a.plan_info[14], items=a.plan_info[0])
                del a
            except Exception as ex:
                d[arm] = dict(error=repr(ex)[:300])
                traceback.print_exc()
        out[c.label()] = d
        log("DIGEST", c.label(), json.dumps(d))
    return out


def pct(ts, q):
    ts = sorted(ts)
    return ts[min(len(ts) - 1, max(0, int(q * len(ts))))]


def bench(cases, arms, R, W, rounds):
    res = {}
    for c in cases:
        armobjs, graphs = {}, {}
        for arm in arms:
            armobjs[arm] = Arm(c, arm)
            graphs[arm] = armobjs[arm].capture(R)
        for arm in arms:
            for _ in range(W):
                graphs[arm].replay()
        torch.cuda.synchronize()
        ts = {arm: [] for arm in arms}
        for rnd in range(rounds):
            order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
            for arm in order:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[arm].replay()
                e1.record()
                e1.synchronize()
                ts[arm].append(e0.elapsed_time(e1) / R)
        rec = {}
        for arm in arms:
            med = statistics.median(ts[arm])
            p10, p90 = pct(ts[arm], 0.1), pct(ts[arm], 0.9)
            rec[arm] = dict(median_ms=med, p10_ms=p10, p90_ms=p90,
                            noise=(p90 - p10) / med if med > 0 else 0.0,
                            cta_tile_q=armobjs[arm].plan_info[3],
                            split_kv=armobjs[arm].plan_info[14],
                            items=armobjs[arm].plan_info[0],
                            graphed=getattr(armobjs[arm], "graphed", None))
        if "off" in rec:
            for arm in arms:
                if arm != "off":
                    rec[arm]["vs_off"] = rec[arm]["median_ms"] / rec["off"]["median_ms"] - 1.0
        res[c.label()] = rec
        log("BENCH", c.label(), json.dumps(rec))
        del graphs, armobjs
        torch.cuda.synchronize()
    return res


def err_stats(out, ref):
    d = out.float() - ref
    return dict(max_abs=float(d.abs().max()), rel_fro=float(d.norm() / ref.norm()),
                finite=bool(torch.isfinite(out.float()).all()))


def check(cases, arms, R, W, rounds, base_digests, ref_limit):
    """Every check named in the module docstring. Returns (passed, report)."""
    report = {}
    passed = True
    for c in cases:
        rec = {}
        outs = {}
        for arm in arms:
            a = Arm(c, arm)
            g = a.capture(R, with_lse=True)
            g.replay()
            torch.cuda.synchronize()
            o0, l0 = a.out.clone(), a.lse.clone()
            bad = 0
            # same warmup + replay count as the timing sequence, no resets in between
            for _ in range(W + rounds - 1):
                g.replay()
                torch.cuda.synchronize()
                if not (torch.equal(a.out, o0) and torch.equal(a.lse, l0)):
                    bad += 1
            # eager run without LSE (TRT-LLM's call) must give the same O
            a.run(with_lse=False)
            torch.cuda.synchronize()
            eager_eq = bool(torch.equal(a.out, o0))
            outs[arm] = (o0, l0)
            rec[arm] = dict(replays=W + rounds, mismatches=bad, eager_no_lse_equal=eager_eq,
                            o=digest(o0), lse=digest(l0), cta_tile_q=a.plan_info[3],
                            split_kv=a.plan_info[14], items=a.plan_info[0])
            if bad or not eager_eq:
                passed = False
            del g, a
        for arm in arms:
            if arm != "off" and "off" in outs:
                eq = bool(torch.equal(outs[arm][0], outs["off"][0]) and
                          torch.equal(outs[arm][1], outs["off"][1]))
                rec[arm]["bitwise_vs_off"] = eq
                rec[arm]["maxabs_vs_off"] = float(
                    (outs[arm][0].float() - outs["off"][0].float()).abs().max())
                if not eq:
                    passed = False
            bd = (base_digests or {}).get(c.label(), {}).get("base")
            if bd and "o" in bd:
                eq = bd["o"] == rec[arm]["o"] and bd["lse"] == rec[arm]["lse"]
                rec[arm]["digest_eq_base"] = eq
                if not eq:
                    passed = False
        if c.total_q * max(c.kv_lens) <= ref_limit:
            ref = c.reference()
            for arm in arms:
                rec[arm]["err_vs_fp32ref"] = err_stats(outs[arm][0], ref)
                if not rec[arm]["err_vs_fp32ref"]["finite"]:
                    passed = False
            del ref
        report[c.label()] = rec
        log("CHECK", c.label(), json.dumps(rec))
        del outs
        torch.cuda.synchronize()
    return passed, report


def flip_checks(dims, arms):
    """LPT 1,0,1 replans on one wrapper; A,B,A plan permutation; plan arrays."""
    res = {}
    ok = True
    if "s1" not in arms:
        return True, res
    for layer in ("sliding", "global"):
        A = Case("1x1000", layer, [1000], [1000], dims)
        B = Case("4x300", layer, [300] * 4, [300] * 4, dims)
        a = Arm(A, "s12", ws_key="flip")
        seq = []
        for arm in ("s12", "off", "s12", "s1", "off", "s12"):
            a.arm = arm
            a.plan(A)
            a.run(with_lse=True, c=A)
            torch.cuda.synchronize()
            seq.append((arm, a.out.clone(), a.lse.clone()))
        eq = all(torch.equal(s[1], seq[0][1]) and torch.equal(s[2], seq[0][2]) for s in seq)
        res[f"{layer}_lpt_flip_1_0_1"] = dict(order=[s[0] for s in seq], bitwise_equal=eq)
        ok &= eq
        # A, B, A on one wrapper (plan cache / workspace reuse across shapes)
        a.arm = "s12"
        a.plan(A)
        a.run(with_lse=True, c=A)
        torch.cuda.synchronize()
        oA, lA = a.out.clone(), a.lse.clone()
        b_out = torch.empty(B.total_q, B.Hq, B.D, dtype=torch.bfloat16, device="cuda")
        a.plan(B)
        set_arm("s12")
        a.w.run(B.q, B.kv, out=b_out)
        a.plan(A)
        a.run(with_lse=True, c=A)
        torch.cuda.synchronize()
        eq2 = bool(torch.equal(a.out, oA) and torch.equal(a.lse, lA))
        res[f"{layer}_ABA"] = dict(bitwise_equal=eq2)
        ok &= eq2
        del a
        for c in (A, B):
            arrs = {}
            for arm in ("off", "s1"):
                x = Arm(c, arm, ws_key="flip")
                arrs[arm] = x.plan_arrays()
                del x
            res[f"plan_arrays_{c.label()}"] = arrs
            log("PLAN", c.label(), json.dumps(arrs))
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("bench", "check"), default="bench")
    ap.add_argument("--arms", default="off,s1,s12")
    ap.add_argument("--report-arm", default="s12")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--W", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--only", default="", help="comma list of case labels to keep")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--ref-limit", type=float, default=2.0e7)
    a = ap.parse_args()
    t0 = time.time()
    arms = a.arms.split(",")
    torch.backends.cuda.matmul.allow_tf32 = False
    dims = load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count, "cc", p.major, p.minor)
    import flashinfer
    log("flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
    log("ARMS", arms, "R", a.R, "W", a.W, "rounds", a.rounds)

    cases = bench_cases(dims)
    extras = [] if a.no_extras else [make_extra(s, dims) for s in extra_case_specs()]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c.label() in keep]
        extras = [c for c in extras if c.label() in keep]

    base_digests = None
    if "base" not in arms and os.path.exists(DIGEST_FILE):
        try:
            blob = json.load(open(DIGEST_FILE))
            if time.time() - blob.get("time", 0) < 3 * 3600:
                base_digests = blob["digests"]
                log("loaded base digests from", DIGEST_FILE)
        except Exception:
            pass

    if a.mode == "bench":
        dig = digest_pass(cases + extras, arms)
        if "base" in arms:
            try:
                json.dump(dict(time=time.time(), digests=dig), open(DIGEST_FILE, "w"))
            except Exception as ex:
                log("could not write", DIGEST_FILE, ex)
        dig_eq = {}
        all_eq = True
        for lab, d in dig.items():
            ref = (base_digests or {}).get(lab, {}).get("base") or d.get("off")
            for arm, v in d.items():
                if arm in ("base", "off") or ref is None:
                    continue
                eq = v.get("o") == ref.get("o") and v.get("lse") == ref.get("lse")
                dig_eq.setdefault(lab, {})[arm] = eq
                all_eq &= eq
        res = bench(cases, arms, a.R, a.W, a.rounds)
        rep = a.report_arm if a.report_arm in arms else arms[0]
        batch = {lab: dict(kernel_ms=r[rep]["median_ms"],
                           noise_floor=max(r[x]["noise"] for x in r)) for lab, r in res.items()}
        summary = {lab: {arm: dict(med=round(r[arm]["median_ms"] * 1e3, 2),
                                   p10=round(r[arm]["p10_ms"] * 1e3, 2),
                                   p90=round(r[arm]["p90_ms"] * 1e3, 2),
                                   vs_off=round(r[arm].get("vs_off", 0.0), 4))
                         for arm in r} for lab, r in res.items()}
        log("SECS", round(time.time() - t0, 1))
        print(json.dumps(dict(batch=batch, output={"class": "exact"}, report_arm=rep,
                              us=summary, digests_equal=dig_eq, digests_all_equal=all_eq,
                              base_digests_loaded=base_digests is not None,
                              digests=dig)))
    else:
        ok1, rep = check(cases + extras, arms, a.R, a.W, a.rounds, base_digests, a.ref_limit)
        ok2, flips = flip_checks(dims, arms)
        log("SECS", round(time.time() - t0, 1))
        print(json.dumps(dict(passed=bool(ok1 and ok2), cases_passed=ok1, flips_passed=ok2,
                              compared="O/LSE bitwise: repeated replays vs first, arms vs "
                              "off (branch main path), digests vs main tree when present; "
                              "FP32 reference error",
                              base_digests_loaded=base_digests is not None,
                              flips=flips, cases=rep)))


if __name__ == "__main__":
    main()
