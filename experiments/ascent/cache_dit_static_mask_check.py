#!/usr/bin/env python3
"""cache-dit 1.5.3 static-mask semantics on a toy MiniMax-H3-shaped transformer (1 GPU, no H3 load).

Stage for step_cache idea 56 (analysis idea 70, branch 27). Answers Q1-Q9 of the brief:
  toy = 50 tiny blocks (Linear on [2,64,128] BF16, cuda) called exactly like H3's loop
  (transformer_minimax_h3.py:1282: block(h, temb, adaln_indices, rotary_emb, key_padding_mask,
  timestep, valid_seq_len=...)), enabled the way enable_cache_dit_for_wan does it (BlockAdapter,
  blocks=[toy.blocks], ForwardPattern.Pattern_3, params_modifiers, cache_dit.enable_cache).
  A request = refresh, then 49 forwards with input x0 + 0.01*i. A forward is "cached" when fewer
  than all 50 blocks ran (only Fn, plus Bn when Bn > 0). Each block counts its calls.
Writes results JSON, quoted cache-dit source lines and real "Cache-DiT stats" payloads (for the table
script's self-test) under $AUTO_DEV_ARTIFACTS/cache_dit_toy. Last stdout line: {"passed": ...}.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import traceback
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ParamsModifier, TaylorSeerCalibratorConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cache_arms_table import DENSE_HEAD, MASKS, check_masks  # noqa: E402

OUT = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "cache_dit_toy")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
NB, D, B, S = 50, 128, 2, 64
N = 49
R: dict = {}
FAIL: list = []


def log(*a):
    print(*a, flush=True)


def expect(cond, what):
    if not cond:
        FAIL.append(what)
        log("  !! EXPECTATION FAILED:", what)
    return bool(cond)


torch.set_grad_enabled(False)
_g = torch.Generator(device="cpu").manual_seed(1234)
W1 = [(torch.randn(D, D, generator=_g) / math.sqrt(D)).to(DEV, torch.bfloat16) for _ in range(NB)]
W2 = [(torch.randn(D, D, generator=_g) / math.sqrt(D)).to(DEV, torch.bfloat16) for _ in range(NB)]
GS = [torch.randn(1, 1, D, generator=_g).to(DEV, torch.bfloat16) for _ in range(NB)]
X0 = torch.randn(B, S, D, generator=_g).to(DEV, torch.bfloat16)
X = [(X0 + 0.01 * i).contiguous() for i in range(N)]
TEMB = torch.randn(B, 1, D, generator=_g).to(DEV, torch.bfloat16)
ADA = torch.arange(S, device=DEV)
ROPE = (torch.randn(S, 64, generator=_g).to(DEV), torch.randn(S, 64, generator=_g).to(DEV))
KPM = torch.ones(B, S, dtype=torch.bool, device=DEV)
TS = torch.full((B,), 500.0, device=DEV)
VSL = S
EXPECTED_ARGS = (id(TEMB), id(ADA), id(ROPE), id(KPM), id(TS), VSL)


# ------------------------------------------------------------------------------------------------
# Toy model
# ------------------------------------------------------------------------------------------------
class ToyBlock(nn.Module):
    """H3-signature block. deferred=True + materialize_output=False returns (x1, ff, gate) and leaves
    x1 + gate*ff pending for the next block (landing 7's contract, pitfall 565)."""

    def __init__(self, i: int, deferred: bool = False):
        super().__init__()
        self.idx = i
        self.w1 = nn.Parameter(W1[i], requires_grad=False)
        self.w2 = nn.Parameter(W2[i], requires_grad=False)
        self.gs = nn.Parameter(GS[i], requires_grad=False)
        self.deferred = deferred
        self.materialize_output = True
        self.calls = 0
        self.last_args = None

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask=None,
                timestep=None, valid_seq_len=None):
        self.calls += 1
        self.last_args = (id(temb), id(adaln_indices), id(rotary_emb), id(key_padding_mask), id(timestep),
                          valid_seq_len)
        if isinstance(hidden_states, tuple):  # pending residual of the previous block
            x1p, ffp, gp = hidden_states
            x = x1p + gp * ffp
        else:
            x = hidden_states
        a = torch.tanh(F.linear(x, self.w1))
        x1 = x + 0.5 * a + 0.01 * temb
        ff = torch.tanh(F.linear(x1, self.w2))
        gate = torch.sigmoid(temb * self.gs)
        if self.deferred and not self.materialize_output:
            return (x1, ff, gate)
        return x1 + gate * ff


def run_blocks(blocks, h, *args, **kwargs):
    """The single block-loop helper (ruling 594 (2)): passes whatever a block returns to the next."""
    for blk in blocks:
        h = blk(h, *args, **kwargs)
    return h


class Group(nn.Module):
    """idea 79 design A: a group of consecutive H3 blocks; no flags, no materialize code."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.calls = 0
        self.last_args = None

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask=None,
                timestep=None, valid_seq_len=None):
        self.calls += 1
        self.last_args = (id(temb), id(adaln_indices), id(rotary_emb), id(key_padding_mask), id(timestep),
                          valid_seq_len)
        return run_blocks(self.blocks, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask,
                          timestep, valid_seq_len=valid_seq_len)


class ToyH3(nn.Module):
    def __init__(self, deferred=False, group_split=None):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([ToyBlock(i, deferred) for i in range(NB)])
        self.use_groups = group_split is not None
        if self.use_groups:
            self.cache_groups = nn.ModuleList([Group(list(self.transformer_blocks[:group_split])),
                                               Group(list(self.transformer_blocks[group_split:]))])

    def assign_block_roles(self, materialize_at):
        """ONE build/regroup site for the per-block materialize switch (ruling 580/594 (1))."""
        for i, blk in enumerate(self.transformer_blocks):
            blk.materialize_output = (materialize_at == "all") or (isinstance(materialize_at, (set, list))
                                                                   and i in materialize_at)

    def forward(self, x, temb, adaln_indices, rotary_emb, key_padding_mask, timestep, valid_seq_len=None):
        blocks = self.cache_groups if self.use_groups else self.transformer_blocks
        h = run_blocks(blocks, x, temb, adaln_indices, rotary_emb, key_padding_mask, timestep,
                       valid_seq_len=valid_seq_len)
        if isinstance(h, tuple):  # the fused norm_out tail applies the last pending term
            h = h[0] + h[2] * h[1]
        return h


def call(toy, i):
    return toy(X[i], TEMB, ADA, ROPE, KPM, TS, valid_seq_len=VSL)


def base_kwargs(head=8, **over):
    kw = dict(Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=head, max_cached_steps=-1,
              max_continuous_cached_steps=3, residual_diff_threshold=0.24, enable_separate_cfg=False)
    kw.update(over)
    return kw


def enable(toy, blocks_ml, cfg_kwargs, calibrator=None):
    """Mirror of enable_cache_dit_for_wan's single-transformer branch, with Pattern_3."""
    db_cfg = DBCacheConfig(**cfg_kwargs)
    adapter = BlockAdapter(
        transformer=toy,
        blocks=[blocks_ml],
        forward_pattern=[ForwardPattern.Pattern_3],
        params_modifiers=[ParamsModifier(cache_config=db_cfg)],
    )
    cache_dit.enable_cache(adapter, cache_config=db_cfg, calibrator_config=calibrator)
    return adapter


def ctx_of(toy):
    return toy._context_manager.get_context(toy._context_names[0])


LIVE_FIELDS = ["Fn_compute_blocks", "Bn_compute_blocks", "max_warmup_steps", "max_cached_steps",
               "max_continuous_cached_steps", "residual_diff_threshold", "num_inference_steps",
               "steps_computation_policy", "enable_separate_cfg", "force_refresh_step_hint"]


def live(toy):
    c = ctx_of(toy)
    cfg = c.cache_config
    d = {k: getattr(cfg, k) for k in LIVE_FIELDS}
    m = cfg.steps_computation_mask
    d["mask_len"] = None if m is None else len(m)
    d["mask_zeros"] = None if m is None else int(sum(1 for v in m if v == 0))
    d["warmup_steps_resolved"] = list(c.warmup_steps)
    d["calibrator"] = repr(c.calibrator) if c.calibrator is not None else None
    d["encoder_calibrator"] = repr(c.encoder_calibrator) if c.encoder_calibrator is not None else None
    return d


def refresh_static(toy, mask, policy="static", **extra):
    cfg = DBCacheConfig().reset(num_inference_steps=len(mask), steps_computation_mask=list(mask),
                                steps_computation_policy=policy, **extra)
    cache_dit.refresh_context(toy, cache_config=cfg, verbose=False)


def count_syncs(fn):
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = fn()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    msgs = [str(x.message) for x in w if "synchroniz" in str(x.message).lower()]
    return len(msgs), (msgs[0][:120] if msgs else None), res


def run_request(toy, n=N, sync=False, per_forward=None):
    outs, cached, ran_counts, syncs = [], [], [], []
    for i in range(n):
        before = [b.calls for b in toy.transformer_blocks]
        if sync:
            cnt, _, out = count_syncs(lambda: call(toy, i))
            syncs.append(cnt)
        else:
            out = call(toy, i)
        after = [b.calls for b in toy.transformer_blocks]
        ran = [j for j in range(NB) if after[j] > before[j]]
        if len(ran) < NB:  # cached: only Fn (+ Bn) blocks ran (Bn>0 runs the last block too)
            cached.append(i)
        ran_counts.append(len(ran))
        outs.append(out.clone())
        if per_forward is not None:
            per_forward(i, toy)
    return {"outs": outs, "cached": cached, "ran": ran_counts, "syncs": syncs}


def zeros_of(mask):
    return [i for i, v in enumerate(mask) if v == 0]


def bitwise_all(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


def maxabs(a, b):
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))


def stats_payload(toy):
    stats = {"transformer_0": cache_dit.summary(toy, details=False, logging=False)}
    return f"{stats}"


def stats_cached_steps(toy):
    s = cache_dit.summary(toy, details=False, logging=False)
    return [list(x.cached_steps) for x in s], [x.accumulated_cached_steps for x in s]


def section(name):
    log("")
    log("=" * 100)
    log(name)
    log("=" * 100)


# ------------------------------------------------------------------------------------------------
# Source quotes (cache-dit 1.5.3 lines this analysis relies on)
# ------------------------------------------------------------------------------------------------
def quote_sources():
    from cache_dit.caching import cache_interface
    from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
    from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
    from cache_dit.caching.cache_contexts.cache_config import BasicCacheConfig
    from cache_dit.caching.cache_contexts.cache_context import CachedContext
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    from cache_dit.caching.cache_contexts.calibrators.taylorseer import TaylorSeerState

    objs = [
        ("refresh_context", cache_interface.refresh_context),
        ("CachedAdapter.maybe_refresh_context", CachedAdapter.maybe_refresh_context),
        ("CachedAdapter._modify_context_params", CachedAdapter._modify_context_params),
        ("BasicCacheConfig.update", BasicCacheConfig.update),
        ("BasicCacheConfig.empty", BasicCacheConfig.empty),
        ("BasicCacheConfig.reset", BasicCacheConfig.reset),
        ("CachedContext.mark_step_begin", CachedContext.mark_step_begin),
        ("CachedContext.warmup_steps", CachedContext.warmup_steps.fget),
        ("CachedContext.is_in_full_compute_steps", CachedContext.is_in_full_compute_steps),
        ("CachedContextManager.reset_context", CachedContextManager.reset_context),
        ("CachedContextManager.maybe_refresh", CachedContextManager.maybe_refresh),
        ("CachedContextManager.can_cache", CachedContextManager.can_cache),
        ("CachedContextManager.similarity", CachedContextManager.similarity),
        ("CachedContextManager.apply_cache", CachedContextManager.apply_cache),
        ("CachedBlocks_Pattern_3_4_5.forward", CachedBlocks_Pattern_3_4_5.forward),
        ("CachedBlocks_Pattern_3_4_5.call_Fn_blocks", CachedBlocks_Pattern_3_4_5.call_Fn_blocks),
        ("CachedBlocks_Pattern_3_4_5.call_Mn_blocks", CachedBlocks_Pattern_3_4_5.call_Mn_blocks),
        ("CachedBlocks_Pattern_3_4_5._process_block_outputs", CachedBlocks_Pattern_3_4_5._process_block_outputs),
        ("TaylorSeerState.derivative", TaylorSeerState.derivative),
        ("TaylorSeerState.approximate", TaylorSeerState.approximate),
    ]
    index = {}
    with open(os.path.join(OUT, "cache_dit_1.5.3_quotes.txt"), "w") as f:
        for name, obj in objs:
            fn = inspect.unwrap(obj)
            try:
                lines, start = inspect.getsourcelines(fn)
                path = inspect.getsourcefile(fn)
            except (OSError, TypeError) as e:
                index[name] = f"unavailable: {e}"
                continue
            rel = path.split("site-packages/")[-1].split("dependencies-1/")[-1]
            index[name] = f"{rel}:{start}-{start + len(lines) - 1}"
            f.write(f"### {name}  ({rel}:{start}-{start + len(lines) - 1})\n")
            for k, ln in enumerate(lines):
                f.write(f"{start + k:5d}: {ln}")
            f.write("\n")
    return index


# ------------------------------------------------------------------------------------------------
def main():
    section("versions")
    R["cache_dit_version"] = getattr(cache_dit, "__version__", None)
    R["torch"] = torch.__version__
    log(R["cache_dit_version"], R["torch"], torch.cuda.get_device_name(0))
    check_masks()
    R["source_index"] = quote_sources()
    for k, v in R["source_index"].items():
        log(f"  {k}: {v}")

    trt = {}
    try:
        from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import (  # noqa: E402
            _refresh_ctx, db_cache_kwargs_from_cache_dit_config)
        from tensorrt_llm.visual_gen.args import CacheDiTConfig  # noqa: E402
        trt = {"_refresh_ctx": _refresh_ctx, "kw": db_cache_kwargs_from_cache_dit_config, "Cfg": CacheDiTConfig}
        log("tensorrt_llm cache_dit_enablers imported")
    except Exception as e:  # noqa: BLE001
        R["trt_import_error"] = repr(e)
        log("tensorrt_llm import failed:", repr(e))

    # Reference: uncached, non-deferred toy; per-forward h8, h50, residual R = h50 - h8.
    ref = ToyH3().to(DEV)
    REF_OUT, H8, RES = [], [], []
    for i in range(N):
        h = X[i]
        for blk in ref.transformer_blocks[:8]:
            h = blk(h, TEMB, ADA, ROPE, KPM, TS, valid_seq_len=VSL)
        h8 = h
        for blk in ref.transformer_blocks[8:]:
            h = blk(h, TEMB, ADA, ROPE, KPM, TS, valid_seq_len=VSL)
        H8.append(h8.clone())
        REF_OUT.append(h.clone())
        RES.append(h.contiguous() - h8)
    full = [call(ref, i).clone() for i in range(N)]
    expect(bitwise_all(full, REF_OUT), "reference: model forward == manual block loop")

    # ------------------------------------------------------------------ six masks, static, base cfg
    section("MASKS: static policy, Fn=8 Bn=0, max_warmup = dense head, max_continuous=3, thr 0.24")
    toy = ToyH3().to(DEV)
    enable(toy, toy.transformer_blocks, base_kwargs())
    R["after_enable_live"] = live(toy)
    log("live after enable:", R["after_enable_live"])
    R["masks"] = {}
    stats_files = {}
    for name in ["w14_c2", "C", "D", "E", "fallback", "W"]:
        m = MASKS[name]
        kw = base_kwargs(head=DENSE_HEAD[name])
        kw.pop("enable_separate_cfg")
        refresh_static(toy, m, **kw)
        r = run_request(toy)
        cached_ran = sorted(set(r["ran"][i] for i in r["cached"]))
        sc, acc = stats_cached_steps(toy)
        ok = expect(r["cached"] == zeros_of(m), f"mask {name}: cached set == mask zeros")
        expect(sc[0] == zeros_of(m) and acc[0] == len(zeros_of(m)), f"mask {name}: stats cached_steps == mask zeros")
        R["masks"][name] = {"n_cached": len(r["cached"]), "expected": len(zeros_of(m)), "cached": r["cached"],
                            "blocks_run_on_cached": cached_ran, "stats_cached_steps_count": acc[0], "ok": ok}
        log(f"  {name:9s} cached {len(r['cached']):2d}/{len(zeros_of(m)):2d} blocks-run-on-cached={cached_ran} "
            f"stats={acc[0]} set={r['cached']}")
        if name in ("w14_c2", "E"):
            p = os.path.join(OUT, f"stats_line_{name}.txt")
            with open(p, "w") as f:
                f.write(stats_payload(toy))
            stats_files[name] = p
    R["stats_line_files"] = stats_files

    # ------------------------------------------------------------------ Q1
    section("Q1: max_continuous_cached_steps 3 vs -1 (static), and dynamic contrast")
    R["Q1"] = {}
    for name in ["E", "D"]:
        for mcc in (3, -1):
            refresh_static(toy, MASKS[name], **{**base_kwargs(), "max_continuous_cached_steps": mcc})
            r = run_request(toy)
            R["Q1"][f"{name}_static_mcc{mcc}"] = r["cached"]
            expect(r["cached"] == zeros_of(MASKS[name]), f"Q1 {name} static mcc {mcc} == mask zeros")
            log(f"  {name} static mcc={mcc:2d}: {len(r['cached'])} cached {r['cached']}")
    for mcc in (3, -1):
        refresh_static(toy, MASKS["E"], policy="dynamic",
                       **{**base_kwargs(), "max_continuous_cached_steps": mcc, "residual_diff_threshold": 10.0})
        r = run_request(toy)
        R["Q1"][f"E_dynamic_thr10_mcc{mcc}"] = r["cached"]
        log(f"  E dynamic thr=10 mcc={mcc:2d}: {len(r['cached'])} cached {r['cached']}")

    # ------------------------------------------------------------------ Q2
    section("Q2: residual_diff_threshold 0.24 / 0.0 / 10.0 under static vs dynamic (mask E, mcc -1)")
    R["Q2"] = {}
    for pol in ("static", "dynamic"):
        for thr in (0.24, 0.0, 10.0):
            refresh_static(toy, MASKS["E"], policy=pol,
                           **{**base_kwargs(), "max_continuous_cached_steps": -1, "residual_diff_threshold": thr})
            r = run_request(toy)
            diffs = dict(ctx_of(toy).residual_diffs)
            R["Q2"][f"{pol}_thr{thr}"] = {"cached": r["cached"], "n": len(r["cached"]),
                                          "residual_diffs_sample": {k: diffs[k] for k in list(diffs)[:4]}}
            if pol == "static":
                expect(r["cached"] == zeros_of(MASKS["E"]), f"Q2 static thr {thr} == mask zeros")
            log(f"  {pol:7s} thr={thr:5}: {len(r['cached'])} cached {r['cached']}  diffs[:4]={R['Q2'][f'{pol}_thr{thr}']['residual_diffs_sample']}")

    # ------------------------------------------------------------------ Q3
    section("Q3: max_warmup_steps on E (dense head 8)")
    R["Q3"] = {}
    for mw in (8, 14, 4, 0):
        refresh_static(toy, MASKS["E"], **{**base_kwargs(), "max_warmup_steps": mw})
        lv = live(toy)
        r = run_request(toy)
        R["Q3"][f"warmup{mw}"] = {"cached": r["cached"], "n": len(r["cached"]),
                                  "warmup_steps_resolved": lv["warmup_steps_resolved"],
                                  "removed_vs_mask": sorted(set(zeros_of(MASKS['E'])) - set(r["cached"]))}
        log(f"  max_warmup={mw:2d}: resolved warmup={lv['warmup_steps_resolved']} cached {len(r['cached'])} "
            f"removed={R['Q3'][f'warmup{mw}']['removed_vs_mask']}")

    # ------------------------------------------------------------------ Q4
    section("Q4 (pitfall 452): which user fields survive each refresh path")
    NONDEF = dict(Fn_compute_blocks=6, Bn_compute_blocks=1, max_warmup_steps=10, max_cached_steps=30,
                  max_continuous_cached_steps=2, residual_diff_threshold=0.33, enable_separate_cfg=False)
    q4 = ToyH3().to(DEV)
    enable(q4, q4.transformer_blocks, NONDEF)
    R["Q4"] = {"enable_values": NONDEF, "after_enable": live(q4)}
    log("  after enable:", R["Q4"]["after_enable"])
    # (i) main's way
    refresh_static(q4, MASKS["E"])
    R["Q4"]["i_main_way"] = live(q4)
    r = run_request(q4)
    R["Q4"]["i_main_way"]["cached"] = r["cached"]
    R["Q4"]["i_main_way"]["blocks_run_on_cached"] = sorted(set(r["ran"][i] for i in r["cached"]))
    expect(r["cached"] == zeros_of(MASKS["E"]) and R["Q4"]["i_main_way"]["blocks_run_on_cached"] == [7],
           "Q4 (i): E cached set with Fn=6+Bn=1 = 7 blocks per cached forward")
    log("  (i) DBCacheConfig().reset(n, mask, static):", R["Q4"]["i_main_way"])
    dropped = [k for k in ("Fn_compute_blocks", "Bn_compute_blocks", "max_warmup_steps", "max_cached_steps",
                           "max_continuous_cached_steps", "residual_diff_threshold")
               if R["Q4"]["i_main_way"][k] != NONDEF[k]]
    R["Q4"]["i_dropped_fields"] = dropped
    log("  (i) dropped fields:", dropped)
    # (ii) explicit kwargs
    if trt:
        cfg = trt["Cfg"](Fn_compute_blocks=6, Bn_compute_blocks=1, max_warmup_steps=10, max_cached_steps=30,
                         max_continuous_cached_steps=2, residual_diff_threshold=0.33)
        kw = trt["kw"](cfg)
        kw.pop("num_inference_steps")
        R["Q4"]["ii_kwargs_from_cfg"] = {k: v for k, v in kw.items()}
        cfg2 = DBCacheConfig().reset(**kw, num_inference_steps=N, steps_computation_mask=list(MASKS["E"]),
                                     steps_computation_policy="static")
        cache_dit.refresh_context(q4, cache_config=cfg2, verbose=False)
        R["Q4"]["ii_explicit"] = live(q4)
        r = run_request(q4)
        R["Q4"]["ii_explicit"]["cached"] = r["cached"]
        R["Q4"]["ii_explicit"]["blocks_run_on_cached"] = sorted(set(r["ran"][i] for i in r["cached"]))
        expect(r["cached"] == zeros_of(MASKS["E"]), "Q4 (ii): E cached set")
        log("  (ii) reset(**db_cache_kwargs(cfg) - n + mask):", R["Q4"]["ii_explicit"])
        # (iii) main's literal _refresh_ctx with a predefined policy
        cfg3 = trt["Cfg"](Fn_compute_blocks=6, Bn_compute_blocks=1, max_warmup_steps=10, max_cached_steps=30,
                          max_continuous_cached_steps=2, residual_diff_threshold=0.33,
                          scm_steps_mask_policy="fast", scm_steps_policy="static")
        trt["_refresh_ctx"](q4, cfg3, N, False)
        R["Q4"]["iii_main_refresh_ctx_fast"] = live(q4)
        R["Q4"]["iii_mask"] = list(ctx_of(q4).cache_config.steps_computation_mask)
        log("  (iii) main _refresh_ctx(fast, static, 49):", R["Q4"]["iii_main_refresh_ctx_fast"])
        log("        predefined 'fast' mask for 49:", "".join(map(str, R["Q4"]["iii_mask"])))
    # (iv) plain refresh_context(num_inference_steps=49) after a mask: does the mask stick?
    cache_dit.refresh_context(q4, num_inference_steps=N, verbose=False)
    R["Q4"]["iv_plain_after_mask"] = live(q4)
    log("  (iv) plain refresh_context(n=49) after a mask:", R["Q4"]["iv_plain_after_mask"])
    # (v) can a refresh clear the mask?  steps_computation_mask=None is skipped by update()
    cache_dit.refresh_context(q4, cache_config=DBCacheConfig().reset(num_inference_steps=N,
                                                                      steps_computation_mask=None), verbose=False)
    R["Q4"]["v_mask_none"] = live(q4)
    log("  (v) reset(mask=None):", R["Q4"]["v_mask_none"])

    # ------------------------------------------------------------------ Q5
    section("Q5: per-request reset E -> w14_c2 -> E, plain and TaylorSeer order 1")
    R["Q5"] = {}
    runs_E_plain = None
    taylor_toy = None
    for label, calib in (("plain", None), ("taylor1", TaylorSeerCalibratorConfig(taylorseer_order=1))):
        t = ToyH3().to(DEV)
        enable(t, t.transformer_blocks, base_kwargs(), calibrator=calib)
        seq = []
        leaks = []
        for name in ("E", "w14_c2", "E"):
            old = ctx_of(t)
            old_cal = old.calibrator
            refresh_static(t, MASKS[name], **{k: v for k, v in base_kwargs(DENSE_HEAD[name]).items()
                                              if k != "enable_separate_cfg"})
            new = ctx_of(t)
            leaks.append({
                "mask": name, "new_ctx_is_new_object": new is not old,
                "old_buffers_after_refresh": len(old.buffers), "new_buffers": len(new.buffers),
                "new_executed_steps": new.executed_steps, "new_cached_steps": list(new.cached_steps),
                "new_continuous_cached_steps": new.continuous_cached_steps,
                "calibrator_rebuilt": (new.calibrator is not old_cal) if old_cal is not None else None,
                "new_calibrator_states": (len(new.calibrator.states) if new.calibrator is not None else None),
            })
            del old, old_cal
            seq.append(run_request(t))
        same = bitwise_all(seq[0]["outs"], seq[2]["outs"])
        same_set = seq[0]["cached"] == seq[2]["cached"]
        R["Q5"][label] = {"third_equals_first_bitwise": same, "cached_sets_equal": same_set,
                          "cached_E": seq[0]["cached"], "cached_w14": seq[1]["cached"], "leak_checks": leaks}
        expect(same and same_set, f"Q5 {label}: third run bitwise == first")
        log(f"  {label}: third==first bitwise={same} cached-set-equal={same_set} nE={len(seq[0]['cached'])} "
            f"nW14={len(seq[1]['cached'])}")
        for lk in leaks:
            log("    after refresh:", lk)
        if label == "plain":
            runs_E_plain = seq[0]
            plain_toy = t
        else:
            taylor_toy = t
            taylor_E = seq[0]
    # fresh-object check: first E vs a brand-new toy's first E
    fresh = ToyH3().to(DEV)
    enable(fresh, fresh.transformer_blocks, base_kwargs())
    refresh_static(fresh, MASKS["E"], **{k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"})
    rf = run_request(fresh)
    R["Q5"]["plain_first_equals_fresh_toy"] = bitwise_all(rf["outs"], runs_E_plain["outs"])
    expect(R["Q5"]["plain_first_equals_fresh_toy"], "Q5 E on a used toy == E on a fresh toy")
    # Plain DBCache semantics: cached output == R_last_compute + h8 (bitwise)
    exp_plain = []
    lastk = None
    for i in range(N):
        if MASKS["E"][i] == 1:
            lastk = i
            exp_plain.append(REF_OUT[i])
        else:
            exp_plain.append((RES[lastk] + H8[i]).contiguous())
    R["Q5"]["plain_cached_equals_Rlast_plus_h8"] = bitwise_all(exp_plain, runs_E_plain["outs"])
    R["Q5"]["plain_compute_equals_reference"] = all(torch.equal(runs_E_plain["outs"][i], REF_OUT[i])
                                                    for i in range(N) if MASKS["E"][i] == 1)
    R["Q5"]["plain_E_maxabs_vs_uncached"] = maxabs(runs_E_plain["outs"], REF_OUT)
    log("  plain cached == R(last compute) + h8 bitwise:", R["Q5"]["plain_cached_equals_Rlast_plus_h8"],
        " compute fwds == reference:", R["Q5"]["plain_compute_equals_reference"],
        " maxabs vs uncached:", R["Q5"]["plain_E_maxabs_vs_uncached"])
    # calibrator handling on refresh
    cal = {}
    refresh_static(taylor_toy, MASKS["E"], **{k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"})
    cal["no_calibrator_kwarg"] = live(taylor_toy)["calibrator"]
    cache_dit.refresh_context(taylor_toy, cache_config=DBCacheConfig().reset(num_inference_steps=N),
                              calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=2), verbose=False)
    c = ctx_of(taylor_toy)
    cal["with_order2"] = {"calibrator": repr(c.calibrator), "n_derivatives": getattr(c.calibrator, "n_derivatives", None)}
    cache_dit.refresh_context(taylor_toy, cache_config=DBCacheConfig().reset(num_inference_steps=N), verbose=False)
    c = ctx_of(taylor_toy)
    cal["then_no_kwarg"] = {"calibrator": repr(c.calibrator), "n_derivatives": getattr(c.calibrator, "n_derivatives", None)}
    cache_dit.refresh_context(taylor_toy, cache_config=DBCacheConfig().reset(num_inference_steps=N),
                              calibrator_config=TaylorSeerCalibratorConfig().reset(enable_calibrator=False),
                              verbose=False)
    c = ctx_of(taylor_toy)
    cal["disable_via_reset_enable_false"] = {"calibrator": repr(c.calibrator),
                                             "encoder_calibrator": repr(c.encoder_calibrator)}
    cache_dit.refresh_context(taylor_toy, cache_config=DBCacheConfig().reset(num_inference_steps=N),
                              calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1), verbose=False)
    cal["restored"] = repr(ctx_of(taylor_toy).calibrator)
    cache_dit.refresh_context(plain_toy, cache_config=DBCacheConfig().reset(num_inference_steps=N),
                              calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1), verbose=False)
    cal["plain_toy_add_calibrator"] = repr(ctx_of(plain_toy).calibrator)
    R["Q5"]["calibrator_refresh"] = cal
    log("  calibrator on refresh:", cal)

    # ------------------------------------------------------------------ Q6
    section("Q6 (pitfall 454): 2-forward warmup request")
    R["Q6"] = {}
    q6 = ToyH3().to(DEV)
    enable(q6, q6.transformer_blocks, base_kwargs())
    refresh_static(q6, [1, 0], max_warmup_steps=1)
    r = run_request(q6, n=2)
    R["Q6"]["a_mask10_warmup1"] = {"cached": r["cached"], "live": live(q6)}
    expect(r["cached"] == [1], "Q6a [1,0] warmup 1 caches forward 1")
    log("  (a) [1,0] + max_warmup_steps=1:", R["Q6"]["a_mask10_warmup1"])
    refresh_static(q6, MASKS["E"])  # only n/mask/policy: does max_warmup_steps=1 persist?
    R["Q6"]["a_then_E_live"] = live(q6)
    log("  (a') then E refresh without warmup kwarg -> live:", R["Q6"]["a_then_E_live"])
    q6b = ToyH3().to(DEV)
    enable(q6b, q6b.transformer_blocks, base_kwargs())
    refresh_static(q6b, [1, 0])
    r = run_request(q6b, n=2)
    R["Q6"]["b_mask10_warmup8"] = {"cached": r["cached"], "live": live(q6b)}
    log("  (b) [1,0] with max_warmup_steps=8 (no override):", R["Q6"]["b_mask10_warmup8"])
    if trt:
        cfgw = trt["Cfg"](Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=8,
                          scm_steps_mask_policy="fast", scm_steps_policy="static")
        q6c = ToyH3().to(DEV)
        enable(q6c, q6c.transformer_blocks, base_kwargs())
        trt["_refresh_ctx"](q6c, cfgw, 2, False)
        r = run_request(q6c, n=2)
        R["Q6"]["c_main_refresh_ctx_2_fresh"] = {"cached": r["cached"], "live": live(q6c)}
        log("  (c) main _refresh_ctx(2) on a fresh toy:", R["Q6"]["c_main_refresh_ctx_2_fresh"])
        refresh_static(q6c, MASKS["E"], **{k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"})
        run_request(q6c)
        trt["_refresh_ctx"](q6c, cfgw, 2, False)
        lv = live(q6c)
        r = run_request(q6c, n=2)
        R["Q6"]["d_main_refresh_ctx_2_after_E"] = {"cached": r["cached"], "live": lv}
        log("  (d) main _refresh_ctx(2) after an E request:", R["Q6"]["d_main_refresh_ctx_2_after_E"])
        refresh_static(q6c, MASKS["E"], **{k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"})
        r = run_request(q6c)
        R["Q6"]["d_next_E_ok"] = r["cached"] == zeros_of(MASKS["E"]) and bitwise_all(r["outs"], runs_E_plain["outs"])
        log("  (d') next E request after the 2-step one: cached ok & bitwise == E:", R["Q6"]["d_next_E_ok"])
    # warmup [1,0] then the timed E: bitwise equal to E on a fresh toy?
    q6e = ToyH3().to(DEV)
    enable(q6e, q6e.transformer_blocks, base_kwargs())
    refresh_static(q6e, [1, 0], max_warmup_steps=1)
    run_request(q6e, n=2)
    refresh_static(q6e, MASKS["E"], **{k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"})
    r = run_request(q6e)
    R["Q6"]["e_E_after_10_warmup_bitwise"] = bitwise_all(r["outs"], runs_E_plain["outs"]) and r["cached"] == zeros_of(MASKS["E"])
    log("  (e) E after a [1,0] warmup request == E fresh:", R["Q6"]["e_E_after_10_warmup_bitwise"])

    # ------------------------------------------------------------------ Q7
    section("Q7: GPU->CPU syncs (torch.cuda.set_sync_debug_mode('warn'))")
    pc, pmsg, _ = count_syncs(lambda: (torch.ones(1, device=DEV).item(), torch.ones(1, device=DEV).item()))
    R["Q7"] = {"positive_control_two_item_calls": pc, "msg": pmsg}
    log("  positive control (2x .item()):", pc, pmsg)
    kw = {k: v for k, v in base_kwargs().items() if k != "enable_separate_cfg"}
    n_ref, _, _ = count_syncs(lambda: refresh_static(toy, MASKS["E"], **kw))
    r = run_request(toy, sync=True)
    R["Q7"]["static_refresh_syncs"] = n_ref
    R["Q7"]["static_compute_fwd_syncs"] = sum(r["syncs"][i] for i in range(N) if MASKS["E"][i] == 1)
    R["Q7"]["static_cached_fwd_syncs"] = sum(r["syncs"][i] for i in r["cached"])
    R["Q7"]["static_cached"] = len(r["cached"])
    n_ref, _, _ = count_syncs(lambda: refresh_static(toy, MASKS["E"], policy="dynamic",
                                                     **{**kw, "max_continuous_cached_steps": -1}))
    r = run_request(toy, sync=True)
    R["Q7"]["dynamic_mask_refresh_syncs"] = n_ref
    R["Q7"]["dynamic_mask_per_forward"] = r["syncs"]
    R["Q7"]["dynamic_mask_cached"] = r["cached"]
    dyn = ToyH3().to(DEV)
    enable(dyn, dyn.transformer_blocks, base_kwargs(max_continuous_cached_steps=-1))
    cache_dit.refresh_context(dyn, num_inference_steps=N, verbose=False)
    r = run_request(dyn, sync=True)
    R["Q7"]["dynamic_nomask_per_forward"] = r["syncs"]
    R["Q7"]["dynamic_nomask_cached"] = r["cached"]
    log("  static: refresh", R["Q7"]["static_refresh_syncs"], "compute fwds", R["Q7"]["static_compute_fwd_syncs"],
        "cached fwds", R["Q7"]["static_cached_fwd_syncs"], f"({R['Q7']['static_cached']} cached)")
    log("  dynamic+mask E: refresh", R["Q7"]["dynamic_mask_refresh_syncs"], "per-forward", R["Q7"]["dynamic_mask_per_forward"])
    log("  dynamic no mask: per-forward", R["Q7"]["dynamic_nomask_per_forward"], "cached", R["Q7"]["dynamic_nomask_cached"])

    # ------------------------------------------------------------------ Q8
    section("Q8 (idea 61): TaylorSeer order 1 on E")
    R["Q8"] = {"cached_set_unchanged": taylor_E["cached"] == zeros_of(MASKS["E"])}
    steps = []

    def rec(i, t):
        c = ctx_of(t)
        if c.calibrator is not None and c.calibrator.states:
            st = next(iter(c.calibrator.states.values()))
            steps.append((i, st.current_step, st.last_non_approximated_step))
    t8 = ToyH3().to(DEV)
    enable(t8, t8.transformer_blocks, base_kwargs(), calibrator=TaylorSeerCalibratorConfig(taylorseer_order=1))
    refresh_static(t8, MASKS["E"], **kw)
    r = run_request(t8, sync=True, per_forward=rec)
    R["Q8"]["cached_set_unchanged_fresh"] = r["cached"] == zeros_of(MASKS["E"])
    R["Q8"]["syncs_compute"] = sum(r["syncs"][i] for i in range(N) if MASKS["E"][i] == 1)
    R["Q8"]["syncs_cached"] = sum(r["syncs"][i] for i in r["cached"])
    R["Q8"]["bitwise_vs_Q5_taylor_E"] = bitwise_all(r["outs"], taylor_E["outs"])
    R["Q8"]["elapsed_at_cached"] = [(i, cs - last) for (i, cs, last) in steps if MASKS["E"][i] == 0]
    # expected outputs under "real gap" (window = k - p) and "unit step" (window = 1)
    real, unit = [], []
    comp = [i for i in range(N) if MASKS["E"][i] == 1]
    for j in range(N):
        if MASKS["E"][j] == 1:
            real.append(REF_OUT[j])
            unit.append(REF_OUT[j])
            continue
        k = max(c for c in comp if c < j)
        p = max(c for c in comp if c < k)
        e = j - k
        for out_list, window in ((real, k - p), (unit, 1)):
            dY = (RES[k] - RES[p]) / window
            o = 0
            o += (1 / math.factorial(0)) * RES[k] * (e ** 0)
            o += (1 / math.factorial(1)) * dY * (e ** 1)
            out_list.append((o + H8[j]).contiguous())
    cached_idx = [j for j in range(N) if MASKS["E"][j] == 0]
    R["Q8"]["bitwise_vs_real_gap_formula"] = all(torch.equal(r["outs"][j], real[j]) for j in cached_idx)
    R["Q8"]["maxabs_vs_real_gap"] = max(float((r["outs"][j].float() - real[j].float()).abs().max()) for j in cached_idx)
    R["Q8"]["maxabs_vs_unit_step"] = max(float((r["outs"][j].float() - unit[j].float()).abs().max()) for j in cached_idx)
    R["Q8"]["maxabs_vs_uncached_taylor"] = maxabs(r["outs"], REF_OUT)
    R["Q8"]["maxabs_vs_uncached_plain"] = R["Q5"]["plain_E_maxabs_vs_uncached"]
    R["Q8"]["calibrator_buffers"] = "dY_prev[0..order] + dY_current[0..order] per stream (residual-sized)"
    log("  Q8:", {k: v for k, v in R["Q8"].items() if k != "elapsed_at_cached"})
    log("  elapsed at cached forwards (i, elapsed):", R["Q8"]["elapsed_at_cached"][:12], "...")

    # ------------------------------------------------------------------ Q9
    section("Q9 (pitfall 565): deferred residual vs Pattern_3; fix 1; grouped fix 2")
    R["Q9"] = {}
    # sanity: deferred chain, no cache, == non-deferred reference
    dnc = ToyH3(deferred=True).to(DEV)
    dnc.assign_block_roles(materialize_at=set())
    o = [call(dnc, i).clone() for i in range(N)]
    R["Q9"]["deferred_nocache_bitwise_vs_ref"] = bitwise_all(o, REF_OUT)
    expect(R["Q9"]["deferred_nocache_bitwise_vs_ref"], "Q9 deferred no-cache == reference")
    log("  deferred chain without cache == reference:", R["Q9"]["deferred_nocache_bitwise_vs_ref"])
    # (a) landing 7 as is (every block defers), and the 'finer fix' (materialize only at Fn-1 and last)
    for label, roles in (("a_all_deferred", set()), ("a_finer_fn_boundary_7_49", {7, 49})):
        dt = ToyH3(deferred=True).to(DEV)
        dt.assign_block_roles(materialize_at=roles)
        enable(dt, dt.transformer_blocks, base_kwargs())
        refresh_static(dt, MASKS["E"], **kw)
        try:
            r = run_request(dt)
            R["Q9"][label] = {"raised": False, "cached": r["cached"],
                              "maxabs_cached_vs_nondeferred": max(float((r["outs"][j].float() - runs_E_plain["outs"][j].float()).abs().max())
                                                                  for j in r["cached"]) if r["cached"] else None}
        except Exception as ex:  # noqa: BLE001
            tb = traceback.extract_tb(ex.__traceback__)
            R["Q9"][label] = {"raised": True, "type": type(ex).__name__, "msg": str(ex)[:300],
                              "frames": [f"{fr.filename.split('dependencies-1/')[-1]}:{fr.lineno}: {fr.line}" for fr in tb[-4:]],
                              "block_calls_before_error": sum(b.calls for b in dt.transformer_blocks)}
        log(f"  ({label}):", R["Q9"][label])
    # (b) fix 1 = branch 2's guard: deferral off (every block materializes) when the cache is on
    f1 = ToyH3(deferred=True).to(DEV)
    f1.assign_block_roles(materialize_at="all")
    enable(f1, f1.transformer_blocks, base_kwargs())
    refresh_static(f1, MASKS["E"], **kw)
    r = run_request(f1)
    R["Q9"]["b_fix1"] = {"bitwise_all_49": bitwise_all(r["outs"], runs_E_plain["outs"]),
                         "cached_same": r["cached"] == runs_E_plain["cached"], "n_cached": len(r["cached"])}
    expect(R["Q9"]["b_fix1"]["bitwise_all_49"] and R["Q9"]["b_fix1"]["cached_same"], "Q9b fix 1 bitwise")
    log("  (b) fix 1:", R["Q9"]["b_fix1"])
    # (c) fix 2 = idea 79 design A: ModuleList[GroupA(blocks[:8]), GroupB(blocks[8:])], Fn=1, Bn=0
    for mname in ("E", "w14_c2"):
        gt = ToyH3(deferred=True, group_split=8).to(DEV)
        gt.assign_block_roles(materialize_at={7, NB - 1})
        enable(gt, gt.cache_groups, base_kwargs(Fn_compute_blocks=1))
        # reference for this mask: non-deferred, ungrouped, Fn=8
        rt = ToyH3().to(DEV)
        enable(rt, rt.transformer_blocks, base_kwargs())
        kwm = {k: v for k, v in base_kwargs(DENSE_HEAD[mname]).items() if k != "enable_separate_cfg"}
        refresh_static(rt, MASKS[mname], **kwm)
        rr = run_request(rt)
        ref_stats = stats_cached_steps(rt)
        args_ok = []

        def check_args(i, tt):
            ok = all(g.last_args == EXPECTED_ARGS for g in tt.cache_groups)
            ok = ok and all(b.last_args == EXPECTED_ARGS for b in tt.transformer_blocks if b.last_args is not None)
            args_ok.append(ok)
        refresh_static(gt, MASKS[mname], **{**kwm, "Fn_compute_blocks": 1})
        lv = live(gt)
        g_calls0 = [g.calls for g in gt.cache_groups]
        r = run_request(gt, per_forward=check_args)
        g_stats = stats_cached_steps(gt)
        shared = all(gt.cache_groups[0].blocks[i] is gt.transformer_blocks[i] for i in range(8)) and \
            all(gt.cache_groups[1].blocks[i - 8] is gt.transformer_blocks[i] for i in range(8, NB))
        n_params_unique = len(list(gt.parameters()))
        n_params_ref = len(list(rt.parameters()))
        sd_keys = len(gt.state_dict())
        sd_keys_ref = len(rt.state_dict())
        res = {
            "bitwise_all_49": bitwise_all(r["outs"], rr["outs"]),
            "cached_same": r["cached"] == rr["cached"], "n_cached": len(r["cached"]),
            "blocks_run_on_cached": sorted(set(r["ran"][i] for i in r["cached"])),
            "group_calls": [g.calls - c0 for g, c0 in zip(gt.cache_groups, g_calls0)],
            "stats_cached_steps_same": g_stats[0][0] == ref_stats[0][0], "stats_count": g_stats[1][0],
            "stats_count_ref": ref_stats[1][0],
            "args_kwargs_reach_every_group_and_block": all(args_ok),
            "params_shared_no_copies": shared, "unique_params": n_params_unique, "unique_params_ref": n_params_ref,
            "state_dict_keys": sd_keys, "state_dict_keys_ref": sd_keys_ref,
            "live_Fn": lv["Fn_compute_blocks"], "live_warmup": lv["warmup_steps_resolved"],
        }
        # switch-all-True grouped == boundary-only grouped (ruling 594 (3))
        gt.assign_block_roles(materialize_at="all")
        refresh_static(gt, MASKS[mname], **{**kwm, "Fn_compute_blocks": 1})
        r2 = run_request(gt)
        res["grouped_all_true_bitwise_vs_boundary_only"] = bitwise_all(r2["outs"], r["outs"])
        R["Q9"][f"c_grouped_{mname}"] = res
        expect(res["bitwise_all_49"] and res["cached_same"], f"Q9c grouped {mname} bitwise")
        log(f"  (c) grouped {mname}:", res)

    # ------------------------------------------------------------------ done
    R["failures"] = FAIL
    with open(os.path.join(OUT, "toy_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    log("")
    log("FAILURES:", FAIL)
    print(json.dumps({"passed": not FAIL, "n_failures": len(FAIL),
                      "results": os.path.join(OUT, "toy_results.json")}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        with open(os.path.join(OUT, "toy_results_partial.json"), "w") as f:
            json.dump(R, f, indent=1, default=str)
        print(json.dumps({"passed": False, "error": traceback.format_exc()[-1500:]}))
        sys.exit(1)
