"""Branch 188 (idea 332): Cake decode_body entry trigger (binding enable_pdl bit 2) on cake_q1/chain.py's chain.

Imports chain.py's builders (World, Chain, Patch, capture, time_arms, flush) unchanged; chain.py is not edited.
40-layer CSA2 decode attention + LowMWO projection chain, TP4 rank shapes, cold L2 (256 MB flush per replay),
CUDA graph replay, merge flag from DeepseekV41Attention._lowm_wo_follows (main's landing-53 route).

Arms (candidate tree; the base tree has no entry trigger and runs A0 only):
  A0 : body entry trigger forced OFF (CakeDecode.body_trigger_entry = False)  == main's launches
  A1 : production default (None -> cake_decode._BODY_TRIGGER_ENTRY_COUNTS)  == the branch's default
  A2 : forced ON (True)

Modes
  time  : interleaved timing of the arms per count (5 rounds x (2 warmup + 20)), plus (--spans) Kineto spans per
          arm: body / merge / K2' / K3' durations, merge exposed tail (merge end - body end), K2'/K3' start/end
          offsets from the merge end, layer span prep start -> K3' end. Last line = evidence JSON
          (kernel_ms = A1, the branch default; base tree: A0 = main).
  check : correctness (candidate): untimed, the harness's sequence (10 warmup + 100 iterations, finite
          sentinels before every replay, Cake scratch shared across layers), every iteration bitwise vs the
          eager entry-OFF, merge-<false> chain (main's code). Arms: A1 (default), A2 + merge<false>,
          env TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY=0, env TRTLLM_CSA2_ATTN_PDL=0 (entry ON), A0, and a skip-wait
          K2' control behind A2 that must FAIL. Then >= 8 advancing decode steps: the entry-ON graph captured at
          the warmup state is replayed on new q / pool rows / selections each step, compared with a fresh eager
          reference of each step. Prints the WAR audit (from source) too.
"""

import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cake_q1_chain", os.path.join(HERE, "chain.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
import torch  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.csa2 import cake_decode as CD  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2 import module as CM  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import store_rows  # noqa: E402

ENTRY = hasattr(CD, "_BODY_TRIGGER_ENTRY_COUNTS")
log = C.log
ARM_ENTRY = {"A0": False, "A1": None, "A2": True}

WAR_AUDIT = [
    {"kernel": "CakeMergeF32Kernel<true|false> (binding.cu)", "launch": "PDL dependent of decode_body",
     "global_writes_before_wait": "none (<true> only fires launch_dependents before griddepcontrol.wait; every "
                                  "mid_out/mid_lse load and output/out_lse store is after the wait)",
     "aliases_body_inputs": False},
    {"kernel": "csa2_lowm_wo_a_kernel (K2', csa2LowMWo.cu)", "launch": "PDL dependent of the merge",
     "global_writes_before_wait": "none (pre-wait: mbarrier init, cp.async.bulk of its wo_a rows into SHARED "
                                  "memory, wo_a scale bytes to shared, launch_dependents at :265; wait :269)",
     "aliases_body_inputs": False},
    {"kernel": "csa2_mxfp8_low_m_gemm_kernel (K3' wo_b, mxfp8LowMGemm.cu, flags=0)",
     "launch": "PDL dependent of K2' (K2' triggers before its wait)",
     "global_writes_before_wait": "none in production (pre-wait: bulk copies of its weights/scales into shared "
                                  "memory; csa2_stamp(qout,0) writes only when the debug ts pointer is non-null, "
                                  "null in production); its trigger (:1257) is AFTER its wait (:1094) unless "
                                  "debug flag bit 1",
     "aliases_body_inputs": False},
    {"kernel": "next kernel after K3' (all-reduce / next layer's first kernel)", "launch": "after K3' triggers",
     "global_writes_before_wait": "n/a: K3' triggers only after its own griddepcontrol.wait, which returns only "
                                  "after K2' completed, which needs the merge complete, which needs the whole "
                                  "decode_body grid complete. Nothing past K3' launches before this layer's "
                                  "body ends, so the next layer's first kernel cannot launch before it.",
     "aliases_body_inputs": False},
]


def set_entry(chain, arm):
    """Set every layer's CakeDecode entry mode (creates the Cake objects with one eager call if needed)."""
    if not ENTRY:
        return
    if any(getattr(L["fi"], "_cake", None) is None for L in chain.layers):
        chain()
        torch.cuda.synchronize()
    value = ARM_ENTRY[arm] if isinstance(arm, str) else arm
    for L in chain.layers:
        L["fi"]._cake.body_trigger_entry = value


def capture_arm(chain, arm, merge="q1", patch=None):
    set_entry(chain, arm)
    return C.capture(chain, merge, patch)


def spans(g, tag, reps=5):
    """Kineto per-layer timeline of one graph (attribution only)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            C.flush()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
    path = C.ART / f"kineto332_{tag}.json"
    prof.export_chrome_trace(str(path))
    evs = json.load(open(path))
    evs = evs.get("traceEvents", evs) if isinstance(evs, dict) else evs
    kern = sorted((float(e["ts"]), float(e.get("dur", 0)), e.get("name", "")) for e in evs
                  if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel")
    kinds = {C.K_PREP: "prep", C.K_BODY: "body", C.K_MERGE: "merge", C.K2: "K2", C.K3: "K3"}

    def kind(n):
        for k, v in kinds.items():
            if k in n:
                return v
        return None

    seq = [(ts, d, kind(n)) for ts, d, n in kern if kind(n)]
    preps = [i for i, s in enumerate(seq) if s[2] == "prep"]
    m = {k: [] for k in ("body_us", "merge_us", "K2_us", "K3_us", "merge_tail_us", "merge_start_minus_body_start",
                         "K2_start_rel_merge_end", "K2_end_rel_merge_end", "K3_start_rel_merge_end",
                         "K3_end_rel_merge_end", "layer_span_us", "K2_start_minus_body_end")}
    for a, b in zip(preps, preps[1:] + [len(seq)]):
        blk = seq[a:b]
        first = {}
        for ts, d, k in blk:
            first.setdefault(k, (ts, d))
        if not all(k in first for k in ("body", "merge", "K2", "K3")):
            continue
        bs, bd = first["body"]
        ms, md = first["merge"]
        k2s, k2d = first["K2"]
        k3s, k3d = [(ts, d) for ts, d, k in blk if k == "K3"][-1]
        me = ms + md
        m["body_us"].append(bd)
        m["merge_us"].append(md)
        m["K2_us"].append(k2d)
        m["K3_us"].append(k3d)
        m["merge_tail_us"].append(me - (bs + bd))
        m["merge_start_minus_body_start"].append(ms - bs)
        m["K2_start_rel_merge_end"].append(k2s - me)
        m["K2_end_rel_merge_end"].append(k2s + k2d - me)
        m["K3_start_rel_merge_end"].append(k3s - me)
        m["K3_end_rel_merge_end"].append(k3s + k3d - me)
        m["layer_span_us"].append(k3s + k3d - blk[0][0])
        m["K2_start_minus_body_end"].append(k2s - (bs + bd))
    out = {k: C.summ(v) for k, v in m.items() if v}
    out["layers_seen"] = len(m["body_us"])
    return out


def default_entry_on(count):
    mode = CD._body_trigger_entry_mode(None)
    return (count in CD._BODY_TRIGGER_ENTRY_COUNTS) if mode is None else mode


def mode_time(args):
    world = C.World(args.ckpt)
    arms = ["A0", "A1"] if ENTRY else ["A0"]
    if ENTRY and args.with_a2:
        arms.append("A2")
    batch, table = {}, {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = C.Chain(world, count, seed=count)
        graphs = {}
        for arm in arms:
            graphs[arm], _ = capture_arm(chain, arm)
        t = C.time_arms(graphs)
        row = {a: {"whole_us": {k: (v * 1e3 if k != "noise" else v) for k, v in s.items()},
                   "per_layer_us": s["median"] * 1e3 / len(C.LAYERS)} for a, s in t.items()}
        if args.spans:
            for arm, g in graphs.items():
                row[arm]["spans"] = spans(g, f"M{count}_{arm}")
        if ENTRY:
            d = t["A1"]["median"] - t["A0"]["median"]
            row["A1_minus_A0_us"] = d * 1e3
            row["A1_vs_A0_rel"] = d / t["A0"]["median"]
            row["A1_entry_on"] = default_entry_on(count)
        table[str(count)] = row
        rep = "A1" if ENTRY else "A0"
        batch[str(count)] = {"kernel_ms": t[rep]["median"], "noise_floor": t[rep]["noise"],
                             "p10_ms": t[rep]["p10"], "p90_ms": t[rep]["p90"],
                             "per_layer_us": t[rep]["median"] * 1e3 / len(C.LAYERS)}
        if ENTRY:
            batch[str(count)]["A0_ms"] = t["A0"]["median"]
            batch[str(count)]["A0_noise"] = t["A0"]["noise"]
        log(f"M={count} " + json.dumps(row)[:9000])
        del graphs, chain
        torch.cuda.empty_cache()
    (C.ART / "table332.json").write_text(json.dumps(table, indent=1))
    hashes = hash_children(args) if args.hash_children else None
    print(json.dumps({"batch": batch, "entry_tree": ENTRY, "reported_arm": "A1" if ENTRY else "A0",
                      "output_sha256": hashes,
                      "default_counts": sorted(CD._BODY_TRIGGER_ENTRY_COUNTS) if ENTRY else None,
                      "layers": len(C.LAYERS), "lease": C.lease_info(), "output": {"class": "exact"}}))


HASH_ENVS = (("default", {}), ("f32_partials0", {"TRTLLM_CSA2_CAKE_F32_PARTIALS": "0"}),
             ("enable_pdl0", {"TRTLLM_ENABLE_PDL": "0"}), ("attn_pdl0", {"TRTLLM_CSA2_ATTN_PDL": "0"}))


def mode_hash(args):
    """sha256 of every output of one graph replay per arm (seeded inputs): equal across trees == bitwise."""
    import hashlib
    world = C.World(args.ckpt)
    arms = ["A0", "A1", "A2"] if ENTRY else ["A0"]
    out = {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = C.Chain(world, count, seed=count)
        row = {}
        for arm in arms:
            g, outs = capture_arm(chain, arm)
            hs = []
            for _ in range(2):
                chain.poison(outs)
                C.flush()
                g.replay()
                torch.cuda.synchronize()
                h = hashlib.sha256()
                for o, L in zip(outs, chain.layers):
                    h.update(L["out"].contiguous().view(torch.uint8).cpu().numpy().tobytes())
                    h.update(o.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                hs.append(h.hexdigest()[:16])
            row[arm] = hs[0] if hs[0] == hs[1] else f"NONDET {hs}"
            del g, outs
        out[str(count)] = row
        del chain
        torch.cuda.empty_cache()
    print(json.dumps({"passed": True, "hash": out}))


def hash_children(args):
    import subprocess
    import sys
    res = {}
    for tag, env in HASH_ENVS:
        cmd = [sys.executable, os.path.abspath(__file__), "hash", args.ckpt, "--counts", args.counts]
        p = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        try:
            res[tag] = json.loads(lines[-1])["hash"]
        except Exception:  # noqa: BLE001
            res[tag] = {"error": p.returncode, "stderr_tail": p.stderr[-1500:]}
        log(f"hash {tag} {env}: {json.dumps(res[tag])}")
    return res


def compare(chain, outs, ref_res, ref_att):
    diff_layers, worst, finite = 0, 0.0, True
    for o, r, L, a in zip(outs, ref_res, chain.layers, ref_att):
        if not (torch.equal(o, r) and torch.equal(L["out"], a)):
            diff_layers += 1
            worst = max(worst, (o.float() - r.float()).abs().max().item(),
                        (L["out"].float() - a.float()).abs().max().item())
        finite &= bool(torch.isfinite(o).all()) and bool(torch.isfinite(L["out"]).all())
    return diff_layers, worst, finite


def eager_reference(chain):
    """Main's code: entry OFF, merge <false>, eager."""
    set_entry(chain, "A0")
    chain.arm = "off"
    res = [r.clone() for r in chain()]
    att = [L["out"].clone() for L in chain.layers]
    torch.cuda.synchronize()
    return res, att


def run_iters(chain, g, outs, ref_res, ref_att, iters):
    bad, max_abs, finite, samples = 0, 0.0, True, []
    for it in range(iters):
        chain.poison(outs)  # reset point: outputs + Cake scratch hold finite sentinels
        C.flush()
        g.replay()
        torch.cuda.synchronize()
        dl, w, f = compare(chain, outs, ref_res, ref_att)
        bad += int(dl > 0)
        max_abs = max(max_abs, w)
        finite &= f
        if it < 2 or (dl and len(samples) < 6):
            samples.append((it, dl, round(w, 4)))
    return dict(iterations=iters, bad_iterations=bad, max_abs_diff=max_abs, finite=finite, samples=samples)


class Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        os.environ.update(self.kv)
        CM._attn_pdl_enabled.cache_clear()
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        CM._attn_pdl_enabled.cache_clear()


def advance(world, chain, step):
    """One decode step's new state, written in place into what the captured graph reads."""
    g = torch.Generator(device="cuda").manual_seed(9000 + step)
    for li, L in enumerate(chain.layers):
        L["q"].copy_((torch.randn(L["q"].shape, generator=g, device="cuda") * 0.5).to(torch.bfloat16))
        sw = L["args"].sparse_backend_args.swa_indices
        sw.copy_(torch.roll(sw, shifts=step * 3 + 1, dims=1))
        rows = torch.randint(0, C.CC.SWA_ROWS, (256,), generator=g, device="cuda")
        store_rows(world.swa_pools[li], rows,
                   torch.randn((256, C.CC.DIM), generator=g, device="cuda").to(torch.bfloat16), "swa")
    seen = set()
    for L in chain.layers:
        lg = L["args"].sparse_backend_args.main_logical_indices
        if lg is not None and lg.data_ptr() not in seen:
            seen.add(lg.data_ptr())
            lg.copy_(torch.roll(lg, shifts=step * 5 + 2, dims=1))
    for mp in world.main_pools.values():
        rows = torch.randint(0, C.CC.MAIN_ROWS, (1024,), generator=g, device="cuda")
        store_rows(mp, rows, torch.randn((1024, C.CC.DIM), generator=g, device="cuda").to(torch.bfloat16), "main")
    torch.cuda.synchronize()


def mode_check(args):
    assert ENTRY, "check runs in the candidate tree"
    print("WAR_AUDIT " + json.dumps(WAR_AUDIT, indent=1), flush=True)
    world = C.World(args.ckpt)
    ok, report = True, {}
    for count in [int(c) for c in args.counts.split(",")]:
        chain = C.Chain(world, count, seed=20 + count)
        set_entry(chain, "A0")
        chain.arm = "off"
        chain()
        torch.cuda.synchronize()
        n_shared = chain.share_scratch()
        ref_res, ref_att = eager_reference(chain)
        flags = [bool(C.DeepseekV41Attention._lowm_wo_follows(L["mod"], L["out"].view(count, -1)))
                 for L in chain.layers]
        rep = {"shared_scratch_layers": n_shared, "q1_flag_layers": sum(flags),
               "A1_default_entry_on": default_entry_on(count)}
        all_arms = (
            ("A1_default_q1", "A1", "q1", None, None, True),
            ("A2_on_merge_false", "A2", "off", None, None, True),
            ("env_entry0", None, "q1", {"TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY": "0"}, None, True),
            ("env_attn_pdl0_entry_on", "A2", "q1", {"TRTLLM_CSA2_ATTN_PDL": "0"}, None, True),
            ("A0_off_q1", "A0", "q1", None, None, True),
            ("A2_skipwait_control", "A2", "q1", None, C.Patch(k2_skip_wait=True), False),
        )
        wanted = set(args.arms.split(",")) if args.arms else None
        arms = [a for a in all_arms if wanted is None or a[0] in wanted]
        for name, arm, merge, env, patch, expect in arms:
            with Env(**(env or {})):
                parsed = None
                if arm is None:  # production parse of the env switch
                    parsed = CD._body_trigger_entry_mode(None)
                    set_entry(chain, bool(parsed) if parsed is not None else default_entry_on(count))
                    g, outs = C.capture(chain, merge, patch)
                else:
                    g, outs = capture_arm(chain, arm, merge, patch)
                chain.arm = merge
                mflags = [bool(chain.flag(L)) for L in chain.layers]
            iters = args.warmup + args.iters if expect else 20
            res = run_iters(chain, g, outs, ref_res, ref_att, iters)
            res["merge_true_layers"] = sum(mflags)
            res["entry_on_layers"] = sum(bool(L["fi"]._cake.body_trigger_entry) for L in chain.layers)
            if parsed is not None:
                res["env_parsed_entry"] = parsed
            if expect:
                res["passed"] = res["bad_iterations"] == 0 and res["finite"]
                ok &= res["passed"]
            else:
                res["control_failed_as_expected"] = res["bad_iterations"] > 0
                ok &= res["control_failed_as_expected"]
            rep[name] = res
            log(f"M={count} {name}: {json.dumps(res)}")
            del g, outs
        if args.steps <= 0:
            report[str(count)] = rep
            del chain
            torch.cuda.empty_cache()
            continue
        # advancing decode steps: entry-ON graph captured at the warmup state, replayed on real states
        g, outs = capture_arm(chain, "A2", "q1")
        steps, prev, step_ok = [], None, True
        for step in range(1, args.steps + 1):
            advance(world, chain, step)
            r_res, r_att = eager_reference(chain)
            changed = prev is None or any(not torch.equal(a, b) for a, b in zip(r_att, prev))
            prev = r_att
            res = run_iters(chain, g, outs, r_res, r_att, 3)
            res["reference_changed_vs_prev_step"] = bool(changed)
            steps.append(res)
            step_ok &= res["bad_iterations"] == 0 and res["finite"] and changed
        rep["advancing_steps"] = {"steps": len(steps), "passed": bool(step_ok),
                                  "max_abs": max(s["max_abs_diff"] for s in steps),
                                  "all_changed": all(s["reference_changed_vs_prev_step"] for s in steps)}
        ok &= step_ok
        log(f"M={count} advancing: {json.dumps(rep['advancing_steps'])}")
        del g, outs
        report[str(count)] = rep
        del chain
        torch.cuda.empty_cache()
    children = {}
    if args.children:
        # Process-level switches (read once per process): each child compares against its own eager
        # entry-OFF reference run under the SAME env (main's code under that env).
        import subprocess
        import sys
        for tag, env in (("enable_pdl0", {"TRTLLM_ENABLE_PDL": "0"}),
                         ("f32_partials0", {"TRTLLM_CSA2_CAKE_F32_PARTIALS": "0"})):
            cmd = [sys.executable, os.path.abspath(__file__), "check", args.ckpt, "--counts", args.counts,
                   "--warmup", str(args.warmup), "--iters", str(args.iters), "--steps", "0",
                   "--arms", "A1_default_q1,A2_on_merge_false,A0_off_q1"]
            log(f"child {tag}: {env}")
            p = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            try:
                res = json.loads(lines[-1])
            except Exception:  # noqa: BLE001
                res = {"passed": False, "rc": p.returncode, "stderr_tail": p.stderr[-3000:]}
            for ln in p.stdout.splitlines()[-40:]:
                print(f"[{tag}] {ln[:600]}", flush=True)
            children[tag] = {"env": env, "passed": bool(res.get("passed")), "rc": p.returncode,
                             "report": res.get("report")}
            ok &= bool(res.get("passed"))
    print(json.dumps({"passed": bool(ok),
                      "children": children,
                      "compared": "40-layer Cake attention + LowMWO projection CUDA-graph chain, each replay's 40 "
                                  "attention outputs and 40 projection outputs bitwise vs the eager entry-OFF, "
                                  "merge<false> chain (main's code): A1 default, A2 entry ON + merge<false>, env "
                                  "TRTLLM_CSA2_CAKE_BODY_TRIGGER_ENTRY=0, env TRTLLM_CSA2_ATTN_PDL=0 with entry ON, "
                                  "A0; 10 warmup + 100 iterations each, finite sentinels before every replay, Cake "
                                  "scratch shared across layers; skip-wait K2' control behind the entry trigger "
                                  "must differ; then advancing decode steps (graph captured at warmup state, "
                                  "replayed on new q / pool rows / selections, vs a fresh eager reference)",
                      "war_audit_dependents": [w["kernel"] for w in WAR_AUDIT],
                      "report": report}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "check", "hash"])
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--spans", action="store_true")
    ap.add_argument("--with-a2", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--children", action="store_true")
    ap.add_argument("--hash-children", action="store_true")
    a = ap.parse_args()
    {"time": mode_time, "check": mode_check, "hash": mode_hash}[a.mode](a)
