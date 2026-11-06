"""Idea 329 correctness (untimed), every knob setting, one process per slot_start.

Driver (no CHECK_CHILD): runs this file once per slot in CHECK_SLOTS (default "0,192") with
CHECK_CHILD=1 and merges the children's last JSON lines.
Child: for every config in CHECK_CONFIGS (default all four of OFF / KACQ / KACQ+ACQREL /
KACQ+EW2+ACQREL plus EW2-only and ACQREL-only), with the knobs set in the environment:
  * Part A (check12.part_a's sequence: capture with 3 eager warmups, CUDA graph, CHECK_REPLAYS
    replays each after a 512 MB read-only flush) at M = 1, 2, 4, 8 (D 0..min(12, 6M)) and
    M = 16, 32 (a few D): after EVERY replay q1/sf1 bitwise == main's fc1_act_q1 output, out
    inside main's own replay band (bf16 scatter-add order slack), finite, and the sync
    workspace (incl. the per-column KACQ counters) + split-K counters all 0; max abs per replay.
  * Part B: Sm12xW4a8DecodeState.warmup then CUDA-graph capture of st.run at batch 1..32 with
    no JIT during capture (compile cache unchanged), no memset node, replay inside main's band.
  * Part C: aux-stream contention rounds, no hang, q1/sf1 bitwise, counters 0.
Also checks that the decode leaf's _WARMED key carries every TRTLLM_SM12X_FI_* knob.
Last line: {"passed": ..., ...}.

History: job 1662 (rc 1, configs {}) was a compile-time assert, not a numerics result: with
EARLY_W2 on, fc1's sC sat at [70656,72704) inside fc2's sB [16384,81920). The kernel now puts
fc1's sC first in the smem pool when EARLY_W2 is on (under fc2's sA), so the ranges are disjoint.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KNOBS = {"KACQ": "TRTLLM_SM12X_FI_FC12_KACQ", "EW2": "TRTLLM_SM12X_FI_FC12_EARLY_W2",
         "ACQREL": "TRTLLM_SM12X_FI_SPLITK_ACQREL"}
DEFAULT_CONFIGS = "OFF:;KACQ:KACQ;ACQREL:ACQREL;EW2:EW2;KACQ_ACQREL:KACQ,ACQREL;ALL:KACQ,EW2,ACQREL"


def slot_guard(limit_gib=4.0, wait_s=180, poll_s=10):
    """WAIT-ONLY dirty-slot guard (top lead rulings 20:5x / 21:0x PST 11-05, pitfalls 3067/3072).

    Polls nvidia-smi memory.used for this job's CUDA_VISIBLE_DEVICES for up to wait_s. If any
    GPU stays above limit_gib, prints 'DIRTY SLOT' with per-GPU memory and exits 3 before any
    CUDA allocation. Never kills processes."""
    import time
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    t0 = time.time()
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception as e:  # no nvidia-smi: report and go on (torch check follows in child)
            print(f"[check_329] slot guard: nvidia-smi unavailable ({e})", flush=True)
            return
        used = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and (not vis or parts[0] in vis or parts[1] in vis):
                used[parts[1]] = float(parts[2]) / 1024.0
        dirty = {k: round(v, 2) for k, v in used.items() if v > limit_gib}
        print(f"[check_329] slot guard t+{time.time() - t0:.0f}s memory.used GiB "
              f"{ {k: round(v, 2) for k, v in used.items()} }", flush=True)
        if not dirty:
            return
        if time.time() - t0 >= wait_s:
            print(f"DIRTY SLOT {dirty}", flush=True)
            print(json.dumps({"passed": False, "error": f"DIRTY SLOT {dirty} (infra, pitfall 3072)"}))
            sys.exit(3)
        time.sleep(poll_s)


def driver():
    slot_guard()
    slots = [s for s in os.environ.get("CHECK_SLOTS", "0,192").split(",") if s]
    results, ok = [], True
    for s in slots:
        env = dict(os.environ, CHECK_CHILD="1", CHECK_SLOT=s)
        if os.environ.get(f"CHECK_CONFIGS_{s}"):
            env["CHECK_CONFIGS"] = os.environ[f"CHECK_CONFIGS_{s}"]
        print(f"[check_329] slot_start {s}: child start", flush=True)
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                             stdout=subprocess.PIPE, text=True)
        last = None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("{"):
                last = line
        rc = p.wait()
        res = json.loads(last) if last else {"passed": False, "error": f"no result, rc {rc}"}
        res["rc"] = rc
        ok = ok and rc == 0 and bool(res.get("passed"))
        results.append(res)
    summary = [{"slot_start": r.get("slot_start"), "passed": r.get("passed"), "rc": r.get("rc"),
                "configs": {c["config"]: c["ok"] for c in r.get("configs", [])}} for r in results]
    print(json.dumps({"passed": ok, "compared": "fused fc12 (each knob config) vs main's "
                      "fc1_act_q1 + fc2_finalize (protected FI ops) and FP32 dequantized reference",
                      "summary": summary, "slots": results}))


def child():
    import torch

    import check12
    from common12 import E, H, SHAPE, SLOT, TOPK, Runner, capture, log, make_state, routing
    import common as _common
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import kernel_moe_mxfp8_mxfp4_fc12_fused as fk
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils import split_k

    torch.manual_seed(0)
    w_raw = _common.make_weights(seed=0)
    check12.W_RAW = w_raw
    leaf, st = make_state(seed=0)
    check12.FLUSH = torch.ones(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    sync = fk.sync_workspace("cuda")
    scratch, cnt = split_k.workspace(torch.device("cuda"),
                                     torch.cuda.get_device_properties(0).multi_processor_count)
    replays = check12.REPLAYS
    cases = [(m, d) for m in (1, 2, 4, 8) for d in range(0, min(12, TOPK * m) + 1)]
    cases += [(16, 12), (16, 24), (16, 48), (32, 24), (32, 64), (32, 96)]
    quick = os.environ.get("CHECK_QUICK", "0") == "1"
    if quick:  # smoke only (gate jobs): a few cases, no parts B/C
        cases = [(1, 1), (1, 4), (1, 5), (2, 7), (4, 12), (8, 12), (32, 96)]
    configs = []
    for part in os.environ.get("CHECK_CONFIGS", DEFAULT_CONFIGS).split(";"):
        name, _, knobs = part.partition(":")
        on = [k for k in knobs.split(",") if k]
        configs.append((name, {v: ("1" if k in on else "0") for k, v in KNOBS.items()}))
    runner = Runner(leaf, st)
    # main's reference per case, once (it does not depend on the knobs)
    refs = {}
    for m, d in cases:
        torch.manual_seed(1000 + 31 * m + d)
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda")
        ids, w = routing(m, d)
        lids = check12.local_ids(ids)
        loc = ((lids >= 0) & (lids < E)).sum(-1).float()
        gb = capture(lambda: runner(x, ids, w, "base"))
        b = runner.buffers(x, ids)[1:]
        base = check12.replay_all(gb, b, replays)
        del gb
        q1r, sf1r, out0 = base[0]
        band = max((r[2].float() - out0.float()).abs().max().item() for r in base)
        stable = all(torch.equal(r[0].view(torch.uint8), q1r.view(torch.uint8))
                     and torch.equal(r[1], sf1r) for r in base)
        reorder = (loc[:, None] - 1).clamp_min(0) * check12.ulp_bf16(
            check12.abs_contrib(x, lids, w, *w_raw))
        ref = _common.reference(x, lids, w, *w_raw)
        refs[(m, d)] = (x, ids, w, q1r, sf1r, out0, band, stable, reorder, ref)
    log(f"[check_329] slot {SLOT}: main references for {len(cases)} cases")
    out_cfg, all_ok = [], True
    for name, env in configs:
        os.environ.update(env)
        ok, rows = True, []
        for m, d in cases:
            x, ids, w, q1r, sf1r, out0, band, stable, reorder, ref = refs[(m, d)]
            gf = capture(lambda: runner(x, ids, w, "fused"))
            b = runner.buffers(x, ids)[1:]
            fused = check12.replay_all(gf, b, replays)
            del gf
            per, case_ok = [], stable
            for q, s, o in fused:
                qeq = torch.equal(q.view(torch.uint8), q1r.view(torch.uint8))
                seq = torch.equal(s, sf1r)
                dlt = (o.float() - out0.float()).abs()
                inband = bool((dlt <= torch.maximum(torch.full_like(dlt, band), reorder)).all().item())
                fin = bool(torch.isfinite(o.float()).all().item())
                zero = int((sync != 0).sum().item()) == 0 and int((cnt != 0).sum().item()) == 0
                per.append(round(dlt.max().item(), 5))
                case_ok = case_ok and qeq and seq and inband and fin and zero
            ef = _common.err_stats(fused[-1][2], ref)
            eb = _common.err_stats(out0, ref)
            rows.append({"M": m, "D": d, "ok": case_ok, "band": band, "per_replay_max_abs": per,
                         "ref_fused_max_abs": ef["max_abs"], "ref_base_max_abs": eb["max_abs"]})
            ok = ok and case_ok
            if not case_ok:
                log(f"[check_329] FAIL slot {SLOT} {name} M{m} D{d}: band {band} per-replay {per}")
        log(f"[check_329] slot {SLOT} config {name} {env}: part A ok={ok} "
            f"({len(cases)} cases x {replays} replays, max |fused-main| "
            f"{max(max(r['per_replay_max_abs']) for r in rows)})")
        if quick:
            st.warmup(TOPK)
            okb, rb, okc = True, {"jit_in_capture": None, "rows": []}, True
        else:
            okb, rb = check12.part_b(leaf, st, fk)
            okc, rc = check12.part_c(leaf, st, fk)
        warmed = list(leaf._WARMED)
        env_in_key = any(isinstance(k[-1], tuple) and all((v, env[v]) in k[-1] for v in env)
                         for k in warmed)
        cfg_ok = ok and okb and okc and env_in_key
        all_ok = all_ok and cfg_ok
        log(f"[check_329] slot {SLOT} config {name}: A {ok} B {okb} (jit_in_capture "
            f"{rb['jit_in_capture']}) C {okc} warmed_key_has_env {env_in_key} -> {cfg_ok}")
        out_cfg.append({"config": name, "env": env, "ok": cfg_ok, "A_ok": ok, "B_ok": okb,
                        "C_ok": okc, "warmed_key_has_env": env_in_key,
                        "B_jit_in_capture": rb["jit_in_capture"],
                        "B_memset_nodes": sum(r["memset_nodes"] for r in rb["rows"]),
                        "A_rows": rows})
    for v in KNOBS.values():
        os.environ.pop(v, None)
    print(json.dumps({"passed": all_ok, "slot_start": SLOT, "shape": SHAPE, "replays": replays,
                      "configs": out_cfg}))


if __name__ == "__main__":
    if os.environ.get("CHECK_CHILD") == "1":
        child()
    else:
        driver()
