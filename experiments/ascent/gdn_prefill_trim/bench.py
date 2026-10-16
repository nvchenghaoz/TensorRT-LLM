"""Idea 146 STEP 0 (gate) + STEP 1 (exact checks): GDN prefill no-op reset skip.

Window per size: 48 layers x [reset_recurrent_state_rows (or skipped via the
production flag)] -> gather_cast_vk_to_fp32_vk -> cast_scatter_fp32_vk_to_vk,
production pool layout (one uint8 pool, bf16 ssm/conv views, block stride
817,152 elements), one shared fp32 workspace pair, T=2048 per prefill.
Arms switch by setting gdn_mixer._SKIP_NOOP_RESET (False == main's code path).
The flag comes from a real Mamba2Metadata.prepare().

Usage: python -u bench.py [all|gate|check]
"""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def tree_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


print("TREE_COMMIT", tree_commit(), flush=True)

from tensorrt_llm._torch.modules.fla.fused_state_io import (  # noqa: E402
    cast_scatter_fp32_vk_to_vk,
    gather_cast_vk_to_fp32_vk,
)
from tensorrt_llm._torch.modules.mamba import gdn_mixer  # noqa: E402
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata  # noqa: E402
from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import (  # noqa: E402
    reset_recurrent_state_rows,
)

log("imports done; default _SKIP_NOOP_RESET =", gdn_mixer._SKIP_NOOP_RESET)

L = 48
NB = 12
H, V, K = 48, 128, 128
SSM_B = H * V * K * 2  # 1,572,864
CONV_B = 10240 * 3 * 2  # 61,440
CAP = 8
T = 2048
CACHED = 32768
SLOTS = [3, 7, 1, 10]
SIZES = [(1, 1, 0), (2, 1, 1), (4, 3, 1)]  # (num_seqs, P, D)
WARM, ITERS = 10, 50
dev = torch.device("cuda")

torch.manual_seed(0)
pool = torch.empty(L, NB, SSM_B + CONV_B, dtype=torch.uint8, device=dev)
ssm_all = pool[:, :, :SSM_B].view(torch.bfloat16).view(L, NB, H, V, K)
conv_all = pool[:, :, SSM_B:].view(torch.bfloat16).view(L, NB, 10240, 3)
for l in range(L):
    ssm_all[l].copy_(torch.randn(NB, H, V, K, device=dev).to(torch.bfloat16))
    conv_all[l].copy_(torch.randn(NB, 10240, 3, device=dev).to(torch.bfloat16))
assert ssm_all[0].stride(0) == 817152 and conv_all[0].stride(0) == 817152
pristine = pool.clone()
ws0 = torch.empty(CAP, H, V, K, dtype=torch.float32, device=dev)
ws1 = torch.randn(CAP, H, V, K, dtype=torch.float32, device=dev)
flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
torch.cuda.synchronize()
log("pool", tuple(pool.shape), "ready")


def make_attn(P, D, cached):
    seq_lens = torch.tensor([T] * P + [1] * D, dtype=torch.int32)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.to(dev),
        num_contexts=P,
        num_ctx_tokens=T * P,
        num_tokens=T * P + D,
        kv_cache_manager=None,
        request_ids=None,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor(cached, dtype=torch.int32)
        ),
    )


def prepared(P, D, cached):
    md = Mamba2Metadata(CAP, 64)
    md.prepare(make_attn(P, D, cached))
    md.state_indices[: P + D].copy_(torch.tensor(SLOTS[: P + D], dtype=torch.int32))
    torch.cuda.synchronize()
    return md


def segment(md, P, n):
    """forward_core's reset gate (verbatim condition) + flashinfer_chunk state I/O."""
    state_indices = md.state_indices[:n]
    state_indices_p, _ = torch.split(state_indices, [P, n - P])
    for l in range(L):
        ssm_states = ssm_all[l]
        conv_states = conv_all[l]
        if P > 0 and not (
            gdn_mixer._SKIP_NOOP_RESET and getattr(md, "all_initial_states", False)
        ):
            reset_recurrent_state_rows(
                ssm_states, state_indices_p, md.has_initial_states[:P], conv_states
            )
        gather_cast_vk_to_fp32_vk(
            ssm_states, state_indices, out_dtype=torch.float32, output=ws0[:n]
        )
        cast_scatter_fp32_vk_to_vk(ws1[:n], ssm_states, state_indices)


def restore_rows(n):
    for s in SLOTS[:n]:
        pool[:, s].copy_(pristine[:, s])


def stats(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return dict(p10=q[0].item(), median=q[1].item(), p90=q[2].item())


def run_gate():
    out = {}
    for n, P, D in SIZES:
        md = prepared(P, D, [CACHED] * P + [CACHED + T + 5] * D)
        assert md.all_initial_states is True, "all-cached prepare must give True"
        times = {"main": [], "skip": []}
        for it in range(WARM + ITERS):
            order = ("main", "skip") if it % 2 == 0 else ("skip", "main")
            for arm in order:
                gdn_mixer._SKIP_NOOP_RESET = arm == "skip"
                restore_rows(n)
                flush_buf.add_(1)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                segment(md, P, n)
                e1.record()
                torch.cuda.synchronize()
                if it >= WARM:
                    times[arm].append(e0.elapsed_time(e1) * 1000.0 / L)  # us/layer
        gdn_mixer._SKIP_NOOP_RESET = True
        sm, ss = stats(times["main"]), stats(times["skip"])
        red = 1.0 - ss["median"] / sm["median"]
        noise = max(
            (sm["p90"] - sm["p10"]) / sm["median"], (ss["p90"] - ss["p10"]) / ss["median"]
        )
        out[str(n)] = dict(
            P=P, D=D, main_us_per_layer=sm, skip_us_per_layer=ss,
            reduction=red, noise=noise,
            main_ms_48=sm["median"] * L / 1000.0, skip_ms_48=ss["median"] * L / 1000.0,
        )
        log(f"num_seqs={n} (P={P},D={D}) us/layer main {sm} skip {ss} reduction {red:.4f}")
    g1 = out["1"]
    go1 = g1["reduction"] >= 0.04 and g1["skip_us_per_layer"]["p90"] < g1["main_us_per_layer"]["p10"]
    not_slower = all(
        out[k]["skip_us_per_layer"]["median"]
        <= out[k]["main_us_per_layer"]["median"] * (1.0 + max(out[k]["noise"], 0.0))
        for k in ("2", "4")
    )
    return dict(sizes=out, go_num_seqs1=go1, not_slower_2_4=not_slower, GO=bool(go1 and not_slower))


def snapshot():
    torch.cuda.synchronize()
    return pool.clone(), ws0.clone()


def eq(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def run_check_case(name, P, D, cached, expect_flag, fresh=()):
    n = P + D
    md = prepared(P, D, cached)
    flag = md.all_initial_states
    res = dict(case=name, P=P, D=D, cached=cached, all_initial_states=flag)
    ok = flag is expect_flag
    # main reference (kill-switch state == main's code path)
    gdn_mixer._SKIP_NOOP_RESET = False
    pool.copy_(pristine)
    segment(md, P, n)
    ref_pool, ref_ws0 = snapshot()
    # sanity on reset semantics: fresh rows were zeroed before gather; conv rows stay zero
    for i in fresh:
        s = SLOTS[i]
        ok &= bool((ref_ws0[i] == 0).all().item())
        ok &= bool((ref_pool[:, s, SSM_B:] == 0).all().item())
    nonfresh_conv_kept = all(
        eq(ref_pool[:, SLOTS[i], SSM_B:], pristine[:, SLOTS[i], SSM_B:])
        for i in range(n) if i not in fresh
    )
    ok &= nonfresh_conv_kept
    # candidate arm, repeated with the timed loop's counts and reset points
    gdn_mixer._SKIP_NOOP_RESET = True
    first = None
    drift = 0
    mism_main = 0
    for it in range(WARM + ITERS):
        restore_rows(n)
        flush_buf.add_(1)
        segment(md, P, n)
        cur_pool, cur_ws0 = snapshot()
        if first is None:
            first = (cur_pool, cur_ws0)
        else:
            if not (eq(cur_pool, first[0]) and eq(cur_ws0, first[1])):
                drift += 1
        if not (eq(cur_pool, ref_pool) and eq(cur_ws0, ref_ws0)):
            mism_main += 1
        if it != 0:
            del cur_pool, cur_ws0
    del first
    ok &= drift == 0 and mism_main == 0
    res.update(iters=WARM + ITERS, drift_vs_iter1=drift, mismatch_vs_main=mism_main,
               fresh_rows_zeroed_ok=bool(ok), nonfresh_conv_kept=nonfresh_conv_kept, passed=bool(ok))
    log("check", res)
    gdn_mixer._SKIP_NOOP_RESET = True
    return res


def run_checks():
    cases = []
    # (i) all cached at 1/2/4 sequences
    for n, P, D in SIZES:
        cases.append(run_check_case(f"all_cached_{n}", P, D,
                                    [CACHED] * P + [CACHED + T + 5] * D, True))
    # (ii) a fresh request mixed in: the reset runs and zeroes its row
    cases.append(run_check_case("fresh_mixed_P2", 2, 0, [CACHED, 0], False, fresh=(1,)))
    cases.append(run_check_case("fresh_mixed_P3D1", 3, 1, [0, CACHED, CACHED, CACHED + T], False,
                                fresh=(0,)))
    # flag semantics
    flags = {}
    with Mamba2Metadata.force_initial_states_for_warmup():
        flags["warmup_override_fresh"] = prepared(1, 0, [0]).all_initial_states
        flags["warmup_override_fresh_P2"] = prepared(2, 0, [0, 0]).all_initial_states
    md = Mamba2Metadata(CAP, 64)
    md.prepare(make_attn(1, 0, [CACHED]))
    flags["before_decode_only"] = md.all_initial_states
    md.prepare(make_attn(0, 2, [CACHED + T, CACHED + T]))
    flags["after_decode_only"] = md.all_initial_states
    flags_ok = (flags["warmup_override_fresh"] is True and flags["warmup_override_fresh_P2"] is True
                and flags["before_decode_only"] is True and flags["after_decode_only"] is False)
    log("flags", flags, "ok", flags_ok)
    # kill switch: env read once at import
    ks = {}
    for val in ("0", "1", None):
        env = dict(os.environ)
        env.pop("TRTLLM_GDN_SKIP_NOOP_STATE_RESET", None)
        if val is not None:
            env["TRTLLM_GDN_SKIP_NOOP_STATE_RESET"] = val
        r = subprocess.run(
            [sys.executable, "-c",
             "from tensorrt_llm._torch.modules.mamba import gdn_mixer as g; print('SKIP', g._SKIP_NOOP_RESET)"],
            env=env, capture_output=True, text=True, timeout=300)
        line = [x for x in r.stdout.splitlines() if x.startswith("SKIP")]
        ks[str(val)] = line[-1] if line else ("ERR " + r.stderr[-300:])
        log("kill switch", val, ks[str(val)])
    ks_ok = ks["0"] == "SKIP False" and ks["1"] == "SKIP True" and ks["None"] == "SKIP True"
    passed = all(c["passed"] for c in cases) and flags_ok and ks_ok
    return dict(cases=cases, flags=flags, flags_ok=flags_ok, kill_switch=ks,
                kill_switch_ok=ks_ok, passed=passed)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    result = {"tree_commit": tree_commit()}
    if mode in ("all", "gate"):
        result["gate"] = run_gate()
    if mode in ("all", "check"):
        result["checks"] = run_checks()
        result["passed"] = result["checks"]["passed"]
    else:
        result["passed"] = True
    result["output"] = {"class": "exact"}
    print(json.dumps(result), flush=True)
