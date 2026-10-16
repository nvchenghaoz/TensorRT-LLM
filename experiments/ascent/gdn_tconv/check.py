# Correctness check for the fused GDN prefill transpose + causal conv1d
# (tensorrt_llm/_torch/modules/mamba/gdn_prefill_conv.py) against main's path:
# extract_transpose_prefill_slice + in-place C++ causal_conv1d_fwd, followed by
# the unchanged fused_gdn_post_conv (and causal_conv1d_update for decode rows of a
# mixed batch). Qwen3.8-27B GDN shapes at TP1: conv_dim 10240 (q 2048 | k 2048 |
# v 6144), in_proj_qkvz row stride 16384, 16 k-heads / 48 v-heads x 128.
#
# Per case: conv output, the WHOLE conv-state pool and post_conv q/k/v/g/beta
# must be bitwise equal (SHA-256 printed). Then an untimed repeat sequence with
# the bench's schedule (10 warmup + 50 iterations, pool restored from a pristine
# copy before each) compares every iteration with iteration 1 and with main.
# Also: the gdn_mixer kill switch TRTLLM_GDN_FUSED_TRANSPOSE_CONV=0 must give
# main's path bitwise, and =1 the fused kernel.
# Usage: python experiments/ascent/gdn_tconv/check.py TREE_COMMIT=<commit>
import hashlib
import json
import os
import subprocess
import sys
import types

print(sys.argv[1] if len(sys.argv) > 1 else "TREE_COMMIT=unlabelled", flush=True)
try:
    _h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    _d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    print("git_head=%s dirty=%d" % (_h.strip(), len(_d.splitlines())), flush=True)
except Exception as _e:  # noqa: BLE001
    print("git unavailable: %r" % (_e,), flush=True)

import torch  # noqa: E402

from tensorrt_llm._torch.modules.mamba.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (  # noqa: E402
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (  # noqa: E402
    gdn_prefill_transpose_conv,
)

H, HV, KD, VD = 16, 48, 128, 128
DIM = 2 * H * KD + HV * VD  # 10240
QKVZ = DIM + HV * VD  # 16384
SLOTS = 12
dev = torch.device("cuda")
WARMUP, ITERS = 10, 50


def log(*a):
    print(*a, flush=True)


def sha(t):
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def beq(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def make_case(lens, init, n_dec, gen, pad_seq=None, strided_pool=False):
    T = sum(lens)
    n = len(lens)
    qkvz = (torch.randn(T + n_dec, QKVZ, device=dev, generator=gen) * 1.5).to(torch.bfloat16)
    ba = torch.randn(T + n_dec, 2 * HV, device=dev, generator=gen).to(torch.bfloat16)
    w = (torch.randn(DIM, 4, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    if strided_pool:
        big = (torch.randn(SLOTS, 2, DIM, 3, device=dev, generator=gen)).to(torch.bfloat16)
        pool0 = big[:, 1]
    else:
        pool0 = (torch.randn(SLOTS, DIM, 3, device=dev, generator=gen)).to(torch.bfloat16)
    perm = torch.randperm(SLOTS, device=dev, generator=gen).to(torch.int32)
    idx_p = perm[:n].clone()
    if pad_seq is not None:
        idx_p[pad_seq] = -1
    idx_d = perm[n : n + n_dec].clone()
    qsl = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev
    )
    hi = torch.tensor(init, dtype=torch.bool, device=dev)
    a_log = torch.log(torch.rand(HV, device=dev, generator=gen) * 15 + 1).float()
    dt_bias = (torch.randn(HV, device=dev, generator=gen) * 0.5).float()
    return dict(
        T=T,
        n_dec=n_dec,
        qkvz=qkvz,
        ba=ba,
        w=w,
        pool0=pool0,
        idx_p=idx_p,
        idx_d=idx_d,
        qsl=qsl,
        hi=hi,
        a_log=a_log,
        dt_bias=dt_bias,
    )


def conv_main(x, w, pool, qsl, idx, hi):
    xt = extract_transpose_prefill_slice(x, x.shape[0], 0, x.shape[1])
    return causal_conv1d_fn(
        xt,
        w,
        None,
        activation="silu",
        conv_states=pool,
        has_initial_state=hi,
        cache_indices=idx,
        query_start_loc=qsl,
    )


def conv_fused(x, w, pool, qsl, idx, hi):
    return gdn_prefill_transpose_conv(x, w, None, pool, qsl, idx, hi, activation="silu")


def run_chain(c, conv, pool):
    """forward_extend's non-verify chain on fresh copies; pool updated in place."""
    qkvz = c["qkvz"].clone()
    T = c["T"]
    mixed = qkvz[:, :DIM]
    if c["n_dec"] > 0:
        x_p, x_d = torch.split(mixed, [T, c["n_dec"]], dim=0)
    else:
        x_p, x_d = mixed, None
    out = conv(x_p, c["w"], pool, c["qsl"], c["idx_p"], c["hi"])
    if x_d is not None:
        x_d = causal_conv1d_update(
            x_d, pool, c["w"], None, activation="silu", conv_state_indices=c["idx_d"]
        )
    b = c["ba"][:, :HV]
    a = c["ba"][:, HV:]
    q, k, v, g, beta = fused_gdn_post_conv(out, x_d, a, b, c["a_log"], c["dt_bias"], H, KD, HV, VD)
    return out, (q, k, v, g, beta)


def fp32_ref(c):
    """Independent fp32 torch conv reference for the prefill part (sanity)."""
    x = c["qkvz"][: c["T"], :DIM].float()
    w = c["w"].float()
    pool = c["pool0"].float()
    outs = []
    for s in range(len(c["hi"])):
        st, en = int(c["qsl"][s]), int(c["qsl"][s + 1])
        slot = int(c["idx_p"][s])
        xs = x[st:en].t()  # [DIM, L]
        if slot == -1:
            outs.append(xs)
            continue
        halo = pool[slot] if bool(c["hi"][s]) else torch.zeros(DIM, 3, device=dev)
        xx = torch.cat([halo, xs], dim=1)
        y = sum(w[:, kk : kk + 1] * xx[:, kk : kk + (en - st)] for kk in range(4))
        outs.append(y * torch.sigmoid(y))
    return torch.cat(outs, dim=1)


def check_case(name, c):
    pool_m = c["pool0"].clone()
    out_m, post_m = run_chain(c, conv_main, pool_m)
    pool_f = c["pool0"].clone()
    out_f, post_f = run_chain(c, conv_fused, pool_f)
    torch.cuda.synchronize()
    names = ["q", "k", "v", "g", "beta"]
    res = {
        "conv_out": beq(out_m, out_f),
        "conv_states": beq(pool_m, pool_f),
    }
    for nm, a, b in zip(names, post_m, post_f):
        res[nm] = beq(a, b)
    hashes = {"conv_out": [sha(out_m), sha(out_f)], "conv_states": [sha(pool_m), sha(pool_f)]}
    for nm, a, b in zip(names, post_m, post_f):
        hashes[nm] = [sha(a), sha(b)]
    ref = fp32_ref(c)
    err_main = (out_m.float() - ref).abs().max().item()
    err_fused = (out_f.float() - ref).abs().max().item()
    if not res["conv_out"]:
        d = out_m.view(torch.int16) != out_f.view(torch.int16)
        log(f"  {name}: conv_out mismatches {int(d.sum())} first {d.nonzero()[:5].tolist()}")
    if not res["conv_states"]:
        d = pool_m.view(torch.int16) != pool_f.view(torch.int16)
        log(f"  {name}: state mismatches {int(d.sum())} first {d.nonzero()[:5].tolist()}")

    # Untimed repeat sequence: same schedule as the bench (10 warmup + 50 iters),
    # pool restored from pristine before each; compare with iteration 1 and main.
    pool = c["pool0"].clone()
    first = None
    drift = 0
    vs_main = 0
    for it in range(WARMUP + ITERS):
        pool.copy_(c["pool0"])
        o, post = run_chain(c, conv_fused, pool)
        cur = [o, pool.clone()] + list(post)
        if first is None:
            first = cur
        drift += sum(0 if beq(x, y) else 1 for x, y in zip(cur, first))
        vs_main += sum(0 if beq(x, y) else 1 for x, y in zip(cur, [out_m, pool_m] + list(post_m)))
    ok = all(res.values()) and drift == 0 and vs_main == 0
    log(
        f"{name}: T={c['T']} dec={c['n_dec']} bitwise={res} repeat_drift={drift} "
        f"repeat_vs_main={vs_main} maxerr_vs_fp32 main={err_main:.3e} fused={err_fused:.3e} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    log(f"  sha256[main,fused]: {hashes}")
    return ok, dict(
        bitwise=res, drift=drift, vs_main=vs_main, sha=hashes, err_main=err_main, err_fused=err_fused
    )


def check_kill_switch(c):
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
    from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (
        gdn_fused_transpose_conv_enabled as enabled,
    )

    fake = types.SimpleNamespace(
        conv1d=types.SimpleNamespace(weight=c["w"], bias=None), activation="silu"
    )
    x = c["qkvz"][: c["T"], :DIM]
    results = {}
    outs = {}
    os.environ.pop("TRTLLM_GDN_FUSED_TRANSPOSE_CONV_MIN_TOKENS", None)
    os.environ["TRTLLM_GDN_FUSED_TRANSPOSE_CONV"] = "1"
    rule_on = (
        (not enabled(2048))
        and (not enabled(4095))
        and enabled(4096)
        and enabled(6144)
        and enabled(8192)
    )
    os.environ["TRTLLM_GDN_FUSED_TRANSPOSE_CONV"] = "0"
    rule_off = not any(enabled(t) for t in (1, 2048, 4096, 8192))
    results["dispatch_rule_default_min4096"] = rule_on
    results["dispatch_rule_switch0_never"] = rule_off
    # force the fused kernel at this small T for the switch=1 arm
    os.environ["TRTLLM_GDN_FUSED_TRANSPOSE_CONV_MIN_TOKENS"] = "0"
    for env in ["0", "1"]:
        os.environ["TRTLLM_GDN_FUSED_TRANSPOSE_CONV"] = env
        pool = c["pool0"].clone()
        o = Qwen3NextGatedDeltaNet._prefill_transpose_conv(
            fake, x, pool, c["hi"], c["idx_p"], c["qsl"]
        )
        outs[env] = (o, pool)
        # verify path stays on main's kernels regardless of the switch
        pool_v = c["pool0"].clone()
        ov = Qwen3NextGatedDeltaNet._prefill_transpose_conv(
            fake, x, pool_v, c["hi"], c["idx_p"], c["qsl"], allow_fused=False
        )
        outs["verify" + env] = (ov, pool_v)
    os.environ["TRTLLM_GDN_FUSED_TRANSPOSE_CONV"] = "1"
    os.environ.pop("TRTLLM_GDN_FUSED_TRANSPOSE_CONV_MIN_TOKENS", None)
    pool_m = c["pool0"].clone()
    om = conv_main(x, c["w"], pool_m, c["qsl"], c["idx_p"], c["hi"])
    pool_f = c["pool0"].clone()
    of = conv_fused(x, c["w"], pool_f, c["qsl"], c["idx_p"], c["hi"])
    results["switch0_equals_main"] = beq(outs["0"][0], om) and beq(outs["0"][1], pool_m)
    results["switch1_equals_fused"] = beq(outs["1"][0], of) and beq(outs["1"][1], pool_f)
    results["verify_path_is_main"] = all(
        beq(outs["verify" + e][0], om) and beq(outs["verify" + e][1], pool_m) for e in "01"
    )
    ok = all(results.values())
    log(f"kill_switch: {results} -> {'PASS' if ok else 'FAIL'}")
    return ok, results


def main():
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    cases = [
        ("1x2048_init0", dict(lens=[2048], init=[False], n_dec=0)),
        ("1x2048_init1", dict(lens=[2048], init=[True], n_dec=0)),
        ("3x2048+1dec", dict(lens=[2048] * 3, init=[True] * 3, n_dec=1)),
        ("4x2048_mixinit", dict(lens=[2048] * 4, init=[True, False, True, False], n_dec=0)),
        ("1x8192", dict(lens=[8192], init=[True], n_dec=0)),
        (
            "ragged_1_2_3_777_2048",
            dict(lens=[1, 2, 3, 777, 2048], init=[True, True, False, True, False], n_dec=0),
        ),
        ("ragged_noinit_short", dict(lens=[1, 2, 3, 64, 65, 63], init=[False] * 6, n_dec=2)),
        ("pad_seq", dict(lens=[5, 300, 64], init=[True, True, True], n_dec=0, pad_seq=1)),
        (
            "strided_pool_ragged",
            dict(lens=[130, 2, 1000], init=[True, False, True], n_dec=1, strided_pool=True),
        ),
    ]
    all_ok = True
    report = {}
    for name, kw in cases:
        c = make_case(gen=gen, **kw)
        ok, info = check_case(name, c)
        all_ok &= ok
        report[name] = dict(
            ok=ok,
            drift=info["drift"],
            vs_main=info["vs_main"],
            bitwise=all(info["bitwise"].values()),
        )
    c = make_case(
        lens=[1, 2, 3, 777, 2048], init=[True, True, False, True, False], n_dec=0, gen=gen, pad_seq=2
    )
    ok, ks = check_kill_switch(c)
    all_ok &= ok
    report["kill_switch"] = ks
    print(
        json.dumps(
            {
                "passed": bool(all_ok),
                "compared": "fused transpose+conv vs main extract_transpose_prefill_slice + C++ "
                "causal_conv1d_fwd (+unchanged fused_gdn_post_conv / causal_conv1d_update): "
                "conv out, whole conv-state pool, q/k/v/g/beta bitwise; 10+50 repeat with "
                "pool restore; kill switch",
                "cases": report,
            }
        )
    )


if __name__ == "__main__":
    main()
