# Arm F exactness probe: fused T1 + gated RMSNorm vs main's T1 kernel followed
# by rms_norm_gated_token_major. Compares (bitwise) o (debug store), the whole
# state pool, and y, for (qk_per, norm_per) reduce-order variants, then dumps
# the TTGIR layouts/reduces of main T1, the norm kernel and F.
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tensorrt_llm._torch.modules.fla import gdn_decode_t1 as t1mod  # noqa
from tensorrt_llm._torch.modules.mamba import layernorm_gated  # noqa: E402

EPS = 1e-6


def log(*a):
    print(*a, flush=True)


def bits(t):
    return t.contiguous().view(torch.int16)


def main_ref(lay):
    common.reset([lay])
    common.make_call(None, None)(lay)
    y = layernorm_gated.rms_norm_gated_token_major(
        lay["out"].reshape(-1, common.V), lay["z"], lay["norm_w"], EPS)
    torch.cuda.synchronize()
    return lay["out"].clone(), lay["flat"].clone(), y.clone()


def run_f(lay, qk_per, norm_per, warps=8):
    common.reset([lay])
    o_dbg = torch.zeros(lay["n"] * common.HV, common.V, dtype=torch.bfloat16,
                        device="cuda")
    y = t1mod.fused_sigmoid_gating_delta_rule_update_t1_norm(
        A_log=lay["A_log"], a=lay["a"], dt_bias=lay["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0, q=lay["q"], k=lay["k"],
        v=lay["v"], b=lay["b"], z=lay["z"], norm_weight=lay["norm_w"],
        eps=EPS, initial_state_source=lay["pool"],
        initial_state_indices=lay["idx"], scale=common.K**-0.5,
        use_qk_l2norm_in_kernel=True, o_debug=o_dbg, num_warps=warps,
        qk_per=qk_per, norm_per=norm_per)
    torch.cuda.synchronize()
    return o_dbg, lay["flat"].clone(), y.clone()


def ulps(a, b):
    ia = bits(a).to(torch.int32)
    ib = bits(b).to(torch.int32)
    d = (ia - ib).abs()
    return int(d.max().item()), int((d != 0).sum().item())


def dump(fn, name):
    fn0 = fn
    while not hasattr(fn0, "device_caches") and hasattr(fn0, "fn"):
        fn0 = fn0.fn
    try:
        for dkey, val in fn0.device_caches.items():
            cache = val[0] if isinstance(val, tuple) else val
            for key, ker in cache.items():
                ttgir = ker.asm.get("ttgir", "")
                log(f"== {name} warps {ker.metadata.num_warps} regs "
                    f"{ker.n_regs} spills {ker.n_spills}")
                for ln in ttgir.splitlines():
                    s = ln.strip()
                    if s.startswith("#blocked") or s.startswith("#linear") \
                            or "tt.reduce" in s or "convert_layout" in s \
                            or "tt.reshape" in s:
                        log("   ", s[:260])
    except Exception as e:  # noqa
        log("dump error", name, repr(e))


def main():
    res = {}
    variants = [(4, 8), (4, 4), (8, 8)]
    for n in (1, 4, 8, 16, 32):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(3000 + n)
        lay = common.make_layer(n, gen, pad=(n >= 4))
        o_r, s_r, y_r = main_ref(lay)
        for qk_per, norm_per in variants:
            o_f, s_f, y_f = run_f(lay, qk_per, norm_per)
            r = dict(o_eq=torch.equal(bits(o_r), bits(o_f)),
                     state_eq=torch.equal(bits(s_r), bits(s_f)),
                     y_eq=torch.equal(bits(y_r), bits(y_f)),
                     o_ulps=ulps(o_r.reshape(-1), o_f.reshape(-1)),
                     y_ulps=ulps(y_r.reshape(-1), y_f.reshape(-1)))
            res[f"N{n}_qk{qk_per}_norm{norm_per}"] = r
            log(f"N={n} qk_per={qk_per} norm_per={norm_per}: {r}")
    dump(t1mod.fused_sigmoid_gating_delta_rule_update_t1_kernel, "main T1")
    dump(layernorm_gated._rms_norm_gated_fwd_multirow_kernel, "norm")
    dump(t1mod.fused_sigmoid_gating_delta_rule_update_t1_norm_kernel, "F")
    print(json.dumps({"fcheck": res}), flush=True)


main()
