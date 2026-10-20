# Idea 198 / branch 118: op-level bitwise check of trtllm::nvfp4_w4a16_gemv_conv against main's
# pair [trtllm::nvfp4_w4a16_gemv -> trtllm::causal_conv1d_update] on the same inputs.
# M = 1, 2, 3; slot patterns with a PAD row (-1) and repeated-pattern slots; conv_state as a
# strided pool view (batch stride > conv_dim * 3, like the V2 cache manager's coalesced pages);
# untouched slots checked; 10 sequential stateful calls compared after every call.
import json
import sys

import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update

dev = torch.device("cuda")
K, N, CONV = 5120, 16384, 10240
SLOTS = 6
G = 1.0 / 128.0


def log(*a):
    print(*a, flush=True)


log("has op:", hasattr(torch.ops.trtllm, "nvfp4_w4a16_gemv_conv"))
gen = torch.Generator(device=dev).manual_seed(7)
wq = torch.randint(0, 256, (N, K // 2), generator=gen, device=dev, dtype=torch.int32).to(torch.uint8)
sf = torch.randint(0x28, 0x40, (N, K // 16), generator=gen, device=dev, dtype=torch.int32).to(torch.uint8)
cw = (torch.randn(CONV, 4, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
# Pool with padding between slots: raw [SLOTS, CONV * 3 + 64] viewed as [SLOTS, CONV, 3].
raw0 = torch.randn(SLOTS, CONV * 3 + 64, generator=gen, device=dev).to(torch.bfloat16)

PATTERNS = dict()
PATTERNS["1"] = [[3], [-1], [0]]
PATTERNS["2"] = [[4, 1], [-1, 2], [5, -1]]
PATTERNS["3"] = [[2, 0, 5], [1, -1, 3], [-1, -1, 4]]

ok_all = True
report = dict()
for m in (1, 2, 3):
    for pat in PATTERNS[str(m)]:
        idx = torch.tensor(pat, dtype=torch.int32, device=dev)
        raw_a = raw0.clone()
        raw_b = raw0.clone()
        cs_a = raw_a[:, :CONV * 3].view(SLOTS, CONV, 3)
        cs_b = raw_b[:, :CONV * 3].view(SLOTS, CONV, 3)
        assert cs_a.stride() == (CONV * 3 + 64, 3, 1)
        step_ok = []
        for step in range(10):
            x = (torch.randn(m, K, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            ref = torch.ops.trtllm.nvfp4_w4a16_gemv(x, wq, sf, G)
            ref_mixed = causal_conv1d_update(ref[:, :CONV], cs_a, cw, None, "silu", conv_state_indices=idx)
            got = torch.ops.trtllm.nvfp4_w4a16_gemv_conv(x, wq, sf, G, cs_b, cw, idx, CONV, -1)
            torch.cuda.synchronize()
            same_out = torch.equal(ref.view(torch.int16), got.view(torch.int16))
            same_mixed = torch.equal(ref_mixed.view(torch.int16), got[:, :CONV].view(torch.int16))
            same_pool = torch.equal(raw_a.view(torch.int16), raw_b.view(torch.int16))
            step_ok.append(bool(same_out and same_mixed and same_pool))
        # Untouched slots (never in idx) and pad bytes equal pristine.
        used = set(v for v in pat if v >= 0)
        untouched = all(torch.equal(raw_b[s].view(torch.int16), raw0[s].view(torch.int16))
                        for s in range(SLOTS) if s not in used)
        pad_bytes = torch.equal(raw_b[:, CONV * 3:].view(torch.int16), raw0[:, CONV * 3:].view(torch.int16))
        changed = all(not torch.equal(raw_b[s].view(torch.int16), raw0[s].view(torch.int16)) for s in used)
        ok = all(step_ok) and untouched and pad_bytes and changed
        ok_all = ok_all and ok
        report["M%d_%s" % (m, "_".join(str(v) for v in pat))] = dict(
            per_step=step_ok, untouched=untouched, pad_bytes=pad_bytes, used_changed=changed, ok=ok)
        log("M=%d slots=%s steps_bitwise=%s untouched=%s pad_bytes=%s used_changed=%s" % (
            m, pat, step_ok, untouched, pad_bytes, changed))

# Rejections: m=4, odd conv_dim, wrong strides.
rej = dict()
for name, fn in (
    ("m4", lambda: torch.ops.trtllm.nvfp4_w4a16_gemv_conv(
        torch.zeros(4, K, dtype=torch.bfloat16, device=dev), wq, sf, G,
        raw0.clone()[:, :CONV * 3].view(SLOTS, CONV, 3), cw,
        torch.zeros(4, dtype=torch.int32, device=dev), CONV, -1)),
    ("odd_conv_dim", lambda: torch.ops.trtllm.nvfp4_w4a16_gemv_conv(
        torch.zeros(1, K, dtype=torch.bfloat16, device=dev), wq, sf, G,
        raw0.clone()[:, :(CONV - 1) * 3].view(SLOTS, CONV - 1, 3), cw[:CONV - 1].contiguous(),
        torch.zeros(1, dtype=torch.int32, device=dev), CONV - 1, -1)),
    ("channel_major_state", lambda: torch.ops.trtllm.nvfp4_w4a16_gemv_conv(
        torch.zeros(1, K, dtype=torch.bfloat16, device=dev), wq, sf, G,
        torch.zeros(SLOTS, 3, CONV, dtype=torch.bfloat16, device=dev).transpose(1, 2), cw,
        torch.zeros(1, dtype=torch.int32, device=dev), CONV, -1)),
):
    try:
        fn()
        torch.cuda.synchronize()
        rej[name] = False
    except RuntimeError as e:
        rej[name] = True
        log("rejected %s: %s" % (name, str(e).splitlines()[0][:160]))
ok_all = ok_all and all(rej.values())
print(json.dumps(dict(passed=ok_all, compared="trtllm::nvfp4_w4a16_gemv_conv vs main's nvfp4_w4a16_gemv -> "
                      "causal_conv1d_update (bitwise: full output, conv pool incl. padding, untouched slots)",
                      cases=report, rejections=rej)))
sys.exit(0 if ok_all else 1)
