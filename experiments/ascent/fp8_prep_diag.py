"""Diagnose quantize-sweep mismatches: which x values differ, their products, both E4M3 codes."""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def _cvt_plain(x_ptr, f, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (x * f).to(tl.float8e4nv), mask=m)


@triton.jit
def _cvt_asm(x_ptr, f, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = x * f
    b = tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $1, $1; cvt.u32.u16 $0, t; }",
        "=r,r", [y], dtype=tl.uint32, is_pure=True, pack=1)
    tl.store(o_ptr + offs, (b & 0xFF).to(tl.uint8), mask=m)


def main():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.visual_gen.attention.backends import fp8_attn_prep as prep

    dev = "cuda"
    allv = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
    allv = torch.cat([allv, -allv])
    report = []
    tot = {"plain": 0, "asm": 0}
    for a in (3.0, 12.0, 0.1787109375, 26.875, 872.0, 24.0, 1.703125):
        at = torch.tensor(a, device=dev).bfloat16()
        vals = allv[allv.float().abs() <= at.float()]
        pad = (-vals.numel()) % 128
        vals = torch.cat([vals, torch.zeros(pad, device=dev, dtype=torch.bfloat16)])
        x = vals.view(1, -1, 1, 128)
        ref, ds = torch.ops.trtllm.quantize_e4m3_per_tensor(x.contiguous())
        new = prep.quantize_qkv_e4m3_per_tensor(x, x.clone(), x.clone())[0]
        diff = (ref.view(torch.uint8) != new.view(torch.uint8)).view(-1).nonzero().view(-1)
        s = ds.float().item()
        f = torch.tensor(1.0, device=dev) / torch.tensor(s, device=dev)
        n = x.numel()
        o_plain = torch.empty(n, dtype=torch.float8_e4m3fn, device=dev)
        o_asm = torch.empty(n, dtype=torch.uint8, device=dev)
        _cvt_plain[(triton.cdiv(n, 1024),)](x.view(-1), f.item(), o_plain, n, BLOCK=1024)
        _cvt_asm[(triton.cdiv(n, 1024),)](x.view(-1), f.item(), o_asm, n, BLOCK=1024)
        rv = ref.view(torch.uint8).view(-1)
        np_ = (o_plain.view(torch.uint8) != rv).sum().item()
        na = (o_asm != rv).sum().item()
        tot["plain"] += np_
        tot["asm"] += na
        rows = []
        for i in diff[:6].tolist():
            xv = x.view(-1)[i].float().item()
            rows.append({"x": xv, "x_hex": hex(x.view(-1).view(torch.int16)[i].item() & 0xFFFF),
                         "prod": xv * f.item(), "prod_f32": (x.view(-1)[i].float() * f).item(),
                         "ref_code": hex(rv[i].item()), "new_code": hex(new.view(torch.uint8).view(-1)[i].item()),
                         "asm_code": hex(o_asm[i].item())})
        print(json.dumps({"amax": a, "scale": s, "factor": f.item(), "n_diff_prod": diff.numel(),
                          "n_diff_plain": np_, "n_diff_asm": na, "rows": rows}), flush=True)
        report.append(rows)
    print(json.dumps({"totals": tot}))


if __name__ == "__main__":
    main()
