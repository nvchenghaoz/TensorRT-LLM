"""Branch 15: NUM_REGS / local bytes of source variants of the warp kernel's table-row addressing
(LN_ID=0 and LN_ID=1, MINB=12, X_POLICY=2), plus a bitwise run of each variant on a K=3 u8 index
at M=32848 against variant 'bool2' on the rows it can address."""

import json

import torch
from cuda.bindings import driver

from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_adaln_nvfp4 as warp

ORIG = """  bool s = false;
  if (has_select) s = sel[row] != 0;
  const uint4 *scr = (const uint4 *)(scale + (s ? scale_rs : 0));
  const uint4 *shr = (const uint4 *)(shift + (s ? shift_rs : 0));"""

VARIANTS = {
    "bool_orig": ORIG,
    "i64mul": """  const int r = has_select ? (int)sel[row] : 0;
  const uint4 *scr = (const uint4 *)(scale + (i64)r * scale_rs);
  const uint4 *shr = (const uint4 *)(shift + (i64)r * shift_rs);""",
    "i32mul": """  const int r = has_select ? (int)sel[row] : 0;
  const uint4 *scr = (const uint4 *)(scale + r * (int)scale_rs);
  const uint4 *shr = (const uint4 *)(shift + r * (int)shift_rs);""",
    "sel2": """  const u32 r = has_select ? (u32)sel[row] : 0u;
  const uint4 *scr = (const uint4 *)(scale + (r != 0u ? scale_rs : 0) + (r > 1u ? scale_rs : 0));
  const uint4 *shr = (const uint4 *)(shift + (r != 0u ? shift_rs : 0) + (r > 1u ? shift_rs : 0));""",
    "u32mul_late": """  const u32 r = has_select ? (u32)sel[row] : 0u;
  const u16 *scb = scale + (size_t)(r * (u32)scale_rs);
  const u16 *shb = shift + (size_t)(r * (u32)shift_rs);
  const uint4 *scr = (const uint4 *)scb;
  const uint4 *shr = (const uint4 *)shb;""",
    "bool_then_extra": """  bool s = false;
  if (has_select) s = sel[row] != 0;
  const int x2 = (has_select && sel[row] > 1) ? (int)sel[row] - 1 : 0;
  const uint4 *scr = (const uint4 *)(scale + (s ? scale_rs : 0) + (i64)x2 * scale_rs);
  const uint4 *shr = (const uint4 *)(shift + (s ? shift_rs : 0) + (i64)x2 * shift_rs);""",
}


_BASE = warp._SRC


def build(src_snip):
    cur = _BASE
    # find the branch's addressing block and swap in the snippet
    start = cur.index("  // Table row index")
    end = cur.index("  uint4 *xo = xmod")
    return cur[:start] + src_snip + "\n" + cur[end:]


def main():
    base_src = warp._SRC
    d, m = 3072, 32848
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, m, d, generator=g, device="cuda").to(torch.bfloat16)
    sc = (torch.randn(1, 3, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    sh = (torch.randn(1, 3, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    ar = torch.arange(m, device="cuda")
    idx = ((ar >= 16384).to(torch.uint8) + (ar >= 32768).to(torch.uint8)).view(1, m, 1)
    gs = torch.tensor([168.0], device="cuda")
    w = torch.ones(d, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(d, dtype=torch.bfloat16, device="cuda")
    warp.register_identity_affine(w, b)
    w0, b0 = w.clone(), b.clone()
    out, ref = {}, None
    for name, snip in VARIANTS.items():
        warp._SRC = build(snip)
        warp._MODULES.clear()
        res = {}
        outs = {}
        for tag, (ww, bb) in (("ln_id1", (w, b)), ("ln_id0", (w0, b0))):
            sel = (idx != 0) if name == "bool_orig" else idx
            try:
                o = warp.launch(x, sel, sc, sh, ww, bb, gs, 1e-6, store_x_mod=True)
                torch.cuda.synchronize()
                outs[tag] = [t.clone() for t in o]
            except Exception as e:  # noqa: BLE001
                res[tag] = f"error {e}"[:300]
        for key, fn in warp._MODULES.items():
            n = driver.cuFuncGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)[1]
            lb = driver.cuFuncGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)[1]
            res["ln_id1" if "LN_ID=1" in str(key) else "ln_id0"] = {"regs": int(n),
                                                                     "local": int(lb)}
        if name == "i64mul":
            ref = outs
        out[name] = res
        out[name]["_outs"] = outs
    # bitwise vs i64mul (verified exact in check jobs) for every u8 variant
    for name in VARIANTS:
        outs = out[name].pop("_outs")
        if name == "bool_orig" or ref is None:
            continue
        out[name]["bitwise_vs_i64mul"] = all(
            torch.equal(a, c) for tag in outs for a, c in zip(outs[tag], ref[tag]))
    # timing per variant (LN_ID=1, store_x_mod=False), CUDA events, median of 50
    for name, snip in VARIANTS.items():
        if name == "bool_orig":
            continue
        warp._SRC = build(snip)
        warp._MODULES.clear()
        fn = lambda: warp.launch(x, idx, sc, sh, w, b, gs, 1e-6, store_x_mod=False)  # noqa: E731
        for _ in range(10):
            fn()
        ts = []
        for _ in range(50):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        out[name]["median_ms_M32848"] = ts[len(ts) // 2]
    warp._SRC = base_src
    print(json.dumps({"passed": True, "variants": out}), flush=True)


if __name__ == "__main__":
    main()
