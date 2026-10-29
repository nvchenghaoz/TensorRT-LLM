"""Branch 165: dump SASS windows of the G3 kernel around STL/LDL (and its TMA/UTMALDG sites)."""
import glob
import re
import subprocess
import sys

from flashinfer.jit import env as jit_env

from tensorrt_llm._torch.visual_gen.attention.backends import fmha_v2_flux3 as lib

lev = tuple(lib.ROUTED_LEVERS)
if "g3" not in lev:
    lev = (*lev, "g3")
lib.get_flux3_d128_module(lev)
uri = f"trtllm_fmha_v2_flux3_d128_{lib._tag(lev, 0)}"
so = [p for p in glob.glob(str(jit_env.FLASHINFER_JIT_DIR / "**" / f"{uri}.so"), recursive=True)][0]
r = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True)
name, lines, funcs = None, [], {}
for line in r.stdout.splitlines():
    m = re.search(r"Function : (\S+)", line)
    if m:
        if name:
            funcs[name] = lines
        name, lines = m.group(1), []
        continue
    if name and line.strip().startswith("/*") and ";" in line:
        lines.append(re.sub(r"\s+/\* 0x[0-9a-f]+ \*/", "", line).strip())
if name:
    funcs[name] = lines
for n, b in funcs.items():
    if not n.endswith("_g3"):
        continue
    print("FUNC", n, len(b))
    hits = [i for i, x in enumerate(b) if "STL" in x or "LDL" in x or "UTMALDG" in x or "CALL" in x
            or "UBLKCP" in x or "UTMAPF" in x]
    shown = set()
    for i in hits[:40]:
        lo, hi = max(0, i - 10), min(len(b), i + 14)
        if i in shown:
            continue
        print(f"----- around {i}")
        for j in range(lo, hi):
            shown.add(j)
            print(f"{j:5d}  {b[j]}")
    # loop body census: count instruction classes overall
    ops = {}
    for x in b:
        m = re.match(r"(?:/\*[0-9a-f]+\*/\s*)?(@!?U?P\d\s+)?([A-Z0-9_.]+)", x)
        if m:
            op = m.group(2).split(".")[0]
            ops[op] = ops.get(op, 0) + 1
    print("OPS", sorted(ops.items(), key=lambda t: -t[1])[:40])
