"""Idea 55: exact trims of the SM120 d512 64-row FA2 paged prefill kernel
(include/flashinfer/attention/prefill_d512_tile64.cuh), Gemma4 global layers.

  A : coalesced quarter-warp cp.async fills. K/V: a quarter-warp loads 2 keys x 64 contiguous
      bytes (4 whole sectors) instead of 8 keys x 16 B (8 half sectors); smem rows get an XOR
      swizzle (row ^ ((2 piece) & 7)) so the fills and the consumer reads stay conflict-free.
      Q: a quarter-warp loads 4 rows x 32 B (whole sectors) instead of 8 rows x 16 B.
  B : K B-fragment as two LDS.32 (words t/2 and t/2 + 2) instead of one 4-lane-broadcast
      LDS.128 + 2 SEL.

Modes
  --worker --inc tree|<dir>   one arm in this process (tree: FlashInfer as the job has it).
  (default) driver            arms (--arms, default main,A,B,AB) as text-patched include
                              copies, one subprocess each with its own FLASHINFER_WORKSPACE_BASE,
                              --rounds interleaved rounds; byte compares of O and LSE vs the
                              main arm, errors vs an FP32 reference, regs/spill per arm.
Arm 'main' is derived from the tree: if the tree already carries AB (the committed candidate),
the AB patches are reversed. Dims come from <checkpoint>/config.json.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

PAGE = 32
KFILE = "include/flashinfer/attention/prefill_d512_tile64.cuh"
CASES = "1x300,4x300,16x300,27x300,1x1000,4x1000,8x1000,1x2000,4x2000,16x512,1x4000,2x4000,1x8000"
EXTRA = "rr1x8000,rr4x2000,rr1x2000,ragged5,ragged2,prefix2,split1x2000,splitprefix"
NAMED = {
    "ragged5": ([37, 300, 1000, 513, 2048], None, {}),
    "ragged2": ([8000, 100], None, {}),
    "prefix2": ([500, 256], [1500, 700], {}),
    "split1x2000": ([2000], None, {"fixed_split_size": 16}),
    "splitprefix": ([700, 129], [2600, 1000], {"fixed_split_size": 8}),
}

# ------------------------------------------------------------------ patches (main -> variant)
P_LDS32 = ("__device__ __forceinline__ void sts_u32(",
           """__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\\n" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ void sts_u32(""")

P_LOADER_DOC = ("""// K/V ring loader: thread -> key `key` of each 64-key tile, 16-byte part `part` (an 8-lane
// phase = 8 keys of one part: 128 contiguous smem bytes; a warp instruction covers 8 keys x
// 64 contiguous global bytes).""",
                """// K/V ring loader: thread -> key `key` = 8 warp + lane / 4 of each 64-key tile and 16-byte
// pieces 4 j + `part` (part = lane % 4). A quarter-warp loads 2 keys x 64 contiguous global
// bytes (4 whole 32-byte sectors). Smem rows are XOR-swizzled: piece p of key r sits at row
// r ^ ((2 p) & 7) (low 3 bits only), so the quarter-warp's 8 stores hit 8 distinct 16-byte bank
// groups, and a consumer's 8 lanes reading 8 aligned keys of one piece still cover 128
// contiguous bytes.""")

P_LOADER = ("""    const uint32_t dst_base = ring_s + slot * kSlotBytes;
    if (st < 4) {
      // smem [8 pieces][64 keys] x 16 B
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t piece = part + 4 * j;
        cp_async_16_zfill(dst_base + (piece * kTileKV + key) * 16,
                          k_data + koff + st * 128 + piece * 16, valid);
      }
    } else if ((warp >> 1) == st - 4) {
      // smem [32 pieces][16 keys] x 16 B, loaded by the two warps that own these keys
      const uint32_t kk = key & 15u;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t piece = part + 4 * j;
        cp_async_16_zfill(dst_base + (piece * 16 + kk) * 16, v_data + voff + piece * 16, valid);
      }
    }""",
            """    const uint32_t dst_base = ring_s + slot * kSlotBytes;
    const uint32_t swz = 2u * part;  // (2 piece) & 7 for piece = 4 j + part
    if (st < 4) {
      // smem [8 pieces][64 keys] x 16 B
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t piece = 4 * j + part;
        cp_async_16_zfill(dst_base + (piece * kTileKV + (key ^ swz)) * 16,
                          k_data + koff + st * 128 + piece * 16, valid);
      }
    } else if ((warp >> 1) == st - 4) {
      // smem [32 pieces][16 keys] x 16 B, loaded by the two warps that own these keys
      const uint32_t kk = key & 15u;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t piece = 4 * j + part;
        cp_async_16_zfill(dst_base + (piece * 16 + (kk ^ swz)) * 16, v_data + voff + piece * 16,
                          valid);
      }
    }""")

P_INIT = ("  loader.key = 8 * warp + (lane & 7u);\n  loader.part = lane >> 3;",
          "  loader.key = 8 * warp + (lane >> 2);\n  loader.part = lane & 3u;")

P_Q = ("""    const uint32_t row = loader.key;  // 8 warps x 8 rows
    uint32_t qi, hi;
    group_size.divmod(qo_tile_idx * kTileQ + row, qi, hi);
    const bool valid = qi < qo_upper_bound64;
    const DTypeQ* src = q_base + (valid ? (qi * q_stride_n + hi * q_stride_h) : 0);
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t chunk = 4 * i + loader.part;""",
       """    // 8 warps x 8 rows; a quarter-warp loads 4 rows x 2 adjacent chunks (one whole 32-byte
    // sector of each row).
    const uint32_t row = 8 * warp + 4 * ((lane >> 3) & 1u) + ((lane >> 1) & 3u);
    const uint32_t chunk0 = 2 * (lane >> 4) + (lane & 1u);
    uint32_t qi, hi;
    group_size.divmod(qo_tile_idx * kTileQ + row, qi, hi);
    const bool valid = qi < qo_upper_bound64;
    const DTypeQ* src = q_base + (valid ? (qi * q_stride_n + hi * q_stride_h) : 0);
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t chunk = 4 * i + chunk0;""")

P_V = ("""  const uint32_t v_lane = ((8 * w + (lane >> 4)) * 16 + frag_row) * 16;""",
       """  // The ring rows are swizzled (RingLoader): key row r of piece p sits at r ^ ((2 p) & 7);
  // for piece 8 w + (lane >> 4) + 2 pp that is r ^ (2 (lane >> 4)) ^ (4 (pp & 1)).
  const uint32_t v_lane = ((8 * w + (lane >> 4)) * 16 + (frag_row & 8u)) * 16;
  const uint32_t v_x = (lane & 7u) ^ (2u * (lane >> 4));""")
P_VLOOP = ("          ldsm_x4_trans(sb + v_lane + pp * 2 * 16 * 16, vr);",
           "          ldsm_x4_trans(sb + v_lane + pp * 2 * 16 * 16 + (v_x ^ (4u * (pp & 1u))) * 16, vr);")

K_LANE_OLD = "  const uint32_t k_lane = (16 * w + g) * 16;"
K_READ_OLD = """            // One 16-byte load of the key's d-step; lane t keeps bytes 2t, 2t+1 (word t/2)
            // and 2t+8, 2t+9 (word t/2 + 2): the B fragment's k positions.
            const uint4 kw = lds_u128(sb + k_lane + (sl * kTileKV + 8 * nt) * 16);
            const uint32_t lo = (t & 2u) ? kw.y : kw.x;
            const uint32_t hi = (t & 2u) ? kw.w : kw.z;"""
K_LANE = {
    "A": "  // K B fragment: key 16 w + 8 nt + g, at swizzled row g ^ ((2 sl) & 7) of piece sl.\n"
         "  const uint32_t k_lane = 16 * w * 16;",
    "B": "  // + word t/2 of the key's 16-byte d-step (the second word, t/2 + 2, is 8 bytes on).\n"
         "  const uint32_t k_lane = (16 * w + g) * 16 + (t >> 1) * 4;",
    "AB": "  // K B fragment: key 16 w + 8 nt + g, at swizzled row g ^ ((2 sl) & 7) of piece sl,\n"
          "  // + word t/2 of its 16-byte d-step (the second word, t/2 + 2, is 8 bytes on).\n"
          "  const uint32_t k_lane = 16 * w * 16 + (t >> 1) * 4;",
}
K_READ = {
    "A": """            // One 16-byte load of the key's d-step; lane t keeps bytes 2t, 2t+1 (word t/2)
            // and 2t+8, 2t+9 (word t/2 + 2): the B fragment's k positions.
            const uint4 kw = lds_u128(sb + k_lane +
                                      (sl * kTileKV + 8 * nt + (g ^ ((2u * sl) & 7u))) * 16);
            const uint32_t lo = (t & 2u) ? kw.y : kw.x;
            const uint32_t hi = (t & 2u) ? kw.w : kw.z;""",
    "B": """            // Lane t needs bytes 2t, 2t+1 (word t/2) and 2t+8, 2t+9 (word t/2 + 2) of the
            // key's d-step: two 4-byte loads, one wavefront each (8 rows x 2 words).
            const uint32_t k_addr = sb + k_lane + (sl * kTileKV + 8 * nt) * 16;
            const uint32_t lo = lds_u32(k_addr);
            const uint32_t hi = lds_u32(k_addr + 8);""",
    "AB": """            // Lane t needs bytes 2t, 2t+1 (word t/2) and 2t+8, 2t+9 (word t/2 + 2) of the
            // key's d-step: two 4-byte loads, one wavefront each (8 rows x 2 words).
            const uint32_t k_addr =
                sb + k_lane + (sl * kTileKV + 8 * nt + (g ^ ((2u * sl) & 7u))) * 16;
            const uint32_t lo = lds_u32(k_addr);
            const uint32_t hi = lds_u32(k_addr + 8);""",
}

CP_OLD = """__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gptr,
                                                  bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(smem_addr), "l"(gptr),
               "r"(pred ? 16 : 0));
}"""
# C: the fixed-size cp.async for in-range rows and a 16-byte zero store for the rest (the
# runtime src-size form may not coalesce across threads)
CP_C = """__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gptr,
                                                  bool pred) {
  // In-range rows: the fixed 16-byte cp.async. Rows past the chunk: 16 zero bytes stored
  // directly (what the zero-fill form writes). On SM120 the runtime src-size (zero-fill) form
  // and the ignore-src form are not coalesced across lanes: each 16-byte lane fetches its own
  // 32-byte sector, so a warp's 512 bytes moved ~34 sectors and ~30 smem wavefronts.
  if (pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"(smem_addr), "l"(gptr));
  } else {
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\\n" ::"r"(smem_addr), "r"(0u));
  }
}"""
# D: the ignore-src predicate form
CP_D = """__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gptr,
                                                  bool pred) {
  asm volatile(
      "{\\n .reg .pred ign;\\n setp.eq.u32 ign, %2, 0;\\n"
      " cp.async.cg.shared.global [%0], [%1], 16, ign;\\n}\\n" ::"r"(smem_addr),
      "l"(gptr), "r"(pred ? 1u : 0u));
}"""
# E: C with the L2 128-byte prefetch hint
CP_E = CP_C.replace("cp.async.cg.shared.global [%0]", "cp.async.cg.shared.global.L2::128B [%0]")

FILL = [P_LOADER_DOC, P_LOADER, P_INIT, P_Q, P_V, P_VLOOP]
AB = [P_LDS32] + FILL + [(K_LANE_OLD, K_LANE["AB"]), (K_READ_OLD, K_READ["AB"])]
ARMS = {
    "main": [],
    "A": FILL + [(K_LANE_OLD, K_LANE["A"]), (K_READ_OLD, K_READ["A"])],
    "B": [P_LDS32, (K_LANE_OLD, K_LANE["B"]), (K_READ_OLD, K_READ["B"])],
    "AB": AB,
    "C": [(CP_OLD, CP_C)],
    "ABC": AB + [(CP_OLD, CP_C)],
    "ABD": AB + [(CP_OLD, CP_D)],
    "ABE": AB + [(CP_OLD, CP_E)],
}
# arms a tree may carry, largest first (main_text reverses the first whose texts all match)
TREE_ARMS = ("ABC", "ABD", "ABE", "AB")


def log(*a):
    print(*a, flush=True)


def heartbeat(stop, tag):
    t0 = time.time()
    while not stop.wait(60):
        log(f"[hb {tag}] {time.time() - t0:.0f}s")


def package_data():
    import importlib.util
    spec = importlib.util.find_spec("flashinfer")
    return pathlib.Path(spec.origin).resolve().parent / "data"


def sha(txt):
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def apply(txt, patches, tag, reverse=False):
    for old, new in patches:
        if reverse:
            old, new = new, old
        n = txt.count(old)
        if n != 1:
            raise RuntimeError(f"{tag}: anchor found {n}x: {old[:90]!r}")
        txt = txt.replace(old, new)
    return txt


def main_text(tree_txt):
    """Main's kernel text: the tree as is, or the tree with its arm's patches reversed."""
    for arm in TREE_ARMS:
        if all(tree_txt.count(new) == 1 for _, new in ARMS[arm]):
            return (apply(tree_txt, list(reversed(ARMS[arm])), f"reverse {arm}", reverse=True),
                    f"tree={arm}")
    for old, _ in ARMS["AB"]:
        if tree_txt.count(old) != 1:
            raise RuntimeError(f"tree is neither main nor AB: {old[:90]!r}")
    return tree_txt, "tree=main"


def make_arm_tree(arm, dst, mtxt):
    data = package_data()
    dst = pathlib.Path(dst)
    for sub in ("include", "csrc"):
        shutil.copytree(data / sub, dst / sub, symlinks=False)
    txt = apply(mtxt, ARMS[arm], arm)
    (dst / KFILE).write_text(txt)
    return dst, sha(txt)


# ------------------------------------------------------------------ cases
def load_dims(ckpt):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    nq = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    ghd = tc.get("global_head_dim") or hd
    gkv = nkv
    if tc.get("attention_k_eq_v", False):
        gkv = tc.get("num_global_key_value_heads") or nkv
    d = dict(num_heads=nq, global_kv_heads=gkv, global_head_dim=ghd, sliding_head_dim=hd,
             sliding_kv_heads=nkv, attention_k_eq_v=tc.get("attention_k_eq_v"),
             num_layers=tc.get("num_hidden_layers"), hidden=tc.get("hidden_size"))
    qpath = os.path.join(ckpt, "hf_quant_config.json")
    if os.path.exists(qpath):
        q = json.load(open(qpath)).get("quantization", {})
        d["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo")
    return d


def read_rows_n(L, dims):
    """Trailing rows per request of layer 59's read-rows sub-call (pitfall 775), tile 64."""
    group = dims["num_heads"] // dims["global_kv_heads"]
    tile = 64
    head = tile // group + 1
    return head + (L - head) % (tile // math.gcd(tile, group))


def case_list(spec, dims):
    out = []
    for cs in [c for c in spec.split(",") if c]:
        if cs in NAMED:
            ql, kl, kw = NAMED[cs]
            out.append((cs, ql, kl, kw))
        elif cs.startswith("rr"):
            b, s = (int(x) for x in cs[2:].split("x"))
            out.append((cs, [read_rows_n(s, dims)] * b, [s] * b, {"disable_split_kv": True}))
        else:
            b, s = (int(x) for x in cs.split("x"))
            out.append((cs, [s] * b, None, {}))
    return out


class Case:
    def __init__(self, label, q_lens, kv_lens, dims, plan_kw, seed=1234):
        import torch
        self.label, self.plan_kw = label, dict(plan_kw)
        self.Hq, self.Hkv, self.D = dims["num_heads"], dims["global_kv_heads"], dims[
            "global_head_dim"]
        self.sm_scale = 1.0 / math.sqrt(self.D)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        self.q_lens = list(q_lens)
        self.kv_lens = list(kv_lens) if kv_lens is not None else list(q_lens)
        self.batch = len(self.q_lens)
        npages = [(L + PAGE - 1) // PAGE for L in self.kv_lens]
        qo, kvp = [0], [0]
        for L, n in zip(self.q_lens, npages):
            qo.append(qo[-1] + L)
            kvp.append(kvp[-1] + n)
        self.total_q = qo[-1]
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device="cuda")
        self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device="cuda")
        self.kv_indices = torch.randperm(kvp[-1], generator=g, device="cuda").to(torch.int32)
        self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in self.kv_lens],
                                          dtype=torch.int32, device="cuda")
        pool = torch.randn(kvp[-1], 2, self.Hkv, PAGE, self.D, generator=g, device="cuda",
                           dtype=torch.float32)
        self.kv_fp8 = pool.to(torch.float8_e4m3fn)
        del pool
        self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device="cuda",
                             dtype=torch.float32).to(torch.bfloat16)

    def reference(self):
        import torch
        out = torch.empty(self.total_q, self.Hq, self.D, dtype=torch.float32, device="cuda")
        grp = self.Hq // self.Hkv
        with torch.no_grad():
            for r in range(self.batch):
                s, e = int(self.qo_indptr[r]), int(self.qo_indptr[r + 1])
                Lq, Lk = e - s, self.kv_lens[r]
                ps, pe = int(self.kv_indptr[r]), int(self.kv_indptr[r + 1])
                pages = self.kv_fp8[self.kv_indices[ps:pe].long()].float()
                k = pages[:, 0].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:Lk]
                v = pages[:, 1].permute(0, 2, 1, 3).reshape(-1, self.Hkv, self.D)[:Lk]
                i = torch.arange(Lq, device="cuda").view(-1, 1) + (Lk - Lq)
                j = torch.arange(Lk, device="cuda").view(1, -1)
                mask = j <= i
                for h in range(self.Hq):
                    sc = (self.q[s:e, h].float() @ k[:, h // grp].T) * self.sm_scale
                    sc.masked_fill_(~mask, float("-inf"))
                    out[s:e, h] = torch.softmax(sc, dim=-1) @ v[:, h // grp]
        return out


def time_fn(fn, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = statistics.median(ts)
    p10 = ts[max(0, int(0.1 * n))]
    p90 = ts[min(n - 1, int(0.9 * n))]
    return dict(median_ms=med, p10_ms=p10, p90_ms=p90, noise=(p90 - p10) / med)


def raw(t):
    import torch
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def res_usage(base):
    """cuobjdump -res-usage of the Tile64 kernels in the JIT modules under base."""
    out = []
    cuobj = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    for so in sorted(pathlib.Path(base).rglob("*.so")):
        if "prefill" not in so.name:
            continue
        try:
            txt = subprocess.run([cuobj, "-res-usage", str(so)], capture_output=True, text=True,
                                 timeout=600).stdout
        except Exception as ex:  # noqa: BLE001
            out.append(f"{so.name}: cuobjdump failed {ex}")
            continue
        lines = txt.splitlines()
        for i, ln in enumerate(lines):
            if "D512Tile64" in ln and i + 1 < len(lines):
                out.append(f"{so.name}: {lines[i + 1].strip()}")
    return out


# ------------------------------------------------------------------ worker
def worker(a):
    if a.fresh_base:
        os.environ["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix=f"idea55_ws_{a.arm}_")
        os.environ["FLASHINFER_JIT_DEBUG"] = "0"
    if a.lineinfo:
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = "-lineinfo"
    if a.patch_arm:
        # build this idea-55 arm's include tree from the job's tree, then run it
        tree_txt = (package_data() / KFILE).read_text()
        mtxt, how = main_text(tree_txt)
        d, s = make_arm_tree(a.patch_arm, tempfile.mkdtemp(prefix=f"idea55_arm_{a.patch_arm}_"),
                             mtxt)
        a.inc = str(d)
        log("PATCH_ARM", a.patch_arm, how, "tile64 sha", s, "tree", d)
    from flashinfer.jit import env as jit_env
    if a.inc != "tree":
        jit_env.FLASHINFER_INCLUDE_DIR = pathlib.Path(a.inc) / "include"
        jit_env.FLASHINFER_CSRC_DIR = pathlib.Path(a.inc) / "csrc"
    jit_env.get_aot_artifacts = lambda name: ()
    import torch
    import flashinfer
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, a.arm), daemon=True).start()
    log("ARM", a.arm, "WS", os.environ.get("FLASHINFER_WORKSPACE_BASE", "<default>"),
        "EXTRA_CUDAFLAGS", os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""))
    inc = pathlib.Path(jit_env.FLASHINFER_INCLUDE_DIR)
    ktxt = (inc / "flashinfer/attention/prefill_d512_tile64.cuh").read_text()
    vtxt = (inc / "flashinfer/vec_dtypes.cuh").read_text()
    log("SRC tile64 sha", sha(ktxt), "swizzle" if "const uint32_t swz = 2u * part;" in ktxt
        else "", "lds32" if "lds_u32(k_addr)" in ktxt else "",
        "| vec_dtypes hw cvt:", "cvt.rn.bf16x2.e4m3x2" in vtxt)
    dims = load_dims(a.ckpt)
    log("DIMS", json.dumps(dims))
    p = torch.cuda.get_device_properties(0)
    log("GPU", p.name, "SMs", p.multi_processor_count)
    ws = torch.zeros(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    save = pathlib.Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    batch, recs, ok = {}, [], True
    if not a.no_warm:
        t0 = time.time()
        c = Case("warm", [300], None, dims, {})
        wo = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        ww = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        ww.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D, PAGE,
                causal=True, sm_scale=c.sm_scale, window_left=-1, q_data_type=torch.bfloat16,
                kv_data_type=torch.float8_e4m3fn)
        ww.run(c.q, c.kv_fp8, out=wo)
        torch.cuda.synchronize()
        del c, wo, ww
        log("WARM jit/load", round(time.time() - t0, 1), "s")
        xa = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        t1 = time.time()
        while time.time() - t1 < 3.0:
            for _ in range(20):
                xa = (xa @ xa).clamp_(-1, 1)
            torch.cuda.synchronize()
        del xa
        torch.cuda.empty_cache()
    for label, ql, kl, kw in case_list(a.cases, dims):
        t0 = time.time()
        c = Case(label, ql, kl, dims, kw)
        out = torch.empty(c.total_q, c.Hq, c.D, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(c.total_q, c.Hq, dtype=torch.float32, device="cuda")
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        w.plan(c.qo_indptr, c.kv_indptr, c.kv_indices, c.last_page_len, c.Hq, c.Hkv, c.D, PAGE,
               causal=True, sm_scale=c.sm_scale, window_left=-1, q_data_type=torch.bfloat16,
               kv_data_type=torch.float8_e4m3fn, **c.plan_kw)
        pi = [int(x) for x in w._plan_info]
        q0, kv0 = c.q.clone(), c.kv_fp8.clone()

        def run():
            w.run(c.q, c.kv_fp8, out=out)

        run()
        torch.cuda.synchronize()
        first = out.clone()
        mism = 0
        if not a.profile:
            # untimed repeated-execution sequence: same call count as the timing run
            for _ in range(a.warmup + a.iters - 1):
                run()
                torch.cuda.synchronize()
                mism += int(not torch.equal(raw(out), raw(first)))
        w.run(c.q, c.kv_fp8, out=out, lse=lse, return_lse=True)
        torch.cuda.synchronize()
        o_lse_same = bool(torch.equal(raw(out), raw(first)))
        rec = dict(case=label, cta_tile_q=pi[3], split_kv=bool(pi[14]), padded_batch=pi[0],
                   repeat=dict(calls=1 if a.profile else a.warmup + a.iters, mismatches=mism),
                   o_lse_call_equal=o_lse_same,
                   q_unchanged=bool(torch.equal(raw(q0), raw(c.q))),
                   kv_unchanged=bool(torch.equal(raw(kv0), raw(c.kv_fp8))),
                   finite=bool(first.float().isfinite().all()))
        ok &= (mism == 0 and o_lse_same and rec["q_unchanged"] and rec["kv_unchanged"]
               and rec["finite"])
        h = hashlib.sha256(raw(first).cpu().numpy().tobytes())
        h.update(raw(lse).cpu().numpy().tobytes())
        rec["o_lse_sha"] = h.hexdigest()[:24]
        if save:
            torch.save({"o": first.cpu(), "lse": lse.clone().cpu()}, save / f"{a.arm}_{label}.pt")
        if not a.no_time and not a.profile:
            t = time_fn(run, a.warmup, a.iters)
            rec.update({k: round(v, 5) for k, v in t.items()})
            batch[f"d512_{label}"] = dict(kernel_ms=round(t["median_ms"], 5),
                                          noise_floor=round(t["noise"], 4),
                                          p10_ms=round(t["p10_ms"], 5),
                                          p90_ms=round(t["p90_ms"], 5))
        rec["wall_s"] = round(time.time() - t0, 1)
        log("RESULT", json.dumps(rec))
        recs.append(rec)
        del c, out, lse, first, w, q0, kv0
        torch.cuda.empty_cache()
    stop.set()
    ru = res_usage(os.environ.get("FLASHINFER_WORKSPACE_BASE", str(jit_env.FLASHINFER_JIT_DIR))) \
        if a.res_usage else []
    for ln in ru:
        log("RES", ln)
    print(json.dumps({"batch": batch, "output": {"class": "exact"}, "passed": bool(ok),
                      "arm": a.arm, "res_usage": ru, "results": recs}), flush=True)


# ------------------------------------------------------------------ driver
def run_worker(arm, inc, a, base, save_dir):
    cmd = [sys.executable, __file__, "--worker", "--arm", arm, "--inc", str(inc), "--ckpt", a.ckpt,
           "--cases", a.cases, "--warmup", str(a.warmup), "--iters", str(a.iters), "--res-usage"]
    if save_dir:
        cmd += ["--save-dir", str(save_dir)]
    if a.no_time:
        cmd += ["--no-time"]
    env = dict(os.environ)
    env["FLASHINFER_WORKSPACE_BASE"] = str(base)
    env["FLASHINFER_JIT_DEBUG"] = "0"
    env.pop("FLASHINFER_FA2_D512_TILE64", None)
    log("SPAWN", arm, "base", base)
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("{"):
            last = line
            continue
        if line.startswith(("[", " ")) and "nvcc" in line:
            continue
        print(f"[{arm}] {line[:1500]}", flush=True)
    rc = p.wait()
    log("ARM_DONE", arm, "rc", rc, "wall_s", round(time.time() - t0, 1))
    return json.loads(last) if (rc == 0 and last) else None


def driver(a):
    import torch
    work = pathlib.Path(tempfile.mkdtemp(prefix="idea55_"))
    tree_txt = (package_data() / KFILE).read_text()
    mtxt, how = main_text(tree_txt)
    log("DRIVER work", work, "tree sha", sha(tree_txt), how, "main sha", sha(mtxt))
    arms = [x for x in a.arms.split(",") if x]
    trees, bases = {}, {}
    for arm in arms:
        d, s = make_arm_tree(arm, work / f"tree_{arm}", mtxt)
        trees[arm] = d
        bases[arm] = pathlib.Path(tempfile.mkdtemp(prefix=f"idea55_ws_{arm}_"))
        log("ARM_TREE", arm, "tile64 sha", s, "== tree" if s == sha(tree_txt) else "")
    save = work / "out"
    res = {arm: [] for arm in arms}
    for r in range(a.rounds):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for arm in order:
            res[arm].append(run_worker(arm, trees[arm], a, bases[arm], save if r == 0 else None))
    passed = all(all(j is not None and j.get("passed") for j in res[arm]) for arm in arms)
    dims = load_dims(a.ckpt)
    numer = {}
    ref_arm = arms[0]
    for label, ql, kl, kw in case_list(a.cases, dims):
        outs = {arm: torch.load(save / f"{arm}_{label}.pt") for arm in arms
                if (save / f"{arm}_{label}.pt").exists()}
        ref = None
        if not a.no_ref:
            c = Case(label, ql, kl, dims, kw)
            ref = c.reference()
            del c
        m = outs.get(ref_arm)
        row = {}
        for arm, d in outs.items():
            o = d["o"].cuda()
            cell = {}
            if ref is not None:
                diff = o.float() - ref
                cell["max_abs_ref"] = float(diff.abs().max())
                cell["rel_fro_ref"] = float(diff.norm() / ref.norm())
            if m is not None and arm != ref_arm:
                mo = m["o"].cuda()
                cell["o_byte_mismatch"] = int((raw(o) != raw(mo)).sum())
                cell["lse_byte_mismatch"] = int((raw(d["lse"]) != raw(m["lse"])).sum())
                cell["maxabs_vs_main"] = float((o.float() - mo.float()).abs().max())
                passed &= cell["o_byte_mismatch"] == 0 and cell["lse_byte_mismatch"] == 0
            row[arm] = cell
        if m is None or len(outs) != len(arms):
            passed = False
        numer[label] = row
        log("NUMERICS", label, json.dumps(row))
        del ref, outs, m
        torch.cuda.empty_cache()
    table = {}
    for label, *_ in case_list(a.cases, dims):
        key = f"d512_{label}"
        table[key] = {arm: [None if j is None else j["batch"].get(key) for j in res[arm]]
                      for arm in arms}
    if not a.no_time:
        log("TABLE case | " + " | ".join(f"{arm} r{r}" for arm in arms for r in range(a.rounds)))
        for key, row in table.items():
            cells = []
            for arm in arms:
                for t in row[arm]:
                    cells.append("-" if t is None else
                                 f"{t['kernel_ms']:.4f}({t['p10_ms']:.4f},{t['p90_ms']:.4f})")
            meds = {arm: statistics.median([t["kernel_ms"] for t in row[arm] if t])
                    for arm in arms if any(row[arm])}
            rel = " ".join(f"{arm}/{ref_arm}={meds[arm] / meds[ref_arm]:.4f}" for arm in meds
                           if arm != ref_arm and ref_arm in meds)
            log("TABLE", key, "|", " | ".join(cells), "|", rel)
    for arm in arms:
        j = res[arm][0]
        log("RES_USAGE", arm, json.dumps(j.get("res_usage") if j else None))
        if j:
            log("TILES", arm, json.dumps({r["case"]: [r["cta_tile_q"], r["split_kv"],
                                                      r["padded_batch"]] for r in j["results"]}))
    print(json.dumps({"passed": bool(passed), "arms": arms, "how": how, "table": table,
                      "numerics": numer,
                      "res_usage": {arm: (res[arm][0] or {}).get("res_usage") for arm in arms}}),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default="tree")
    ap.add_argument("--inc", default="tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", default=CASES + "," + EXTRA)
    ap.add_argument("--arms", default="main,A,B,AB")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--profile", action="store_true", help="one run + one LSE call per case")
    ap.add_argument("--lineinfo", action="store_true")
    ap.add_argument("--res-usage", action="store_true")
    ap.add_argument("--fresh-base", action="store_true")
    ap.add_argument("--patch-arm", default="", help="worker: run this arm (text-patched copy)")
    a = ap.parse_args()
    if a.worker:
        worker(a)
    else:
        driver(a)


if __name__ == "__main__":
    main()
