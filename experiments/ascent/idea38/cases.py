"""Case specs for idea 38: 'BxS' (uniform) or 'r<l1>+<l2>+...' (ragged batch of lengths).

Builds on fa_census.Case (HND paged FP8 E4M3 KV, page 32, BF16 Q, random page permutation)."""
import fa_census as fc
import torch


class RaggedCase(fc.Case):
    """fc.Case with per-request lengths; reference()/gather_kv() are inherited unchanged."""

    def __init__(self, layer, lens, dims, seed=0):  # noqa: D107  (no super().__init__)
        _init_body(self, layer, list(lens), dims, seed)

    def label(self):
        return f"{self.layer}_d{self.D}_r" + "+".join(str(L) for L in self.lens)


def _init_body(self, layer, lens, dims, seed):
    import math
    PAGE = fc.PAGE
    self.layer = layer
    self.batch = len(lens)
    self.seqlen = max(lens)
    self.Hq = dims["num_heads"]
    if layer == "sliding":
        self.Hkv = dims["sliding_kv_heads"]
        self.D = dims["sliding_head_dim"]
        self.window_left = dims["sliding_window"] - 1
    else:
        self.Hkv = dims["global_kv_heads"]
        self.D = dims["global_head_dim"]
        self.window_left = -1
    self.sm_scale = 1.0 / math.sqrt(self.D)
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    self.lens = lens
    npages = [(L + PAGE - 1) // PAGE for L in lens]
    total_pages = sum(npages)
    self.total_q = sum(lens)
    qo, kvp = [0], [0]
    for L, n in zip(lens, npages):
        qo.append(qo[-1] + L)
        kvp.append(kvp[-1] + n)
    self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
    self.kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
    perm = torch.randperm(total_pages, generator=g, device=dev).to(torch.int32)
    self.kv_indices = perm
    self.last_page_len = torch.tensor([(L - 1) % PAGE + 1 for L in lens], dtype=torch.int32,
                                      device=dev)
    maxp = max(npages)
    bt = torch.zeros(len(lens), maxp, dtype=torch.int32, device=dev)
    for r in range(len(lens)):
        bt[r, :npages[r]] = perm[kvp[r]:kvp[r + 1]]
    self.block_tables = bt
    self.seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    pool = torch.randn(total_pages, 2, self.Hkv, PAGE, self.D, generator=g, device=dev,
                       dtype=torch.float32)
    self.kv_fp8 = pool.to(torch.float8_e4m3fn)
    self.kv_bf16 = self.kv_fp8.to(torch.bfloat16)
    self.q = torch.randn(self.total_q, self.Hq, self.D, generator=g, device=dev,
                         dtype=torch.float32).to(torch.bfloat16)
    self.q_fp8 = self.q.to(torch.float8_e4m3fn)
    cnt = 0
    for L in lens:
        if self.window_left < 0:
            cnt += L * (L + 1) // 2
        else:
            W = self.window_left + 1
            cnt += sum(min(i + 1, W) for i in range(L))
    self.pairs = cnt
    self.flops = 4.0 * self.Hq * self.D * cnt


def make_case(spec, dims, layer="sliding", seed=1234):
    if spec.startswith("r"):
        lens = [int(x) for x in spec[1:].split("+")]
        return RaggedCase(layer, lens, dims, seed=seed)
    b, s = (int(x) for x in spec.split("x"))
    return fc.Case(layer, b, s, dims, seed=seed)


def bytes_equal(a, b):
    """Raw byte compare (torch.equal treats -0.0 == +0.0 and NaN != NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a = a.contiguous().view(torch.uint8) if a.element_size() == 1 else a.contiguous().view(
        {2: torch.int16, 4: torch.int32}[a.element_size()])
    b = b.contiguous().view(a.dtype)
    return bool(torch.equal(a, b))
