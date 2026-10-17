import json, subprocess, time, torch
from types import SimpleNamespace
print("TREE_COMMIT", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)
from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk, gather_cast_vk_to_fp32_vk
from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows
print("HAS_SKIP_ATTR", hasattr(gdn_mixer, "_SKIP_NOOP_RESET"), getattr(gdn_mixer, "_SKIP_NOOP_RESET", None), flush=True)
L = 48; NB = 12; H = 48; V = 128; K = 128; SSM_B = H * V * K * 2; CONV_B = 10240 * 3 * 2; CAP = 8; T = 2048; CACHED = 32768; SLOTS = [3, 7, 1, 10]
dev = torch.device("cuda")
torch.manual_seed(0)
pool = torch.empty(L, NB, SSM_B + CONV_B, dtype=torch.uint8, device=dev)
ssm_all = pool[:, :, :SSM_B].view(torch.bfloat16).view(L, NB, H, V, K)
conv_all = pool[:, :, SSM_B:].view(torch.bfloat16).view(L, NB, 10240, 3)
for l in range(L):
    ssm_all[l].copy_(torch.randn(NB, H, V, K, device=dev).to(torch.bfloat16))
    conv_all[l].copy_(torch.randn(NB, 10240, 3, device=dev).to(torch.bfloat16))
pristine = pool.clone()
ws0 = torch.empty(CAP, H, V, K, dtype=torch.float32, device=dev)
ws1 = torch.randn(CAP, H, V, K, dtype=torch.float32, device=dev)
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
print("pool ready", flush=True)
def seg(md, P, n):
    st = md.state_indices[:n]
    stp, _ = torch.split(st, [P, n - P])
    for l in range(L):
        s = ssm_all[l]; c = conv_all[l]
        if P > 0 and not (getattr(gdn_mixer, "_SKIP_NOOP_RESET", False) and getattr(md, "all_initial_states", False)):
            reset_recurrent_state_rows(s, stp, md.has_initial_states[:P], c)
        gather_cast_vk_to_fp32_vk(s, st, out_dtype=torch.float32, output=ws0[:n])
        cast_scatter_fp32_vk_to_vk(ws1[:n], s, st)
batch = dict()
for n, P, D in [(1, 1, 0), (2, 1, 1), (4, 3, 1)]:
    sl = torch.tensor([T] * P + [1] * D, dtype=torch.int32)
    am = SimpleNamespace(seq_lens=sl, seq_lens_cuda=sl.to(dev), num_contexts=P, num_ctx_tokens=T * P, num_tokens=T * P + D, kv_cache_manager=None, request_ids=None, kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=torch.tensor([CACHED] * P + [CACHED + T] * D, dtype=torch.int32)))
    md = Mamba2Metadata(CAP, 64)
    md.prepare(am)
    md.state_indices[:n].copy_(torch.tensor(SLOTS[:n], dtype=torch.int32))
    ts = []
    for it in range(70):
        for s in SLOTS[:n]:
            pool[:, s].copy_(pristine[:, s])
        flush.add_(1)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); seg(md, P, n); e1.record(); torch.cuda.synchronize()
        if it >= 10:
            ts.append(e0.elapsed_time(e1))
    q = torch.quantile(torch.tensor(ts, dtype=torch.float64), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
    print("num_seqs", n, "P", P, "D", D, "all_initial_states", getattr(md, "all_initial_states", None), "ms_48layers p10/med/p90", q, flush=True)
    batch[str(n)] = dict(kernel_ms=q[1], noise_floor=(q[2] - q[0]) / q[1], p10_ms=q[0], p90_ms=q[2])
print(json.dumps(dict(batch=batch, output=dict([("class", "exact")]))), flush=True)
