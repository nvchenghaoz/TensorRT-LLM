#!/usr/bin/env python3
# GRID PROOF for idea 93 (B=1 XQA multi-block cap): runs an eager TrtllmAttention generation step (served decode
# shape: 24 Q / 4 KV heads, head_dim 256, FP8 e4m3 paged KV, 32-token pages, max_seq_len 36864) at
# num_cached_tokens 36863 for B=1 and B=2 through the built TRT-LLM library of the current tree, profiles it with
# torch.profiler and reads kernel_mha's launch grid from the Kineto trace. TRTLLM_XQA_BLOCKS_PER_SEQUENCE is unset.
# Usage: b1_grid_proof.py <expected m at B=1> <expected m at B=2>
import json
import os
import sys
import tempfile

os.environ.pop("TRTLLM_XQA_BLOCKS_PER_SEQUENCE", None)

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402
from tensorrt_llm._torch.attention.backends import TrtllmAttention, TrtllmAttentionMetadata  # noqa: E402
from tensorrt_llm._torch.metadata import KVCacheParams  # noqa: E402
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager  # noqa: E402
from tensorrt_llm.bindings import DataType  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

NQ, NKV, HD, TPB, MAXSEQ = 24, 4, 256, 32, 36864
CACHED = 36863


def log(*a):
    print(*a, flush=True)


def main():
    exp = {1: int(sys.argv[1]), 2: int(sys.argv[2])}
    log("device", torch.cuda.get_device_name(), "sm count", torch.cuda.get_device_properties(0).multi_processor_count,
        "trtllm", tensorrt_llm.__version__, "file", tensorrt_llm.__file__)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    nblk = 2 * (MAXSEQ // TPB) + 16
    mgr = KVCacheManager(KvCacheConfig(max_tokens=nblk * TPB, enable_block_reuse=False),
                         tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF, num_layers=1, num_kv_heads=NKV,
                         head_dim=HD, tokens_per_block=TPB, max_seq_len=MAXSEQ, max_batch_size=2, mapping=mapping,
                         dtype=DataType.FP8)
    mgr.add_dummy_requests([0, 1], token_nums=[CACHED + 1] * 2, is_gen=True)
    log("kv cache manager ready")
    attn = TrtllmAttention(layer_idx=0, num_heads=NQ, head_dim=HD, num_kv_heads=NKV,
                           quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8), kv_cache_dtype="fp8")
    log("has_fp8_kv_cache", attn.has_fp8_kv_cache)
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    res, passed = {}, True
    for b in (1, 2):
        md = TrtllmAttentionMetadata(max_num_requests=2, max_num_tokens=8192, kv_cache_manager=mgr, mapping=mapping)
        md.seq_lens = torch.ones(b, dtype=torch.int)
        md.beam_width = 1
        md.num_contexts = 0
        md.request_ids = list(range(b))
        md.prompt_lens = [CACHED] * b
        md.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[CACHED] * b)
        md.prepare()
        qkv = (torch.randn(b, (NQ + 2 * NKV) * HD, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        out = attn.forward(qkv, None, None, md, is_fused_qkv=True)  # warm (JIT compile)
        torch.cuda.synchronize()
        trace = os.path.join(tempfile.mkdtemp(), f"b{b}.json")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = attn.forward(qkv, None, None, md, is_fused_qkv=True)
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace)
        o = out[0] if isinstance(out, (tuple, list)) else out
        ev = json.load(open(trace))
        evs = ev["traceEvents"] if isinstance(ev, dict) else ev
        kern = [e for e in evs if e.get("cat") == "kernel"]
        mha = [e for e in kern if "kernel_mha" in e.get("name", "")]
        grids = [e.get("args", {}).get("grid") for e in mha]
        ok = len(grids) == 1 and grids[0] == [exp[b], NKV, b] and bool(torch.isfinite(o.float()).all().item())
        passed &= ok
        res[str(b)] = dict(kernel_mha_grids=grids, expected_grid=[exp[b], NKV, b],
                           kernel_mha_us=[e.get("dur") for e in mha], kernels=[e.get("name")[:80] for e in kern],
                           output_finite=bool(torch.isfinite(o.float()).all().item()), passed=ok)
        log(f"B={b}: kernel_mha grids {grids} (expected {[exp[b], NKV, b]}), kernels {res[str(b)]['kernels']} -> "
            f"{'PASS' if ok else 'FAIL'}")
    mgr.shutdown()
    print(json.dumps({"passed": bool(passed), "grid": res, "num_cached_tokens": CACHED,
                      "env_TRTLLM_XQA_BLOCKS_PER_SEQUENCE": os.environ.get("TRTLLM_XQA_BLOCKS_PER_SEQUENCE"),
                      "compared": "kernel_mha launch grid (Kineto trace) vs expected {m,4,B}"}), flush=True)


if __name__ == "__main__":
    main()
