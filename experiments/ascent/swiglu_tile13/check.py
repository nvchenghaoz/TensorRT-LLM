"""Branch 13 correctness (untimed). Runs unchanged in main and the branch tree
(pass the file content inline: python -c "$(cat check.py)").

Per M (decode rows: BF16 input, tile 1; prefill rows: fused_gemma_add_rmsnorm
_nvfp4 input, tile 0), two synthetic Qwen3.8-27B dense MLP layers (H=5120,
I=17408, random FP4 weights + E4M3 SF, GatedMLP's own load-time permutation):
  * sha256 of down_proj's input (FP4 bytes + swizzled SF incl. padding rows)
    and of the MLP output, from the tree's own dispatch;
  * repeated execution, 10 warmup + 50 runs, same reset points as timing:
    decode rows as CUDA-graph replays with PDL (outputs NaN/0xFF-filled before
    each replay), prefill rows eager with the allocator pool poisoned; every
    run compared bitwise with run 1 and with the first eager call.
REF_HASHES (json from main's run) makes it compare against main: passed needs
every hash equal to main's and zero repeat mismatches.
"""
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "ascent", "swiglu_epi"))
from mlp_common import EPS, GS_X, H, DownInputTap, build_mlp, dev, sha  # noqa: E402

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402

MS = [int(v) for v in os.environ.get("CHECK_MS", "41,48,64,96,128,447,448,2112,4160,6208,8192").split(",")]
WARM, ITERS = 10, 50
REF = json.loads(os.environ.get("REF_HASHES", "{}"))


def log(*a):
    print(*a, flush=True)


def poison_pool():
    t = torch.empty(1536 * 1024 * 1024, dtype=torch.uint8, device=dev)
    t.fill_(0xFF)
    del t


def main():
    mlps = [build_mlp(300 + i, 1.0) for i in range(2)]
    taps = [DownInputTap(m) for m in mlps]
    gs = torch.tensor([GS_X], dtype=torch.float32, device=dev)
    gamma = (torch.randn(H, generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.3).to(torch.bfloat16)
    ok = all(bool(getattr(m, "_swiglu_epi_permuted", False)) for m in mlps)
    hashes, rep = {}, {}
    for m in MS:
        g = torch.Generator(device=dev).manual_seed(1000 + m)
        x = (torch.randn((m, H), generator=g, device=dev) * 0.7).to(torch.bfloat16)
        r = (torch.randn((m, H), generator=g, device=dev) * 3.0).to(torch.bfloat16)
        prefill = m >= 1024

        def fwd(i):
            if prefill:
                # the fused add+norm updates the residual in place: reset it per call
                q, s = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r.clone(), gamma, EPS, gs)
                return mlps[i](Fp4QuantizedTensor(q, s))
            return mlps[i](x)

        firsts = []
        for i in range(2):
            o = fwd(i)
            torch.cuda.synchronize()
            fq, fsf = taps[i].last
            assert fsf is not None, "down_proj input is not NVFP4 (fused path not taken)"
            firsts.append((o.view(torch.int16).clone(), fq.clone(), fsf.clone()))
            key = f"{m}/{i}"
            hashes[key] = {"out": sha(firsts[-1][0]), "fp4": sha(fq), "sf": sha(fsf),
                           "fp4_nonzero": int((fq != 0).sum())}
            if REF:
                same = REF.get(key) == hashes[key]
                ok &= same
                log("HASH", key, json.dumps(hashes[key]), "eq_main", same)
            else:
                log("HASH", key, json.dumps(hashes[key]))
            ok &= hashes[key]["fp4_nonzero"] > 0
        tile = GatedMLP._swiglu_epi_tile(m)
        if not prefill:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    [fwd(i) for i in range(2)]
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                gouts = [fwd(i) for i in range(2)]
            gtaps = [t.last for t in taps]
            torch.cuda.synchronize()

            def step():
                for go, (tq, ts) in zip(gouts, gtaps):
                    go.view(torch.int16).fill_(-1)
                    tq.fill_(0xFF)
                    ts.fill_(0xFF)
                gr.replay()
                torch.cuda.synchronize()
                return [(go.view(torch.int16).clone(), tq.clone(), ts.clone()) for go, (tq, ts) in zip(gouts, gtaps)]
            mode = "cuda_graph_replay_pdl"
        else:
            def step():
                poison_pool()
                res = []
                for i in range(2):
                    o = fwd(i)
                    torch.cuda.synchronize()
                    res.append((o.view(torch.int16).clone(), taps[i].last[0].clone(), taps[i].last[1].clone()))
                return res
            mode = "eager_poisoned_pool"

        def eq(a, b):
            return all(torch.equal(u, v) for pa, pb in zip(a, b) for u, v in zip(pa, pb))
        run1, n1, ne, maxdiff = None, 0, 0, 0
        for it in range(WARM + ITERS):
            cur = step()
            if run1 is None:
                run1 = cur
            if it >= WARM:
                n1 += not eq(cur, run1)
                ne += not eq(cur, firsts)
                maxdiff = max(maxdiff, max(int((c[1].int() - f[1].int()).abs().max()) for c, f in zip(cur, firsts)))
        rep[str(m)] = {"mode": mode, "tile": tile, "iters": ITERS, "neq_run1": n1, "neq_first_eager": ne,
                       "max_fp4_byte_diff": maxdiff}
        ok &= n1 == 0 and ne == 0
        log("REPEAT", m, json.dumps(rep[str(m)]))
        if not prefill:
            del gr, gouts, gtaps
        torch.cuda.empty_cache()
    log("REF_HASHES_JSON", json.dumps(hashes, separators=(",", ":")))
    print(json.dumps({"passed": bool(ok), "against_main": bool(REF),
                      "compared": "down_proj input FP4+SF (padding rows incl.) and MLP output sha256 vs main's "
                                  "kernel on identical inputs; 10+50 repeats (graph+PDL replay decode, eager "
                                  "poisoned pool prefill) bitwise vs run 1 and first call",
                      "repeat": rep}))


if __name__ == "__main__":
    main()
