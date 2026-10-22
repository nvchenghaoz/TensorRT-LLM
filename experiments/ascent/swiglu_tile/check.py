"""Branch 11 correctness (untimed), REAL checkpoint weights of one dense MLP.

Main's dispatch (d0d87a23: SwiGLU epilogue from 17 rows, tile 0 from 1024) is
reproduced in-process through GatedMLP._SWIGLU_EPI_MIN_ROWS/_swiglu_epi_tile,
the only things this branch changes. Per M in {8,16,24,32,512,2079,4127}:
  * down_proj input (FP4 bytes + swizzled SF) and MLP output of the branch's
    dispatch == main's dispatch, bitwise (autotuned tactics, same process);
  * rows where the branch moves from fused to unfused (17..40): every valid
    gate_up nvfp4_gemm tactic + fused_silu_mul_nvfp4_permuted -> down equals
    main's fused output bitwise (covers any tactic the serving tuner picks);
  * repeated execution, 10 warmup + 50 runs: decode rows (<= 32) as CUDA-graph
    replays (outputs/taps NaN-filled before each replay), prefill rows eager with
    the allocator pool poisoned; each run == run 1 and == main's output.
Prefill rows (>= 2048) take an Fp4QuantizedTensor from
fused_gemma_add_rmsnorm_nvfp4 as in serving; smaller rows take BF16.
"""
import glob
import json
import os
import sys

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "swiglu_epi"))
from mlp_common import DownInputTap, EPS, H, I, dev, sha  # noqa: E402

from tensorrt_llm._torch.autotuner import autotune  # noqa: E402
from tensorrt_llm._torch.custom_ops.torch_custom_ops import NVFP4GemmUnifiedRunner  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP  # noqa: E402
from tensorrt_llm._torch.utils import Fp4QuantizedTensor  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

CKPT = sys.argv[1]
LAYER = int(os.environ.get("CHECK_LAYER", "5"))
MS = [8, 16, 24, 32, 512, 2079, 4127]
WARM, ITERS = 10, 50
NEW_MIN, NEW_TILE = GatedMLP._SWIGLU_EPI_MIN_ROWS, GatedMLP._swiglu_epi_tile


def log(*a):
    print(*a, flush=True)


def set_main():
    GatedMLP._SWIGLU_EPI_MIN_ROWS = 17
    GatedMLP._swiglu_epi_tile = staticmethod(lambda m: 0 if m >= 1024 else 1)


def set_new():
    GatedMLP._SWIGLU_EPI_MIN_ROWS = NEW_MIN
    GatedMLP._swiglu_epi_tile = staticmethod(NEW_TILE)


def load_tensors():
    want = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if f"layers.{LAYER}.mlp." in k and "experts" not in k:
                    want[k] = sf.get_tensor(k)
    log("KEYS", json.dumps({k: [str(v.dtype), list(v.shape)] for k, v in want.items()}))
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        d = {}
        for k, v in want.items():
            if f".mlp.{proj}." in k:
                d[k.rsplit(".", 1)[1]] = v
        out[proj] = d
    return out


def build_real():
    w = load_tensors()
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    mlp = GatedMLP(hidden_size=H, intermediate_size=I, bias=False, dtype=torch.bfloat16,
                   config=ModelConfig(quant_config=qc), layer_idx=LAYER).to(dev)
    mlp.gate_up_proj.load_weights([w["gate_proj"], w["up_proj"]])
    mlp.down_proj.load_weights([w["down_proj"]])
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        lin.process_weights_after_loading()
    mlp.post_load_weights()
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        lin.post_load_weights()
    return mlp


def make_x(mlp, m, gamma):
    g = torch.Generator(device=dev).manual_seed(1000 + m)
    x = torch.randn((m, H), generator=g, device=dev).to(torch.bfloat16)
    r = (torch.randn((m, H), generator=g, device=dev) * 2.0).to(torch.bfloat16)
    if m >= 2048:
        q, s = torch.ops.trtllm.fused_gemma_add_rmsnorm_nvfp4(x, r, gamma, EPS, mlp.gate_up_proj.input_scale)
        return Fp4QuantizedTensor(q, s)
    return x


def run(mlp, tap, x):
    o = mlp(x)
    torch.cuda.synchronize()
    fq, fsf = tap.last
    return o.view(torch.int16).clone(), fq.clone(), (fsf.clone() if fsf is not None else None)


def eq3(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and x.shape == y.shape
                                             and torch.equal(x, y)) for x, y in zip(a, b))


def poison_pool():
    t = torch.empty(1536 * 1024 * 1024, dtype=torch.uint8, device=dev)
    t.fill_(0xFF)
    del t


def main():
    mlp = build_real()
    log("permuted", bool(getattr(mlp, "_swiglu_epi_permuted", False)), "new_min", NEW_MIN,
        "tiles", {m: NEW_TILE(m) for m in MS})
    tap = DownInputTap(mlp)
    gamma = (torch.randn(H, device=dev) * 0.3).to(torch.bfloat16)
    ok = bool(getattr(mlp, "_swiglu_epi_permuted", False))
    res = {"cases": {}, "per_tactic": {}, "repeat": {}}
    for m in MS:
        x = make_x(mlp, m, gamma)
        set_main()
        with autotune():
            mlp(x)
        torch.cuda.synchronize()
        ref = run(mlp, tap, x)
        set_new()
        cand = run(mlp, tap, x)
        e = eq3(cand, ref)
        fp4_nz = int((ref[1] != 0).sum())
        res["cases"][str(m)] = {"eq_main": e, "sha_out": sha(ref[0]), "sha_fp4": sha(ref[1]), "sha_sf": sha(ref[2]),
                                "fp4_nonzero": fp4_nz,
                                "path_main": "fused" if m >= 17 else "unfused",
                                "path_new": "fused" if m >= NEW_MIN else "unfused"}
        ok &= e and fp4_nz > 0
        log("CASE", m, json.dumps(res["cases"][str(m)]))
        # every gate_up tactic on the unfused path where the branch switches it
        if 17 <= m < NEW_MIN:
            runner = NVFP4GemmUnifiedRunner(0, torch.bfloat16, ["cutlass", "cublaslt", "cuda_core"])
            gu = mlp.gate_up_proj
            a, asf = torch.ops.trtllm.fp4_quantize(x, gu.input_scale, 16, False)
            ins = [a, gu.weight, asf, gu.weight_scale, gu.alpha]
            tacs = runner.get_valid_tactics(ins, None)
            bad = []
            for t in tacs:
                h = runner(ins, tactic=t)
                q, s = torch.ops.trtllm.fused_silu_mul_nvfp4_permuted(h, mlp.down_proj.input_scale)
                o = mlp.down_proj(Fp4QuantizedTensor(q, s))
                torch.cuda.synchronize()
                if not eq3((o.view(torch.int16), q.reshape(ref[1].shape), s), ref):
                    bad.append(str(t))
            res["per_tactic"][str(m)] = {"tactics": len(tacs), "bad": bad}
            ok &= not bad and len(tacs) > 0
            log("PER_TACTIC", m, json.dumps(res["per_tactic"][str(m)]))
        # repeated execution with the branch's dispatch
        if m <= 32:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    mlp(x)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = mlp(x)
            gtap = tap.last
            torch.cuda.synchronize()

            def step():
                gout.view(torch.int16).fill_(-1)
                gtap[0].fill_(0xFF)
                if gtap[1] is not None:
                    gtap[1].fill_(0xFF)
                g.replay()
                torch.cuda.synchronize()
                return (gout.view(torch.int16).clone(), gtap[0].clone(),
                        gtap[1].clone() if gtap[1] is not None else None)
            mode = "cuda_graph_replay"
        else:
            def step():
                poison_pool()
                return run(mlp, tap, x)
            mode = "eager_poisoned_pool"
        first, n1, nb = None, 0, 0
        for it in range(WARM + ITERS):
            cur = step()
            if first is None:
                first = cur
            if it >= WARM:
                n1 += not eq3(cur, first)
                nb += not eq3(cur, ref)
        res["repeat"][str(m)] = {"mode": mode, "iters": ITERS, "neq_run1": n1, "neq_main": nb}
        ok &= n1 == 0 and nb == 0
        log("REPEAT", m, json.dumps(res["repeat"][str(m)]))
        if m <= 32:
            del g, gout
    print(json.dumps({"passed": bool(ok), "layer": LAYER,
                      "compared": "real-checkpoint MLP: down_proj input FP4+SF and output, branch dispatch vs "
                                  "main's dispatch (bitwise), per-tactic unfused vs main fused at 17..40 rows, "
                                  "10+50 repeats (graph replay <=32 rows, eager poisoned pool prefill)",
                      "result": res}))


if __name__ == "__main__":
    main()
