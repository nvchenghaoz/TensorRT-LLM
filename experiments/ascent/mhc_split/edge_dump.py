"""Acceptance item 3 (event 4606): edge type into K1 in a captured CUDA graph that goes through
the REAL fork (mHC.fused_hc_lagged_split -> maybe_execute_in_parallel_deferred_join).

A stand-in all-reduce kernel (Triton, launched with the programmatic-stream-serialization
attribute, griddepcontrol.wait first and launch_dependents last) writes x_prev; then the
boundary runs. The graph is captured with torch.cuda.CUDAGraph(keep_graph=True) and walked
before instantiation with the driver API (cuGraphGetNodes / cuGraphGetEdges_v2 edge data /
cuGraphNodeGetType / cuGraphKernelNodeGetParams_v2 + cuFuncGetName). For every K1 node
(_mhc_lagged_layer_input_kernel) and every mHC CUDA kernel node, prints each incoming edge:
source node type and kernel name, from/to port, type (0 Default, 1 Programmatic).

Arms per M (1, 2, 4, 8):
  SPLIT    : the branch's fork (K1 on main, K2 = mhc_fused_hc on the aux stream).
  MAINPATH : split off (main's path): mhc_fused_hc directly after the stand-in AR.
  NOFORK   : control, K1 launched directly after the stand-in AR (no event record).
and REAL_LAYER (M=1): 3 real DeepseekV41DecoderLayer.forward (chain.py stand-in sublayers,
each ending with the stand-in AR), covering attention (hc_ffn) and FFN (hc_attn) boundaries.
Diagnostic only; not timing or numerics evidence.
"""

import ctypes
import json
import os
import sys

os.environ.setdefault("CHAIN_ENTRY_IDS", "")

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber
from tensorrt_llm._torch.modules.mhc import mhc_lagged_split as _split
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

K1_NAME = "_mhc_lagged_layer_input_kernel"
AR_NAME = "_ar_standin_kernel"
NODE_TYPES = {0: "KERNEL", 1: "MEMCPY", 2: "MEMSET", 3: "HOST", 4: "GRAPH", 5: "EMPTY",
              6: "WAIT_EVENT", 7: "EVENT_RECORD", 8: "EXT_SEM_SIGNAL", 9: "EXT_SEM_WAIT",
              10: "MEM_ALLOC", 11: "MEM_FREE", 12: "BATCH_MEM_OP", 13: "CONDITIONAL"}
EDGE_TYPES = {0: "Default", 1: "Programmatic"}
PORTS = {0: "DEFAULT", 1: "PROGRAMMATIC", 2: "LAUNCH_ORDER"}


def log(*a):
    print(*a, flush=True)


@triton.jit
def _ar_standin_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(y_ptr + offs, v, mask=m)
    tl.extra.cuda.gdc_launch_dependents()


def ar_standin(x, y):
    n = x.numel()
    _ar_standin_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024, launch_pdl=True)
    return y


# ---------------- driver API graph walk ----------------
cu = ctypes.CDLL("libcuda.so.1")


class EdgeData(ctypes.Structure):
    _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte), ("type", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte * 5)]


class KParams(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p), ("gx", ctypes.c_uint), ("gy", ctypes.c_uint), ("gz", ctypes.c_uint),
                ("bx", ctypes.c_uint), ("by", ctypes.c_uint), ("bz", ctypes.c_uint), ("smem", ctypes.c_uint),
                ("kernelParams", ctypes.c_void_p), ("extra", ctypes.c_void_p), ("kern", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p)]


def chk(r, what):
    if r != 0:
        raise RuntimeError(f"{what} -> CUresult {r}")


def graph_nodes(g):
    n = ctypes.c_size_t(0)
    chk(cu.cuGraphGetNodes(ctypes.c_void_p(g), None, ctypes.byref(n)), "cuGraphGetNodes(count)")
    arr = (ctypes.c_void_p * n.value)()
    chk(cu.cuGraphGetNodes(ctypes.c_void_p(g), arr, ctypes.byref(n)), "cuGraphGetNodes")
    return [int(v) for v in arr[: n.value]]


def graph_edges(g):
    fn = cu.cuGraphGetEdges_v2
    n = ctypes.c_size_t(0)
    chk(fn(ctypes.c_void_p(g), None, None, None, ctypes.byref(n)), "cuGraphGetEdges_v2(count)")
    frm = (ctypes.c_void_p * n.value)()
    to = (ctypes.c_void_p * n.value)()
    ed = (EdgeData * n.value)()
    chk(fn(ctypes.c_void_p(g), frm, to, ed, ctypes.byref(n)), "cuGraphGetEdges_v2")
    return [(int(frm[i]), int(to[i]), ed[i].from_port, ed[i].to_port, ed[i].type) for i in range(n.value)]


def node_type(node):
    t = ctypes.c_int(-1)
    chk(cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(t)), "cuGraphNodeGetType")
    return t.value


def kernel_name(node):
    p = KParams()
    r = cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p))
    if r != 0:
        return f"<params err {r}>"
    name = ctypes.c_char_p()
    if p.func:
        r = cu.cuFuncGetName(ctypes.byref(name), ctypes.c_void_p(p.func))
    elif p.kern:
        r = cu.cuKernelGetName(ctypes.byref(name), ctypes.c_void_p(p.kern))
    else:
        return "<no func>"
    if r != 0 or not name.value:
        return f"<name err {r}>"
    return name.value.decode(errors="replace")


def describe(node, cache):
    if node not in cache:
        t = node_type(node)
        nm = kernel_name(node) if t == 0 else ""
        cache[node] = (NODE_TYPES.get(t, str(t)), nm)
    return cache[node]


def short(nm):
    return nm if len(nm) <= 70 else nm[:67] + "..."


def walk(g_handle, tag):
    nodes = graph_nodes(g_handle)
    edges = graph_edges(g_handle)
    cache = {}
    for n in nodes:
        describe(n, cache)
    counts = {}
    for n in nodes:
        t, nm = cache[n]
        counts[t] = counts.get(t, 0) + 1
    incoming = {}
    outgoing = {}
    for e in edges:
        incoming.setdefault(e[1], []).append(e)
        outgoing.setdefault(e[0], []).append(e)

    def edge_rec(e, src_side=True):
        other = e[0] if src_side else e[1]
        t, nm = cache.get(other, describe(other, cache))
        return {"node_type": t, "kernel": short(nm), "from_port": PORTS.get(e[2], e[2]),
                "to_port": PORTS.get(e[3], e[3]), "type": EDGE_TYPES.get(e[4], e[4])}

    k1 = [n for n in nodes if cache[n][1] and K1_NAME in cache[n][1]]
    mhc = [n for n in nodes if cache[n][1] and "mhc" in cache[n][1].lower() and K1_NAME not in cache[n][1]]
    ar = [n for n in nodes if cache[n][1] and AR_NAME in cache[n][1]]
    out = {"nodes": len(nodes), "edges": len(edges), "node_types": counts,
           "k1": [[edge_rec(e) for e in incoming.get(n, [])] for n in k1],
           "mhc_cuda": [{"kernel": short(cache[n][1]), "in": [edge_rec(e) for e in incoming.get(n, [])]}
                        for n in mhc],
           "ar_out": [[edge_rec(e, src_side=False) for e in outgoing.get(n, [])] for n in ar]}
    log(f"[{tag}] nodes={len(nodes)} edges={len(edges)} types={counts}")
    for i, ins in enumerate(out["k1"]):
        log(f"[{tag}]   K1#{i} incoming: {json.dumps(ins)}")
    for i, rec in enumerate(out["mhc_cuda"]):
        log(f"[{tag}]   MHC#{i} {rec['kernel']} incoming: {json.dumps(rec['in'])}")
    for i, outs in enumerate(out["ar_out"]):
        log(f"[{tag}]   AR#{i} outgoing: {json.dumps(outs)}")
    return out


def dot_dump(g_handle, path):
    for flags in ((1 | (1 << 14)), 1):
        r = cu.cuGraphDebugDotPrint(ctypes.c_void_p(g_handle), path.encode(), ctypes.c_uint(flags))
        if r == 0:
            return flags
    return None


def capture_keep(body):
    with with_multi_stream(True):
        body()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            res = body()
    torch.cuda.synchronize()
    return g, res


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    art = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")
    os.makedirs(art, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    hidden = 4096
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import chain as _chain
        cfg = _chain.read_cfg(ckpt)
        hidden = int(cfg.get("hidden_size", hidden))
    except Exception as e:  # noqa: BLE001
        log("chain import/config failed:", repr(e))
        _chain, cfg = None, {}
    log(f"hidden={hidden}, PDL env={os.environ.get('TRTLLM_ENABLE_PDL', '1')}")
    mhc = mHC(4, hidden, 20, dtype=torch.float32, eps=1e-6, norm_eps=1e-6, sinkhorn_eps=1e-6,
              post_mult_value=2.0).to(dev)
    with torch.no_grad():
        mhc.fn.normal_(0, 1e-2)
        mhc.base.normal_(0, 0.1)
        mhc.scale.copy_(torch.tensor([0.5, 0.7, 0.9]))
    nw = (1 + 0.1 * torch.randn(hidden, device=dev)).to(torch.bfloat16)
    aux = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()

    results = {}
    k1_types = {}
    ok = True
    for M in (1, 2, 4, 8):
        x_src = torch.randn(M, hidden, device=dev).to(torch.bfloat16)
        x = torch.empty_like(x_src)
        res = torch.randn(M, 4, hidden, device=dev).to(torch.bfloat16)
        post = (2 * torch.rand(M, 4, 1, device=dev)).float()
        comb = torch.rand(M, 4, 4, device=dev)
        comb = (comb / comb.sum(-2, keepdim=True)).float().contiguous()
        pre = (torch.rand(M, 4, 1, device=dev) + 1e-3).float()
        ar_standin(x_src, x)
        with autotune(skip_dynamic_tuning_buckets=True):
            mhc.fused_hc_lagged(x, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)
        torch.cuda.synchronize()
        holder = {}

        def body_split():
            ar_standin(x_src, x)
            split = mhc.fused_hc_lagged_split(x, res, post, comb, pre, nw, 1e-6, aux, ev0, ev1)
            assert split is not None, "split path did not engage"
            outs, join = split
            li = outs[4] * 1  # main-stream consumer of K1's layer_input
            join.join()
            return li

        from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_fused_hc_tactic
        holder["P"] = _split.reducer_threads_for_tactic(mhc_fused_hc_tactic(
            x, res, post.reshape(M, 4), comb.reshape(M, 4, 4), mhc.fn.contiguous(),
            mhc.scale, mhc.base, 4, hidden, mhc.norm_eps, mhc.eps, mhc.sinkhorn_eps, mhc.post_mult_value,
            mhc.sinkhorn_iters, norm_weight=nw, norm_eps=1e-6, pre_mix_ext=pre.reshape(M, 4)))
        log(f"M={M} reducer threads P={holder['P']}")

        def body_main():
            ar_standin(x_src, x)
            outs = mhc.fused_hc_lagged(x, res, post, comb, pre, norm_weight=nw, norm_eps=1e-6)
            return outs[4] * 1

        def body_nofork():
            ar_standin(x_src, x)
            P = holder["P"]
            li = _split.mhc_lagged_layer_input(x.reshape(M, hidden), res.reshape(M, 4, hidden).contiguous(),
                                               post.reshape(M, 4), comb.reshape(M, 4, 4), pre.reshape(M, 4),
                                               nw, 1e-6, P)
            return li * 1

        arms = {}
        lis = {}
        for arm, body in (("SPLIT", body_split), ("MAINPATH", body_main), ("NOFORK", body_nofork)):
            prev = _split._SPLIT_ENV_ON
            _split._SPLIT_ENV_ON = arm != "MAINPATH"
            try:
                g, li = capture_keep(body)
                h = g.raw_cuda_graph()
                arms[arm] = walk(h, f"M={M} {arm}")
                if M == 1 and arm in ("SPLIT", "MAINPATH"):
                    p = os.path.join(art, f"graph_M1_{arm}.dot")
                    log(f"dot dump {p}: flags {dot_dump(h, p)}")
                g.replay()
                torch.cuda.synchronize()
                lis[arm] = li.clone()
            except Exception as e:  # noqa: BLE001
                ok = False
                arms[arm] = {"error": repr(e)}
                log(f"M={M} {arm} FAILED: {e!r}")
            finally:
                _split._SPLIT_ENV_ON = prev
        bitwise = {}
        if "MAINPATH" in lis:
            for arm in ("SPLIT", "NOFORK"):
                if arm in lis:
                    bitwise[arm] = int((lis[arm].view(torch.int16) != lis["MAINPATH"].view(torch.int16)).sum().item())
        arms["layer_input_bad_vs_mainpath"] = bitwise
        log(f"M={M} layer_input mismatches vs main path: {bitwise}")
        results[str(M)] = arms
        for arm in ("SPLIT", "NOFORK"):
            k1s = arms.get(arm, {}).get("k1", [])
            k1_types[f"M{M}/{arm}"] = [[(e["node_type"], e["kernel"][:24], e["type"], e["from_port"], e["to_port"])
                                        for e in ins] for ins in k1s]
            if not k1s:
                ok = False
        if any(bitwise.values()):
            ok = False

    # REAL_LAYER: 3 real decoder layers, stand-in sublayers ending with the stand-in AR.
    if _chain is not None:
        try:
            orig_run = _chain.StandIn.run

            def run_with_ar(self, xin):
                y = orig_run(self, xin)
                return ar_standin(y, torch.empty_like(y))

            _chain.StandIn.run = run_with_ar
            gen = torch.Generator(device=dev)
            gen.manual_seed(7)
            layers, model, hid = _chain.build_chain(cfg, 3, dev, gen)
            residual0 = torch.empty(1, 4, hid, device=dev, dtype=torch.bfloat16)
            pre0 = torch.empty(1, 4, 1, device=dev, dtype=torch.float32)
            _chain.fill_inputs(residual0, pre0, 1)
            _chain.tune(layers, model, residual0, pre0)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), with_multi_stream(True):
                for _ in range(2):
                    _chain.run_chain(layers, model, residual0, pre0)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with with_multi_stream(True):
                with torch.cuda.graph(g):
                    outp = _chain.run_chain(layers, model, residual0, pre0)
            torch.cuda.synchronize()
            h = g.raw_cuda_graph()
            rl = walk(h, "M=1 REAL_LAYER")
            p = os.path.join(art, "graph_M1_REAL_LAYER.dot")
            log(f"dot dump {p}: flags {dot_dump(h, p)}")
            g.replay()
            torch.cuda.synchronize()
            rl["finite"] = bool(torch.isfinite(outp.float()).all().item())
            results["REAL_LAYER_M1"] = rl
            k1_types["REAL_LAYER_M1"] = [[(e["node_type"], e["kernel"][:24], e["type"], e["from_port"],
                                           e["to_port"]) for e in ins] for ins in rl["k1"]]
            if not rl["k1"]:
                ok = False
        except Exception as e:  # noqa: BLE001
            ok = False
            results["REAL_LAYER_M1"] = {"error": repr(e)}
            log(f"REAL_LAYER FAILED: {e!r}")

    # Plain statement.
    split_ar_types = set()
    for key, lst in k1_types.items():
        if "NOFORK" in key:
            continue
        for ins in lst:
            for (nt, kn, et, fp, tp) in ins:
                if AR_NAME[:24] in kn or kn.startswith("_ar_standin"):
                    split_ar_types.add(et)
    if split_ar_types == {"Programmatic"}:
        statement = "the AR->K1 edge is Programmatic (event0.record() in the fork keeps it)"
    elif split_ar_types == {"Default"}:
        statement = "the fork's event0.record() serializes it to a Default edge"
    elif not split_ar_types:
        statement = "no direct AR->K1 edge found (see k1 incoming edges)"
    else:
        statement = f"mixed AR->K1 edge types {sorted(split_ar_types)}"
    log("STATEMENT:", statement)
    with open(os.path.join(art, "edge_dump.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({"passed": bool(ok), "statement": statement, "k1_incoming_edges": k1_types,
                      "compared": "graph edge data into K1 (cuGraphGetEdges_v2) in graphs captured through "
                                  "mHC.fused_hc_lagged_split / maybe_execute_in_parallel_deferred_join; "
                                  "layer_input bitwise vs main path per M"}))


if __name__ == "__main__":
    main()
