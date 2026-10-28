"""Branch 103: prove the served route of the NVFP4 tail pair and its calls per request.

python route_count.py <checkpoint>
VisualGen runs the model in worker processes, so the counters are installed by a sitecustomize.py
(on PYTHONPATH of every child process) that wraps, on import, flux3_sm120_lpt_pair.run (the one-launch
LPT pair) and flux3_primsts._sm120_tail_pair (main's two-stream pair), and appends one JSON line per
call (pid, down/attn M, whether the stream is capturing) to a log.
Arms (fresh child process each, benchmark settings + engine.json, ONE 1024^2 / 16-step request):
  B_eager  : default env, cuda_graph disabled -> every step's pairs run Python (expect 512 LPT calls)
  A_eager  : TRTLLM_FLUX3_SM120_PAIR_KERNEL=0, cuda_graph disabled (expect 512 two-stream, 0 LPT)
  B_graph  : default env, served engine.json (cuda graph on): calls made during capture/eager warmup
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_v2 as g  # noqa: E402

SITE = r'''
import importlib.abc, importlib.util, json, os, sys
_LOG = os.environ.get("B103_ROUTE_LOG")
_T = {"tensorrt_llm._torch.visual_gen.models.flux3.flux3_sm120_lpt_pair": ("run", "lpt"),
      "tensorrt_llm._torch.visual_gen.models.flux3.flux3_primsts": ("_sm120_tail_pair", "two_stream")}

def _write(rec):
    with open(_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

def _wrap(mod, attr, tag):
    orig = getattr(mod, attr)
    def w(*a, **k):
        try:
            import torch
            cap = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            cap = None
        shp = None
        if tag == "lpt":
            shp = [int(a[0].shape[0]), int(a[6].shape[0])]
        _write({"pid": os.getpid(), "tag": tag, "M": shp, "capturing": cap})
        return orig(*a, **k)
    setattr(mod, attr, w)
    _write({"pid": os.getpid(), "patched": mod.__name__ + "." + attr})

class _F(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _T:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module
        attr, tag = _T[name]
        def exec_module(module, orig_exec=orig_exec):
            orig_exec(module)
            _wrap(module, attr, tag)
        spec.loader.exec_module = exec_module
        return spec

if _LOG:
    sys.meta_path.insert(0, _F())
'''


def child(arm_dir, ckpt, graph):
    from tensorrt_llm import VisualGen
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(g.WORKTREE, "experiments", "ascent-config", "engine.json")))
    bare = {k: v for k, v in engine.items() if k not in ("visual_gen_args", "env")}
    vga = g.deep_merge(g.BASE_VGA, engine.get("visual_gen_args", {}))
    vga = g.deep_merge(vga, bare)
    if graph == "0":
        vga["cuda_graph_config"] = {"enable": False}
    print("VGA", json.dumps(vga), flush=True)
    model = VisualGen(model=ckpt, args=VisualGenArgs(**vga))
    log = os.environ["B103_ROUTE_LOG"]
    n0 = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    with open(os.path.join(arm_dir, "mark.json"), "w") as f:
        json.dump({"lines_before_request": n0}, f)
    key, seed, prompt = g.CHECK[0]
    params = model.default_params
    params.height, params.width, params.seed = 1024, 1024, seed
    params.num_inference_steps = 16
    out = model.generate(inputs=prompt, params=params)
    if getattr(out, "error", None):
        raise RuntimeError(out.error)
    print("generated", key, flush=True)
    try:
        model.shutdown()
    except Exception:  # noqa: BLE001
        pass


def tally(recs):
    t = {}
    for r in recs:
        if "tag" not in r:
            continue
        k = f"{r['tag']}" + (f"_M{r['M'][0]}_{r['M'][1]}" if r.get("M") else "") + \
            ("_capturing" if r.get("capturing") else "")
        t[k] = t.get(k, 0) + 1
    return t


def parent(ckpt):
    out = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp/b103"), "b103_route")
    os.makedirs(out, exist_ok=True)
    site = tempfile.mkdtemp(prefix="b103_site_")
    open(os.path.join(site, "sitecustomize.py"), "w").write(SITE)
    head = subprocess.run(["git", "-C", g.WORKTREE, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    anc = subprocess.run(["git", "-C", g.WORKTREE, "merge-base", "--is-ancestor", g.MAIN, "HEAD"]).returncode
    print(f"HEAD {head} ancestry rc={anc}", flush=True)
    res = {}
    for arm, val, graph in (("B_eager", None, "0"), ("A_eager", "0", "0"), ("B_graph", None, "1")):
        d = os.path.join(out, arm)
        os.makedirs(d, exist_ok=True)
        log = os.path.join(d, "route.jsonl")
        env = dict(os.environ)
        env["B103_ROUTE_LOG"] = log
        env["PYTHONPATH"] = site + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.pop("TRTLLM_FLUX3_SM120_PAIR_KERNEL", None)
        if val is not None:
            env["TRTLLM_FLUX3_SM120_PAIR_KERNEL"] = val
        print(f"=== {arm} PAIR_KERNEL={env.get('TRTLLM_FLUX3_SM120_PAIR_KERNEL')} cuda_graph={graph}",
              flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", d, ckpt, graph],
                           env=env, capture_output=True, text=True)
        open(os.path.join(d, "log.txt"), "w").write(p.stdout + p.stderr)
        recs = [json.loads(x) for x in open(log)] if os.path.exists(log) else []
        mark = json.load(open(os.path.join(d, "mark.json"))) if os.path.exists(
            os.path.join(d, "mark.json")) else {"lines_before_request": 0}
        n0 = mark["lines_before_request"]
        r = {"rc": p.returncode, "s": round(time.time() - t0),
             "patched": sorted({x["patched"] for x in recs if "patched" in x}),
             "pids": sorted({x["pid"] for x in recs if "tag" in x}),
             "load_warmup_capture": tally(recs[:n0]), "request": tally(recs[n0:])}
        if p.returncode != 0:
            r["tail"] = (p.stdout + p.stderr)[-2500:]
        res[arm] = r
        print(arm, json.dumps(r), flush=True)
    be, ae = res["B_eager"]["request"], res["A_eager"]["request"]
    lpt_b = sum(v for k, v in be.items() if k.startswith("lpt"))
    ok = (all(r["rc"] == 0 for r in res.values()) and lpt_b == 512 and be.get("two_stream", 0) == 0
          and ae.get("two_stream", 0) == 512 and not any(k.startswith("lpt") for k in ae))
    print(json.dumps({"passed": ok, "head": head, "ancestry_rc": anc,
                      "B_eager_lpt_calls_per_request": lpt_b,
                      "A_eager_two_stream_calls_per_request": ae.get("two_stream", 0),
                      "arms": res}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        parent(sys.argv[1])
