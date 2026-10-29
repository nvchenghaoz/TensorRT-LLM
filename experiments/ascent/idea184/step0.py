"""Idea 184 STEP 0 (+ idea 186 STEP 0(b)): in-model nsys of the attention -> attn_out quant ->
LPT pair fork, one request after warmup, served engine.json, CUDA graphs on,
--cuda-graph-trace=node, a fresh process per arm.

Arms (letters in <order>):
  A : main as served (LPT pair).
  B : timing only. flux3_primsts.nvfp4_quantize_prepared is monkeypatched so that, when called
      from nvfp4_out_pair, it returns a preallocated (data, scales) of the real shapes (filled
      once, eagerly, before model load, from a real cute-dsl quantize of randn). The LPT path
      still calls it; only the quantize launch disappears.

usage: python step0.py <ckpt> <outdir> [order, default AAB]
       python step0.py --child <arm> <ckpt>
"""

import hashlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
import types

PROMPT = "A red fox standing in fresh snow at sunrise, photorealistic"
SEED = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NSYS = "/usr/local/cuda/bin/nsys"


def log(msg):
    print(msg, flush=True)


def load_pipe(ckpt):
    import torch

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    engine = json.load(open(os.path.join(WT, "experiments", "ascent-config", "engine.json")))
    pipeline_config = dict(engine.pop("pipeline_config", {}))
    pipeline_config["variant"] = "nvfp4"
    args = VisualGenArgs(
        model=ckpt,
        pipeline_config=pipeline_config,
        compilation_config={"resolutions": [[1024, 1024]], "num_frames": [1], "modes": ["t2i"]},
        **engine,
    )
    t0 = time.time()
    pipe = PipelineLoader(args, device="cuda:0").load()
    torch.cuda.synchronize()
    log(f"[load] {time.time() - t0:.0f}s engine={engine}")
    return pipe


def install_dummy_quant():
    import torch
    from flashinfer import SfLayout
    from flashinfer import nvfp4_quantize as quantize

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as fp

    cache = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    for s in (4176, 4096):
        x = torch.randn((s, 3072), device="cuda", dtype=torch.bfloat16, generator=g)
        d, sc = quantize(x, 1.0, sfLayout=SfLayout.layout_128x4, do_shuffle=False,
                         backend="cute-dsl")
        cache[s] = (d.view(torch.uint8).contiguous(), sc.view(torch.uint8).reshape(-1).contiguous())
    torch.cuda.synchronize()
    orig = fp.nvfp4_quantize_prepared
    counters = {"dummy": 0, "real": 0}

    def patched(activation, weight):
        caller = sys._getframe(1).f_code.co_name
        if caller == "nvfp4_out_pair" and activation.shape[0] in cache \
                and activation.shape[1] == 3072:
            counters["dummy"] += 1
            return cache[activation.shape[0]]
        counters["real"] += 1
        return orig(activation, weight)

    fp.nvfp4_quantize_prepared = patched
    return counters


def child(arm, ckpt):
    import torch

    counters = install_dummy_quant() if arm == "B" else None
    pipe = load_pipe(ckpt)
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_sm120_lpt_pair as lpt

    from tensorrt_llm.visual_gen.params import VisualGenParams

    log(f"[arm {arm}] lpt.enabled={lpt.enabled()} sched={lpt.sched_mode()} "
        f"SWIGLU_SCHED={os.environ.get('TRTLLM_FLUX3_SM120_SWIGLU_SCHED', '(unset=dynamic)')} "
        f"SIDE_GATE_UP={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP', '(unset=3)')} "
        f"PAIR_KERNEL={os.environ.get('TRTLLM_FLUX3_SM120_PAIR_KERNEL', '(unset=1)')} "
        f"SIDE_GATE_UP_ORDER={os.environ.get('TRTLLM_FLUX3_SIDE_GATE_UP_ORDER', '(unset=fmha_first)')} "
        f"ATTN_NVFP4_OUT={os.environ.get('TRTLLM_FLUX3_ATTN_NVFP4_OUT')}")

    def gen():
        params = VisualGenParams(height=1024, width=1024, seed=SEED, num_inference_steps=16)
        out = pipe.infer(types.SimpleNamespace(prompt=PROMPT, params=params))
        torch.cuda.synchronize()
        img = out.image.detach().cpu().contiguous()
        return out.denoise * 1e3, hashlib.sha256(img.numpy().tobytes()).hexdigest()[:16]

    for i in range(2):
        d, h = gen()
        log(f"[arm {arm}] warm {i}: denoise {d:.2f} ms img {h}")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    d, h = gen()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log(f"[arm {arm}] profiled denoise {d:.2f} ms img {h} counters={counters}")
    print("CHILD_JSON " + json.dumps({"arm": arm, "denoise_ms": d, "img": h,
                                      "counters": counters}), flush=True)


def is_quant(n):
    return "Quantize" in n or "quantize" in n


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stat(v):
    return {"n": len(v), "median": round(statistics.median(v), 2) if v else None,
            "p10": round(pct(v, 0.1), 2) if v else None, "p90": round(pct(v, 0.9), 2) if v else None,
            "mean": round(statistics.mean(v), 2) if v else None}


def analyze(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    rows = []
    for start, end, name_id, short_id, gx, gy, stream in cur.execute(
            "SELECT start, end, demangledName, shortName, gridX, gridY, streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        rows.append((start, end, strings.get(name_id, "?"), strings.get(short_id, "?"), gx, gy,
                     stream))
    rows.sort()
    names = {}
    for r in rows:
        k = r[2][:110]
        names[k] = names.get(k, 0) + 1
    args_idx = [i for i, r in enumerate(rows) if "flux3_lpt_pair_args" in r[2]]
    calls = []
    gate_names = {}
    quant_names = {}
    for ai in args_idx:
        a = rows[ai]
        gemm = None
        for r in rows[ai + 1: ai + 12]:
            if "flux3_lpt_pair_args" in r[2]:
                break
            if "Gemm" in r[2] or "gemm" in r[2] or "cutlass" in r[2].lower():
                gemm = r
                break
        fmha = None
        for r in reversed(rows[max(0, ai - 40): ai]):
            if "fmha_v2_flux3" in r[2]:
                fmha = r
                break
        if fmha is None or gemm is None:
            continue
        window = rows[max(0, ai - 40): ai]
        between = [r for r in window if r[0] >= fmha[0] and r is not fmha]
        quants = [r for r in between if is_quant(r[2])]
        for q in quants:
            quant_names[q[2][:110]] = quant_names.get(q[2][:110], 0) + 1
        cand = [r for r in window
                if r is not fmha and not is_quant(r[2]) and r[0] < fmha[1] and r[1] > fmha[0]]
        gate = max(cand, key=lambda r: r[1] - r[0]) if cand else None
        if gate is not None:
            gate_names[gate[2][:110]] = gate_names.get(gate[2][:110], 0) + 1
        join = max(fmha[1], gate[1]) if gate else fmha[1]
        first_tail = quants[0][0] if quants else a[0]
        calls.append({
            "fork_span_us": (join - min(fmha[0], gate[0] if gate else fmha[0])) / 1e3,
            "gate_end_minus_fmha_end_us": (gate[1] - fmha[1]) / 1e3 if gate else None,
            "tail_span_with_quant_us": (gemm[1] - first_tail) / 1e3,
            "tail_span_args_to_lpt_end_us": (gemm[1] - a[0]) / 1e3,
            "S": "4176" if fmha[4] == 66 else ("4096" if fmha[4] == 64 else str(fmha[4])),
            "fmha_us": (fmha[1] - fmha[0]) / 1e3,
            "gate_us": (gate[1] - gate[0]) / 1e3 if gate else None,
            "last": "fmha" if (gate is None or fmha[1] >= gate[1]) else "gate_up",
            "fmha_end_minus_gate_end_us": (fmha[1] - gate[1]) / 1e3 if gate else None,
            "gate_start_minus_fmha_start_us": (gate[0] - fmha[0]) / 1e3 if gate else None,
            "n_quant": len(quants),
            "quant_us": sum((q[1] - q[0]) for q in quants) / 1e3,
            "join_to_args_us": (a[0] - join) / 1e3,
            "lpt_gemm_us": (gemm[1] - gemm[0]) / 1e3,
            "join_to_lpt_end_us": (gemm[1] - join) / 1e3,
            "between": [(r[3][:40], round((r[0] - join) / 1e3, 2), round((r[1] - r[0]) / 1e3, 2))
                        for r in between if r[1] > join or is_quant(r[2])][:6],
        })
    out = {"args_calls": len(args_idx), "calls_matched": len(calls), "gate_up_names": gate_names,
           "quant_names": quant_names,
           "fmha_total": sum(1 for r in rows if "fmha_v2_flux3" in r[2]),
           "fmha_names": sorted({r[2][:140] for r in rows if "fmha_v2_flux3" in r[2]}),
           "top_kernels": sorted(names.items(), key=lambda kv: -kv[1])[:40]}
    for s in ("4176", "4096"):
        cs = [c for c in calls if c["S"] == s]
        if not cs:
            continue
        o = {"calls": len(cs),
             "fmha_last_pct": round(100 * sum(c["last"] == "fmha" for c in cs) / len(cs), 1),
             "n_quant_total": sum(c["n_quant"] for c in cs)}
        for k in ("fmha_us", "gate_us", "quant_us", "join_to_args_us", "lpt_gemm_us",
                  "join_to_lpt_end_us", "fmha_end_minus_gate_end_us",
                  "gate_start_minus_fmha_start_us", "fork_span_us", "gate_end_minus_fmha_end_us",
                  "tail_span_with_quant_us", "tail_span_args_to_lpt_end_us"):
            v = [c[k] for c in cs if c[k] is not None]
            o[k] = stat(v)
        o["residual_tail_idle_at_join_us"] = stat(
            [abs(c["fmha_end_minus_gate_end_us"]) for c in cs
             if c["fmha_end_minus_gate_end_us"] is not None])
        o["residual_fmha_alone_us"] = stat(
            [max(0.0, c["fmha_end_minus_gate_end_us"]) for c in cs
             if c["fmha_end_minus_gate_end_us"] is not None])
        o["example"] = cs[len(cs) // 2]
        out["S" + s] = o
    return out


def parent(ckpt, outdir, order):
    if outdir == "@art":
        outdir = os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "idea184_step0")
    os.makedirs(outdir, exist_ok=True)
    try:
        for c in ("0b4693c4", "434118cf"):
            rc = subprocess.run(["git", "-C", WT, "merge-base", "--is-ancestor", c, "HEAD"]).returncode
            log(f"[ancestry] {c} is_ancestor rc={rc}")
        log("[head] " + subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip())
    except Exception as exc:  # noqa: BLE001
        log(f"[ancestry] failed: {exc}")
    res = {}
    for i, arm in enumerate(order):
        tag = f"{i}_{arm}"
        rep = os.path.join(outdir, f"step0_{tag}")
        cmd = [NSYS, "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true",
               "-o", rep, sys.executable, __file__, "--child", arm, ckpt]
        # A / B: main's path (env 0; B adds the dummy quant); N: this branch's nvfp4_out route.
        env = dict(os.environ, TRTLLM_FLUX3_ATTN_NVFP4_OUT="1" if arm == "N" else "0")
        log(f"=== {tag}: {' '.join(cmd)}")
        t0 = time.time()
        child_json = None
        with open(os.path.join(outdir, f"log_{tag}.txt"), "w") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=env)
            for line in p.stdout:
                f.write(line)
                if line.startswith("CHILD_JSON "):
                    child_json = json.loads(line[len("CHILD_JSON "):])
                if line.startswith(("[arm", "[load", "Traceback")) or "Error" in line:
                    log(f"  {tag}| {line.rstrip()[:300]}")
            p.wait()
        log(f"=== {tag}: rc={p.returncode} {time.time() - t0:.0f}s")
        s = subprocess.run([NSYS, "export", "--type", "sqlite", "-f", "true", "-o",
                            rep + ".sqlite", rep + ".nsys-rep"], capture_output=True, text=True)
        log(s.stdout[-400:] + s.stderr[-400:])
        r = analyze(rep + ".sqlite")
        r["child"] = child_json
        res[tag] = r
        with open(os.path.join(outdir, f"analysis_{tag}.json"), "w") as f:
            json.dump(r, f, indent=1)
        short = {k: v for k, v in r.items() if k != "top_kernels"}
        log(f"[analysis {tag}] " + json.dumps(short)[:6000])
    summ = {}
    for tag, r in res.items():
        summ[tag] = {"denoise_ms": (r.get("child") or {}).get("denoise_ms")}
        for s in ("S4176", "S4096"):
            if s in r:
                summ[tag][s] = {k: (r[s][k]["median"] if isinstance(r[s][k], dict)
                                    and "median" in r[s][k] else r[s][k])
                                for k in r[s] if k != "example"}
    log("[summary] " + json.dumps(summ))
    print(json.dumps({"passed": True, "summary": summ}))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "AAB")
