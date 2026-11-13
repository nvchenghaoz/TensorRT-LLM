"""Idea 40 stage 0b: NCCL env winner in branch 13's chunked TP-overlap regime (landing 8 on main).

One 2-GPU job, sequential subprocesses on the same pair:
  0. pair label (physical ids, topo relation, link gen/width read under a peer copy) and a quick AR
     re-check of default vs NCCL_P2P_USE_CUDA_MEMCPY=1 with a default A/A (ar_inplace_bench --quick),
  1. tp_overlap/block_pair.py --mode time (real MiniMaxH3TransformerBlock, torch.compile, chunked async
     AR, default chunks) under arms default, CE, default (A/A), CE (repeat),
  2. tp_overlap/block_pair.py --mode check --chunks 4 under CE: compiled serial and chunked block vs the
     eager serialized block, bitwise per iteration over 5 warmup + 20 iterations, both ranks.
Prints one JSON line.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_WIN = {"NCCL_P2P_USE_CUDA_MEMCPY": "1"}


def run(cmd, env_add, tag, timeout=1500):
    env = dict(os.environ)
    env.update(env_add)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === {tag}: env={env_add} cmd={' '.join(cmd[1:])}", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = None
    lines = []
    for ln in p.stdout:
        ln = ln.rstrip("\n")
        lines.append(ln)
        if ln.startswith("{"):
            last = ln
        if not ln.startswith("{") or len(ln) < 400:
            print(f"  [{tag}] {ln[-300:]}", flush=True)
    p.wait(timeout=timeout)
    try:
        res = json.loads(last) if last else {"error": "no json", "tail": lines[-20:]}
    except Exception as ex:
        res = {"error": repr(ex), "tail": lines[-20:]}
    res["_wall_s"] = round(time.time() - t0, 1)
    res["_rc"] = p.returncode
    print(f"[{time.strftime('%H:%M:%S')}] === {tag} done rc={p.returncode} in {res['_wall_s']} s", flush=True)
    return res


def main():
    ckpt = sys.argv[1]
    py = sys.executable
    out = {}
    ar = run([py, os.path.join(HERE, "ar_inplace_bench.py"), "--mode", "envsweep", "--checkpoint", ckpt,
              "--m", "37966", "--warmup", "5", "--iters", "30", "--quick", "--no-rsag",
              "--configs", "default,p2p_cuda_memcpy,default_rep"], {}, "ar_quick", timeout=900)
    if ar.get("wrong_pair"):
        print(json.dumps({"wrong_pair": True, "pair": ar.get("pair")}), flush=True)
        return
    out["ar_quick"] = ar
    bp = os.path.join(HERE, "tp_overlap", "block_pair.py")
    arms = [("default", {}), ("ce", ENV_WIN), ("default_aa", {}), ("ce_rep", ENV_WIN)]
    out["block"] = {}
    for tag, env in arms:
        r = run([py, bp, ckpt, "--mode", "time"], env, tag)
        out["block"][tag] = {"step_ms": r.get("batch", {}).get("1", {}).get("step_ms"),
                             "noise": r.get("batch", {}).get("1", {}).get("noise_floor"),
                             "detail": r.get("detail"), "rc": r["_rc"], "err": r.get("error")}
    chk = run([py, bp, ckpt, "--mode", "check", "--chunks", "4"], ENV_WIN, "check_ce")
    out["check_ce"] = chk
    b = out["block"]
    d = [b[k]["step_ms"] for k in ("default", "default_aa") if b[k]["step_ms"]]
    c = [b[k]["step_ms"] for k in ("ce", "ce_rep") if b[k]["step_ms"]]
    summ = {}
    if d and c:
        dm, cm = sum(d) / len(d), sum(c) / len(c)
        summ = {"default_ms": d, "ce_ms": c, "aa_delta": abs(d[0] - d[-1]) / dm if len(d) > 1 else None,
                "ce_rep_delta": abs(c[0] - c[-1]) / cm if len(c) > 1 else None,
                "gain": 1 - cm / dm, "per_iter_noise": [b[k]["noise"] for k in b]}
    out["summary"] = summ
    out["passed"] = bool(chk.get("passed")) and bool(ar.get("passed"))
    with open(os.path.join(os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp"), "nccl_env_block.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"passed": out["passed"], "summary": summ, "label": ar.get("label"),
                      "ar_table": ar.get("table"), "ar_aa": ar.get("aa"), "ce_ref_ms": ar.get("ce_ms"),
                      "check_ce": {k: chk.get(k) for k in ("passed", "rank0", "eager_serial_vs_compiled_serial")},
                      "block": {k: {"step_ms": v["step_ms"], "noise": v["noise"]} for k, v in b.items()}}),
          flush=True)


if __name__ == "__main__":
    main()
