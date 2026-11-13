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
NCH8_ONLY = False


ART = os.environ.get("AUTO_DEV_ARTIFACTS", "/tmp")


def clean(s):
    return "".join(c if (32 <= ord(c) < 127 or c == "\t") else "?" for c in s)


def run(cmd, env_add, tag, timeout=900):
    import signal
    import threading

    env = dict(os.environ)
    env.update(env_add)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === {tag}: env={env_add} cmd={' '.join(cmd[1:])}", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace", start_new_session=True)

    def killer():
        while True:
            try:
                p.wait(timeout=60)
                return
            except subprocess.TimeoutExpired:
                el = time.time() - t0
                print(f"[{tag}] heartbeat {el:.0f} s", flush=True)
                if el > timeout:
                    break
        print(f"[{tag}] TIMEOUT after {timeout} s: killing process group", flush=True)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass

    threading.Thread(target=killer, daemon=True).start()
    last = None
    lines = []
    logf = open(os.path.join(ART, f"log_{tag}.txt"), "w")
    for ln in p.stdout:
        ln = clean(ln.rstrip("\n"))
        lines.append(ln)
        logf.write(ln + "\n")
        logf.flush()
        if ln.startswith("{"):
            last = ln
        if (not ln.startswith("{") or len(ln) < 400) and "frame #" not in ln:
            print(f"  [{tag}] {ln[-300:]}", flush=True)
    logf.close()
    p.wait()
    try:
        res = json.loads(last) if last else {"error": "no json", "tail": lines[-20:]}
    except Exception as ex:
        res = {"error": repr(ex), "tail": lines[-20:]}
    res["_wall_s"] = round(time.time() - t0, 1)
    res["_rc"] = p.returncode
    keys = ("NCCL WARN", "rror", "Watchdog", "timeout", "Timeout", "block forward", "check chunks", "Traceback")
    res["_key_lines"] = [ln[-400:] for ln in lines if any(k in ln for k in keys) and "frame #" not in ln][:25]
    print(f"[{time.strftime('%H:%M:%S')}] === {tag} done rc={p.returncode} in {res['_wall_s']} s", flush=True)
    return res


def chan_phase(ckpt, py, bp, dbg, out):
    """CE hang probe, then channel-count winners in the chunked block regime."""
    n16 = {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}
    n8 = {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"}
    out["probe"] = {}
    if NCH8_ONLY:
        # nch8 confirmation: alternating replicates in the chunked block regime, check under nch8
        arms = [("default", dbg), ("nch8", {**n8, **dbg}), ("default_aa", dbg), ("nch8_rep", {**n8, **dbg}),
                ("default_aa2", dbg), ("nch8_rep2", {**n8, **dbg})]
        n16 = n8
    else:
        probe = os.path.join(HERE, "nccl_ce_hang_probe.py")
        for v, env in (("plain", ENV_WIN), ("async4", ENV_WIN), ("sync", ENV_WIN), ("malloc", ENV_WIN),
                       ("free", ENV_WIN), ("free_noCE", {})):
            r = run([py, probe, v.replace("_noCE", "")], {**env, **dbg}, f"probe_{v}", timeout=100)
            out["probe"][v] = {"ok": bool(r.get("ok")), "rc": r["_rc"], "wall_s": r["_wall_s"]}
        arms = [("default", dbg), ("nch16", {**n16, **dbg}), ("nch8", {**n8, **dbg}), ("default_aa", dbg),
                ("nch16_rep", {**n16, **dbg})]
    out["block"] = {}
    for tag, env in arms:
        r = run([py, bp, ckpt, "--mode", "time"], env, tag)
        out["block"][tag] = {"step_ms": r.get("batch", {}).get("1", {}).get("step_ms"),
                             "noise": r.get("batch", {}).get("1", {}).get("noise_floor"),
                             "detail": r.get("detail"), "rc": r["_rc"], "key_lines": r.get("_key_lines")}
    chk = run([py, bp, ckpt, "--mode", "check", "--chunks", "4"], {**n16, **dbg},
              "check_nch8" if NCH8_ONLY else "check_nch16")
    out["check_nch16"] = chk
    b = out["block"]
    d = [b[k]["step_ms"] for k in b if k.startswith("default") and b[k]["step_ms"]]
    s = {}
    if d:
        dm = sum(d) / len(d)
        c8 = [b[k]["step_ms"] for k in b if k.startswith("nch8") and b[k]["step_ms"]]
        s = {"default_ms": d, "aa_delta": (max(d) - min(d)) / dm, "nch8_ms": c8,
             "nch8_mean_rel": round(sum(c8) / len(c8) / dm - 1, 4) if c8 else None,
             "rel_vs_default": {k: (round(v["step_ms"] / dm - 1, 4) if v["step_ms"] else None) for k, v in b.items()},
             "noise": {k: v["noise"] for k, v in b.items()}}
    out["summary"] = s
    with open(os.path.join(ART, "nccl_env_block_chan.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"passed": bool(chk.get("passed")), "label": out["ar_quick"].get("label"),
                      "ar_table": out["ar_quick"].get("table"), "probe": out["probe"], "summary": s,
                      "block": {k: {"step_ms": v["step_ms"], "noise": v["noise"]} for k, v in b.items()},
                      "check_nch16": {k: chk.get(k) for k in ("passed", "rank0")}}), flush=True)


def main():
    ckpt = sys.argv[1]
    py = sys.executable
    out = {}
    global NCH8_ONLY
    NCH8_ONLY = len(sys.argv) > 3 and sys.argv[3] == "nch8"
    cfgs = "default,nch16,nch8,default_rep" if (len(sys.argv) > 2 and sys.argv[2] == "chan") \
        else "default,p2p_cuda_memcpy,default_rep"
    if NCH8_ONLY:
        cfgs = "default,nch8,default_rep"
    ar = run([py, os.path.join(HERE, "ar_inplace_bench.py"), "--mode", "envsweep", "--checkpoint", ckpt,
              "--m", "37966", "--warmup", "5", "--iters", "30", "--quick", "--no-rsag",
              "--configs", cfgs], {}, "ar_quick", timeout=900)
    if ar.get("wrong_pair") or (NCH8_ONLY and "PIX" not in str(ar.get("label"))):
        print(json.dumps({"wrong_pair": True, "pair": ar.get("pair")}), flush=True)
        return
    out["ar_quick"] = ar
    bp = os.path.join(HERE, "tp_overlap", "block_pair.py")
    dbg = {"NCCL_DEBUG": "WARN"}
    if len(sys.argv) > 2 and sys.argv[2] == "chan":
        return chan_phase(ckpt, py, bp, dbg, out)
    arms = [("ce", {**ENV_WIN, **dbg}), ("default", dbg), ("ce_rep", {**ENV_WIN, **dbg}), ("default_aa", dbg)]
    out["block"] = {}
    for tag, env in arms:
        r = run([py, bp, ckpt, "--mode", "time"], env, tag)
        out["block"][tag] = {"step_ms": r.get("batch", {}).get("1", {}).get("step_ms"),
                             "noise": r.get("batch", {}).get("1", {}).get("noise_floor"),
                             "detail": r.get("detail"), "rc": r["_rc"], "err": r.get("error"),
                             "key_lines": r.get("_key_lines")}
        if tag == "ce" and out["block"][tag]["step_ms"] is None:
            # diagnose: the serialized path (no chunked overlap) under CE
            r2 = run([py, bp, ckpt, "--mode", "time"], {**ENV_WIN, **dbg, "TLLM_H3_TP_OVERLAP_CHUNKS": "0"},
                     "ce_serial")
            out["block"]["ce_serial"] = {"step_ms": r2.get("batch", {}).get("1", {}).get("step_ms"),
                                         "rc": r2["_rc"], "key_lines": r2.get("_key_lines")}
            r3 = run([py, bp, ckpt, "--mode", "time"], {**dbg, "TLLM_H3_TP_OVERLAP_CHUNKS": "0"}, "default_serial")
            out["block"]["default_serial"] = {"step_ms": r3.get("batch", {}).get("1", {}).get("step_ms"),
                                              "rc": r3["_rc"]}
            with open(os.path.join(ART, "nccl_env_block.json"), "w") as f:
                json.dump(out, f, indent=1)
            print(json.dumps({"passed": False, "ce_chunked_failed": True,
                              "label": ar.get("label"), "ar_table": ar.get("table"),
                              "block": {k: {kk: v.get(kk) for kk in ("step_ms", "rc", "key_lines")}
                                        for k, v in out["block"].items()}}), flush=True)
            return
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
