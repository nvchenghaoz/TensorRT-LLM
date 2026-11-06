"""Branch 185 pair arm (same script in BASE and CAND trees).

1. Dirty-slot guard (pitfall 3067/3072): torch.cuda.mem_get_info per visible GPU; if > 4 GiB is
   already used, print 'DIRTY SLOT <GiB>' and exit 3 before allocating anything.
2. cpu.max / cpu.stat (nr_throttled, throttled_usec) before/after each sub-run.
3. DEFAULT: bench.py pair (digest of >= 10 alternating c1/c4 steps, then timing).
4. DIAG: bench.py diag (copy_ / H2D enqueue / event.synchronize split per upload).
5. THREADS1: bench.py pair with OMP_NUM_THREADS=1 (torch intra-op pool of 1).
Last stdout line: DEFAULT's JSON plus the THREADS1 rows and the cgroup counters.
"""

import json
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ARM"


def log(*a):
    print(*a, flush=True)


def cgroup():
    out = {}
    for name in ("cpu.max", "cpu.stat"):
        try:
            with open(f"/sys/fs/cgroup/{name}") as f:
                text = f.read().strip()
        except OSError as e:
            text = f"<unavailable: {e}>"
        if name == "cpu.stat":
            kv = dict(line.split() for line in text.splitlines() if len(line.split()) == 2)
            out[name] = {k: kv.get(k) for k in ("usage_usec", "nr_periods", "nr_throttled",
                                                 "throttled_usec")}
        else:
            out[name] = text
    return out


def guard():
    import torch

    used_max = 0.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1 << 30)
        log(f"GUARD gpu {i} {torch.cuda.get_device_name(i)} used {used:.2f} GiB of "
            f"{total / (1 << 30):.2f}")
        used_max = max(used_max, used)
    if used_max > 4.0:
        log(f"DIRTY SLOT {used_max:.2f}")
        sys.exit(3)


def run(tag, mode, extra_env):
    env = dict(os.environ, **extra_env)
    before = cgroup()
    log(f"===== {LABEL} {tag} ({mode}) env {extra_env} cgroup before {json.dumps(before)}")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "bench.py"), CKPT, mode], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            last = line
        log(f"[{LABEL} {tag}] {line}")
    rc = proc.wait()
    after = cgroup()
    log(f"===== {LABEL} {tag} rc {rc} cgroup after {json.dumps(after)}")
    try:
        result = json.loads(last)
    except ValueError:
        result = None
    return rc, result, {"before": before, "after": after}


def main():
    os.environ.setdefault("B185_KV_GB", "12")
    log(f"{LABEL} host {socket.gethostname()} CVD {os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"nproc {os.cpu_count()} affinity {len(os.sched_getaffinity(0))}")
    guard()
    rc, default, cg_default = run("DEFAULT", "pair", {})
    if rc or default is None:
        log(f"DEFAULT failed rc {rc}")
        sys.exit(rc or 1)
    rc_d, _, cg_diag = run("DIAG", "diag", {})
    rc_t, threads1, cg_t1 = run("THREADS1", "pair", {"OMP_NUM_THREADS": "1"})
    for shape, row in default["detail"].items():
        t1 = (threads1 or {}).get("detail", {}).get(shape, {})
        log(f"ROW {LABEL} {shape} DEFAULT a {row['a_prepare']} | THREADS1 a {t1.get('a_prepare')}")
    default["label"] = LABEL
    default["threads1"] = {s: r["a_prepare"] for s, r in (threads1 or {}).get("detail", {}).items()}
    default["threads1_digest"] = (threads1 or {}).get("digest")
    default["diag_rc"] = rc_d
    default["cgroup"] = {"default": cg_default, "diag": cg_diag, "threads1": cg_t1}
    print(json.dumps(default))


if __name__ == "__main__":
    main()
