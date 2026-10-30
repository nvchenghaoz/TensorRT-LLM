"""Idea 74 correctness (untimed): coordinator -> rank0 reference hop, two real processes.

Producer = this process (DiffusionRequest.refs_to_shm), consumer = a spawned child that
receives the pickled request over a Pipe (as over ZMQ) and runs refs_from_shm, then
(round section) decodes the restored PNG with pipeline_flux3._load_image_pixels on cuda.

Sections (every one also asserts no /dev/shm/trtllm_vgref_<our pid>_* file is left):
 sizes     : sizes 1..33 MB over image/video/audio slots and several indices: sha256 of
             every restored payload == input; handle kind posix; also with
             TRTLLM_VISUALGEN_REF_POSIX_SHM=0 (torch handles) and with a forced producer
             OSError (SHM_DIR -> missing dir; torch fallback).
 many      : 1000 i2i requests (refs A/B alternating), sha256 each.
 consumer_exc : index out of range and a size mismatch: RuntimeError with main's prefix,
             the other payloads restored, files unlinked.
 enqueue_fail : VisualGen.generate_async with an executor whose enqueue raises.
 discard   : DiffusionRemoteClient._discard_unsent_request.
 shutdown  : a producer child mints 3 requests' handles, never sends, exits: files gone.
 rounds    : >= 50 rounds of t2i -> i2i(A) -> t2i -> i2i(B); each i2i runs 3 arms (default,
             REF_POSIX_SHM=0, PNG_FAST=0 in the consumer). Bytes sha == input every round;
             pixels (sha of the cuda tensor's bytes) == round 0 and == every arm.
"""

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "idea63"))

N_ROUNDS = int(os.environ.get("IDEA74_ROUNDS", "50"))
SLOTS = ("image_reference", "video_reference", "audio_reference")


def log(*a):
    print(f"[idea74 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def leftovers(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        return sorted(f for f in os.listdir("/dev/shm") if f.startswith(f"trtllm_vgref_{pid}_"))
    except OSError:
        return ["<no /dev/shm>"]


def make_request(rid, refs_by_slot):
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest
    params = types.SimpleNamespace(seed=1)
    for slot in SLOTS:
        setattr(params, slot, [types.SimpleNamespace(content=c, format="bytes")
                               for c in refs_by_slot.get(slot, [])] or None)
    return DiffusionRequest(request_id=rid, prompt=["p"], params=params)


# ----------------------------------------------------------------------------- child
def child(conn):
    import torch
    from tensorrt_llm._torch.visual_gen.models.flux3 import pipeline_flux3 as flux3
    dev = torch.device("cuda")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        req, decode, png_fast_off = msg
        out = {"error": None, "shas": {}, "types": {}, "pix": None}
        try:
            req.refs_from_shm()
        except RuntimeError as exc:
            out["error"] = str(exc)
        out["handles_none"] = req.ref_handles is None
        for slot in SLOTS:
            for i, r in enumerate(getattr(req.params, slot, None) or []):
                out["shas"][f"{slot}[{i}]"] = sha(r.content)
                out["types"][f"{slot}[{i}]"] = type(r.content).__name__
        if decode and req.params.image_reference:
            if png_fast_off:
                os.environ["TRTLLM_FLUX3_PNG_FAST"] = "0"
            try:
                px = flux3._load_image_pixels(req.params.image_reference[0].content, device=dev)
                torch.cuda.synchronize()
                out["pix"] = sha(px.contiguous().cpu().numpy().tobytes()) + str(tuple(px.stride()))
            finally:
                os.environ.pop("TRTLLM_FLUX3_PNG_FAST", None)
        conn.send(out)


def producer_child(conn, payload):
    """Mint handles for 3 requests, report the files, never send, exit normally."""
    reqs = []
    for i in range(3):
        r = make_request(i, {"image_reference": [payload]})
        r.refs_to_shm()
        reqs.append(r)
    conn.send(leftovers())
    conn.close()


# ----------------------------------------------------------------------------- main
def main():
    import numpy as np
    from pngtest import find_refs
    from tensorrt_llm._torch.visual_gen import ref_shm
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient

    refs = find_refs()
    pristine = {k: sha(v) for k, v in refs.items()}
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=child, args=(b,), daemon=True)
    p.start()
    res = {}
    ok = True

    def hop(req, decode=False, png_fast_off=False):
        a.send((req, decode, png_fast_off))
        return a.recv()

    def check(name, cond, detail=None):
        nonlocal ok
        res[name] = {"ok": bool(cond)} if detail is None else {"ok": bool(cond), "detail": detail}
        ok &= bool(cond)
        if not cond:
            log("FAIL", name, detail)

    rng = np.random.default_rng(74)
    # ---- sizes / slots
    sizes = [1, 3, 4095, 4096, 4097, 65537, 1 << 20, len(refs["A"]), len(refs["B"]), 33554433]
    payloads = [rng.integers(0, 256, n, dtype=np.uint8).tobytes() for n in sizes]
    payloads[7], payloads[8] = refs["A"], refs["B"]
    by_slot = {"image_reference": payloads[:6], "video_reference": payloads[6:9],
               "audio_reference": payloads[9:]}
    want = {f"{s}[{i}]": sha(c) for s, cs in by_slot.items() for i, c in enumerate(cs)}
    for arm in ("posix", "killswitch", "producer_oserror"):
        saved_dir = ref_shm.SHM_DIR
        if arm == "killswitch":
            os.environ["TRTLLM_VISUALGEN_REF_POSIX_SHM"] = "0"
        if arm == "producer_oserror":
            ref_shm.SHM_DIR = "/nonexistent_idea74_dir"
        try:
            req = make_request(1, by_slot)
            req.refs_to_shm()
            kinds = sorted({"posix" if ref_shm.is_handle(h["handle"]) else "torch"
                            for h in req.ref_handles})
            emptied = all(r.content == b"" for s in SLOTS for r in getattr(req.params, s))
        finally:
            os.environ.pop("TRTLLM_VISUALGEN_REF_POSIX_SHM", None)
            ref_shm.SHM_DIR = saved_dir
        out = hop(req)
        expect_kind = ["posix"] if arm == "posix" else ["torch"]
        check(f"sizes_{arm}", out["error"] is None and out["shas"] == want and kinds == expect_kind
              and emptied and out["handles_none"] and set(out["types"].values()) == {"bytes"}
              and not leftovers(), {"kinds": kinds, "error": out["error"], "left": leftovers()})
    log("sizes done", ok)
    # ---- many
    bad = 0
    for i in range(1000):
        k = "AB"[i % 2]
        req = make_request(i, {"image_reference": [refs[k]]})
        req.refs_to_shm()
        out = hop(req)
        bad += out["shas"] != {"image_reference[0]": pristine[k]} or out["error"] is not None
    check("many_1000", bad == 0 and not leftovers(), {"bad": bad, "left": leftovers()})
    log("many done", ok)
    # ---- consumer exceptions
    req = make_request(2, {"image_reference": [refs["A"], refs["B"]]})
    req.refs_to_shm()
    req.ref_handles[0]["index"] = 99
    out = hop(req)
    check("consumer_exc_index", out["error"] is not None
          and out["error"].startswith("failed to restore reference payloads from shared memory: "
                                      "image_reference[99]:")
          and out["shas"].get("image_reference[1]") == pristine["B"] and out["handles_none"]
          and not leftovers(), {"error": out["error"], "left": leftovers()})
    req = make_request(3, {"image_reference": [refs["A"], refs["B"]]})
    req.refs_to_shm()
    req.ref_handles[1]["handle"]["size"] += 1
    out = hop(req)
    check("consumer_exc_size", out["error"] is not None and "image_reference[1]" in out["error"]
          and out["shas"].get("image_reference[0]") == pristine["A"] and not leftovers(),
          {"error": out["error"], "left": leftovers()})
    # ---- enqueue failure through VisualGen.generate_async
    from tensorrt_llm.visual_gen.visual_gen import VisualGen

    class _FailExec:
        def enqueue_requests(self, requests):
            raise RuntimeError("forced enqueue failure")

    class _VG(VisualGen):
        default_params = property(lambda self: self._dp)

    vg = _VG.__new__(_VG)
    vg.executor = _FailExec()
    vg._req_counter = itertools.count()
    vg._dp = types.SimpleNamespace(
        seed=5, image_reference=[types.SimpleNamespace(content=refs["A"], format="bytes")])
    raised = None
    try:
        vg.generate_async("p")
    except RuntimeError as exc:
        raised = str(exc)
    vg.executor = None
    check("enqueue_fail", raised == "forced enqueue failure" and not leftovers()
          and sha(vg._dp.image_reference[0].content) == pristine["A"],
          {"raised": raised, "left": leftovers()})
    # ---- discard unsent
    req = make_request(4, {"image_reference": [refs["B"]]})
    req.refs_to_shm()
    had = leftovers()
    DiffusionRemoteClient._discard_unsent_request(req)
    check("discard_unsent", len(had) == 1 and not leftovers(), {"had": had, "left": leftovers()})
    # ---- shutdown with unconsumed handles (producer exits; atexit sweep)
    pa, pb = ctx.Pipe()
    q = ctx.Process(target=producer_child, args=(pb, refs["A"]))
    q.start()
    minted = pa.recv()
    q.join()
    check("shutdown_unconsumed", len(minted) == 3 and q.exitcode == 0 and not leftovers(q.pid),
          {"minted": minted, "left": leftovers(q.pid)})
    log("lifecycle done", ok)
    # ---- rounds
    pix0 = {}
    mism = 0
    arms = (("default", False, False), ("ref_posix_shm_0", True, False), ("png_fast_0", False, True))
    for rnd in range(N_ROUNDS):
        for case in ("t2i", "A", "t2i", "B"):
            if case == "t2i":
                req = make_request(10, {})
                req.refs_to_shm()
                out = hop(req, decode=True)
                mism += req.ref_handles is not None or out["shas"] != {} or out["error"] is not None
                continue
            for arm, ks, pf in (arms if rnd % 2 == 0 else arms[::-1]):
                if ks:
                    os.environ["TRTLLM_VISUALGEN_REF_POSIX_SHM"] = "0"
                try:
                    req = make_request(11, {"image_reference": [refs[case]]})
                    req.refs_to_shm()
                    kind = "posix" if ref_shm.is_handle(req.ref_handles[0]["handle"]) else "torch"
                finally:
                    os.environ.pop("TRTLLM_VISUALGEN_REF_POSIX_SHM", None)
                out = hop(req, decode=True, png_fast_off=pf)
                bad_r = (out["error"] is not None
                         or out["shas"] != {"image_reference[0]": pristine[case]}
                         or kind != ("torch" if ks else "posix"))
                pix0.setdefault(case, out["pix"])
                bad_r |= out["pix"] != pix0[case] or out["pix"] is None
                if bad_r:
                    log("round mismatch", rnd, case, arm, out["error"])
                mism += bad_r
        if any(sha(v) != pristine[k] for k, v in refs.items()):
            mism += 1
        if rnd % 10 == 0:
            log("round", rnd, "mismatches", mism)
    check("rounds", mism == 0 and not leftovers(), {"rounds": N_ROUNDS, "mismatches": mism,
                                                    "left": leftovers()})
    a.send(None)
    p.join()
    print(json.dumps({"passed": bool(ok), "compared": "sha256 of every restored payload vs the "
                      "pristine input (frozen inputs.json refs A/B + random sizes), pixels via "
                      "_load_image_pixels vs round 0 and vs both kill switches, /dev/shm leftovers",
                      "sections": res}))


if __name__ == "__main__":
    main()
