import asyncio, base64, json, os, queue, random, statistics, subprocess, sys, threading, time

# VisualGen post-generation host path microbenchmark (idea 16). Self-contained
# (run inline with python -c in both pair arms).
# Part A: real DiffusionRemoteClient coordinator loop + real ZeroMqQueue/HMAC
#   sockets, fake worker subprocess that blocks on its request socket.
#   dispatch = worker get - enqueue_requests; delivery = server-loop return - worker put.
# Part B: b64_json response render (ImageObject -> HTTP body bytes), 1.9 MB PNG.
N_BATCHES, PER_BATCH = 5, 20

WORKER_SRC = r'''
import json, sys, time, zmq
from tensorrt_llm._torch.visual_gen import executor as ex
from tensorrt_llm.executor.ipc import ZeroMqQueue
req_addr, req_key, resp_addr, resp_key, gen_path = sys.argv[1:6]
gen_ms = {int(k): v for k, v in json.load(open(gen_path)).items()}
req_q = ZeroMqQueue((req_addr, bytes.fromhex(req_key)), is_server=False, socket_type=zmq.PULL)
resp_q = ZeroMqQueue((resp_addr, bytes.fromhex(resp_key)), is_server=False, socket_type=zmq.PUSH)
resp_q.put(ex.DiffusionResponse(request_id=-1, output={"default_generation_params": {}}))
while True:
    req = req_q.get()
    t_get = time.perf_counter()
    if req is None:
        break
    d = gen_ms.get(req.request_id, 0.0)
    if d > 0:
        time.sleep(d / 1000.0)
    t_put = time.perf_counter()
    resp_q.put(ex.DiffusionResponse(request_id=req.request_id, output={"t_get": t_get, "t_put": t_put}))
req_q.close(); resp_q.close()
'''


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_client(ex):
    c = ex.DiffusionRemoteClient.__new__(ex.DiffusionRemoteClient)
    req_port, resp_port = ex.find_free_port(), ex.find_free_port()
    c.n_workers = 1
    c.request_queue_addr = f"tcp://127.0.0.1:{req_port}"
    c.response_queue_addr = f"tcp://127.0.0.1:{resp_port}"
    c.req_addr_connect, c.resp_addr_connect = c.request_queue_addr, c.response_queue_addr
    c.req_hmac_key, c.resp_hmac_key = os.urandom(32), os.urandom(32)
    c.requests_ipc = c.responses_ipc = None
    c.pending_requests = queue.Queue()
    c.completed_responses = {}
    c._abandoned_request_ids = set()
    c._iter_stats = ex._IterationStatsTracker()
    c._event_loop = c.response_event = c.lock = None
    c.shutdown_event = threading.Event()
    c.event_loop_ready = threading.Event()
    c._shutdown_lock = threading.Lock()
    c._shutdown_started = False
    c._shutdown_complete = threading.Event()
    c._shutdown_error = None
    c._shutdown_thread = None
    c.worker_processes = []
    c._worker_spawner = None
    c._ext_worker_thread = None
    c._monitor_worker_liveness = False
    c._worker_failure = None
    c._request_to_send = None
    c.default_generation_params, c.extra_param_specs = {}, {}
    c.supports_image_edit, c.ref_slot_specs = False, {}
    c.background_thread = threading.Thread(target=c._serve_forever_thread, daemon=True)
    c.background_thread.start()
    c.event_loop_ready.wait()
    return c


def bench_render(n_iter=40):
    from starlette.responses import JSONResponse
    from tensorrt_llm.serve import openai_server as osrv
    from tensorrt_llm.serve.openai_protocol import ImageGenerationResponse, ImageObject
    png = random.Random(1234).randbytes(1_904_700)
    prompt = 'A "quoted" prompt with \\ backslash,\nnewline and café 漢字'
    headers = {"Server-Timing": "total;dur=1234.5"}
    splicer_cls = getattr(osrv, "_B64JsonSplicer", None)

    def build(b64_value):
        return ImageGenerationResponse(created=1700000000,
                                       data=[ImageObject(b64_json=b64_value, revised_prompt=prompt)],
                                       output_format="png", size="1024x1024")

    def render_base(b64):
        return JSONResponse(content=build(b64.decode("utf-8")).model_dump(), headers=headers)

    def render_cand(b64):
        sp = splicer_cls()
        return sp.response(build(sp.placeholder(b64)).model_dump(), headers)

    fn = render_cand if splicer_cls is not None else render_base
    b64 = base64.b64encode(png)
    ref, out = render_base(b64), fn(b64)
    identical = out.body == ref.body and out.raw_headers == ref.raw_headers
    times = []
    for _ in range(n_iter):
        b64 = base64.b64encode(png)
        t0 = time.perf_counter()
        r = fn(b64)
        times.append((time.perf_counter() - t0) * 1000)
        del r
    times.sort()
    return {"path": "splice" if splicer_cls is not None else "JSONResponse",
            "median_ms": statistics.median(times), "p10_ms": times[len(times) // 10],
            "p90_ms": times[(9 * len(times)) // 10],
            "body_identical_to_JSONResponse": identical, "body_bytes": len(out.body)}


def stats(xs):
    s = sorted(xs)
    return {"mean": statistics.mean(s), "p50": statistics.median(s),
            "p90": s[(9 * len(s)) // 10], "max": s[-1]}


def main():
    log("importing executor")
    from tensorrt_llm._torch.visual_gen import executor as ex
    log(f"SERVE_IDLE_TIMEOUT={getattr(ex, 'SERVE_IDLE_TIMEOUT', None)}")
    rng = random.Random(20261027)
    n_total = N_BATCHES * PER_BATCH
    gen_ms = {i: rng.uniform(20.0, 60.0) for i in range(n_total + 4)}
    burst_ids = list(range(n_total + 4, n_total + 20))
    for i in burst_ids:
        gen_ms[i] = 0.0
    gap_ms = [rng.uniform(5.0, 40.0) for _ in range(n_total)]
    gen_path = f"/tmp/coord_gen_{os.getpid()}.json"
    with open(gen_path, "w") as f:
        json.dump(gen_ms, f)

    c = make_client(ex)
    w = subprocess.Popen([sys.executable, "-c", WORKER_SRC, c.req_addr_connect,
                          c.req_hmac_key.hex(), c.resp_addr_connect, c.resp_hmac_key.hex(),
                          gen_path])
    c._wait_ready()
    log("coordinator ready")

    async def run_one(rid):
        t_enq = time.perf_counter()
        c.enqueue_requests([ex.DiffusionRequest(request_id=rid, prompt=["x"])])
        fut = asyncio.run_coroutine_threadsafe(c.await_responses(rid), c._event_loop)
        resp = await asyncio.wrap_future(fut)
        t_ret = time.perf_counter()
        assert resp is not None and resp.request_id == rid, resp
        return (resp.output["t_get"] - t_enq) * 1000, (t_ret - resp.output["t_put"]) * 1000

    async def run_all():
        for rid in range(n_total, n_total + 4):
            await run_one(rid)
        res = []
        for i in range(n_total):
            res.append(await run_one(i))
            await asyncio.sleep(gap_ms[i] / 1000)
            if (i + 1) % PER_BATCH == 0:
                log(f"  {i + 1}/{n_total} requests")
        return res

    res = asyncio.run(run_all())

    async def burst():
        t0 = time.perf_counter()
        c.enqueue_requests([ex.DiffusionRequest(request_id=r, prompt=["x"]) for r in burst_ids])
        fut = asyncio.run_coroutine_threadsafe(c.await_responses(burst_ids, timeout=10.0),
                                               c._event_loop)
        resps = await asyncio.wrap_future(fut)
        ok = all(r is not None and r.request_id == i for r, i in zip(resps, burst_ids))
        return ok, (time.perf_counter() - t0) * 1000

    burst_ok, burst_ms = asyncio.run(burst())
    log(f"burst of 16: ok={burst_ok} {burst_ms:.1f} ms")

    time.sleep(0.5)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    time.sleep(3.0)
    idle_cpu_pct = (time.process_time() - cpu0) / (time.perf_counter() - wall0) * 100
    log(f"idle coordinator-process CPU: {idle_cpu_pct:.3f}% of one core")

    t_sd = time.perf_counter()
    c.shutdown()
    shutdown_ms = (time.perf_counter() - t_sd) * 1000
    w.wait(timeout=10)
    log(f"shutdown {shutdown_ms:.1f} ms, worker exit {w.returncode}")

    log("render benchmark")
    render = bench_render()
    log(f"render: {render}")

    dispatch = [r[0] for r in res]
    delivery = [r[1] for r in res]
    total = [a + b for a, b in res]
    bm = [statistics.mean(total[b * PER_BATCH:(b + 1) * PER_BATCH]) for b in range(N_BATCHES)]
    step_ms = statistics.mean(total) + render["median_ms"]
    noise = (max(bm) - min(bm) + render["p90_ms"] - render["p10_ms"]) / step_ms
    print(json.dumps({
        "batch": {"1": {"step_ms": step_ms, "noise_floor": noise}},
        "output": {"class": "exact"},
        "dispatch_ms": stats(dispatch), "delivery_ms": stats(delivery),
        "dispatch_plus_delivery_ms": stats(total), "coord_batch_means_ms": bm,
        "render": render, "burst16_ok": burst_ok, "burst16_ms": burst_ms,
        "idle_cpu_pct_one_core": idle_cpu_pct, "shutdown_ms": shutdown_ms,
        "worker_exit": w.returncode,
    }), flush=True)


main()
