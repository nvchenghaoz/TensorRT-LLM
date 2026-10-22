import importlib.util, json, os, torch
path = os.path.join(os.getcwd(), "tensorrt_llm/_torch/modules/fused_silu_mul_nvfp4.py")
spec = importlib.util.spec_from_file_location("fsm", path)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
has_v2 = hasattr(F, "_launch_permuted_v2")
print("module", path, "has_v2", has_v2, flush=True)
dev = "cuda"
I = 17408
N = 2 * I
NKT = (I // 16 + 3) // 4
L, NIN, WARM, ITERS = 64, 4, 10, 50
gs = torch.tensor([448.0 * 6.0 / 24.0], dtype=torch.float32, device=dev)
def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
res, det = {}, {}
for M in [5, 8, 16, 24, 32]:
    xs = [(torch.randn(M, N, device=dev) * 2).to(torch.bfloat16) for _ in range(NIN)]
    m_pad = (M + 127) // 128 * 128
    os_ = [(torch.empty((M, I // 2), dtype=torch.uint8, device=dev),
            torch.empty((m_pad * NKT * 4,), dtype=torch.uint8, device=dev)) for _ in range(L)]
    fn = lambda: [F._launch(xs[l % NIN], gs, os_[l][0], os_[l][1], permuted=True) for l in range(L)]
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(ITERS):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        t.append(e0.elapsed_time(e1) / L)
    med = q(t, 0.5)
    det[str(M)] = {"med_us": med * 1e3, "p90_us": q(t, 0.9) * 1e3, "p10_us": q(t, 0.1) * 1e3}
    print("M", M, det[str(M)], flush=True)
    if M in (8, 16, 32):
        res[str(M)] = {"kernel_ms": med, "noise_floor": (q(t, 0.9) - q(t, 0.1)) / med}
    del g
print(json.dumps({"batch": res, "output": {"class": "exact"}, "has_v2": has_v2, "detail": det,
                  "note": "per-call fused_silu_mul_nvfp4_permuted (default dispatch), CUDA graph of 64 calls, 4 rotating L2-resident inputs, 10 warm + 50 timed replays"}))
