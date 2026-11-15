"""Experiment-only hook (branch 32): time Gemma4InputProcessor.call_with_text_prompt in-server.

Active only when ASCENT_TOK_LOG is set. Writes one JSON line per call to
$ASCENT_TOK_LOG/tok_<pid>.jsonl: duration, id count, pid/ppid, live piece-pool threads, the
TOKENIZERS_PARALLELISM env. Chains to any other sitecustomize on sys.path.
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG = os.environ.get("ASCENT_TOK_LOG")
_TARGET = "tensorrt_llm._torch.models.modeling_gemma4mm"


def _patch(mod):
    import json
    import threading
    import time

    cls = getattr(mod, "Gemma4InputProcessor", None)
    if cls is None or getattr(cls, "_ascent_patched", False):
        return
    orig = cls.call_with_text_prompt
    lock = threading.Lock()

    def wrapped(self, inputs, sampling_params):
        before = sum(1 for t in threading.enumerate() if t.name.startswith("gemma4_tok"))
        a = time.perf_counter()
        out = orig(self, inputs, sampling_params)
        dt = time.perf_counter() - a
        after = sum(1 for t in threading.enumerate() if t.name.startswith("gemma4_tok"))
        rec = {"t_end": time.time(), "ms": 1e3 * dt, "n_ids": len(out[0]),
               "chars": len(inputs.get("prompt") or ""), "pid": os.getpid(), "ppid": os.getppid(),
               "pool_threads_before": before, "pool_threads_after": after,
               "thread": threading.current_thread().name,
               "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM")}
        with lock:
            with open(os.path.join(_LOG, f"tok_{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        return out

    cls.call_with_text_prompt = wrapped
    cls._ascent_patched = True


if _LOG:
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(name, path, target) if hasattr(f, "find_spec") else None
                if spec is not None and spec.loader is not None:
                    loader = spec.loader
                    orig_exec = loader.exec_module

                    def exec_module(module, _orig=orig_exec):
                        _orig(module)
                        try:
                            _patch(module)
                        except Exception as e:  # noqa: BLE001
                            print(f"[ascent tok hook] patch failed: {e!r}", file=sys.stderr)

                    loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())

# Chain to the environment's own sitecustomize, if any.
_saved = list(sys.path)
try:
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _DIR]
    _me = sys.modules.pop("sitecustomize", None)
    try:
        import sitecustomize  # noqa: F401
    except ImportError:
        pass
    finally:
        if _me is not None:
            sys.modules["sitecustomize"] = _me
finally:
    sys.path[:] = _saved
