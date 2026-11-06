"""JIT watch for idea 237 item 4 (imported at interpreter start via PYTHONPATH; active only when JITWATCH_DIR is set).

Every Python process of the server (main + MPI ranks) writes <JITWATCH_DIR>/<pid>.log lines
  <unix time> <event> <detail>
events: 'hook' (installed, argv), 'patched' (module wrapped), 'triton.compile' (one call of
triton.compiler.compiler.compile = an in-memory Triton cache miss, disk hit or not), 'cute.compile' (one call of
cutlass.cute.compile = a CuTe-DSL JIT), 'driver.cuModuleLoad*' / 'driver.cuLibraryLoad*' (Python callers of
cuda.bindings.driver only: CuTe-DSL module loads; C/C++ callers such as Triton's launcher or CUDA lazy loading of
TRT-LLM's own kernels are not visible here), 'triton.load_binary' (Triton's launcher cuModuleLoadData of a
compiled kernel, triton.backends.nvidia.driver.CudaUtils.load_binary). The wrappers only log and forward; they change no arguments or results.
Any other sitecustomize on sys.path is executed afterwards, so the environment is otherwise unchanged.
"""
import os
import sys
import time

_DIR = os.environ.get("JITWATCH_DIR")


def _emit(kind, detail=""):
    try:
        with open(os.path.join(_DIR, f"{os.getpid()}.log"), "a") as f:
            f.write(f"{time.time():.3f} {kind} {str(detail)[:300].replace(chr(10), ' ')}\n")
    except Exception:  # noqa: BLE001
        pass


def _desc(args, kwargs):
    out = []
    for a in list(args)[:2]:
        fn = getattr(a, "fn", None)
        if fn is not None:
            out.append(getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__)
        elif callable(a) and hasattr(a, "__name__"):
            out.append(a.__name__)
        else:
            out.append(type(a).__name__)
    if "options" in kwargs:
        out.append("opts")
    return "|".join(out)


class _CompileProxy:
    def __init__(self, inner, tag="cute.compile"):
        self._inner = inner
        self._tag = tag

    def __call__(self, *args, **kwargs):
        _emit(self._tag, _desc(args, kwargs))
        return self._inner(*args, **kwargs)

    def __getitem__(self, item):
        return _CompileProxy(self._inner[item], self._tag)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch(name, module):
    try:
        if name == "triton.compiler.compiler" and hasattr(module, "compile"):
            inner = module.compile

            def compile(*args, **kwargs):  # noqa: A001
                _emit("triton.compile", _desc(args, kwargs))
                return inner(*args, **kwargs)

            compile.__wrapped__ = inner
            module.compile = compile
            _emit("patched", name)
        elif name == "cutlass.cute" and hasattr(module, "compile"):
            module.compile = _CompileProxy(module.compile)
            _emit("patched", name)
        elif name == "triton.backends.nvidia.driver" and hasattr(module, "CudaUtils"):
            cls = module.CudaUtils
            orig_init = cls.__init__

            def __init__(self, *a, _orig=orig_init, **k):
                _orig(self, *a, **k)
                lb = getattr(self, "load_binary", None)
                if lb is not None and not getattr(lb, "_jitwatch", False):
                    def load_binary(*args, _lb=lb):
                        _emit("triton.load_binary", args[0] if args and isinstance(args[0], str) else "")
                        return _lb(*args)

                    load_binary._jitwatch = True
                    self.load_binary = load_binary

            cls.__init__ = __init__
            _emit("patched", name)
        elif name == "cuda.bindings.driver":
            done = []
            for attr in dir(module):
                if attr.startswith(("cuModuleLoad", "cuLibraryLoad")):
                    inner = getattr(module, attr)

                    def wrap(*args, _inner=inner, _attr=attr, **kwargs):
                        _emit("driver." + _attr, "")
                        return _inner(*args, **kwargs)

                    setattr(module, attr, wrap)
                    done.append(attr)
            _emit("patched", f"{name} {','.join(done)}")
    except Exception as exc:  # noqa: BLE001
        _emit("patch_error", f"{name} {exc!r}")


def _install():
    import importlib.abc
    import importlib.machinery

    targets = {"triton.compiler.compiler", "cutlass.cute", "cuda.bindings.driver", "triton.backends.nvidia.driver"}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in targets:
                return None
            targets.discard(fullname)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig, _name=fullname):
                _orig(module)
                _patch(_name, module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _emit("hook", " ".join(sys.argv)[:200])


def _chain():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sys.path:
        try:
            if not p or os.path.abspath(p) == here:
                continue
            cand = os.path.join(p, "sitecustomize.py")
            if os.path.isfile(cand):
                with open(cand) as f:
                    code = f.read()
                exec(compile(code, cand, "exec"), {"__name__": "sitecustomize", "__file__": cand})
                return
        except Exception:  # noqa: BLE001
            return


if _DIR:
    try:
        os.makedirs(_DIR, exist_ok=True)
        _install()
    except Exception as _exc:  # noqa: BLE001
        _emit("install_error", repr(_exc))
_chain()
