# Prints the public API of tests/microbenchmarks/attention_perf/attention_perf_harness.py (read-only use).
import dataclasses
import importlib.util
import inspect
import sys

path = "tests/microbenchmarks/attention_perf/attention_perf_harness.py"
spec = importlib.util.spec_from_file_location("attention_perf_harness", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["attention_perf_harness"] = mod
spec.loader.exec_module(mod)
print("DOC:", (mod.__doc__ or "")[:3000])
for name, obj in vars(mod).items():
    if getattr(obj, "__module__", None) != "attention_perf_harness":
        continue
    if inspect.isclass(obj):
        print(f"CLASS {name}")
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                print(f"   field {f.name}: {f.type} = {f.default!r}")
        for mn, mo in vars(obj).items():
            if callable(mo) and not mn.startswith("__"):
                try:
                    print(f"   method {mn}{inspect.signature(mo)}")
                except Exception:
                    print(f"   method {mn}")
    elif inspect.isfunction(obj):
        print(f"FUNC {name}{inspect.signature(obj)}  :: {(obj.__doc__ or '').strip()[:300]}")
src = inspect.getsource(mod)
lines = src.splitlines()
print("NLINES", len(lines))
for want in ("def main", "def run_case", "def bench", "def _run", "def run_attention", "def make", "def build"):
    for i, l in enumerate(lines):
        if l.lstrip().startswith(want):
            print(f"---- {want} at L{i + 1}")
            print("\n".join(lines[i:i + 60]))
