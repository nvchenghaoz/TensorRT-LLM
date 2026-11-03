"""pytest plugin: run the CSA2 fused store/gather/staging tests on SM120 too.

tests/ is protected, so instead of editing test_kernel.py's ``_sm100()`` helper
(which skips unless capability major == 10) this plugin swaps that helper, at
collection time, for one that admits major in (10, 12) -- exactly the widened
gates in csa2/quantization.py. The SM100-only packed-attention tests keep their
own inline ``pytest.skip("SM100 packed attention")`` and stay skipped.

Use: python -m pytest -p sm120_fused_tests_plugin ... (with experiments/ascent on
PYTHONPATH).
"""

import pytest
import torch


def _sm100_or_sm120():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import dsl_available

    if not torch.cuda.is_available() or not dsl_available():
        pytest.skip("CUDA and the CuTe DSL are required")
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Fused CSA2 store/gather is enabled on SM100/SM120 only")


def pytest_collection_modifyitems(session, config, items):
    patched = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and hasattr(module, "_sm100") and module not in patched:
            module._sm100 = _sm100_or_sm120
            patched.add(module)
    for module in patched:
        print(f"[sm120-plugin] patched _sm100 in {module.__name__}")
