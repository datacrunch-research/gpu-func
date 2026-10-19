from types import ModuleType

import pytest
from test_triton_kernel import JIT, Autotuner, Config, add
from test_triton_kernel import triton as _triton_fixture  # noqa: F401

import gfaas
from gfaas.triton_compat import source_bundle


def tlx_task():
    return None


def imported_tlx_kernel(X, N, BLOCK):
    with tlx_task():
        pass


@pytest.fixture
def tlx_runtime(monkeypatch):
    import sys

    module = ModuleType("triton.language.extra.tlx")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(tlx_task, "__module__", module.__name__)
    module.tlx_task = tlx_task
    return module


def test_tlx_kernel_uses_common_options_and_native_configs(tlx_runtime):
    native = Autotuner(
        JIT(add), ["X", "N", "BLOCK"], [Config({"BLOCK": 64}), Config({"BLOCK": 128})], [], [], []
    )
    kernel = gfaas.TLXKernel(native, variants_per_job=1, max_concurrent_jobs=2)
    assert isinstance(kernel, gfaas.TritonKernel)
    assert isinstance(kernel, gfaas.Kernel)
    assert kernel.tuning.replication_factor == 3
    assert kernel.variants_per_job == 1
    assert kernel.max_concurrent_jobs == 2
    assert kernel._source_bundle(native.fn).startswith("import triton.language.extra.tlx")


def test_tlx_missing_runtime_fails_before_work(monkeypatch):
    from gfaas import tlx_kernel

    def unavailable(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(tlx_kernel.importlib, "import_module", unavailable)
    with pytest.raises(gfaas.UnsupportedTritonKernelError, match="TLX-enabled"):
        gfaas.TLXKernel(JIT(add))


def test_tlx_rejects_custom_autotune_callbacks(tlx_runtime):
    native = Autotuner(
        JIT(add),
        ["X", "N", "BLOCK"],
        [Config({"BLOCK": 64}, pre_hook=lambda args: None)],
        [],
        [],
        [],
    )
    with pytest.raises(gfaas.UnsupportedTritonKernelError, match="pre_hook"):
        gfaas.TLXKernel(native)


def test_tlx_imported_primitives_are_reconstructed(tlx_runtime):
    source = source_bundle(JIT(imported_tlx_kernel))
    assert "from triton.language.extra.tlx import tlx_task as tlx_task" in source
    assert "with tlx_task():" in source
    assert "def tlx_task" not in source
