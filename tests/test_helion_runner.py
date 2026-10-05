import sys
from types import ModuleType, SimpleNamespace

import pytest

from gfaas import helion_runner, triton_quick_runner


def test_launch_matching_uses_types_and_options_when_names_and_constants_match(
    tmp_path, monkeypatch
):
    def device(x):
        pass

    jit = SimpleNamespace(
        arg_names=["x"], fn=device, params=[SimpleNamespace(name="x", is_constexpr=False)]
    )
    calls = []

    def host(x, _launcher):
        _launcher(jit, (1,), x, num_warps=4)
        return x

    module = SimpleNamespace(device=jit, host=host)
    triton = ModuleType("triton")
    triton.compile = lambda *a, **kw: SimpleNamespace(hash=kw["options"]["hash"])
    compiler = ModuleType("triton.compiler")

    class ASTSource:
        def __init__(self, jit, signature, constexprs):
            pass

    compiler.ASTSource = ASTSource
    backend = ModuleType("triton.backends.compiler")
    backend.GPUTarget = lambda **kw: kw
    for name, value in [
        ("triton", triton),
        ("triton.compiler", compiler),
        ("triton.backends.compiler", backend),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    units = []
    for index, kind in enumerate(("i32", "fp32")):
        cache = tmp_path / str(index)
        cache.mkdir()
        spec = {
            "kernel_name": "device",
            "signature": {"x": kind},
            "constants": {},
            "options": {"num_warps": 4, "hash": str(index)},
        }
        # Compiler-only hash is removed from actual options below.
        spec["options"].pop("hash")
        units.append({"cache": str(cache), "launch": spec, "compiled": {"cache_hash": str(index)}})
    sequence = iter(("0", "1"))
    triton.compile = lambda *a, **kw: SimpleNamespace(hash=next(sequence))
    monkeypatch.setattr(
        triton_quick_runner,
        "bound_candidate",
        lambda binary, *a: lambda *args, **kw: calls.append(binary.hash),
    )
    candidate = helion_runner.load_candidate(
        {"module": module, "record": {"kernel_name": "host"}, "units": units}, {"arch": 103}
    )
    assert candidate(1.5) == 1.5
    assert candidate(3) == 3
    assert calls == ["1", "0"]
    with pytest.raises(RuntimeError, match="unprepared"):
        candidate(True)


@pytest.mark.parametrize("name", ["helion_prepare", "helion_runner", "helion_compiler_runner"])
def test_handlers_import_when_vfunc_loads_the_source_as_a_standalone_module(name):
    import importlib.util
    from pathlib import Path

    path = Path(helion_runner.__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("standalone_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(
        getattr(
            module,
            "prepare"
            if name == "helion_prepare"
            else "compile_batch"
            if name == "helion_compiler_runner"
            else "benchmark_cycle",
        )
    )


def test_loaded_signature_restores_function_order_after_sorted_json(tmp_path, monkeypatch):
    def device(x, output, size):
        pass

    jit = SimpleNamespace(fn=device, arg_names=["x", "output", "size"])
    module = SimpleNamespace(device=jit, host=lambda **kw: None)
    observed = []
    compiler = ModuleType("triton.compiler")

    class ASTSource:
        def __init__(self, fn, signature, constexprs):
            observed.append(list(signature))

    compiler.ASTSource = ASTSource
    backend = ModuleType("triton.backends.compiler")
    backend.GPUTarget = lambda **kw: kw
    triton = ModuleType("triton")
    triton.compile = lambda *a, **kw: SimpleNamespace(hash="hash")
    for name, value in [
        ("triton", triton),
        ("triton.compiler", compiler),
        ("triton.backends.compiler", backend),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    spec = {
        "kernel_name": "device",
        "signature": {"output": "*fp32", "size": "constexpr", "x": "*fp32"},
        "constants": {"size": 1024},
        "options": {},
    }
    helion_runner.load_candidate(
        {
            "module": module,
            "record": {"kernel_name": "host"},
            "units": [{"cache": str(tmp_path), "launch": spec, "compiled": {"cache_hash": "hash"}}],
        },
        {},
    )
    assert observed == [["x", "output", "size"]]


@pytest.mark.parametrize(
    "dtype,kind", [("uint32", "u32"), ("int32", "i32"), ("int64", "i64"), ("float32", "fp32")]
)
def test_generated_integer_tensor_arguments_use_triton_compiler_types(monkeypatch, dtype, kind):
    from gfaas.triton_compat import argument_type

    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    setattr(language, dtype, {"float32": "fp32"}.get(dtype, dtype))
    monkeypatch.setitem(sys.modules, "triton.language", language)
    tensor = SimpleNamespace(dtype="torch." + dtype, data_ptr=lambda: None)
    assert argument_type(tensor, SimpleNamespace(is_constexpr=False)) == "*" + kind


def test_l2_flush_buffer_uses_the_current_replica_gpu(monkeypatch):
    allocations = []
    attributes = []

    def attribute(size, name, device):
        size._obj.value = 1024
        attributes.append((name, device))
        return 0

    monkeypatch.setattr(
        triton_quick_runner.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cuDeviceGetAttribute=attribute),
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: 2),
        uint8="uint8",
        empty=lambda *a, **kw: allocations.append((a, kw)),
    )
    triton_quick_runner.l2_flush_buffer(torch)
    assert attributes == [(38, 2)]
    assert allocations == [((2048,), {"dtype": "uint8", "device": "cuda:2"})]
