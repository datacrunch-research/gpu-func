import sys
from types import ModuleType, SimpleNamespace

import pytest

from gfaas import helion_runner, triton_quick_runner


def test_launch_matching_uses_types_and_options_when_names_and_constants_match(
    tmp_path, monkeypatch
):
    def device(x):
        pass

    jit = SimpleNamespace(fn=device, params=[SimpleNamespace(name="x", is_constexpr=False)])
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
