import inspect
import sys
from types import ModuleType, SimpleNamespace

import cloudpickle

from gfaas import helion_prepare, triton_inputs, triton_quick_runner


def test_preparation_captures_multiple_launches_without_executing_device_code(monkeypatch):
    executed = []

    def first(x, block):
        executed.append("first")

    def second(x, block):
        executed.append("second")

    def jit(fn):
        return SimpleNamespace(
            fn=fn,
            params=[
                SimpleNamespace(name=p, is_constexpr=p == "block")
                for p in inspect.signature(fn).parameters
            ],
        )

    first_jit, second_jit = jit(first), jit(second)

    class Bound:
        def to_triton_code(self, config):
            if config["block"] == 0:
                raise ValueError("invalid config")
            return "generated source"

        def compile_config(self, config):
            def run(x, _launcher):
                _launcher(first_jit, (1,), x, config["block"], num_warps=4)
                _launcher(second_jit, (1,), x, config["block"], num_warps=4)
                _launcher(first_jit, (1,), x, config["block"], num_warps=4)

            return run

    helion = ModuleType("helion")
    helion.__version__ = "1.4.0"
    helion.Settings = lambda **kw: kw
    helion.Config = lambda **kw: kw
    helion.kernel = lambda *a, **kw: SimpleNamespace(bind=lambda args: Bound())
    triton = ModuleType("triton")
    triton.__version__ = "3.8.0"
    codecache = ModuleType("torch._inductor.codecache")
    codecache.PyCodeCache = SimpleNamespace(
        load=lambda source: SimpleNamespace(example=lambda x: x)
    )
    for name, module in [
        ("helion", helion),
        ("triton", triton),
        ("torch._inductor.codecache", codecache),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(triton_inputs, "SnapshotInputs", lambda *a: lambda metadata: ((1,), {}))
    monkeypatch.setattr(triton_quick_runner, "probe_target", lambda: {"arch": 103})
    result = helion_prepare.prepare(
        source="example",
        kernel_name="example",
        settings=cloudpickle.dumps({}),
        configurations=[{"block": 64}, {"block": 0}],
        inputs={"metadata": {}},
        helion_version="1.4.0",
    )
    assert executed == []
    valid, bad = result["variants"]
    assert valid["status"] == "prepared"
    assert [s["kernel_name"] for s in valid["launches"]] == ["first", "second"]
    assert valid["launches"][0]["signature"] == {"x": "i32", "block": "constexpr"}
    assert valid["launches"][0]["constants"] == {"block": 64}
    assert bad["status"] == "failed" and "invalid config" in bad["diagnostics"]
