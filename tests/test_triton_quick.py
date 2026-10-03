from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import cloudpickle
import pytest

from gfaas import TritonTuning
from gfaas import triton_quick_runner as runner
from gfaas.triton_policy import portable_callable


@pytest.mark.parametrize(
    "options",
    [
        dict(quick_benchmark_delta=-1),
        dict(quick_benchmark_delta=float("nan")),
        dict(pruning_min_runtime_us=-1),
        dict(pruning_min_runtime_us=float("inf")),
        dict(evaluate=1),
        dict(quick_benchmark_variants_per_job=0),
        dict(quick_benchmark_max_concurrent_jobs=0),
        dict(quick_benchmark_max_concurrent_jobs=33),
    ],
)
def test_invalid_tuning_policy(options):
    with pytest.raises((ValueError, TypeError)):
        TritonTuning(**options)


def test_final_pruning_uses_final_valid_best_and_configurable_short_kernel_exemption():
    rows = [
        {"id": "invalid", "status": "invalid", "runtime_us": 1},
        {"id": "best", "status": "measured", "runtime_us": 10},
        {"id": "near", "status": "measured", "runtime_us": 11},
        {"id": "far", "status": "measured", "runtime_us": 12},
    ]
    assert runner.select_rows(rows, 10, 0.10, 0) == ["best", "near"]
    assert runner.select_rows(rows, 10, 0.10, 100) == ["best", "near", "far"]
    assert runner.select_rows(rows, None, 0.10, 100) == []


def test_python_callback_transport_does_not_require_author_module():
    module = ModuleType("private_author_module")
    exec(
        "def helper(x): return x + 1\ndef evaluate(candidate, x): return helper(x) == candidate(x)",
        module.__dict__,
    )
    sys.modules[module.__name__] = module
    callback = cloudpickle.dumps(portable_callable(module.evaluate))
    del sys.modules[module.__name__]
    assert cloudpickle.loads(callback)(lambda x: x + 1, 5)


def test_quick_benchmark_invalid_fast_kernels_never_establish_pruning_reference(monkeypatch):
    target = {"backend": "cuda", "arch": 103, "warp_size": 32}
    variants = [
        {
            "id": str(block),
            "constants": {"N": 1, "BLOCK": block},
            "signature": {"X": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
            "options": {},
        }
        for block in [1, 2, 3, 4]
    ]
    modules = {
        name: ModuleType(name)
        for name in ["torch", "triton", "triton.compiler", "triton.backends.compiler"]
    }
    state = {"block": 0}
    times = {1: 1, 2: 100, 3: 0.5, 4: 105}

    class Binary:
        def __init__(self, block):
            self.block, self.hash = block, str(block)

        def __getitem__(self, grid):
            def launch(*args):
                state["block"] = self.block
                args[0]["block"] = self.block

            return launch

    def compile(ast, **kwargs):
        state["block"] = ast.constants["BLOCK"]
        return Binary(state["block"])

    class AST:
        def __init__(self, fn, signature, constexprs):
            self.constants = constexprs

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return times[state["block"]] / 1000

    def jit(fn):
        return SimpleNamespace(fn=fn, arg_names=["X", "N", "BLOCK"])

    modules["triton"].__version__ = "test"
    modules["triton"].jit = jit
    modules["triton"].compile = compile
    modules["triton.compiler"].ASTSource = AST
    modules["triton.backends.compiler"].GPUTarget = lambda **kw: kw
    loads = []

    def load(*a, **kw):
        inputs = ({"block": 0},)
        loads.append(inputs)
        return inputs, {"N": 1}

    modules["torch"].load = load
    modules["torch"].cuda = SimpleNamespace(
        synchronize=lambda: None,
        Event=Event,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, "probe_target", lambda: target)
    monkeypatch.setattr(
        runner,
        "l2_flush_buffer",
        lambda _: SimpleNamespace(zero_=lambda: None, numel=lambda: 1024, element_size=lambda: 1),
    )
    monkeypatch.setattr(
        runner,
        "restore_caches",
        lambda *a: {
            v["id"]: {"id": v["id"], "status": "compiled", "cache_hash": v["id"]} for v in variants
        },
    )

    def evaluate(candidate, *args, **kwargs):
        candidate(*args, **kwargs)
        return args[0]["block"] not in [1, 3]

    result = runner.quick_benchmark(
        source="import triton\n@triton.jit\ndef kernel(X, N, BLOCK): pass\n",
        kernel_name="kernel",
        variants=json.dumps(variants),
        artifacts=[],
        inputs=b"",
        callbacks=cloudpickle.dumps(((1,), evaluate)),
        target=target,
        triton_version="test",
        delta=0.10,
        minimum_us=0,
    )
    assert len(loads) == 4  # One shared benchmark set and three fresh evaluations.
    loads.clear()
    assert result["best_id"] == "2" and result["best_runtime_us"] == 100
    assert result["retained_ids"] == ["2", "4"]
    assert [r["evaluation"] for r in result["results"]] == [
        "failed",
        "passed",
        "failed",
        "not_needed",
    ]
    result = runner.quick_benchmark(
        source="import triton\n@triton.jit\ndef kernel(X, N, BLOCK): pass\n",
        kernel_name="kernel",
        variants=json.dumps(variants),
        artifacts=[],
        inputs=b"",
        callbacks=cloudpickle.dumps(((1,), None)),
        target=target,
        triton_version="test",
        delta=0.10,
        minimum_us=0,
    )
    assert len(loads) == 1  # All four variants share their benchmark inputs.
    assert result["best_id"] == "3" and result["retained_ids"] == ["3"]
    assert all(r["evaluation"] == "skipped" for r in result["results"])


def test_quick_timing_order_flush_and_fastest_of_five():
    log = []
    durations = iter([0.009, 0.003, 0.008, 0.004, 0.005])

    roles = iter(["start", "end"] * 5)

    class Event:
        def __init__(self, **kwargs):
            self.role = next(roles)

        def record(self):
            log.append(self.role)

        def elapsed_time(self, other):
            assert log[-1] == "sync"
            return next(durations)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    )
    flush = SimpleNamespace(zero_=lambda: log.append("flush"))
    trials = runner.measure_quick(
        lambda x, *, y: log.append(("call", x, y)), (1,), {"y": 2}, flush, torch
    )
    assert log[:3] == ["sync", ("call", 1, 2), "sync"]
    assert log[3:-1] == ["start", ("call", 1, 2), "end", "flush"] * 5
    assert log[-1] == "sync" and trials == [9, 3, 8, 4, 5]
    assert min(trials) == 3
