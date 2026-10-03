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
        dict(quick_benchmark_group_size=0),
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
            self.block = state["block"]

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return times[other.block] / 1000

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
    durations = iter([9, 3, 8, 4, 5])

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
    assert log[3:103] == ["flush"] * 100
    assert log[103:-1] == ["start", ("call", 1, 2), "end", "flush"] * 5
    assert log[-1] == "sync" and trials["trial_us"] == [9000, 3000, 8000, 4000, 5000]
    assert trials["runtime_us"] == 3000
    assert trials["refinement_iterations"] == 0


def test_quick_refinement_uses_ten_ms_iteration_count_and_minimum():
    log = []
    durations = iter([1, 2, 3, 4, 5] + [2] * 9 + [0.5])

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, other):
            assert log[-1] == "sync"
            return next(durations)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    )
    result = runner.measure_quick(
        lambda: log.append("call"),
        (),
        {},
        SimpleNamespace(zero_=lambda: log.append("flush")),
        torch,
    )
    assert result["runtime_us"] == 500
    assert result["refinement_iterations"] == 10
    assert result["pilot_trial_us"] == [1000, 2000, 3000, 4000, 5000]
    assert result["trial_us"] == [2000] * 9 + [500]
    assert log.count("flush") == 100 + 5 + 100 + 10
    assert log.count("sync") == 4


def test_group_timing_interleaves_candidates_and_handles_different_counts():
    log = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, other):
            return 0.1

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    )
    values = runner.measure_group(
        [lambda: log.append("a"), lambda: log.append("b")],
        (),
        {},
        SimpleNamespace(zero_=lambda: log.append("flush")),
        torch,
        [3, 2],
    )
    assert [x for x in log if x in ["a", "b"]] == ["a", "b", "a", "b", "a"]
    assert log[:100] == ["flush"] * 100
    assert log.count("sync") == 1
    assert values == [[100] * 3, [100] * 2]


def test_tiered_pruning_uses_valid_best_across_batches_then_regroups(monkeypatch):
    timings = {"invalid": 1, "best": 100, "cutoff": 200, "near": 150, "exempt": 250, "slow": 400}
    calls = []
    entries = []
    for name in timings:

        def candidate():
            pass

        candidate.identity = name
        entries.append(({"id": name}, candidate))

    def measure(candidates, args, kwargs, flush, torch, counts):
        calls.append(([c.identity for c in candidates], counts))
        return [[timings[c.identity]] * n for c, n in zip(candidates, counts, strict=True)]

    monkeypatch.setattr(runner, "measure_group", measure)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    best, identity = runner.benchmark_candidates(
        entries, (), {}, lambda: ((), {}), lambda c: c.identity != "invalid", None, torch, 2, 300
    )
    # Pilot all three groups first; 200 and 250 us survive because they are <300 us.
    assert [ids for ids, _ in calls[:3]] == [
        ["invalid", "best"],
        ["cutoff", "near"],
        ["exempt", "slow"],
    ]
    assert [ids for ids, _ in calls[3:]] == [["best", "cutoff"], ["near", "exempt"]]
    assert best == 100 and identity == "best"
    assert entries[0][0]["status"] == "invalid"
    assert entries[-1][0]["status"] == "pilot_pruned"
    calls.clear()
    for row, _ in entries:
        row.clear()
    for name, (row, _) in zip(timings, entries, strict=True):
        row["id"] = name
    runner.benchmark_candidates(
        entries, (), {}, lambda: ((), {}), lambda c: c.identity != "invalid", None, torch, 2, 0
    )
    assert [ids for ids, _ in calls[3:]] == [["best", "near"]]
    assert entries[2][0]["status"] == "pilot_pruned"  # Exactly 2x is pruned.
