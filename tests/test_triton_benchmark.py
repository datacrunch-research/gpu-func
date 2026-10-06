from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gfaas import TritonPruning, TritonTuning
from gfaas.triton_inputs import capture_inputs
from gfaas.triton_quick_runner import InputRing, final_benchmark, pruning_limit, ring_trials
from gfaas.triton_replication import TritonBenchmarkError, benchmark_shards


@pytest.mark.parametrize(
    "relative,absolute", [(-1, 0), (0, -1), (float("nan"), 0), (0, float("inf"))]
)
def test_invalid_pruning(relative, absolute):
    with pytest.raises(ValueError):
        TritonPruning(relative, absolute)


def test_absolute_and_relative_allowances_and_independent_disabling():
    assert pruning_limit(5, TritonPruning(0.1, 1).request()) == 6
    assert pruning_limit(100, TritonPruning(0.1, 1).request()) == 110
    assert pruning_limit(100, None) == float("inf")
    assert TritonTuning(pilot_pruning=None).refined_pruning is not None
    assert TritonTuning(refined_pruning=None).pilot_pruning is not None


class Tensor:
    requires_grad = False
    shape = (2,)
    dtype = "torch.float32"
    device = SimpleNamespace(type="cuda", index=0)
    layout = "torch.strided"

    def __init__(self, pointer):
        self.pointer = pointer

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: 8)

    def stride(self):
        return (1,)

    def storage_offset(self):
        return 0

    def numel(self):
        return 2

    def element_size(self):
        return 4


def test_ring_exceeds_l2_and_rejects_cross_set_aliases_and_limits():
    torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(current_device=lambda: 0))
    metadata = capture_inputs((Tensor(1),), {"N": 2})
    pointers = iter(range(100, 200))
    ring = InputRing(
        lambda _: ((Tensor(next(pointers)),), {"N": 2}),
        lambda *a, **k: None,
        metadata,
        torch,
        16,
        10,
        100,
    )
    assert len(ring.sets) == 3 and ring.footprint_bytes * len(ring.sets) > 16
    assert ring.allocated_bytes == 24
    assert ring.next() is ring.sets[0]
    assert ring.next() is ring.sets[1]
    with pytest.raises(ValueError, match="independent"):
        InputRing(
            lambda _: ((Tensor(2),), {"N": 2}), lambda *a, **k: None, metadata, torch, 16, 10, 100
        )
    for max_sets, max_bytes in [(2, 100), (10, 16)]:
        bounded = InputRing(
            lambda _: ((Tensor(next(pointers)),), {"N": 2}),
            lambda *a, **k: None,
            metadata,
            torch,
            16,
            max_sets,
            max_bytes,
        )
        assert len(bounded.sets) == 1 and bounded.requires_flush
        assert bounded.allocated_bytes == 8
    with pytest.raises(ValueError, match="max_ring_bytes"):
        InputRing(
            lambda _: ((Tensor(next(pointers)),), {"N": 2}),
            lambda *a, **k: None,
            metadata,
            torch,
            16,
            10,
            4,
        )


def test_ring_metadata_and_alias_contract():
    torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(current_device=lambda: 0))
    metadata = capture_inputs((Tensor(1), Tensor(1)), {})
    with pytest.raises(ValueError, match="aliasing"):
        InputRing(
            lambda _: ((Tensor(2), Tensor(3)), {}), lambda *a: None, metadata, torch, 0, 1, 100
        )
    with pytest.raises(ValueError, match="scalar"):
        InputRing(
            lambda _: ((), {"N": 3}),
            lambda **k: None,
            capture_inputs((), {"N": 2}),
            torch,
            0,
            1,
            100,
        )


def test_ring_event_order_reset_excluded_and_no_intermediate_flushes():
    log = []

    class Event:
        def __init__(self, **kw):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, other):
            return 0.01

    ring = SimpleNamespace(next=lambda: ((1,), {}), reset=lambda *a: log.append("reset"))
    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    )
    trials = ring_trials(
        [lambda *a: log.append("a"), lambda *a: log.append("b")],
        [2, 1],
        ring,
        torch,
        SimpleNamespace(zero_=lambda: log.append("flush")),
    )
    assert log[:100] == ["flush"] * 100
    assert log[100:-1] == [
        "reset",
        "event",
        "a",
        "event",
        "reset",
        "event",
        "b",
        "event",
        "reset",
        "event",
        "a",
        "event",
    ]
    assert log[-1] == "sync"
    assert trials == [[10, 10], [10]]


def test_long_kernel_final_events_have_at_least_25_iterations(monkeypatch):
    from gfaas import triton_quick_runner as runner

    calls = []
    monkeypatch.setattr(
        runner, "ring_trials", lambda c, counts, *a: calls.append(counts) or [[4000] * counts[0]]
    )
    result = final_benchmark(None, 4000, None, None, None)
    assert calls == [[25]] and result["runtime_us"] == 4000
    assert result["method"] == "events-ring"


@pytest.mark.parametrize(
    "settings,sets,graphs,calls,iterations,runtime",
    [
        (None, 300, 3, 100, 50, 2),
        ({"graph_duration_ms": 0.1, "final_duration_ms": 2.0}, 220, 11, 20, 20, 10),
    ],
)
def test_graph_ring_pads_and_cycles_distinct_graphs_with_resets_outside_capture(
    settings, sets, graphs, calls, iterations, runtime
):
    log = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class Stream:
        def wait_stream(self, other):
            pass

        def synchronize(self):
            pass

    class Event:
        def __init__(self, **kw):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, other):
            return 0.2

    class Graph:
        instances = []

        def __init__(self):
            self.identity = len(self.instances)
            self.instances.append(self)

        def replay(self):
            log.append(("replay", self.identity))

    class Ring:
        sets = [((i,), {}) for i in range(201)]

        def ensure(self, n):
            self.sets.extend(((i,), {}) for i in range(len(self.sets), n))

        def reset(self, *a):
            log.append("reset")

    ring = Ring()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            Stream=Stream,
            current_stream=Stream,
            stream=lambda s: Context(),
            graph_pool_handle=lambda: 1,
            CUDAGraph=Graph,
            graph=lambda *a, **k: Context(),
            Event=Event,
        )
    )
    result = final_benchmark(
        lambda *a: None, 5, ring, torch, SimpleNamespace(zero_=lambda: None), settings
    )
    assert len(ring.sets) == sets and result["graph_count"] == graphs
    assert result["calls_per_graph"] == calls and result["iterations"] == iterations
    assert result["runtime_us"] == runtime
    replays = [identity for item in log if isinstance(item, tuple) for identity in [item[1]]]
    assert replays[:6] == [i % graphs for i in range(6)]


class FakeBenchmark:
    def __init__(self, devices):
        self.devices = iter(devices)
        self.requests = []

    def spawn(self, **kwargs):
        self.requests.append(kwargs)
        gpu = next(self.devices)
        variants = json.loads(kwargs["variants"])
        if gpu in kwargs["excluded_gpu_uuids"]:
            result = {"status": "duplicate_gpu", "gpu_uuid": gpu, "results": []}
        else:
            result = {
                "status": "passed",
                "gpu_uuid": gpu,
                "results": [
                    {
                        "id": v["id"],
                        "status": "measured",
                        "runtime_us": 10 if v["id"] == "a" else 30,
                        "refined_us": 10 if v["id"] == "a" else 30,
                        "evaluation": "passed",
                    }
                    for v in variants
                ],
            }
        return SimpleNamespace(call_id=f"call_{len(self.requests)}", wait=lambda: result)


def test_replication_verifies_distinct_gpus_and_prunes_globally():
    fn = FakeBenchmark(["gpu1", "gpu1", "gpu2", "gpu3"])
    calls = ["compile"]
    report = benchmark_shards(
        fn, [{"id": "a"}, {"id": "b"}], {}, TritonTuning(replication_factor=3), calls
    )
    assert report["best_id"] == "a" and report["best_runtime_us"] == 10
    assert [r["gpu_uuid"] for r in report["results"][0]["replicas"]] == ["gpu1", "gpu2", "gpu3"]
    assert report["results"][1]["status"] == "globally_pruned"
    assert len(calls) == 5
    assert json.loads(fn.requests[1]["variants"]) == [{"id": "a"}]
    assert fn.requests[3]["excluded_gpu_uuids"] == ["gpu1", "gpu2"]


def test_replication_fails_instead_of_counting_duplicate_gpus():
    fn = FakeBenchmark(["gpu1", "gpu1", "gpu1"])
    with pytest.raises(TritonBenchmarkError, match="distinct GPU") as error:
        benchmark_shards(
            fn,
            [{"id": "a"}],
            {},
            TritonTuning(replication_factor=2, replication_max_attempts=2),
            [],
        )
    assert len(error.value.call_ids) == 3
    assert len(error.value.report["results"][0]["replicas"]) == 1


def test_replication_rejects_incomplete_reports_and_retains_call_id():
    class Missing:
        def spawn(self, **kwargs):
            return SimpleNamespace(
                call_id="broken", wait=lambda: {"gpu_uuid": "gpu1", "results": []}
            )

    with pytest.raises(TritonBenchmarkError, match="Incomplete") as error:
        benchmark_shards(Missing(), [{"id": "a"}], {}, TritonTuning(), [])
    assert error.value.call_ids == ["broken"]


def test_replica_calls_reserve_gpu_count_and_average_verified_bundle():
    from gfaas import App, Image
    from gfaas.triton_quick_runner import benchmark_cycle

    class Client:
        requests = []

        def submit(self, **kwargs):
            self.requests.append(kwargs)
            if kwargs["function"].__name__ == "benchmark_cycle":
                result = {
                    "status": "passed",
                    "gpu_uuid": "gpu1",
                    "results": [
                        {"id": "a", "status": "measured", "runtime_us": 10, "refined_us": 11}
                    ],
                }
            else:
                assert kwargs["function"].__name__ == "benchmark_replicas"
                assert kwargs["gpu_count"] == 3
                assert kwargs["kwargs"]["device_count"] == 3
                result = {
                    "status": "replica_bundle",
                    "replica_reports": [
                        {"status": "duplicate_gpu", "gpu_uuid": "gpu1", "results": []},
                        {
                            "status": "passed",
                            "gpu_uuid": "gpu2",
                            "results": [{"id": "a", "status": "measured", "runtime_us": 12}],
                        },
                        {
                            "status": "passed",
                            "gpu_uuid": "gpu3",
                            "results": [{"id": "a", "status": "measured", "runtime_us": 8}],
                        },
                    ],
                }
            return SimpleNamespace(call_id=f"call{len(self.requests)}", wait=lambda: result)

    client = Client()
    app = App("replicas", image=Image("registered"), client=client)
    scope = app.function(gpu_count=1, gpu_type="gb300")
    fn = scope.bind(benchmark_cycle)
    report = benchmark_shards(fn, [{"id": "a"}], {}, TritonTuning(refined_pruning=None), [])
    assert report["best_runtime_us"] == 10
    assert {r["gpu_uuid"] for r in report["results"][0]["replicas"]} == {"gpu1", "gpu2", "gpu3"}
    assert report["call_ids"] == ["call1", "call2"]


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("dtype", "torch.float16"),
        ("shape", (3,)),
        ("device", SimpleNamespace(type="cuda", index=1)),
    ],
)
def test_generated_inputs_reject_metadata_or_device_mismatch(attribute, value):
    torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(current_device=lambda: 0))
    metadata = capture_inputs((Tensor(1),), {})
    generated = Tensor(2)
    setattr(generated, attribute, value)
    with pytest.raises(ValueError, match="differs|another GPU"):
        InputRing(lambda _: ((generated,), {}), lambda *a: None, metadata, torch, 0, 1, 100)


def test_evaluation_inputs_cannot_alias_the_benchmark_ring():
    torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(current_device=lambda: 0))
    ring = InputRing(
        lambda _: ((Tensor(2),), {}),
        lambda *a: None,
        capture_inputs((Tensor(1),), {}),
        torch,
        0,
        1,
        100,
    )
    with pytest.raises(ValueError, match="Evaluation inputs share"):
        ring.fresh()


def test_custom_final_budget_and_iteration_cap_reach_gpu_runner(monkeypatch):
    from gfaas import TritonBenchmark
    from gfaas import triton_quick_runner as runner

    counts = []
    monkeypatch.setattr(runner, "ring_trials", lambda c, n, *a: counts.append(n) or [[4000] * n[0]])
    settings = TritonBenchmark(final_duration_ms=1000, min_final_trials=5, max_final_trials=7)
    request = TritonTuning(benchmark=settings).request()
    result = final_benchmark(None, 4000, None, None, None, request["benchmark"])
    assert counts == [[7]]
    assert result["iterations"] == 7 and result["method"] == "events-ring"


@pytest.mark.parametrize(
    "options",
    [
        {"pilot_trials": 0},
        {"final_duration_ms": float("nan")},
        {"graph_duration_ms": -1},
        {"max_calls_per_graph": 5},
        {"min_final_trials": 100_001},
        {"l2_flush_iterations": True},
    ],
)
def test_invalid_benchmark_budgets(options):
    from gfaas import TritonBenchmark

    with pytest.raises(ValueError):
        TritonBenchmark(**options)


def test_replica_accuracy_disagreement_raises_with_retained_results():
    class Disagreement(FakeBenchmark):
        def spawn(self, **kwargs):
            result = super().spawn(**kwargs)
            original = result.wait

            def wait():
                report = original()
                if report["gpu_uuid"] == "gpu2":
                    report["results"][0].update(status="invalid", evaluation="failed")
                return report

            result.wait = wait
            return result

    with pytest.raises(TritonBenchmarkError, match="failed evaluation.*passing") as error:
        benchmark_shards(
            Disagreement(["gpu1", "gpu2"]),
            [{"id": "a"}],
            {},
            TritonTuning(replication_factor=2),
            [],
        )
    assert len(error.value.report["results"][0]["replicas"]) == 1


def test_graph_padding_falls_back_to_events_when_ring_memory_is_bounded(monkeypatch):
    ring = SimpleNamespace(sets=[None], max_sets=1, allocated_bytes=1024, max_bytes=1024)
    monkeypatch.setattr("gfaas.triton_quick_runner.ring_trials", lambda *a: [[5.0, 6.0]])
    result = final_benchmark(None, 5, ring, None, None)
    assert result["method"] == "events-ring"
    assert result["calls_per_graph"] == 0


def test_bounded_ring_evicts_outside_event_timing_and_disables_graphs(monkeypatch):
    log = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            log.append("event")

        def elapsed_time(self, other):
            return 0.005

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"))
    )
    ring = SimpleNamespace(
        requires_flush=True,
        sets=[((), {})],
        max_sets=1,
        allocated_bytes=8,
        max_bytes=8,
        next=lambda: ((), {}),
        reset=lambda *a, **k: log.append("reset"),
    )
    flush = SimpleNamespace(zero_=lambda: log.append("flush"))
    timing = final_benchmark(
        lambda: log.append("kernel"),
        5,
        ring,
        torch,
        flush,
        {"l2_flush_iterations": 2, "max_final_trials": 3, "min_final_trials": 1},
    )
    assert timing["method"] == "events-ring-flush"
    assert timing["calls_per_graph"] == 0
    assert log == ["flush"] * 2 + ["reset", "flush", "event", "kernel", "event"] * 3 + ["sync"]


def test_l2_eviction_buffer_uses_current_replica_device(monkeypatch):
    from gfaas import triton_quick_runner as runner

    def attribute(result, name, device):
        assert name == 38 and device == 2
        result._obj.value = 4096
        return 0

    monkeypatch.setattr(
        runner.ctypes, "CDLL", lambda _: SimpleNamespace(cuDeviceGetAttribute=attribute)
    )
    allocations = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: 2),
        uint8="uint8",
        empty=lambda size, **kwargs: allocations.append((size, kwargs)) or "buffer",
    )
    assert runner.l2_flush_buffer(torch) == "buffer"
    assert allocations == [(8192, {"dtype": "uint8", "device": "cuda:2"})]
