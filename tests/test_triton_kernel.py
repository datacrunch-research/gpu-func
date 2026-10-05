from __future__ import annotations

import inspect
import json
import sys
from threading import Barrier, Lock
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gfaas import (
    App,
    Image,
    TritonCompilationError,
    TritonExecutionNotImplementedError,
    UnsupportedTritonKernelError,
)
from gfaas import (
    TritonKernel as NativeTritonKernel,
)
from gfaas.errors import GfaasError


def make_inputs(metadata):
    return (), {}


def reset_inputs(*args, **kwargs):
    pass


def TritonKernel(*args, **kwargs):
    kernel = NativeTritonKernel(*args, **kwargs)
    # Compiler-only tests isolate the dispatch phase, rather than run tuning.
    if "tuning" not in kwargs:
        kernel.tuning = None
    return kernel


def add(X, N, BLOCK):
    pass


class JIT:
    def __init__(self, fn: Any):
        self.fn = fn
        self.arg_names = list(inspect.signature(fn).parameters)
        self.params = [
            SimpleNamespace(name=n, is_constexpr=n in ("N", "BLOCK")) for n in self.arg_names
        ]
        self._repr = None
        self.pre_run_hooks = []


class Config:
    def __init__(self, kwargs: dict[str, Any], pre_hook: Any = None):
        self.kwargs = kwargs
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return {**self.kwargs, "num_warps": 4}


class Autotuner:
    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value):
        self.fn, self.arg_names, self.configs = fn, arg_names, configs
        self.reset_to_zero = list(reset_to_zero or [])
        self.restore_value = list(restore_value or [])
        self.pre_hook = lambda kwargs: 0
        self.post_hook = lambda kwargs: 0
        if self.reset_to_zero or self.restore_value:

            def default_reset(kwargs):
                return self.reset_to_zero

            self.pre_hook = default_reset
        self.perf_model = None
        self.configs_top_k = 1.0


class Client:
    def __init__(self, fail=False):
        self.submission = None
        self.fail = fail

    def submit(self, **kwargs):
        if kwargs["function"].__name__ == "probe_target":
            return SimpleNamespace(
                call_id="call_probe", wait=lambda: {"backend": "cuda", "arch": 103, "warp_size": 32}
            )
        from gfaas.artifacts import collect_artifact_ids

        assert collect_artifact_ids((), kwargs["kwargs"]) == []
        kwargs["kwargs"]["variants"] = json.loads(kwargs["kwargs"]["variants"])
        self.submission = kwargs
        rows = [
            {"id": v["id"], "status": "failed" if self.fail else "compiled"}
            for v in kwargs["kwargs"]["variants"]
        ]
        return SimpleNamespace(call_id="call_test", wait=lambda: {"results": rows})


@pytest.fixture(autouse=True)
def triton(monkeypatch, request):
    modules = {
        name: ModuleType(name)
        for name in (
            "triton",
            "triton.runtime",
            "triton.runtime.jit",
            "triton.runtime.autotuner",
            "triton.language",
        )
    }
    modules["triton"].__version__ = "3.8.0"
    modules["triton"].Config = Config
    modules["triton.runtime.jit"].JITFunction = JIT
    modules["triton.runtime.autotuner"].Autotuner = Autotuner
    modules["triton.language"].float32 = "fp32"
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    if request.node.name not in {
        "test_benchmark_gpu_phase_receives_metadata_and_compiler_artifacts",
        "test_generator_and_reset_are_removed",
    } and not request.node.name.startswith("test_launch_"):
        # These tests isolate compilation at its boundary, independently of
        # tensor transport, benchmarking, and the new execution phase.
        monkeypatch.setattr(
            "gfaas.triton_kernel.snapshot_inputs",
            lambda *a: {"metadata": {"args": [], "kwargs": {}}, "storages": {}},
        )
        original = NativeTritonKernel._dispatch

        def compiler_boundary(self, *args):
            report, identity = original(self, *args)
            raise TritonExecutionNotImplementedError(report, identity)

        monkeypatch.setattr(NativeTritonKernel, "_dispatch", compiler_boundary)


def configuration(client=None, **options):
    return dict(app=App("kernels", image=Image("compiler-image"), client=client), **options)


def configured_kernel(native, *, app, **options):
    policy = {
        k: options.pop(k)
        for k in list(options)
        if k in {"variants_per_job", "max_concurrent_jobs", "cache_compression_level", "tuning"}
    }
    options.pop("target_arch", None)
    kernel = TritonKernel(native, **policy)
    scope = app.function(gpu_count=1, gpu_type="gb300", **options)

    class ScopedKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                with scope:
                    return kernel[grid](*args, **kwargs)

            return launch

        def __getattr__(self, name):
            return getattr(kernel, name)

    return ScopedKernel()


def test_single_compile_never_evaluates_grid_or_transports_tensor():
    client = Client()
    tensor = SimpleNamespace(dtype="torch.float32", data_ptr=lambda: pytest.fail("pointer read"))
    kernel = configured_kernel(JIT(add), **configuration(client))
    with pytest.raises(TritonExecutionNotImplementedError) as error:
        kernel[lambda meta: pytest.fail("grid executed")](tensor, 256, 64)
    request = client.submission
    assert request["gpu_count"] == 0
    assert request["kwargs"]["variants"][0]["signature"] == {
        "X": "*fp32",
        "N": "constexpr",
        "BLOCK": "constexpr",
    }
    assert tensor not in request["kwargs"].values()
    assert error.value.call_id == "call_test"


def test_autotune_compiles_every_config_and_partial_failures_have_report():
    client = Client(fail=True)
    native = Autotuner(
        JIT(add),
        ["X", "N", "BLOCK"],
        [Config({"BLOCK": 64}), Config({"BLOCK": 128})],
        ["N"],
        None,
        None,
    )
    kernel = configured_kernel(native, **configuration(client))
    tensor = SimpleNamespace(dtype="torch.float32", data_ptr=lambda: 0)
    with pytest.raises(TritonCompilationError) as error:
        kernel[(1,)](tensor, N=256)
    assert len(error.value.report["results"]) == 2
    assert [v["constants"]["BLOCK"] for v in client.submission["kwargs"]["variants"]] == [64, 128]


@pytest.mark.parametrize("modifier", ["pre_hook", "perf_model", "configs_top_k"])
def test_modifiers_rejected_before_submission(modifier):
    client = Client()
    native = Autotuner(JIT(add), ["X", "N", "BLOCK"], [Config({"BLOCK": 64})], ["N"], None, None)
    setattr(native, modifier, 2 if modifier == "configs_top_k" else lambda *a: None)
    with pytest.raises(UnsupportedTritonKernelError):
        configured_kernel(native, **configuration(client))
    assert client.submission is None


def test_config_hooks_nested_wrappers_and_mutation_rejected():
    native = Autotuner(JIT(add), [], [Config({}, pre_hook=lambda: None)], [], None, None)
    with pytest.raises(UnsupportedTritonKernelError, match="pre_hook"):
        configured_kernel(native, **configuration())
    with pytest.raises(UnsupportedTritonKernelError):
        configured_kernel(SimpleNamespace(fn=JIT(add)), **configuration())
    jit = JIT(add)
    kernel = configured_kernel(jit, **configuration())
    jit.pre_run_hooks.append(lambda: None)
    with pytest.raises(UnsupportedTritonKernelError):
        kernel[(1,)](None, 1, 64)


def test_positional_configuration_conflict_rejected_without_network():
    client = Client()
    native = Autotuner(JIT(add), [], [Config({"BLOCK": 64})], [], None, None)
    kernel = configured_kernel(native, **configuration(client))
    with pytest.raises(ValueError, match="conflicts"):
        kernel[(1,)](None, 1, 128)
    assert client.submission is None


def test_new_autotune_modifier_is_rejected_even_without_named_check():
    native = Autotuner(JIT(add), [], [Config({"BLOCK": 64})], [], None, None)
    native.future_callback = lambda: None
    with pytest.raises(UnsupportedTritonKernelError, match="Unrecognized"):
        configured_kernel(native, **configuration())


def test_legacy_default_repr_closure_is_accepted_and_custom_one_rejected():
    jit = JIT(add)
    del jit._repr

    def attach(repr):
        return lambda _: add.__name__ if repr is None else repr(_)

    jit.repr = attach(None)
    configured_kernel(jit, **configuration())
    jit.repr = attach(lambda _: "custom")
    with pytest.raises(UnsupportedTritonKernelError, match="repr"):
        configured_kernel(jit, **configuration())


def test_ir_override_launch_rejected_before_remote_work():
    client = Client()
    kernel = configured_kernel(JIT(add), **configuration(client))
    with pytest.raises(UnsupportedTritonKernelError, match="ir_override"):
        kernel[(1,)](None, 1, 64, ir_override="custom.ptx")
    assert client.submission is None


def test_wrong_variant_identity_cannot_report_success():
    class WrongReportClient(Client):
        def submit(self, **kwargs):
            if kwargs["function"].__name__ == "probe_target":
                return SimpleNamespace(
                    call_id="call_probe",
                    wait=lambda: {"backend": "cuda", "arch": 103, "warp_size": 32},
                )
            return SimpleNamespace(
                call_id="call_test",
                wait=lambda: {"results": [{"id": "different-variant", "status": "compiled"}]},
            )

    kernel = configured_kernel(JIT(add), **configuration(WrongReportClient()))
    with pytest.raises(GfaasError, match="incomplete variant report"):
        kernel[(1,)](None, 1, 64)


def test_large_configuration_set_fits_bounded_artifact_scanner():
    client = Client()
    native = Autotuner(
        JIT(add), [], [Config({"BLOCK": block}) for block in range(1536)], [], None, None
    )
    wrapped = configured_kernel(native, **configuration(client))
    with pytest.raises(TritonExecutionNotImplementedError) as outcome:
        wrapped[(1,)](None, N=256)
    assert len(outcome.value.report["results"]) == 1536


def test_sharding_bounds_inflight_jobs_preserves_order_and_call_identities():
    gate, lock = Barrier(3), Lock()

    class ShardedClient:
        def __init__(self):
            self.requests = []
            self.active = self.peak = 0

        def submit(self, **kwargs):
            if kwargs["function"].__name__ == "probe_target":
                return SimpleNamespace(
                    call_id="call_probe",
                    wait=lambda: {"backend": "cuda", "arch": 103, "warp_size": 32},
                )
            with lock:
                index = len(self.requests)
                self.requests.append(kwargs)

            def wait():
                with lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                if index < 3:
                    gate.wait(timeout=5)
                rows = [
                    {"id": v["id"], "status": "compiled"}
                    for v in json.loads(kwargs["kwargs"]["variants"])
                ]
                with lock:
                    self.active -= 1
                return {"results": rows}

            return SimpleNamespace(call_id=f"call_{index}", wait=wait)

    client = ShardedClient()
    options = configuration(client, variants_per_job=2, max_concurrent_jobs=3)
    native = Autotuner(JIT(add), [], [Config({"BLOCK": i}) for i in range(13)], [], None, None)
    with pytest.raises(TritonExecutionNotImplementedError) as outcome:
        configured_kernel(native, **options)[(1,)](None, N=256)
    report = outcome.value.report
    assert client.peak == 3
    assert len(client.requests) == 7
    assert len(set(outcome.value.call_ids)) == 8
    assert [v["constants"]["BLOCK"] for v in report["variants"]] == list(range(13))
    assert [r["id"] for r in report["results"]] == [v["id"] for v in report["variants"]]
    assert sum(len(json.loads(r["kwargs"]["variants"])) for r in client.requests) == 13
    assert all(r["gpu_count"] == 0 and r["cpu_millicores"] <= 2000 for r in client.requests)


def test_failed_shard_keeps_successful_results_and_failed_call_identity():
    class FailedShardClient:
        def __init__(self):
            self.index = 0
            self.lock = Lock()

        def submit(self, **kwargs):
            if kwargs["function"].__name__ == "probe_target":
                return SimpleNamespace(
                    call_id="call_probe",
                    wait=lambda: {"backend": "cuda", "arch": 103, "warp_size": 32},
                )
            with self.lock:
                index = self.index
                self.index += 1

            def wait():
                if index == 0:
                    raise RuntimeError("worker terminated")
                return {
                    "results": [
                        {"id": v["id"], "status": "compiled"}
                        for v in json.loads(kwargs["kwargs"]["variants"])
                    ]
                }

            return SimpleNamespace(call_id=f"call_{index}", wait=wait)

    options = configuration(FailedShardClient(), variants_per_job=2)
    native = Autotuner(JIT(add), [], [Config({"BLOCK": i}) for i in range(6)], [], None, None)
    with pytest.raises(TritonCompilationError) as outcome:
        configured_kernel(native, **options)[(1,)](None, N=256)
    rows = outcome.value.report["results"]
    assert sum(r["status"] == "compiled" for r in rows) == 4
    assert sum(r["status"] == "failed" for r in rows) == 2
    assert "call_0" in outcome.value.call_ids
    assert any("worker terminated" in r.get("diagnostics", "") for r in rows)


def test_structural_failure_stops_queued_shards_without_losing_variants():
    class BrokenClient:
        def __init__(self):
            self.submissions = 0

        def submit(self, **kwargs):
            if kwargs["function"].__name__ == "probe_target":
                return SimpleNamespace(
                    call_id="call_probe",
                    wait=lambda: {"backend": "cuda", "arch": 103, "warp_size": 32},
                )
            self.submissions += 1
            raise GfaasError("service rejected submission")

    client = BrokenClient()
    options = configuration(client, variants_per_job=2, max_concurrent_jobs=1)
    native = Autotuner(JIT(add), [], [Config({"BLOCK": i}) for i in range(12)], [], None, None)
    with pytest.raises(TritonCompilationError) as outcome:
        configured_kernel(native, **options)[(1,)](None, N=256)
    assert client.submissions == 1
    assert outcome.value.call_ids == ["call_probe"]
    assert len(outcome.value.report["results"]) == 12
    assert sum(s["report"].get("not_submitted", False) for s in outcome.value.report["shards"]) == 5


@pytest.mark.parametrize(
    ("count", "jobs"), [(1, 1), (128, 1), (129, 2), (1024, 8), (1536, 8), (2048, 8), (2049, 9)]
)
def test_automatic_sharding_balances_chunks_and_dispatch_window(count, jobs):
    kernel = configured_kernel(JIT(add), **configuration())
    assert kernel._job_count(count) == jobs
    assert kernel.max_concurrent_jobs == 8
    assert kernel.cache_compression_level == 1


def test_explicit_chunk_limit_overrides_automatic_policy():
    assert configured_kernel(JIT(add), **configuration(variants_per_job=100))._job_count(1536) == 16
    assert configured_kernel(JIT(add), **configuration(variants_per_job=2048))._job_count(1536) == 1
    assert configured_kernel(JIT(add), **configuration(max_concurrent_jobs=4))._job_count(1024) == 4


def test_existing_app_function_environment_and_resource_settings_are_forwarded():
    client = Client()
    inherited = Image.from_remote("compiler", {"image_digest": "sha256:" + "a" * 64})
    app = App("existing-app", image=inherited, client=client)
    kernel = TritonKernel(JIT(add))
    with (
        app.function(
            gpu="gb300",
            cpu_millicores=2500,
            memory_bytes=2 * 1024**3,
            timeout=91,
            capacity_wait=17,
            ephemeral_storage_bytes=3 * 1024**3,
            shared_memory_bytes=128 * 1024**2,
            max_output_bytes=64 * 1024**2,
            max_log_bytes=1024,
            env={"EXAMPLE": "value"},
        ),
        pytest.raises(TritonExecutionNotImplementedError),
    ):
        kernel[(1,)](None, 1, 64)
    request = client.submission
    assert request["image"] is inherited and request["app_name"] == "existing-app"
    assert request["timeout_s"] == 91 and request["capacity_wait_s"] == 17
    assert request["cpu_millicores"] == 1000 and request["gpu_type"] == "gb300"
    assert request["memory_bytes"] == 2 * 1024**3
    assert request["ephemeral_storage_bytes"] == 3 * 1024**3
    assert request["shared_memory_bytes"] == 128 * 1024**2
    assert request["max_output_bytes"] == 64 * 1024**2 and request["max_log_bytes"] == 1024
    assert request["env"] == {"EXAMPLE": "value"} and app.client is client
    assert not hasattr(kernel, "compiler")  # invocation settings do not leak onto shared kernel


def test_context_required_and_image_override_is_resolved_at_invocation():
    kernel = TritonKernel(JIT(add))
    with pytest.raises(RuntimeError, match="inside with app.function"):
        kernel[(1,)](None, 1, 64)
    client = Client()
    app = App("existing", image=Image("default"), client=client)
    with (
        app.function(gpu="gb300", image=Image("override")),
        pytest.raises(TritonExecutionNotImplementedError),
    ):
        kernel[(1,)](None, 1, 64)
    assert client.submission["image"].name == "override"
    with App("missing").function(gpu="gb300"), pytest.raises(ValueError, match="no image"):
        kernel[(1,)](None, 1, 64)


def test_compile_only_does_not_upload_inputs():
    client = Client()
    kernel = TritonKernel(JIT(add))
    with (
        App("compile", image=Image("compiler"), client=client).function(gpu="gb300"),
        pytest.raises(TritonExecutionNotImplementedError) as outcome,
    ):
        kernel[(1,)](None, 1, 64)
    assert "benchmark" not in outcome.value.report


def test_benchmark_gpu_phase_receives_metadata_and_compiler_artifacts():
    import cloudpickle

    import gfaas

    class BenchmarkClient(Client):
        def submit(self, **kwargs):
            if kwargs["function"].__name__ == "execute_winner":
                return SimpleNamespace(
                    call_id="call_execute", wait=lambda: kwargs["kwargs"]["inputs"]
                )
            if kwargs["function"].__name__ == "benchmark_cycle":
                self.benchmark_request = kwargs
                variants = json.loads(kwargs["kwargs"]["variants"])
                report = {
                    "status": "passed",
                    "gpu_uuid": "gpu_0",
                    "results": [
                        {"id": v["id"], "status": "measured", "runtime_us": 10, "refined_us": 11}
                        for v in variants
                    ],
                }
                return SimpleNamespace(call_id="call_benchmark", wait=lambda: report)
            return super().submit(**kwargs)

        def get_call_result(self, identity):
            assert identity == "call_test"
            return {"artifacts": [{"name": "compiled-triton", "artifact_id": "art_compiled"}]}

    client = BenchmarkClient()
    kernel = TritonKernel(JIT(add), tuning=gfaas.TritonTuning(replication_factor=1))
    app = App("benchmark", image=Image("compiler"), client=client)
    with app.function(gpu="gb300", timeout=44, env={"A": "B"}):
        assert kernel[(1,)](None, 1, 64) is None
    report = next(iter(kernel.tuning_results.values()))
    request = client.benchmark_request
    assert request["gpu_count"] == 1 and request["gpu_type"] == "gb300"
    assert request["timeout_s"] == 44 and request["env"] == {"A": "B"}
    data = request["kwargs"]
    assert data["metadata"]["args"][0] == {"kind": "value", "value": None}
    assert data["inputs"]["storages"] == {}
    assert data["artifacts"][0].artifact_id == "art_compiled"
    assert cloudpickle.loads(data["callbacks"])[0] == (1,)
    assert report["call_ids"] == ("call_probe", "call_test", "call_benchmark")
    assert report["benchmark"]["best_runtime_us"] == 10


def test_generator_and_reset_are_removed():
    NativeTritonKernel(JIT(add))
    with pytest.raises(TypeError, match="make_inputs"):
        NativeTritonKernel(JIT(add), make_inputs=make_inputs)
    with pytest.raises(TypeError, match="reset_inputs"):
        NativeTritonKernel(JIT(add), reset_inputs=reset_inputs)


def test_launch_cached_specializations_execute_every_call_and_reports_are_immutable():
    import gfaas

    class LaunchClient(Client):
        def __init__(self):
            super().__init__()
            self.calls = []

        def submit(self, **request):
            name = request["function"].__name__
            self.calls.append(name)
            if name == "execute_winner":
                return SimpleNamespace(call_id="execute", wait=lambda: request["kwargs"]["inputs"])
            if name == "benchmark_cycle":
                rows = [
                    {"id": v["id"], "status": "measured", "runtime_us": 10, "refined_us": 11}
                    for v in json.loads(request["kwargs"]["variants"])
                ]
                return SimpleNamespace(
                    call_id="bench",
                    wait=lambda: {"status": "passed", "gpu_uuid": "gpu0", "results": rows},
                )
            return super().submit(**request)

        def get_call_result(self, identity):
            return {"artifacts": [{"name": "compiled-triton", "artifact_id": "art_compiled"}]}

    client = LaunchClient()
    kernel = NativeTritonKernel(JIT(add), tuning=gfaas.TritonTuning(replication_factor=1))
    app = App("launch", image=Image("compiler"), client=client)
    with app.function(gpu="gb300"):
        assert kernel[(1,)](None, 1, 64) is None
        assert kernel[(1,)](None, 1, 64) is None
    assert client.calls.count("probe_target") == 1
    assert client.calls.count("compile_batch") == 1
    assert client.calls.count("benchmark_cycle") == 1
    assert client.calls.count("execute_winner") == 2
    assert len(kernel.tuning_results) == 1
    key, report = next(iter(kernel.tuning_results.items()))
    assert report["specialization"] == key
    with pytest.raises(TypeError):
        kernel.tuning_results[key] = {}
    with pytest.raises(TypeError):
        report["benchmark"]["best_runtime_us"] = 0
    assert not hasattr(kernel, "last_tuning")
    with app.function(gpu="gb300"):
        kernel[(1,)](None, 2, 64)
    assert len(kernel.tuning_results) == 2
    with app.function(gpu="gb300", image=Image("other")):
        kernel[(1,)](None, 2, 64)
    assert len(kernel.tuning_results) == 3


def test_launch_single_configuration_executes_without_benchmarking():
    class LaunchClient(Client):
        def submit(self, **request):
            if request["function"].__name__ == "execute_winner":
                return SimpleNamespace(call_id="execute", wait=lambda: request["kwargs"]["inputs"])
            return super().submit(**request)

        def get_call_result(self, identity):
            return {"artifacts": [{"name": "compiled-triton", "artifact_id": "art_compiled"}]}

    kernel = NativeTritonKernel(JIT(add))
    with App("single", image=Image("compiler"), client=LaunchClient()).function(gpu="gb300"):
        assert kernel[(1,)](None, 1, 64) is None
    assert "benchmark" not in next(iter(kernel.tuning_results.values()))


def test_launch_accepts_declarative_reset_but_rejects_custom_hooks():
    native = Autotuner(JIT(add), ["X", "N", "BLOCK"], [Config({"BLOCK": 64})], ["N"], ["X"], None)
    from gfaas.triton_compat import reset_arguments

    assert reset_arguments(native) == (["X"], [])
    NativeTritonKernel(native)
    native.pre_hook = lambda *a: None
    with pytest.raises(UnsupportedTritonKernelError, match="callback"):
        NativeTritonKernel(native)


@pytest.mark.parametrize("launch_first", [False, True])
def test_launch_benchmark_autotunes_when_cold_reuses_winner_and_replicates(launch_first):
    import gfaas

    class BenchmarkClient(Client):
        def __init__(self):
            super().__init__()
            self.phases = []
            self.sequence = 0

        def submit(self, **request):
            name = request["function"].__name__
            self.phases.append(name)
            data = request["kwargs"]
            if name == "execute_winner":
                return SimpleNamespace(call_id="execute", wait=lambda: data["inputs"])
            if name in ("benchmark_cycle", "benchmark_selected_replicas", "benchmark_replicas"):
                self.sequence += 1
                variants = json.loads(data["variants"])

                def report(gpu, timing):
                    return {
                        "status": "passed",
                        "gpu_uuid": gpu,
                        "results": [
                            {
                                "id": variant["id"],
                                "status": "measured",
                                "runtime_us": timing,
                                "refined_us": timing,
                                "evaluation": "passed",
                            }
                            for variant in variants
                        ],
                    }

                if name == "benchmark_cycle":
                    result = report("gpu0", 10)
                elif name == "benchmark_selected_replicas":
                    assert request["gpu_count"] == 2
                    result = {"replica_reports": [report("gpu0", 12), report("gpu1", 14)]}
                else:
                    result = {"replica_reports": [report("gpu1", 14), report("gpu2", 13)]}
                return SimpleNamespace(call_id=f"bench_{self.sequence}", wait=lambda: result)
            return super().submit(**request)

        def get_call_result(self, identity):
            return {"artifacts": [{"name": "compiled-triton", "artifact_id": "art_compiled"}]}

    client = BenchmarkClient()
    kernel = NativeTritonKernel(JIT(add), tuning=gfaas.TritonTuning(replication_factor=1))
    assert isinstance(kernel, gfaas.Kernel)
    app = App("bench", image=Image("compiler"), client=client)
    options = gfaas.KernelBenchmark(replication_factor=2)
    with app.function(gpu="gb300"):
        if launch_first:
            kernel[(1,)](None, 1, 64)
        first = gfaas.benchmark(kernel[(1,)], None, 1, 64, options=options)
        second = gfaas.benchmark(kernel[(1,)], None, 1, 64, options=options)
    assert first["autotuned"] is (not launch_first)
    assert first["reused_specialization"] is launch_first
    assert second["reused_specialization"] is True
    assert client.phases.count("compile_batch") == 1
    assert client.phases.count("benchmark_cycle") == 1
    assert client.phases.count("execute_winner") == int(launch_first)
    assert client.phases.count("benchmark_selected_replicas") == 2
    assert client.phases.count("benchmark_replicas") == 0
    assert first["runtime_us"] == 13
    assert {r["gpu_uuid"] for r in first["replicas"]} == {"gpu0", "gpu1"}
    assert first["configuration"] == second["configuration"]
    assert next(iter(kernel.tuning_results.values()))["benchmark"]["best_runtime_us"] == 10
    with pytest.raises(TypeError):
        first["runtime_us"] = 1


def test_launch_benchmark_options_type_check_happens_before_remote_work():
    import gfaas

    kernel = NativeTritonKernel(JIT(add))
    with pytest.raises(TypeError, match="options"):
        gfaas.benchmark(kernel[(1,)], None, 1, 64, options={})


def test_launch_cached_benchmark_requests_only_winners_compiler_shard():
    import gfaas

    class ShardedClient(Client):
        def submit(self, **request):
            name = request["function"].__name__
            data = request["kwargs"]
            if name == "compile_batch":
                (variant,) = json.loads(data["variants"])
                block = variant["constants"]["BLOCK"]
                return SimpleNamespace(
                    call_id=f"compile_{block}",
                    wait=lambda: {"results": [{"id": variant["id"], "status": "compiled"}]},
                )
            if name in ("benchmark_cycle", "benchmark_selected_replicas"):
                variants = json.loads(data["variants"])
                report = {
                    "status": "passed",
                    "gpu_uuid": "gpu0",
                    "results": [
                        {
                            "id": v["id"],
                            "status": "measured",
                            "runtime_us": v["constants"]["BLOCK"],
                            "refined_us": v["constants"]["BLOCK"],
                            "evaluation": "passed",
                        }
                        for v in variants
                    ],
                }
                if name == "benchmark_selected_replicas":
                    assert [a.artifact_id for a in data["artifacts"]] == ["art_compile_32"]
                    report = {"replica_reports": [report]}
                return SimpleNamespace(call_id=name, wait=lambda: report)
            return super().submit(**request)

        def get_call_result(self, identity):
            return {"artifacts": [{"name": "compiled-triton", "artifact_id": "art_" + identity}]}

    client = ShardedClient()
    native = Autotuner(
        JIT(add), ["X", "N", "BLOCK"], [Config({"BLOCK": 32}), Config({"BLOCK": 64})], ["N"], [], []
    )
    kernel = NativeTritonKernel(
        native,
        variants_per_job=1,
        max_concurrent_jobs=1,
        tuning=gfaas.TritonTuning(replication_factor=1),
    )
    app = App("bench", image=Image("compiler"), client=client)
    with app.function(gpu="gb300"):
        result = gfaas.benchmark(
            kernel[(1,)], None, 64, options=gfaas.KernelBenchmark(replication_factor=1)
        )
    assert result["configuration"]["constants"]["BLOCK"] == 32
    assert len(result["benchmark_call_ids"]) == 1
