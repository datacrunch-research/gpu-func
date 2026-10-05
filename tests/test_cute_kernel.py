import functools
import hashlib
import json
import types
from types import SimpleNamespace

import pytest

import gfaas
from gfaas.app import App
from gfaas.cute_backend import load_artifacts
from gfaas.cute_compat import source_bundle
from gfaas.cute_kernel import CuteDSLKernel
from gfaas.image import Image

cute = types.ModuleType("cutlass.cute")
cutlass = types.ModuleType("cutlass")


class Constexpr:
    pass


cutlass.Constexpr = Constexpr


def jit(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raise AssertionError("Client must never launch a native kernel")

    return wrapped


cute.jit = jit
cute.kernel = jit


@cute.kernel
def device(a, BLOCK: cutlass.Constexpr):
    pass


@cute.jit
def host(a, BLOCK: cutlass.Constexpr):
    device(a, BLOCK)


@cute.jit
def defaults(a, BLOCK: cutlass.Constexpr = 64, scale: float = 2.0):
    device(a, BLOCK)


class CallableKernel:
    def __init__(self):
        self.block = 128

    @cute.jit
    def __call__(self, a):
        device(a, self.block)


class FakeClient:
    def __init__(self):
        self.phases = []
        self.requests = []

    def submit(self, **request):
        self.requests.append(request)
        name, data = request["function"].__name__, request["kwargs"]
        self.phases.append(name)
        if name == "probe_environment":
            report = {
                "target": {"backend": "cuda", "arch": 103, "warp_size": 32},
                "cute_version": "4.8.0",
                "cpu_arch": "aarch64",
            }
        elif name == "compile_batch":
            report = {
                "results": [
                    {"id": v["id"], "status": "compiled"} for v in json.loads(data["variants"])
                ]
            }
        elif name == "execute_winner":
            report = data["inputs"]
        else:
            variants = json.loads(data["variants"])

            def result(gpu, runtime):
                return {
                    "status": "passed",
                    "gpu_uuid": gpu,
                    "results": [
                        {
                            "id": v["id"],
                            "status": "measured",
                            "evaluation": "passed",
                            "refined_us": runtime,
                            "runtime_us": runtime,
                        }
                        for v in variants
                    ],
                }

            report = (
                {"replica_reports": [result("gpu0", 10), result("gpu1", 11), result("gpu2", 12)]}
                if name == "benchmark_selected_replicas"
                else {"replica_reports": [result("gpu1", 11), result("gpu2", 12)]}
                if name == "benchmark_replicas"
                else result("gpu0", 10)
            )
        return SimpleNamespace(call_id=f"call{len(self.phases)}", wait=lambda: report)

    def get_call_result(self, identity):
        return {"artifacts": [{"name": "compiled-cute", "artifact_id": "art_cute"}]}


@pytest.fixture(autouse=True)
def version(monkeypatch):
    monkeypatch.setattr("gfaas.cute_backend.version", lambda: "4.8.0")


def test_source_bundles_device_dependencies_and_callable_state():
    source, name = source_bundle(host)
    assert name == "host"
    assert "def device" in source and "import cutlass.cute as cute" in source
    source, name = source_bundle(CallableKernel())
    assert "class CallableKernel" in source and f"{name}.block = 128" in source


@pytest.mark.parametrize(
    "options",
    [
        {"configurations": []},
        {"configurations": [{"missing": 1}]},
        {"configurations": [{"a": 1}]},
        {"max_concurrent_jobs": 0},
        {"compile_options": "--gpu-arch sm_80"},
        {"reset_to_zero": ["missing"]},
    ],
)
def test_bad_definition_rejected_before_submission(options):
    with pytest.raises((ValueError, TypeError)):
        CuteDSLKernel(host, **options)


def test_device_entry_rejected():
    with pytest.raises(ValueError, match="host entry"):
        CuteDSLKernel(device)


@pytest.mark.parametrize("launch_first", [False, True])
def test_cold_and_cached_calls_share_tuning_and_distinct_gpu_benchmarks(launch_first):
    client = FakeClient()
    app = App("cute", image=Image("test"), client=client)
    kernel = CuteDSLKernel(
        host,
        configurations=[{"BLOCK": 64}, {"BLOCK": 128}],
        tuning=gfaas.KernelTuning(replication_factor=3),
    )
    with app.function(gpu="gb300"):
        if launch_first:
            kernel(1)
        first = gfaas.benchmark(kernel, 1)
        second = gfaas.benchmark(kernel, 1)
        kernel(1)
    assert first["autotuned"] is not launch_first
    assert second["reused_specialization"]
    assert first["runtime_us"] == 11
    assert {r["gpu_uuid"] for r in first["replicas"]} == {"gpu0", "gpu1", "gpu2"}
    assert client.phases.count("compile_batch") == 2
    assert client.phases.count("probe_environment") == 1
    assert client.phases.count("benchmark_cycle") == 1
    assert client.phases.count("benchmark_selected_replicas") == 2
    assert client.phases.count("execute_winner") == 1 + int(launch_first)
    assert next(iter(kernel.tuning_results.values()))["benchmark"]["best_runtime_us"] == 11
    assert first["configuration"] == second["configuration"]
    with pytest.raises(TypeError):
        first["runtime_us"] = 0
    assert all(
        r["gpu_count"] == 0 for r in client.requests if r["function"].__name__ == "compile_batch"
    )


def test_config_can_override_a_default_and_runtime_defaults_are_preserved():
    client = FakeClient()
    app = App("cute", image=Image("test"), client=client)
    kernel = CuteDSLKernel(defaults, configurations=[{"BLOCK": 128}])
    with app.function(gpu="gb300"):
        kernel(1)
    request = next(r for r in client.requests if r["function"].__name__ == "compile_batch")
    variant = json.loads(request["kwargs"]["variants"])[0]
    assert variant["constants"] == {"BLOCK": 128}
    assert variant["defaults"] == {"scale": 2.0}


def test_selected_object_checks_identity_and_integrity(tmp_path):
    source = "source"
    variant = {"id": "a" * 64}
    root = tmp_path / variant["id"]
    root.mkdir()
    obj = root / "kernel.o"
    obj.write_bytes(b"binary")
    target = {"backend": "cuda", "arch": 103, "warp_size": 32}
    manifest = {
        "schema": "vfunc.cute-compilation/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "target": target,
        "cute_version": "4.8.0",
        "results": [
            {
                "id": variant["id"],
                "status": "compiled",
                "sha256": hashlib.sha256(b"binary").hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_artifacts([tmp_path], source, target, "4.8.0", [variant]) == {variant["id"]: obj}
    obj.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="checksum"):
        load_artifacts([tmp_path], source, target, "4.8.0", [variant])


def test_nested_tile_shapes_roundtrip_without_turning_tuples_into_lists():
    from gfaas.cute_compat import decode_constants, encode_constant

    value = ((64, 128), [2, (3, 4)])
    assert decode_constants(json.loads(json.dumps(encode_constant(value)))) == value
    kernel = CuteDSLKernel(host, configurations=[{"BLOCK": (64, 128)}])
    assert kernel.configurations[0]["BLOCK"] == (64, 128)


def test_exported_runtime_status_is_not_a_python_host_return(monkeypatch, tmp_path):
    import sys

    from gfaas.cute_backend import candidate

    calls = []
    loaded = SimpleNamespace(vfunc_kernel=lambda *args: calls.append(args) or 0)
    runtime = types.ModuleType("cutlass.cute.runtime")
    runtime.from_dlpack = lambda value, **kwargs: value
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    monkeypatch.setitem(sys.modules, "cutlass.cute.runtime", runtime)
    monkeypatch.setattr(
        cute, "runtime", SimpleNamespace(load_module=lambda _: loaded), raising=False
    )
    cuda = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    driver = types.ModuleType("cuda.bindings.driver")
    cuda.bindings = bindings
    bindings.driver = driver
    for name, module in (
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        ("cuda.bindings.driver", driver),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    launch = candidate(tmp_path / "kernel.o", {"signature": {"a": "scalar"}}, ["a"])
    assert launch(3) is None
    assert calls == [(3,)]
    assert launch.module is loaded


def test_single_job_cute_replication_verifies_objects_once(monkeypatch, tmp_path):
    import sys
    from contextlib import nullcontext

    from gfaas import cute_backend, triton_quick_runner

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(device_count=lambda: 3, device=lambda d: nullcontext(d))
    monkeypatch.setitem(sys.modules, "torch", torch)
    prepared = {"winner": tmp_path / "kernel.o"}
    calls = []

    def load(artifacts, source, target, version, variants):
        assert version == "4.8.0" and variants == [{"id": "winner"}]
        calls.append("load")
        return prepared

    def measure(**kwargs):
        assert kwargs["prepared_cache"][1] is prepared
        assert kwargs["backend"] == "cute" and kwargs["triton_version"] == "4.8.0"
        assert kwargs["final_only"]
        calls.append("measure")
        return {"status": "passed"}

    monkeypatch.setattr(cute_backend, "load_artifacts", load)
    monkeypatch.setattr(triton_quick_runner, "benchmark_cycle", measure)
    report = triton_quick_runner.benchmark_selected_replicas(
        device_count=3,
        variants='[{"id":"winner"}]',
        artifacts=[],
        source="source",
        target={},
        cute_version="4.8.0",
        final_only=True,
    )
    assert calls == ["load", "measure", "measure", "measure"]
    assert len(report["replica_reports"]) == 3
    assert report["prepared_configurations"] == 1
