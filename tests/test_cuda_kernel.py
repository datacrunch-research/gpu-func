import hashlib
import json
from types import SimpleNamespace

import pytest

from gfaas import App, CUDAConfig, CUDAKernel, Image, KernelTuning, benchmark
from gfaas.cuda_kernel_runner import load_binaries, scalar_value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block": (0,)},
        {"block": (1025,)},
        {"block": (16, 16, 16)},
        {"shared_memory": -1},
        {"defines": {"invalid name": 1}},
        {"defines": {"B": float("inf")}},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        CUDAConfig(**kwargs)


def test_configuration_is_immutable_and_signature_checked():
    d = {"B": 128}
    config = CUDAConfig(defines=d)
    d["B"] = 32
    assert config.defines["B"] == 128
    with pytest.raises(TypeError):
        config.defines["B"] = 32
    with pytest.raises(ValueError):
        CUDAKernel("source", name="kernel", signature={"x": "int16"})
    with pytest.raises(ValueError):
        CUDAKernel("source", name="kernel", signature={"n": "int32"}, reset_to_zero=("n",))
    with pytest.raises(ValueError):
        CUDAKernel("source", name="kernel", signature={"n": "int32"}, nvcc_flags=("-o",))


def test_scalar_abi_does_not_truncate():
    assert scalar_value("int64", 2**40).value == 2**40
    assert scalar_value("uint32", 2**32 - 1).value == 2**32 - 1
    for kind, value in [("int32", 2**31), ("uint32", -1), ("int64", True), ("float32", "1")]:
        with pytest.raises((ValueError, TypeError)):
            scalar_value(kind, value)


def test_only_requested_binary_loaded_and_checksum_verified(tmp_path):
    data = b"compiled cubin"
    (tmp_path / "winner.cubin").write_bytes(data)
    # Other compiled configurations need not even have staged binary files.
    manifest = {
        "schema": "vfunc.cuda-kernel/v1",
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "kernel_name": "entry",
        "target": {"arch": 103},
        "results": [
            {"id": "winner", "status": "compiled", "sha256": hashlib.sha256(data).hexdigest()},
            {"id": "other", "status": "compiled", "sha256": "ignored"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert (
        load_binaries([tmp_path], "source", "entry", {"arch": 103}, {"winner"})["winner"]["binary"]
        == data
    )
    with pytest.raises(RuntimeError, match="match"):
        load_binaries([tmp_path], "different", "entry", {"arch": 103}, {"winner"})
    with pytest.raises(RuntimeError, match="Missing"):
        load_binaries([tmp_path], "source", "entry", {"arch": 103}, {"missing"})
    (tmp_path / "winner.cubin").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        load_binaries([tmp_path], "source", "entry", {"arch": 103}, {"winner"})


def test_compile_tune_benchmark_cache_and_execute_share_environment():
    class Client:
        def __init__(self):
            self.calls = []

        def submit(self, **request):
            function, data = request["function"].__name__, request["kwargs"]
            self.calls.append((function, request))
            variants = json.loads(data.get("variants", "[]"))
            if function == "probe_target":
                result = {"backend": "cuda", "arch": 103, "warp_size": 32}
            elif function == "compile_batch":
                assert request["gpu_count"] == 0
                result = {"results": [{"id": v["id"], "status": "compiled"} for v in variants]}
            elif function == "execute":
                result = data["inputs"]
            else:

                def report(index):
                    return {
                        "gpu_uuid": f"gpu{index}",
                        "status": "passed",
                        "results": [
                            {
                                "id": v["id"],
                                "status": "measured",
                                "runtime_us": 10 + index,
                                "refined_us": 10,
                                "evaluation": "passed",
                            }
                            for v in variants
                        ],
                    }

                result = (
                    {
                        "status": "replica_bundle",
                        "replica_reports": [
                            (
                                {"status": "duplicate_gpu", "gpu_uuid": f"gpu{i}"}
                                if f"gpu{i}" in data.get("excluded_gpu_uuids", [])
                                else report(i)
                            )
                            for i in range(3)
                        ],
                    }
                    if data.get("device_count")
                    else report(0)
                )
            return SimpleNamespace(call_id=f"call{len(self.calls)}", wait=lambda: result)

        def get_call_result(self, identity):
            return {
                "artifacts": [{"name": "compiled-cuda-kernel", "artifact_id": "art_" + identity}]
            }

    client = Client()
    app = App("cuda-test", image=Image("image"), client=client)
    kernel = CUDAKernel(
        'extern "C" __global__ void entry(float* x, int n) {}',
        name="entry",
        signature={"x": "pointer", "n": "int32"},
        configs=[CUDAConfig((128,)), CUDAConfig((256,))],
        tuning=KernelTuning(replication_factor=3),
    )
    with app.function(gpu="gb300", timeout=180):
        cold = benchmark(kernel[(1,)], None, 1)
        tuned = next(iter(kernel.tuning_results.values()))
        warm = benchmark(kernel[(1,)], None, 1)
        kernel[(1,)](None, 1)
    with app.function(gpu="gb300", env={"CUDA_DEVICE_MAX_CONNECTIONS": "1"}):
        changed = benchmark(kernel[(1,)], None, 1)
    assert not changed["reused_specialization"]
    assert changed["specialization"] != cold["specialization"]
    assert cold["autotuned"] and not cold["reused_specialization"]
    assert warm["reused_specialization"] and not warm["autotuned"]
    assert warm["runtime_us"] == 11
    assert len({r["gpu_uuid"] for r in warm["replicas"]}) == 3
    assert tuned is kernel.tuning_results[cold["specialization"]]
    assert len([c for c, _ in client.calls if c == "compile_batch"]) == 2
    assert len([c for c, _ in client.calls if c == "execute"]) == 1
    assert all(request["image"].name == "image" for _, request in client.calls)
    with pytest.raises(TypeError):
        warm["runtime_us"] = 2
    with pytest.raises(TypeError):
        kernel[(1,)](None)  # Missing required argument fails before submission.


def test_cpu_compilation_discovers_wheel_nvcc_and_retains_variant_failures(monkeypatch, tmp_path):
    import gfaas.cuda_kernel_runner as runner

    toolkit = tmp_path / "wheel/bin/nvcc"
    toolkit.parent.mkdir(parents=True)
    toolkit.write_text("compiler")
    toolkit.chmod(0o755)
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "gfaas.cuda_runner._which", lambda _: (_ for _ in ()).throw(RuntimeError("not on PATH"))
    )
    monkeypatch.setattr("gfaas.cuda_runner._host_cxx_flags", lambda: ["-ccbin", "g++"])
    monkeypatch.setattr("gfaas.cuda_runner._subprocess_env", lambda _: {})
    distribution = SimpleNamespace(
        metadata={"Name": "nvidia-cuda-nvcc"},
        files=["wheel/bin/nvcc"],
        locate_file=lambda entry: tmp_path / entry,
    )
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [distribution])
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        broken = "-DBROKEN=1" in command
        if not broken:
            from pathlib import Path

            Path(command[-1]).write_bytes(b"cubin")
        return SimpleNamespace(returncode=int(broken), stderr="compile error" if broken else "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "nvcc version")
    result = runner.compile_batch(
        source="source",
        kernel_name="entry",
        variants=json.dumps([{"id": "ok", "defines": {}}, {"id": "bad", "defines": {"BROKEN": 1}}]),
        target={"arch": 103},
        workers=2,
        nvcc_flags=[],
    )
    assert {r["id"]: r["status"] for r in result["results"]} == {"ok": "compiled", "bad": "failed"}
    assert all(c[0] == str(toolkit) and "--cubin" in c and "-arch=sm_103" in c for c in commands)
    assert (tmp_path / "compiled-cuda-kernel/manifest.json").exists()


@pytest.mark.parametrize("shared_memory", [0, 512])
@pytest.mark.parametrize("active_context", [False, True])
def test_driver_launch_preserves_pointer_scalar_widths_and_stream(
    monkeypatch, shared_memory, active_context
):
    import ctypes
    import sys

    import gfaas.cuda_kernel_runner as runner

    captured = {}
    initialized = []
    attributes = []

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    def output(pointer, value):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = value
        return 0

    def launch(function, *args):
        captured["function"] = function.value
        captured["dimensions"] = args[:6]
        captured["stream"] = args[7].value
        params = args[8]
        captured["pointer"] = ctypes.cast(params[0], ctypes.POINTER(ctypes.c_void_p)).contents.value
        captured["integer"] = ctypes.cast(params[1], ctypes.POINTER(ctypes.c_int64)).contents.value
        captured["double"] = ctypes.cast(params[2], ctypes.POINTER(ctypes.c_double)).contents.value
        return 0

    cuda = SimpleNamespace(
        cuCtxGetCurrent=Function(lambda pointer: output(pointer, int(active_context))),
        cuFuncSetAttribute=Function(
            lambda function, attribute, value: attributes.append((attribute, value)) or 0
        ),
        cuModuleLoadData=Function(lambda pointer, *a: output(pointer, 42)),
        cuModuleGetFunction=Function(lambda pointer, *a: output(pointer, 43)),
        cuLaunchKernel=Function(launch),
    )
    monkeypatch.setattr(runner.ctypes, "CDLL", lambda _: cuda)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            empty=lambda *args, **kwargs: initialized.append((args, kwargs)),
            cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=2**40)),
        ),
    )
    tensor = SimpleNamespace(is_cuda=True, data_ptr=lambda: 2**45 + 16)
    variant = {
        "signature": {"x": "pointer", "n": "int64", "scale": "float64"},
        "defines": {},
        "block": [128, 1, 1],
        "shared_memory": shared_memory,
    }
    candidate = runner.load_candidate(b"cubin", "entry", variant, lambda meta: (1,))
    candidate(tensor, 2**40, 1.25)
    assert bool(initialized) is (not active_context)
    assert attributes == ([(8, shared_memory)] if shared_memory else [])
    assert captured == {
        "function": 43,
        "dimensions": (1, 1, 1, 128, 1, 1),
        "stream": 2**40,
        "pointer": 2**45 + 16,
        "integer": 2**40,
        "double": 1.25,
    }


def test_compile_only_has_no_input_or_tuning_dependency():
    calls = []

    class Client:
        def submit(self, **request):
            calls.append(request)
            name = request["function"].__name__
            result = (
                {"backend": "cuda", "arch": 103, "warp_size": 32}
                if name == "probe_target"
                else {
                    "results": [
                        {"id": v["id"], "status": "compiled"}
                        for v in json.loads(request["kwargs"]["variants"])
                    ]
                }
            )
            return SimpleNamespace(call_id=f"call{len(calls)}", wait=lambda: result)

        def get_call_result(self, call_id):
            return {
                "artifacts": [{"name": "compiled-cuda-kernel", "artifact_id": "art_" + call_id}]
            }

    kernel = CUDAKernel(
        'extern "C" __global__ void k(int n) {}',
        name="k",
        signature={"n": "int32"},
        configs=[CUDAConfig((128,)), CUDAConfig((256,))],
        variants_per_job=1,
    )
    app = App("compile-only", image=Image("image"), client=Client())
    with app.function(gpu="gb300"):
        result = kernel.compile()
    assert len(result["artifacts"]) == 2 and len(result["results"]) == 2
    assert [r["gpu_count"] for r in calls] == [1, 0, 0]
    assert not kernel.tuning_results
    assert all("inputs" not in r["kwargs"] for r in calls)


def test_cuda_benchmark_pipeline_runs_without_triton(monkeypatch, tmp_path):
    """CUDA must use prepared cubins through the shared timing/ranking pipeline."""
    import sys

    import cloudpickle

    from gfaas import cuda_kernel_runner, triton_quick_runner

    target = {"arch": 103}
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            init=lambda: None,
            get_device_properties=lambda _: SimpleNamespace(uuid="gpu-0"),
            synchronize=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(triton_quick_runner, "probe_target", lambda _: target)
    monkeypatch.setattr(
        triton_quick_runner, "l2_flush_buffer", lambda _: SimpleNamespace(numel=lambda: 256)
    )
    ring = SimpleNamespace(sets=[0, 1], footprint_bytes=256, allocated_bytes=512)
    monkeypatch.setattr(triton_quick_runner, "InputRing", lambda *args: ring)
    monkeypatch.setattr(cloudpickle, "loads", lambda _: (None, None, None, None))
    records = {"candidate": {"status": "compiled", "binary": b"cubin"}}
    prepared = []

    def prepare(actual, variants, name, grid, actual_ring, reset):
        assert actual is records and actual_ring is ring
        assert variants == [{"id": "candidate"}] and name == "add"
        prepared.append(name)
        return [({"id": "candidate"}, lambda: None)]

    monkeypatch.setattr(cuda_kernel_runner, "prepare_entries", prepare)
    monkeypatch.setattr(
        triton_quick_runner,
        "ring_trials",
        lambda candidates, counts, *args: [[5.0] * n for n in counts],
    )
    monkeypatch.setattr(triton_quick_runner, "final_benchmark", lambda *args: {"runtime_us": 4.0})
    result = triton_quick_runner.benchmark_cycle(
        source="CUDA source",
        kernel_name="add",
        variants=json.dumps([{"id": "candidate"}]),
        artifacts=[],
        target=target,
        triton_version="",
        metadata={},
        callbacks=b"callbacks",
        policy=KernelTuning().request(),
        backend="cuda",
        prepared_cache=(str(tmp_path), records),
    )
    assert prepared == ["add"]
    assert result["status"] == "passed"
    assert result["best_id"] == "candidate"
    assert result["best_runtime_us"] == 4.0
    assert result["results"][0]["pilot_us"] == 5.0
    assert result["results"][0]["evaluation"] == "skipped"
