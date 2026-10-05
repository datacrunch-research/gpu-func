from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gfaas import (
    App,
    Image,
    KernelBenchmark,
    ThunderKittensConfig,
    ThunderKittensKernel,
    TritonTuning,
    benchmark,
)
from gfaas.cuda_kernel_runner import load_binaries, scalar_value
from gfaas.thunderkittens_kernel import dimensions3


@pytest.fixture
def headers(tmp_path):
    (tmp_path / "kittens.cuh").write_text("// fixture headers")
    return tmp_path


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block": (0,)},
        {"block": (1025,)},
        {"block": (True,)},
        {"shared_memory": -1},
        {"defines": {"X": "4"}},
        {"defines": {"X;bad": 1}},
        {"defines": {"KITTENS_SM103": 1}},
        {"defines": {"X": float("nan")}},
    ],
)
def test_configuration_rejects_invalid_launch_or_macros(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ThunderKittensConfig(**kwargs)


def test_configuration_and_header_snapshot_are_immutable(headers):
    values = {"TILE": 64}
    config = ThunderKittensConfig(values, block=(32,))
    values["TILE"] = 128
    assert config.defines["TILE"] == 64
    with pytest.raises(TypeError):
        config.defines["TILE"] = 128
    first = ThunderKittensKernel("source", "launch", signature={"X": "pointer"}, headers=headers)
    second = ThunderKittensKernel("source", "launch", signature={"X": "pointer"}, headers=headers)
    assert first.headers_sha256 == second.headers_sha256
    (headers / "kittens.cuh").write_text("// edited")
    assert first.headers_sha256 == second.headers_sha256
    third = ThunderKittensKernel("source", "launch", signature={"X": "pointer"}, headers=headers)
    assert third.headers_sha256 != first.headers_sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entrypoint_kind": "bad"},
        {"signature": {"X": "float16"}},
        {"reset_to_zero": ["N"]},
        {"nvcc_flags": ["-o", "/tmp/other"]},
        {"nvcc_flags": ["-arch=sm_90"]},
        {"variants_per_job": 0},
        {"max_concurrent_jobs": 0},
    ],
)
def test_invalid_api_fails_before_remote_work(headers, kwargs):
    params = {"signature": {"X": "pointer", "N": "int32"}, "headers": headers, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        ThunderKittensKernel("source", "launch", **params)


def test_scalar_ranges_are_checked():
    assert scalar_value("int32", -1).value == -1
    with pytest.raises(ValueError):
        scalar_value("uint32", -1)
    with pytest.raises(ValueError):
        scalar_value("int64", 2**63)
    with pytest.raises(ValueError):
        scalar_value("int32", True)
    assert dimensions3((2, 3)) == (2, 3, 1)


def test_loads_only_selected_binary_and_checks_its_integrity(tmp_path):
    data = b"winning cubin"
    identity = "winner"
    row = {
        "id": identity,
        "status": "compiled",
        "extension": ".cubin",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest = {
        "schema": "vfunc.cuda-kernel/v1",
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "kernel_name": "launch",
        "target": {"arch": 103},
        "results": [row, {"id": "other", "status": "compiled"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "winner.cubin").write_bytes(data)
    # No other.cubin exists; it must not be opened.
    result = load_binaries([tmp_path], "source", "launch", {"arch": 103}, {identity})
    assert result[identity]["binary"] == data
    (tmp_path / "winner.cubin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        load_binaries([tmp_path], "source", "launch", {"arch": 103}, {identity})


class FakeClient:
    def __init__(self):
        self.phases = []
        self.rows = {}
        self.sequence = 0
        self.uploads = 0

    def upload_artifact_file(self, path, **kwargs):
        self.uploads += 1
        with tarfile.open(fileobj=io.BytesIO(Path(path).read_bytes()), mode="r:gz") as tar:
            assert tar.extractfile("kittens.cuh").read() == b"// fixture headers"
        return {"id": "art_headers"}

    def submit(self, **request):
        phase = request["function"].__name__
        self.phases.append(phase)
        data = request["kwargs"]
        self.sequence += 1
        identity = f"call_{self.sequence}"
        if phase == "probe_target":
            output = {"backend": "cuda", "arch": 103, "warp_size": 32}
        elif phase == "compile_batch":
            variants = json.loads(data["variants"])
            output = {"results": [{"id": v["id"], "status": "compiled"} for v in variants]}
            self.rows[identity] = variants
        elif phase == "execute":
            output = data["inputs"]
        else:
            variants = json.loads(data["variants"])

            def report(gpu):
                return {
                    "status": "passed",
                    "gpu_uuid": f"gpu{gpu}",
                    "results": [
                        {
                            "id": v["id"],
                            "status": "measured",
                            "runtime_us": 10.0 + gpu,
                            "refined_us": 11.0,
                            "evaluation": "passed",
                        }
                        for v in variants
                    ],
                }

            output = (
                {"replica_reports": [report(i) for i in range(data["device_count"])]}
                if phase == "benchmark_replicas"
                else report(0)
            )
        return SimpleNamespace(call_id=identity, wait=lambda: output)

    def get_call_result(self, identity):
        return {"artifacts": [{"name": "compiled-cuda-kernel", "artifact_id": "art_" + identity}]}


def test_cold_and_cached_benchmark_one_job_then_execution(headers):
    client = FakeClient()
    app = App("test", image=Image("image"), client=client)
    kernel = ThunderKittensKernel(
        "source",
        "launch",
        signature={"X": "pointer", "N": "int32"},
        headers=headers,
        configs=[ThunderKittensConfig({"TILE": 64}), ThunderKittensConfig({"TILE": 128})],
        tuning=TritonTuning(replication_factor=1),
    )
    with app.function(gpu="gb300"):
        cold = benchmark(kernel[(1,)], None, 64, options=KernelBenchmark(replication_factor=3))
        warm = benchmark(kernel[(1,)], None, 64, options=KernelBenchmark(replication_factor=3))
        kernel[(1,)](None, 64)
    assert cold["autotuned"] and not cold["reused_specialization"]
    assert warm["reused_specialization"]
    assert client.phases.count("compile_batch") == 1
    assert client.phases.count("benchmark_cycle") == 1
    assert client.phases.count("benchmark_replicas") == 2
    assert client.phases.count("execute") == 1
    assert client.uploads == 1
    assert warm["runtime_us"] == 11
    assert len({r["gpu_uuid"] for r in warm["replicas"]}) == 3
    assert len(warm["benchmark_call_ids"]) == 1
    assert next(iter(kernel.tuning_results.values()))["benchmark"]["best_runtime_us"] == 10
    with pytest.raises(TypeError):
        warm["runtime_us"] = 0


def test_nvcc_discovery_uses_image_package_without_path(monkeypatch, tmp_path):
    import importlib.metadata

    from gfaas.cuda_kernel_runner import find_nvcc

    compiler = tmp_path / "nvcc"
    compiler.write_text("fixture")
    compiler.chmod(0o700)
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        "gfaas.cuda_runner._which", lambda name: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            files=[Path("nvidia/cu13/bin/nvcc")], locate_file=lambda name: compiler
        ),
    )
    assert find_nvcc() == str(compiler)


def test_entrypoint_abi_checks_cover_host_bridge_and_scalar_types():
    from gfaas.cuda_kernel_runner import entrypoint_checks

    checks = entrypoint_checks(
        "launch", {"X": "pointer", "N": "int32", "alpha": "float32"}, "launcher"
    )
    assert "== 11" in checks
    assert "vFunc argument 0 ABI mismatch" in checks
    assert "vFunc argument 1 ABI mismatch" in checks
    assert "vFunc argument 2 ABI mismatch" in checks
    assert "cudaStream_t" in checks
    raw = entrypoint_checks("kernel", {"X": "pointer"}, "kernel")
    assert "== 1" in raw and "result, void" in raw
