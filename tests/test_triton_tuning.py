from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gfaas import TritonCandidate, TritonCase, spawn_triton_tuning
from gfaas import triton_tuning_runner as runner


class FakeClient:
    def __init__(self) -> None:
        self.submission: dict[str, Any] = {}

    def submit(self, **kwargs: Any) -> object:
        self.submission = kwargs
        return object()


def _request(client: FakeClient) -> object:
    return spawn_triton_tuning(
        source="def kernel(): pass",
        kernel_name="kernel",
        signature={"X": "*fp32", "BLOCK": "constexpr"},
        candidates=[TritonCandidate("small", {"BLOCK": 64})],
        cases=[TritonCase("n=64", {"n": 64})],
        target_arch=103,
        image="triton-runtime",
        gpu="GB300",
        client=client,  # type: ignore[arg-type]
    )


def test_tuning_submits_one_call_with_cpu_compile_and_one_gpu_stage() -> None:
    client = FakeClient()
    result = _request(client)
    assert result is not None
    submission = client.submission
    assert submission["gpu_count"] == 1
    assert submission["kwargs"]["triton_version"] == "3.6.0"
    stages = submission["stages"]
    assert [stage.name for stage in stages] == ["compile", "tune"]
    assert stages[0].resources["gpu"]["count"] == 0
    assert stages[0].resources["cpu_millicores"] == 4000
    assert stages[0].outputs[0].name == "compiled-triton"
    assert stages[1].resources["gpu"]["count"] == 1
    assert stages[1].artifacts[0].from_stage == "compile"


def test_tuning_rejects_duplicate_keys_and_cpu_oversubscription() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="tuning keys must be distinct"):
        spawn_triton_tuning(
            source="pass",
            kernel_name="kernel",
            signature={"X": "*fp32"},
            candidates=[TritonCandidate("one")],
            cases=[TritonCase("same"), TritonCase("same")],
            target_arch=103,
            image="triton-runtime",
            gpu="GB300",
            client=client,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="CPU stage resource envelope"):
        spawn_triton_tuning(
            source="pass",
            kernel_name="kernel",
            signature={"X": "*fp32"},
            candidates=[TritonCandidate("one")],
            cases=[TritonCase("one")],
            target_arch=103,
            image="triton-runtime",
            gpu="GB300",
            compile_workers=4,
            compile_cpu_millicores=2000,
            client=client,  # type: ignore[arg-type]
        )
    assert client.submission == {}


def test_transferred_cache_is_relocated_and_hash_checked(monkeypatch: Any, tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "art_demo"
    source_cache = artifact / "cache" / "key"
    source_cache.mkdir(parents=True)
    binary = source_cache / "kernel.cubin"
    binary.write_bytes(b"cubin")
    group = source_cache / "__grp__kernel.json"
    group.write_text(json.dumps({"child_paths": {"kernel.cubin": str(binary)}}))
    source = "kernel source"
    signature = {"X": "*fp32"}
    cases = [{"key": "case", "params": {}, "constexprs": {}}]
    candidates = [{"name": "candidate", "constexprs": {}, "num_warps": 4, "num_stages": 2}]
    manifest = {
        "schema": "vfunc.triton-tuning/v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "triton_version": "3.6.0",
        "target_arch": 103,
        "target_gpu_pool": None,
        "build_image": {"name": None, "digest": None},
        "signature": signature,
        "candidates": candidates,
        "cases": cases,
        "records": [{"key": "case", "candidate": "candidate", "compile_ms": 1.0}],
        "cache_files": runner._cache_files(artifact / "cache"),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("GFAAS_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("GFAAS_COMPILED_TRITON_ARTIFACT_ID", "art_demo")
    destination = tmp_path / "local-cache"
    runner._load_artifact(source, signature, candidates, cases, 103, "3.6.0", destination)
    transferred = json.loads((destination / "key" / "__grp__kernel.json").read_text())
    assert transferred["child_paths"]["kernel.cubin"] == str(destination / "key" / "kernel.cubin")

    shutil.rmtree(destination)
    binary.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner._load_artifact(source, signature, candidates, cases, 103, "3.6.0", destination)


def test_all_failed_candidates_return_errors_without_a_winner(
    monkeypatch: Any, tmp_path: Path
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _index: (10, 3),
        get_device_properties=lambda _index: SimpleNamespace(uuid="GPU-one"),
        get_device_name=lambda _index: "GB300",
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="13.0"))
    )
    monkeypatch.setattr(runner, "_driver_version", lambda: 13000)
    cases = [{"key": "n=64", "params": {}, "constexprs": {}}]
    candidates = [{"name": "invalid", "constexprs": {}, "num_warps": 4, "num_stages": 2}]
    manifest = {
        "records": [
            {"key": "n=64", "candidate": "invalid", "compile_ms": 3.0, "error": "invalid kernel"}
        ]
    }
    monkeypatch.setattr(runner, "_load_artifact", lambda *_args: manifest)
    monkeypatch.setattr(
        runner,
        "_triton",
        lambda *_args: SimpleNamespace(
            knobs=SimpleNamespace(compilation=SimpleNamespace(listener=None))
        ),
    )
    monkeypatch.setattr(
        runner,
        "_source_module",
        lambda *_args: SimpleNamespace(kernel=object(), make_inputs=lambda _case: {}),
    )
    report = runner.tune_stage(
        source="pass",
        kernel_name="kernel",
        signature={},
        candidates=candidates,
        cases=cases,
        target_arch=103,
        triton_version="3.6.0",
        compile_workers=1,
        warmup=0,
        trials=1,
        max_adaptive_candidates=0,
    )
    assert report["status"] == "failed"
    assert report["cases"][0]["winner"] is None
    assert report["cases"][0]["candidates"][0]["error"] == "invalid kernel"


@pytest.mark.parametrize("valid", [True, False])
def test_benchmark_resets_each_launch_and_restores_after_candidate(valid: bool) -> None:
    events: list[str] = []
    inputs = {"X": object()}
    case = {"key": "case", "params": {}, "constexprs": {}}
    candidate = {"name": "candidate", "constexprs": {}, "num_warps": 4, "num_stages": 2}

    class Compiled:
        def __getitem__(self, grid: tuple[int, int, int]) -> Any:
            assert grid == (1, 1, 1)

            def launch(*args: Any) -> None:
                assert args == (inputs["X"],)
                events.append("launch")

            return launch

    class Event:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _end: Any) -> float:
            return 0.5

    def validate(_case: Any, _inputs: Any) -> bool:
        events.append("validate")
        return valid

    module = SimpleNamespace(
        grid=lambda _case, _candidate: (1,),
        reset_inputs=lambda _case, _inputs: events.append("reset"),
        restore_inputs=lambda _case, _inputs: events.append("restore"),
        validate=validate,
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: None))
    kernel = SimpleNamespace(arg_names=["X"])
    if valid:
        assert runner._benchmark(
            Compiled(), module, case, candidate, kernel, inputs, 1, 2, torch
        ) == [0.5, 0.5]
        assert events == [
            "reset",
            "launch",
            "validate",
            "reset",
            "launch",
            "reset",
            "launch",
            "reset",
            "launch",
            "restore",
        ]
    else:
        with pytest.raises(RuntimeError, match="correctness"):
            runner._benchmark(Compiled(), module, case, candidate, kernel, inputs, 1, 2, torch)
        assert events == ["reset", "launch", "validate", "restore"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Triton requires Python 3.11 or newer")
def test_cpu_compilation_prepares_multiple_variants_without_a_gpu(
    monkeypatch: Any, tmp_path: Path
) -> None:
    triton = pytest.importorskip("triton")
    if triton.__version__ != "3.6.0":
        pytest.skip("the proof is pinned to Triton 3.6.0")
    source = """
import triton
import triton.language as tl
@triton.jit
def kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, tl.load(X + i, i < N, 0), i < N)
"""
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(output))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = runner.compile_stage(
        source=source,
        kernel_name="kernel",
        signature={"X": "*fp32", "OUT": "*fp32", "N": "constexpr", "BLOCK": "constexpr"},
        candidates=[
            {"name": "small", "constexprs": {"BLOCK": 64}, "num_warps": 4, "num_stages": 2},
            {"name": "large", "constexprs": {"BLOCK": 128}, "num_warps": 4, "num_stages": 2},
            {"name": "invalid", "constexprs": {"BLOCK": 96}, "num_warps": 4, "num_stages": 2},
        ],
        cases=[{"key": "n=256", "params": {}, "constexprs": {"N": 256}}],
        target_arch=121,
        triton_version="3.6.0",
        compile_workers=2,
        warmup=1,
        trials=2,
        max_adaptive_candidates=0,
    )
    assert report["compiled"] == 2
    assert report["errors"] == 1
    manifest = json.loads((output / "compiled-triton" / "manifest.json").read_text())
    assert "error" in manifest["records"][2]
    assert len(manifest["cache_files"]) > 2
