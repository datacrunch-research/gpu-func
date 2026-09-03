from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gfaas import cuda_runner, local_cuda, python_runner
from gfaas_cli import main as cli


class FakeRemoteResult:
    def __init__(
        self,
        *,
        call_id: str = "call_test",
        result: Any = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.call_id = call_id
        self.result = result
        self.events = events or []
        self.cancellations: list[str | None] = []

    def iter_events(self, **_kwargs: Any) -> Iterator[dict[str, Any]]:
        yield from self.events

    def wait(self) -> Any:
        return self.result

    def cancel(self, *, reason: str | None = None) -> dict[str, Any]:
        self.cancellations.append(reason)
        return {"id": self.call_id, "state": "cancelling"}


class FakeClient:
    def __init__(self, remote: FakeRemoteResult | None = None) -> None:
        self.remote = remote or FakeRemoteResult()
        self.submissions: list[dict[str, Any]] = []
        self.cancellations: list[tuple[str, str | None]] = []
        self.artifact_metadata: dict[str, Any] = {}
        self.file_downloads: list[tuple[str, Path]] = []
        self.call_artifacts: list[dict[str, Any]] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, **kwargs: Any) -> FakeRemoteResult:
        self.submissions.append(kwargs)
        return self.remote

    def get_call(self, call_id: str) -> dict[str, Any]:
        return {"id": call_id, "state": "running"}

    def cancel_call(self, call_id: str, *, reason: str | None = None) -> dict[str, Any]:
        self.cancellations.append((call_id, reason))
        return {"id": call_id, "state": "cancelling"}

    def get_call_logs(self, call_id: str) -> dict[str, Any]:
        return {"call_id": call_id, "stdout": "hello\n", "stderr": "", "truncated": False}

    def list_call_artifacts(self, call_id: str) -> dict[str, Any]:
        return {"call_id": call_id, "items": self.call_artifacts}

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return {"id": artifact_id, **self.artifact_metadata}

    def download_artifact_file(self, artifact_id: str, destination: Path) -> tuple[Path, str]:
        self.file_downloads.append((artifact_id, destination))
        return destination, "application/octet-stream"

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "gpu_pools": [
                {
                    "name": "gb300",
                    "status": "ready",
                    "connected_workers": 2,
                    "available_workers": 1,
                }
            ]
        }


def _factory(client: FakeClient):
    return lambda: client


def _local_toolchain() -> local_cuda.LocalCudaToolchain:
    gpu = local_cuda.LocalGpu(
        index=0,
        uuid="GPU-local",
        name="NVIDIA GB10",
        compute_capability="12.1",
        architecture="sm_121",
        memory_total_mib=None,
        driver_version="580.42",
    )
    return local_cuda.LocalCudaToolchain(
        nvcc="/usr/local/cuda/bin/nvcc",
        nvcc_version="13.0.88",
        cuda_root="/usr/local/cuda",
        nvidia_smi="/usr/bin/nvidia-smi",
        ncu="/usr/local/cuda/bin/ncu",
        host_compiler="/usr/bin/g++",
        supported_architectures=("sm_121",),
        gpus=(gpu,),
        selected_gpu=gpu,
        cuda_visible_devices=None,
    )


def test_run_submits_cuda_file_with_resources_and_program_arguments(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    client = FakeClient()

    status = cli.main(
        [
            "run",
            str(source),
            "--detach",
            "--gpu-type",
            "gb300",
            "--memory",
            "2GiB",
            "--output",
            "benchmark=results.csv",
            "--nvcc-flag=-O3",
            "--",
            "--problem-size",
            "4096",
        ],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out == "call_test\n"
    assert len(client.submissions) == 1
    submission = client.submissions[0]
    assert submission["image"] == "cuda-nvcc"
    assert submission["function"] is cuda_runner.run
    assert submission["gpu_type"] == "gb300"
    assert submission["gpu_count"] == 1
    assert submission["memory_bytes"] == 2 * 1024**3
    assert [output.name for output in submission["outputs"]] == ["benchmark"]
    assert submission["outputs"][0].relative_path == "results.csv"
    assert submission["kwargs"] == {
        "source": "int main() { return 0; }\n",
        "profile": False,
        "ncu_args": [],
        "nvcc_flags": ["-O3"],
        "program_args": ["--problem-size", "4096"],
    }


def test_run_submits_python_script_with_declared_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("print('hello')\n")
    client = FakeClient()

    status = cli.main(
        [
            "run",
            str(source),
            "--detach",
            "--image",
            "pytorch-cu130",
            "--output",
            "report=reports/result.json",
            "--output-directory",
            "profiles=profiles",
            "--",
            "--steps",
            "10",
        ],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out == "call_test\n"
    submission = client.submissions[0]
    assert submission["function"] is python_runner.run_script
    assert submission["kwargs"] == {
        "source": "print('hello')\n",
        "filename": "experiment.py",
        "program_args": ["--steps", "10"],
    }
    assert [output.name for output in submission["outputs"]] == ["report", "profiles"]
    assert submission["outputs"][1].layout == "tree"


def test_run_does_not_forward_client_credentials_to_the_workload(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("print('hello')\n")
    monkeypatch.setenv("GFAAS_API_KEY", "client-secret")
    client = FakeClient()

    status = cli.main(
        ["run", str(source), "--detach", "--env", "MODE=benchmark"],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out == "call_test\n"
    assert client.submissions[0]["env"] == {"MODE": "benchmark"}


def test_run_submits_python_callable_without_importing_source(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "function.py"
    source.write_text("raise RuntimeError('must not execute locally')\n")
    client = FakeClient()

    status = cli.main(
        ["run", f"{source}:train", "--detach", "--", "input.json"],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out == "call_test\n"
    submission = client.submissions[0]
    assert submission["function"] == ("function", "train")
    assert submission["source_file"] == source
    assert submission["args"] == ("input.json",)


def test_explicit_python_runtime_accepts_a_nonstandard_script_suffix(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "experiment.txt"
    source.write_text("print('hello')\n")
    client = FakeClient()

    status = cli.main(
        ["run", str(source), "--runtime", "python", "--detach"],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out == "call_test\n"
    assert client.submissions[0]["function"] is python_runner.run_script
    assert client.submissions[0]["kwargs"]["filename"] == "experiment.txt"


def test_python_callable_rejects_a_non_importable_source_filename(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "not-importable.py"
    source.write_text("def train():\n    return None\n")
    client = FakeClient()

    status = cli.main(
        ["run", f"{source}:train", "--detach"],
        client_factory=_factory(client),
    )

    assert status == 1
    assert "importable module name" in capsys.readouterr().err
    assert client.submissions == []


def test_foreground_json_output_is_json_lines(tmp_path: Path, capsys) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    remote = FakeRemoteResult(
        result={"phase": "run", "returncode": 0, "compile_ms": 12, "run_ms": 4},
        events=[{"cursor": "1", "type": "state", "state": "running"}],
    )
    client = FakeClient(remote)

    assert (
        cli.main(
            ["run", str(source), "--json"],
            client_factory=_factory(client),
        )
        == 0
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0] == {"call_id": "call_test", "type": "submitted"}
    assert records[1]["type"] == "call_event"
    assert records[2]["type"] == "result"
    assert records[2]["value"]["run_ms"] == 4


def test_foreground_human_output_summarizes_lifecycle_events(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    remote = FakeRemoteResult(
        result={"phase": "run", "returncode": 0, "compile_ms": 12, "run_ms": 4},
        events=[
            {"cursor": "0", "type": "state", "state": "queued"},
            {
                "cursor": "1",
                "type": "diagnostic",
                "attributes": {
                    "type": "placement_rejection",
                    "reason": "gpu_occupancy",
                    "worker_id": "gb300-01",
                    "placement_generation": 2,
                },
            },
            {
                "cursor": "2",
                "type": "preparation",
                "attributes": {
                    "phase": "bundle_uploaded",
                    "completed_files": 0,
                    "completed_bytes": 0,
                    "details": {"size_bytes": 32656},
                    "worker_id": "gb300-03",
                },
            },
            {
                "cursor": "3",
                "type": "state",
                "state": "succeeded",
                "attributes": {
                    "worker_id": "gb300-03",
                    "timing": {"total_ms": 1675, "execute_ms": 1260},
                },
            },
            {
                "cursor": "4",
                "type": "artifact",
                "attributes": {"artifact_id": "art_result", "role": "result"},
            },
        ],
    )

    assert cli.main(["run", str(source)], client_factory=_factory(FakeClient(remote))) == 0

    captured = capsys.readouterr()
    assert "[vfunc] state=queued" in captured.err
    assert (
        "[vfunc] waiting for capacity reason=gpu_occupancy worker=gb300-01 generation=2"
        in captured.err
    )
    assert (
        "[vfunc] preparation phase=bundle_uploaded worker=gb300-03 files=0 bytes=0B "
        "size=31.9KiB" in captured.err
    )
    assert "[vfunc] state=succeeded worker=gb300-03 total=1.68s execute=1.26s" in captured.err
    assert "[vfunc] artifact role=result id=art_result" in captured.err
    assert '"attempt_id"' not in captured.err


def test_call_cancel_passes_the_reason(capsys) -> None:
    client = FakeClient()

    assert (
        cli.main(
            ["call", "cancel", "call_1", "--reason", "superseded", "--json"],
            client_factory=_factory(client),
        )
        == 0
    )

    assert client.cancellations == [("call_1", "superseded")]
    assert json.loads(capsys.readouterr().out) == {"id": "call_1", "state": "cancelling"}


def test_call_show_logs_and_artifacts_are_available(capsys) -> None:
    client = FakeClient()
    client.call_artifacts = [{"name": "result", "artifact": {"id": "art_1"}}]

    assert cli.main(["call", "show", "call_1", "--json"], client_factory=_factory(client)) == 0
    assert json.loads(capsys.readouterr().out) == {"id": "call_1", "state": "running"}

    assert cli.main(["call", "logs", "call_1"], client_factory=_factory(client)) == 0
    assert capsys.readouterr().out == "hello\n"

    assert cli.main(["call", "artifacts", "call_1", "--json"], client_factory=_factory(client)) == 0
    artifacts = json.loads(capsys.readouterr().out)
    assert artifacts["items"][0]["artifact"]["id"] == "art_1"


def test_artifact_download_refuses_to_replace_an_existing_path(
    tmp_path: Path,
    capsys,
) -> None:
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"keep")
    client = FakeClient()

    status = cli.main(
        ["artifact", "download", "art_1", str(destination)],
        client_factory=_factory(client),
    )

    assert status == 1
    assert "destination already exists" in capsys.readouterr().err
    assert destination.read_bytes() == b"keep"
    assert client.file_downloads == []


def test_artifact_download_uses_the_requested_destination(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "result.bin"
    client = FakeClient()

    status = cli.main(
        ["artifact", "download", "art_1", str(destination)],
        client_factory=_factory(client),
    )

    assert status == 0
    assert capsys.readouterr().out.strip() == str(destination)
    assert client.file_downloads == [("art_1", destination)]


def test_pool_list_has_human_and_json_output(capsys) -> None:
    client = FakeClient()
    assert cli.main(["pool", "list"], client_factory=_factory(client)) == 0
    human = capsys.readouterr().out
    assert "gb300" in human
    assert "ready" in human

    assert cli.main(["pool", "list", "--json"], client_factory=_factory(client)) == 0
    assert json.loads(capsys.readouterr().out)["gpu_pools"][0]["name"] == "gb300"


def test_completion_writes_shell_setup_without_creating_a_client(capsys) -> None:
    def unexpected_client() -> FakeClient:
        raise AssertionError("completion must not create a client")

    expected_markers = {
        "bash": "_python_argcomplete",
        "fish": "__fish_vfunc_complete",
        "zsh": "_python_argcomplete",
        "powershell": "Register-ArgumentCompleter",
    }
    for shell, marker in expected_markers.items():
        assert cli.main(["completion", shell], client_factory=unexpected_client) == 0
        output = capsys.readouterr().out
        assert marker in output
        assert "vfunc" in output


def test_local_info_does_not_create_a_remote_client(monkeypatch, capsys) -> None:
    monkeypatch.setattr(local_cuda, "discover", lambda **_kwargs: _local_toolchain())

    def unexpected_client() -> FakeClient:
        raise AssertionError("local info must not create a client")

    assert cli.main(["local", "info", "--json"], client_factory=unexpected_client) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["type"] == "local_cuda_info"
    assert report["selected_gpu"]["architecture"] == "sm_121"


def test_local_run_uses_native_architecture_and_checks_outputs(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_cuda, "discover", lambda **_kwargs: _local_toolchain())
    calls: list[dict[str, Any]] = []

    def local_run(_source: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        (kwargs["workdir"] / "results.csv").write_text("name,value\nlocal,1\n")
        return {
            "phase": "run",
            "stdout": "done\n",
            "stderr": "",
            "returncode": 0,
            "compile_ms": 10,
            "run_ms": 2,
            "ncu_csv": None,
        }

    monkeypatch.setattr(local_cuda, "run", local_run)

    def unexpected_client() -> FakeClient:
        raise AssertionError("local run must not create a client")

    status = cli.main(
        [
            "local",
            "run",
            str(source),
            "--env",
            "MODE=benchmark",
            "--nvcc-flag=-O3",
            "--output",
            "benchmark=results.csv",
            "--json",
            "--",
            "--size",
            "4096",
        ],
        client_factory=unexpected_client,
    )

    assert status == 0
    assert calls[0]["nvcc_flags"] == ["-arch=sm_121", "-O3"]
    assert calls[0]["program_args"] == ["--size", "4096"]
    assert calls[0]["environment"]["MODE"] == "benchmark"
    result = json.loads(capsys.readouterr().out)
    assert result["call_id"] == "local"
    assert result["value"]["architecture"] == "sm_121"
    assert result["value"]["outputs"][0]["exists"] is True


def test_local_run_refuses_to_replace_an_existing_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    output = tmp_path / "results.csv"
    output.write_text("keep\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_cuda, "discover", lambda **_kwargs: _local_toolchain())

    def unexpected_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an existing output must stop local execution")

    monkeypatch.setattr(local_cuda, "run", unexpected_run)

    status = cli.main(["local", "run", str(source), "--output", "benchmark=results.csv"])

    assert status == 1
    assert "local output path already exists" in capsys.readouterr().err
    assert output.read_text() == "keep\n"
