"""Keep CLI hardware requirements intact through Environment creation."""

import json

import httpx
import pytest

from gfaas import Client, ClientConfig
from gfaas_cli.client import GfaasClient
from gfaas_cli.main import main


@pytest.mark.parametrize(
    "command",
    ["python", "callable", "cuda", "custom", "compile", "exercise", "benchmark"],
)
def test_cli_submissions_send_hardware_requirements(command, tmp_path, monkeypatch):
    (tmp_path / "work.py").write_text("def work():\n    return 42\n")
    (tmp_path / "kernel.cu").write_text("int main() { return 0; }\n")
    (tmp_path / "run.py").write_text("# exercise runner\n")
    (tmp_path / "runner").mkdir()
    (tmp_path / "runner" / "cli.py").write_text("# runner support\n")
    commands = {
        "python": ["run", str(tmp_path / "work.py")],
        "callable": ["run", str(tmp_path / "work.py") + ":work"],
        "cuda": ["run", str(tmp_path / "kernel.cu")],
        "custom": ["custom", "run", str(tmp_path / "kernel.cu"), "--arch", "sm_103"],
        "compile": ["custom", "compile", str(tmp_path / "kernel.cu"), "--arch", "sm_103"],
        "exercise": ["exercise", "sample", "test", "--exercise-dir", str(tmp_path)],
        "benchmark": ["benchmark", "--exercise-dir", str(tmp_path)],
    }
    bodies = {}

    def handle(request):
        path = request.url.path.rsplit("/", 1)[-1]
        if path == "capabilities":
            return httpx.Response(
                200,
                json={
                    "environment_compatibility": {
                        "cpu_architectures": ["amd64", "arm64"],
                        "gpu_compute_capabilities": "exact",
                    }
                },
            )
        if path == "artifacts":
            return httpx.Response(201, json={"id": "art_input"})
        bodies[path] = json.loads(request.content)
        return httpx.Response(
            201 if path != "calls" else 202,
            json={
                "id": {"environments": "env_test", "functions": "fn_test", "calls": "call_test"}[
                    path
                ]
            },
        )

    client = Client(
        ClientConfig(
            api_base="https://example.test/api",
            api_key="test-key",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    client._http.close()
    client._http = httpx.Client(base_url=client.cfg.api_base, transport=httpx.MockTransport(handle))
    # Workspace transfer is covered separately; exercise the real CLI-to-SDK submission path.
    monkeypatch.setattr(
        client, "upload_artifact_directory", lambda *a, **kw: {"id": "art_workspace"}
    )
    monkeypatch.setattr(GfaasClient, "from_args", lambda args: GfaasClient(client, poll_interval=0))
    with client:
        assert (
            main(
                commands[command]
                + [
                    "--detach",
                    "--gpu-type",
                    "gb300",
                    "--cpu-architecture",
                    "arm64",
                    "--gpu-compute-capability",
                    "10.0",
                    "--gpu-compute-capability",
                    "10.3",
                ],
                client_factory=lambda: client,
            )
            == 0
        )
    assert bodies["environments"]["compatibility"] == {
        "cpu_architecture": "arm64",
        "gpu_compute_capabilities": ["10.0", "10.3"],
    }
    if command == "compile":
        assert bodies["functions"]["default_resources"]["gpu"]["count"] == 0


@pytest.mark.parametrize("command", [["run", "work.py"], ["custom", "run", "kernel.cu"]])
@pytest.mark.parametrize(
    "flags",
    [
        ["--cpu-architecture", "x86"],
        ["--gpu-compute-capability", "sm_103"],
        ["--gpu-compute-capability", "10.3"] * 2,
        sum((["--gpu-compute-capability", f"{major}.0"] for major in range(1, 34)), []),
    ],
)
def test_invalid_requirements_fail_before_client_creation(command, flags, monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("invalid requirements reached client creation")

    monkeypatch.setattr(GfaasClient, "from_args", forbidden)
    with pytest.raises(SystemExit) as error:
        main(command + flags, client_factory=forbidden)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
