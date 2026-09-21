from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gfaas.artifacts import ArtifactOutput
from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import GfaasError
from gfaas.stages import CallStage, StageArtifactBinding


def test_submit_packages_source_for_a_named_callable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("def train(value):\n    return value\n")
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="secret",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    uploads: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    def upload_artifact(data: bytes, **kwargs: Any) -> dict[str, Any]:
        uploads.append({"data": data, **kwargs})
        return {"id": f"art_{len(uploads)}"}

    monkeypatch.setattr(client, "upload_artifact", upload_artifact)
    monkeypatch.setattr(client, "create_environment", lambda _definition: {"id": "env_1"})

    def create_function(definition: dict[str, Any]) -> dict[str, Any]:
        functions.append(definition)
        return {"id": "fn_1"}

    monkeypatch.setattr(client, "create_function", create_function)
    monkeypatch.setattr(client, "create_call", lambda _request, **_kwargs: {"id": "call_1"})

    try:
        result = client.submit(
            image="pytorch-cu130",
            function=("experiment", "train"),
            args=("input",),
            source_file=source,
        )
    finally:
        client.close()

    assert result.call_id == "call_1"
    assert uploads[0]["filename"] == "experiment.py"
    assert uploads[0]["kind"] == "source"
    assert functions[0]["executable"] == {
        "kind": "python_callable",
        "module": "experiment",
        "qualname": "train",
        "source_artifact_id": "art_1",
    }


def test_submit_rejects_a_named_callable_with_the_wrong_source_module(
    tmp_path: Path,
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("def train():\n    return None\n")
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key=None,
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )

    try:
        with pytest.raises(GfaasError, match="does not match source module"):
            client.submit(
                image="pytorch-cu130",
                function=("other", "train"),
                source_file=source,
            )
    finally:
        client.close()


def test_submit_pins_and_marks_an_image_qualification_call(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "qualification.py"
    source.write_text("def check():\n    return True\n")
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="secret",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    environments: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        client,
        "upload_artifact",
        lambda _data, **_kwargs: {"id": "art_source"},
    )

    def create_environment(definition: dict[str, Any]) -> dict[str, Any]:
        environments.append(definition)
        return {"id": "env_1"}

    monkeypatch.setattr(client, "create_environment", create_environment)
    monkeypatch.setattr(client, "create_function", lambda _definition: {"id": "fn_1"})

    def create_call(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append(request)
        return {"id": "call_1"}

    monkeypatch.setattr(client, "create_call", create_call)
    digest = f"sha256:{'a' * 64}"
    try:
        result = client.submit(
            image="candidate",
            function=("qualification", "check"),
            source_file=source,
            qualification_image_digest=digest,
        )
    finally:
        client.close()

    assert result.call_id == "call_1"
    assert environments[0]["name"] == "candidate"
    assert environments[0]["source"] == {
        "kind": "registered_image",
        "name": digest,
    }
    assert calls[0]["qualification"] == {"image_digest": digest}


def test_submit_serializes_one_logical_staged_call(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "pipeline.py"
    source.write_text("def compile(): pass\ndef execute(): pass\n")
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="secret",
            poll_interval_s=0.01,
            request_timeout_s=1,
        )
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        client,
        "upload_artifact",
        lambda _data, **_kwargs: {"id": "art_source"},
    )
    monkeypatch.setattr(client, "create_environment", lambda _definition: {"id": "env_1"})
    monkeypatch.setattr(client, "create_function", lambda _definition: {"id": "fn_1"})

    def create_call(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append(request)
        return {"id": "call_1"}

    monkeypatch.setattr(client, "create_call", create_call)
    output = ArtifactOutput("binary", "binary", kind="other")
    try:
        client.submit(
            image="cuda-nvcc",
            function=("pipeline", "execute"),
            source_file=source,
            gpu_count=1,
            stages=(
                CallStage(
                    "compile",
                    "compile",
                    resources={"gpu": {"count": 0}},
                    outputs=(output,),
                ),
                CallStage(
                    "execute",
                    "execute",
                    artifacts=(StageArtifactBinding("compile", "binary", "BINARY_ID"),),
                ),
            ),
        )
    finally:
        client.close()

    assert calls[0]["function_id"] == "fn_1"
    assert calls[0]["stages"] == [
        {
            "name": "compile",
            "executable": {"module": "pipeline", "qualname": "compile"},
            "resources": {"gpu": {"count": 0}},
            "outputs": [output.request()],
        },
        {
            "name": "execute",
            "executable": {"module": "pipeline", "qualname": "execute"},
            "artifacts": [{"from_stage": "compile", "output": "binary", "env": "BINARY_ID"}],
        },
    ]
