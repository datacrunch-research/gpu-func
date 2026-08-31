from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import GfaasError


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
