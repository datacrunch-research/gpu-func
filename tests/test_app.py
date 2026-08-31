from __future__ import annotations

from typing import Any

import gfaas


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return object()


def test_function_decorator_forwards_an_explicit_gpu_count() -> None:
    client = FakeClient()
    app = gfaas.App("multi-gpu", image=gfaas.Image("pytorch"), client=client)

    @app.function(gpu_count=4, gpu_type="gb300")
    def train() -> None:
        pass

    train.spawn()

    assert client.calls[0]["gpu"] is None
    assert client.calls[0]["gpu_count"] == 4
    assert client.calls[0]["gpu_type"] == "gb300"
