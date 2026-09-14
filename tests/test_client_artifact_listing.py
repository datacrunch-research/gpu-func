"""Owner-scoped Artifact listing follows the server cursor across pages."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from gfaas.client import Client, GfaasError
from gfaas.config import ClientConfig


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="key-1",
            poll_interval_s=0.001,
            request_timeout_s=5.0,
        )
    )
    client._http.close()
    client._http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=client.cfg.api_base,
        headers={"X-API-Key": client.cfg.api_key},
    )
    return client


def test_iter_artifacts_follows_the_cursor_and_sends_the_page_limit():
    seen_queries: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/artifacts"
        after = request.url.params.get("after", "")
        seen_queries.append((after, request.url.params["limit"]))
        if not after:
            return httpx.Response(
                200,
                json={
                    "items": [{"id": "art_1"}, {"id": "art_2"}],
                    "next_cursor": "1700000000000:art_2",
                },
            )
        assert after == "1700000000000:art_2"
        return httpx.Response(200, json={"items": [{"id": "art_3"}]})

    with _client(handler) as client:
        ids = [item["id"] for item in client.iter_artifacts(limit=2)]

    assert ids == ["art_1", "art_2", "art_3"]
    assert seen_queries == [("", "2"), ("1700000000000:art_2", "2")]


def test_list_artifacts_surfaces_a_rejected_cursor():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/artifacts"
        return httpx.Response(
            400,
            json={
                "type": "about:blank",
                "title": "Bad request",
                "status": 400,
                "code": "invalid_request",
                "request_id": "req_1",
            },
        )

    with _client(handler) as client, pytest.raises(GfaasError):
        client.list_artifacts(after="not-a-cursor")
