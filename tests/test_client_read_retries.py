"""API reads ride out a coordinator restart or a proxy's brief error page; writes are sent once."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

import gfaas.transfers
from gfaas.client import READ_ATTEMPTS, Client, GfaasError
from gfaas.config import ClientConfig

# What Cloudflare answers while the coordinator behind it restarts.
PROXY_ERROR_PAGE = httpx.Response(
    502, headers={"content-type": "text/html"}, text="<html>Bad gateway</html>"
)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gfaas.transfers.time, "sleep", lambda _seconds: None)


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


def test_waiting_on_a_call_rides_out_a_coordinator_restart():
    """While the coordinator restarts, its proxy answers with an error page and then refuses
    connections. A client waiting on the Call keeps waiting, and returns its result."""
    answers = iter(
        [
            httpx.Response(200, json={"id": "call_1", "state": "running"}),
            PROXY_ERROR_PAGE,
            httpx.ConnectError("connection refused"),
            PROXY_ERROR_PAGE,
            httpx.Response(200, json={"id": "call_1", "state": "succeeded"}),
        ]
    )
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/v1/calls/call_1/result":
            return httpx.Response(
                200, json={"outcome": "succeeded", "output": {"storage": "inline", "value": 42}}
            )
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    with _client(handler) as client:
        assert client.wait_for_result("call_1", timeout_s=60) == 42

    assert requests.count("/api/v1/calls/call_1") == 5


def test_a_read_that_keeps_failing_gives_up_after_its_budget():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return PROXY_ERROR_PAGE

    with _client(handler) as client, pytest.raises(GfaasError, match="502"):
        client.get_call("call_1")

    assert attempts == READ_ATTEMPTS


def test_a_read_refused_for_good_is_not_sent_again():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, json={"title": "Not found"})

    with _client(handler) as client, pytest.raises(GfaasError, match="404"):
        client.get_call("call_1")

    assert attempts == 1


def test_a_write_is_sent_once_even_when_the_proxy_fails():
    """Sending a write again could repeat its effect, such as admitting a Call twice."""
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return PROXY_ERROR_PAGE

    with _client(handler) as client, pytest.raises(GfaasError, match="502"):
        client.cancel_call("call_1", reason="done")

    assert attempts == 1
