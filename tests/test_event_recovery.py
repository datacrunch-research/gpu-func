from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import gfaas.events as events
from gfaas.client import Client, RemoteResult
from gfaas.config import ClientConfig
from gfaas.errors import GfaasError


class Clock:
    now = 0.0

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(events.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(events.time, "time", lambda: 1_700_000_000 + clock.now)
    monkeypatch.setattr(events.time, "sleep", clock.sleep)
    return clock


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    client = Client(ClientConfig("https://gpu.example.com/api", None, 0.1, 10))
    client._http.close()

    def read_only(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path in {"/api/v1/calls/call_1", "/api/v1/calls/call_1/events"}
        return handler(request)

    client._http = httpx.Client(
        base_url=client.cfg.api_base, transport=httpx.MockTransport(read_only)
    )
    return client


def event(cursor: str, *, terminal: bool = False) -> dict:
    if terminal:
        return {"cursor": cursor, "type": "state", "state": "succeeded"}
    return {"cursor": cursor, "type": "stdout", "stream_data": f"line {cursor}\n"}


def sse(*items: dict) -> bytes:
    return b"".join(f"data: {json.dumps(item)}\n\n".encode() for item in items)


def page(*items: dict, next_cursor: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={"items": items, "next_cursor": next_cursor})


@pytest.mark.parametrize("timeout_s", [None, 30])
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_http_recovery_resumes_cursor_and_honors_retry_after(clock, timeout_s, status):
    cursors = []

    def handler(request):
        if request.url.path.endswith("/call_1"):
            return httpx.Response(200, json={"state": "succeeded"})
        cursors.append(request.url.params.get("after"))
        if len(cursors) == 1:
            return httpx.Response(status, headers={"Retry-After": "2"})
        if len(cursors) == 2:
            if timeout_s is None:
                return httpx.Response(200, content=sse(event("2", terminal=True)))
            return page(event("2", terminal=True))
        return page()

    with client_for(handler) as client:
        assert list(client.iter_events("call_1", after="1", timeout_s=timeout_s)) == [
            event("2", terminal=True)
        ]
    assert cursors[:2] == ["1", "1"]
    assert clock.sleeps == [2]


@pytest.mark.parametrize("timeout_s", [None, 30])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 501])
def test_permanent_errors_preserve_status_without_retry(clock, timeout_s, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"detail": "secret must not appear"})

    with client_for(handler) as client, pytest.raises(GfaasError) as failure:
        list(client.iter_events("call_1", timeout_s=timeout_s))
    assert failure.value.status_code == status
    assert "secret" not in str(failure.value)
    assert len(requests) == 1
    assert not clock.sleeps


@pytest.mark.parametrize("timeout_s", [None, 300])
def test_repeated_failures_exhaust_budget(clock, timeout_s):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError("offline", request=request)

    with client_for(handler) as client, pytest.raises(httpx.ConnectError, match="offline"):
        list(client.iter_events("call_1", timeout_s=timeout_s))
    assert len(requests) == 6
    assert clock.sleeps == [0.5, 1, 2, 4, 8]


@pytest.mark.parametrize("retry_after", ["2", formatdate(1_700_000_002, usegmt=True)])
def test_retry_after_cannot_extend_total_deadline(clock, retry_after):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, headers={"Retry-After": retry_after})

    with client_for(handler) as client, pytest.raises(TimeoutError):
        list(client.iter_events("call_1", timeout_s=1))
    assert len(requests) == 1
    assert clock.sleeps == [1]


def test_retry_after_beyond_recovery_budget_does_not_retry_early(clock):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    with client_for(handler) as client, pytest.raises(GfaasError) as failure:
        list(client.iter_events("call_1"))
    assert failure.value.status_code == 429
    assert len(requests) == 1
    assert clock.sleeps == []


def test_requests_and_delivery_use_remaining_deadline(clock):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.extensions["timeout"]["read"] == pytest.approx(
            1 if len(requests) == 1 else 0.5
        )
        if len(requests) == 1:
            raise httpx.ReadTimeout("temporary", request=request)
        clock.now += 0.6
        return page(event("1", terminal=True))

    with client_for(handler) as client, pytest.raises(TimeoutError):
        list(client.iter_events("call_1", timeout_s=1))
    assert len(requests) == 2


def test_expired_deadline_does_not_start_request(clock):
    with (
        client_for(lambda _: pytest.fail("unexpected request")) as client,
        pytest.raises(TimeoutError),
    ):
        list(client.iter_events("call_1", timeout_s=0))


@pytest.mark.parametrize("timeout_s", [None, 30])
def test_completed_status_drains_events_written_before_status_read(clock, timeout_s):
    paths = []
    cursors = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/call_1"):
            return httpx.Response(200, json={"state": "succeeded"})
        cursors.append(request.url.params.get("after"))
        if len(cursors) == 1:
            return httpx.Response(200, content=b"") if timeout_s is None else page()
        if len(cursors) == 2:
            return page(event("3"), next_cursor="3")
        return page(event("4"))

    with client_for(handler) as client:
        assert list(client.iter_events("call_1", after="2", timeout_s=timeout_s)) == [
            event("3"),
            event("4"),
        ]
    assert cursors == ["2", "2", "3"]
    assert len(paths) == 4
    assert not clock.sleeps


@pytest.mark.parametrize("timeout_s", [None, 30])
def test_nonfollowing_watch_drains_json_pages_without_status_or_sse(clock, timeout_s):
    cursors = []

    def handler(request):
        assert request.url.path.endswith("/events")
        assert request.url.params["follow"] == "false"
        cursors.append(request.url.params.get("after"))
        if len(cursors) == 1:
            return page(event("1"), next_cursor="1")
        return page(event("2"))

    with client_for(handler) as client:
        assert list(client.iter_events("call_1", follow=False, timeout_s=timeout_s)) == [
            event("1"),
            event("2"),
        ]
    assert cursors == [None, "1"]
    assert not clock.sleeps


class InterruptedStream(httpx.SyncByteStream):
    def __init__(self, error: BaseException | None) -> None:
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield sse(event("1"))
        yield b'data: {"cursor": "2"'  # Incomplete frame must be replayed, not delivered.
        if self.error:
            raise self.error

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("error", [None, httpx.RemoteProtocolError("disconnected")])
def test_sse_disconnect_or_eof_recovers_partial_frame(clock, error):
    stream = InterruptedStream(error)
    cursors = []
    statuses = 0

    def handler(request):
        nonlocal statuses
        if request.url.path.endswith("/call_1"):
            statuses += 1
            return httpx.Response(
                200, json={"state": "running" if len(cursors) == 1 else "succeeded"}
            )
        cursors.append(request.url.params.get("after"))
        if len(cursors) == 1:
            return httpx.Response(200, stream=stream)
        if len(cursors) == 2:
            return httpx.Response(200, content=sse(event("2"), event("3", terminal=True)))
        return page()

    with client_for(handler) as client:
        result = RemoteResult(client, "call_1", "any")
        assert list(result.iter_logs()) == [("stdout", "line 1\n"), ("stdout", "line 2\n")]
    assert cursors == [None, "1", "3"]
    assert stream.closed
    assert clock.sleeps == [0.5]


def test_polling_resumes_after_page_and_status_failures(clock):
    cursors = []
    statuses = 0

    def handler(request):
        nonlocal statuses
        if request.url.path.endswith("/call_1"):
            statuses += 1
            if statuses == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"state": "running"})
        cursors.append(request.url.params.get("after"))
        if len(cursors) == 1:
            return page(event("1"))
        if len(cursors) == 2:
            raise httpx.ReadTimeout("temporary", request=request)
        return page(event("2", terminal=True))

    with client_for(handler) as client:
        assert list(client.iter_events("call_1", timeout_s=10)) == [
            event("1"),
            event("2", terminal=True),
        ]
    assert cursors == [None, "1", "1"]
    assert clock.sleeps == [0.5, 1]


@pytest.mark.parametrize("timeout_s", [None, 30])
def test_close_stops_requests_and_releases_stream(clock, timeout_s):
    stream = InterruptedStream(AssertionError("must not read ahead"))
    requests = []

    def handler(request):
        requests.append(request)
        if timeout_s is None:
            return httpx.Response(200, stream=stream)
        return page(event("1"), event("2"))

    with client_for(handler) as client:
        iterator = client.iter_events("call_1", timeout_s=timeout_s)
        assert next(iterator) == event("1")
        iterator.close()
    assert len(requests) == 1
    assert not clock.sleeps
    assert timeout_s is not None or stream.closed


@pytest.mark.parametrize("timeout_s", [None, 30])
def test_interrupt_during_backoff_is_not_swallowed(clock, monkeypatch, timeout_s):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(events.time, "sleep", interrupt)
    with client_for(handler) as client, pytest.raises(KeyboardInterrupt):
        list(client.iter_events("call_1", timeout_s=timeout_s))
    assert len(requests) == 1


@pytest.mark.parametrize("timeout_s", [None, 10])
def test_real_http_disconnect_resumes_after_delivered_event(timeout_s):
    cursors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            url = urlsplit(self.path)
            if not url.path.endswith("/events"):
                body = json.dumps(
                    {"state": "running" if len(cursors) < 3 else "succeeded"}
                ).encode()
            else:
                params = parse_qs(url.query)
                cursors.append(params.get("after", [None])[0])
                if len(cursors) == 2:
                    self.close_connection = True  # Disconnect without an HTTP response.
                    return
                items = [event("1")] if len(cursors) == 1 else [event("2", terminal=True)]
                if len(cursors) > 3:
                    items = []
                if params["follow"] == ["true"]:
                    body = sse(*items)
                else:
                    body = json.dumps({"items": items, "next_cursor": None}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with Client(
            ClientConfig(f"http://127.0.0.1:{server.server_port}", None, 0.01, 2)
        ) as client:
            assert list(client.iter_events("call_1", timeout_s=timeout_s)) == [
                event("1"),
                event("2", terminal=True),
            ]
        assert cursors[:3] == [None, "1", "1"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_response_chunks_cannot_extend_watch_deadline(clock):
    class SlowBody(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b'{"items": ['
            clock.now += 2
            yield b"]}"
            pytest.fail("must stop reading after deadline")

        def close(self):
            self.closed = True

    stream = SlowBody()
    with (
        client_for(lambda _: httpx.Response(200, stream=stream)) as client,
        pytest.raises(TimeoutError),
    ):
        list(client.iter_events("call_1", timeout_s=1))
    assert stream.closed


def test_recovery_time_budget_bounds_slow_retries(clock):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) > 1:
            clock.now += 30
        return httpx.Response(503)

    with client_for(handler) as client, pytest.raises(GfaasError) as failure:
        list(client.iter_events("call_1"))
    assert failure.value.status_code == 503
    assert len(requests) == 3


@pytest.mark.parametrize("timeout_s", [None, 10])
def test_invalid_events_fail_without_retry(clock, timeout_s):
    requests = []

    def handler(request):
        requests.append(request)
        bad_event = {"type": "stdout", "stream_data": "no cursor"}
        if timeout_s is None:
            return httpx.Response(200, content=sse(bad_event))
        return page(bad_event)

    with client_for(handler) as client, pytest.raises(GfaasError, match="cursor"):
        list(client.iter_events("call_1", timeout_s=timeout_s))
    assert len(requests) == 1
    assert not clock.sleeps


def test_repeated_sse_eof_is_bounded_even_when_status_succeeds(clock):
    streams = []

    def handler(request):
        if request.url.path.endswith("/call_1"):
            return httpx.Response(200, json={"state": "running"})
        streams.append(request)
        return httpx.Response(200, content=b"")

    with client_for(handler) as client, pytest.raises(httpx.RemoteProtocolError):
        list(client.iter_events("call_1"))
    assert len(streams) == 6
