"""Recovery for read-only Call observation. No submission or cancellation requests."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .errors import GfaasError

TERMINAL_CALL_STATES = {"succeeded", "failed", "timed_out", "cancelled"}
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class _ObservationError(GfaasError):
    def __init__(self, response: httpx.Response) -> None:
        self.retry_after = response.headers.get("retry-after")
        # Do not echo response bodies, which can include credentials or proxy HTML.
        detail = "authentication rejected" if response.status_code in {401, 403} else "HTTP error"
        super().__init__(
            f"Call observation failed: {detail} ({response.status_code})",
            status_code=response.status_code,
        )


class EventWatcher:
    """One iterator's cursor, deadline, and consecutive-failure budget."""

    def __init__(
        self,
        http: httpx.Client,
        call_id: str,
        *,
        after: str | None,
        follow: bool,
        timeout_s: float | None,
        request_timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be non-negative and finite")
        self.http = http
        self.path = f"/v1/calls/{call_id}"
        self.cursor = after
        self.follow = follow
        self.timeout_s = timeout_s
        self.deadline = None if timeout_s is None else time.monotonic() + timeout_s
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s
        self.failures = 0
        self.recovery_deadline: float | None = None
        self.last_error: Exception | None = None

    def _remaining(self) -> float:
        now = time.monotonic()
        remaining = math.inf if self.deadline is None else self.deadline - now
        if remaining <= 0:
            raise TimeoutError(f"{self.path} event stream exceeded {self.timeout_s}s")
        if self.recovery_deadline is not None:
            recovery_remaining = self.recovery_deadline - now
            if recovery_remaining <= 0 and self.last_error is not None:
                raise self.last_error
            remaining = min(remaining, recovery_remaining)
        return remaining

    def _timeout(self) -> float:
        return min(self.request_timeout_s, self._remaining())

    def _sleep(self, delay: float) -> None:
        time.sleep(min(delay, self._remaining()))
        self._remaining()

    def _progress(self) -> None:
        self.failures = 0
        self.recovery_deadline = None
        self.last_error = None

    def _retry(self, error: Exception) -> None:
        self._remaining()
        if isinstance(error, _ObservationError) and error.status_code not in _RETRY_STATUSES:
            raise error
        now = time.monotonic()
        if self.recovery_deadline is None:
            self.recovery_deadline = now + 60
        if self.failures >= 5 or now >= self.recovery_deadline:
            raise error
        delay = min(0.5 * 2**self.failures, 8)
        if isinstance(error, _ObservationError):
            delay = max(delay, _retry_after_seconds(error.retry_after))
        # Never retry earlier than Retry-After, or wait beyond the recovery budget.
        if delay >= self.recovery_deadline - now:
            raise error
        self.failures += 1
        self.last_error = error
        self._sleep(delay)

    def _json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        with self.http.stream("GET", path, timeout=self._timeout(), **kwargs) as response:
            self._remaining()
            if not response.is_success:
                raise _ObservationError(response)
            chunks = []
            for chunk in response.iter_bytes():
                self._remaining()
                chunks.append(chunk)
            self._remaining()
        body = json.loads(b"".join(chunks))
        if not isinstance(body, dict):
            raise GfaasError("Call observation response must be an object")
        return body

    def _completed(self) -> bool:
        state = self._json(self.path).get("state")
        if not isinstance(state, str):
            raise GfaasError("Call status response has no state")
        return state in TERMINAL_CALL_STATES

    def _params(self, *, follow: bool) -> dict[str, str]:
        params = {"follow": str(follow).lower(), "limit": "1000"}
        if self.cursor is not None:
            params["after"] = self.cursor
        return params

    def _delivered(self, event: dict[str, Any]) -> None:
        self._remaining()
        cursor = event.get("cursor")
        if not isinstance(cursor, str) or not cursor:
            raise GfaasError("Call event has no valid cursor; cannot resume safely")
        self.cursor = cursor
        self._progress()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.deadline is not None or not self.follow:
            yield from self._poll()
            return
        while True:
            try:
                with self.http.stream(
                    "GET",
                    f"{self.path}/events",
                    params=self._params(follow=True),
                    headers={"accept": "text/event-stream"},
                    timeout=self._timeout(),
                ) as response:
                    if not response.is_success:
                        raise _ObservationError(response)
                    for event in _decode_event_stream(response.iter_lines()):
                        self._delivered(event)
                        yield event
                if self._completed():
                    # Status can become terminal after the stream closes. Drain again
                    # so events written between EOF and the status read are retained.
                    break
                raise httpx.RemoteProtocolError("event stream ended before Call completion")
            except (*_TRANSIENT_ERRORS, _ObservationError) as error:
                self._retry(error)
        yield from self._poll(completed=True)

    def _poll(self, *, completed: bool = False) -> Iterator[dict[str, Any]]:
        while True:
            try:
                page = self._json(f"{self.path}/events", params=self._params(follow=False))
                for event in page.get("items", []):
                    self._delivered(event)
                    yield event
                    if event.get("type") == "state" and event.get("state") in TERMINAL_CALL_STATES:
                        completed = True
                if page.get("next_cursor") is not None:
                    self._progress()
                    continue
                if completed or not self.follow:
                    return
                if self._completed():
                    completed = True
                    continue  # One final drain after observing terminal status.
                self._progress()
                self._sleep(self.poll_interval_s)
            except (*_TRANSIENT_ERRORS, _ObservationError) as error:
                self._retry(error)


def _retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0
    try:
        if value.strip().isdigit():
            return float(value)
        return max(0, parsedate_to_datetime(value).timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        return 0


def _decode_event_stream(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                try:
                    event = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as error:
                    raise GfaasError("event stream returned invalid JSON") from error
                if not isinstance(event, dict):
                    raise GfaasError("event stream payload must be an object")
                yield event
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
    # An incomplete frame is not delivered. Reconnection starts after the last
    # complete event, and replays that frame from the durable event log.
