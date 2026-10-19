"""Retirement previews stay read-only until the caller explicitly applies them."""

from __future__ import annotations

import httpx
import pytest

from gfaas.client import Client, GfaasError
from gfaas.config import ClientConfig
from gfaas_cli import main as cli


def test_cli_retirement_requires_apply_and_reports_server_blockers(capsys):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/functions/fn_example/retirement"
        requests.append(request.method)
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "function_has_active_calls"})
        return httpx.Response(
            200, json={"blockers": ["function_has_active_calls"], "retired_at": None}
        )

    def factory() -> Client:
        client = Client(
            ClientConfig(
                api_base="https://example.test/api",
                api_key="key",
                request_timeout_s=5,
                poll_interval_s=0.1,
            )
        )
        client._http.close()
        client._http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url=client.cfg.api_base
        )
        return client

    assert cli.main(["retire", "function", "fn_example", "--json"], client_factory=factory) == 0
    assert requests == ["GET"]
    assert "function_has_active_calls" in capsys.readouterr().out
    assert cli.main(["retire", "function", "fn_example", "--apply"], client_factory=factory) != 0
    assert requests == ["GET", "POST"]
    with factory() as client:
        with pytest.raises(GfaasError, match="function_has_active_calls"):
            client.retire_resource("function", "fn_example")
        with pytest.raises(ValueError):
            client.retire_resource("function", "fn_example/../fn_other")
