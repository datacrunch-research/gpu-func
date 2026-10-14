from __future__ import annotations

import base64
import hashlib
import json
import stat
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from gfaas import credentials
from gfaas.config import ClientConfig
from gfaas_cli import auth
from gfaas_cli.errors import CliError
from gfaas_cli.parser import build_parser

API = "https://api.example.com/api"
OTHER = "https://other.example.com/api"
MINTED = {
    "api_key": "vfk_minted-key",
    "key_id": "key_1",
    "email": "person@example.com",
    "principal_id": "internal-users",
    "expires_at_ms": 4_102_444_800_000,
}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "vfunc" / "credentials.json"
    monkeypatch.setenv(credentials.PATH_VARIABLE, str(path))
    for name in (
        "GFAAS_API_KEY",
        "GFAAS_API_BASE",
        credentials.PROFILE_VARIABLE,
        "SSH_CONNECTION",
        "SSH_TTY",
    ):
        monkeypatch.delenv(name, raising=False)
    return path


def run(argv: list[str], handler: Any) -> int:
    args = build_parser().parse_args(argv)
    return auth.dispatch(
        args,
        http_factory=lambda base: httpx.Client(
            base_url=base, transport=httpx.MockTransport(handler)
        ),
    )


def problem(code: str) -> httpx.Response:
    return httpx.Response(400, json={"code": code})


def test_profiles_are_private_and_the_last_login_is_the_default(store: Path) -> None:
    credentials.save(store, credentials.Profile("dev", OTHER, "vfk_dev"))
    credentials.save(store, credentials.Profile("gb300dev", API, "vfk_gb300dev"))

    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.parent.stat().st_mode) == 0o700
    stored = credentials.load(store)
    assert stored.default_profile == "gb300dev"
    assert set(stored.profiles) == {"dev", "gb300dev"}

    assert credentials.remove(store, "gb300dev")
    assert credentials.load(store).default_profile == "dev"
    assert not credentials.remove(store, "gb300dev")


def test_a_stored_key_is_sent_only_to_its_api_base(store: Path) -> None:
    credentials.save(store, credentials.Profile("dev", OTHER, "vfk_dev"))
    credentials.save(store, credentials.Profile("gb300dev", API, "vfk_gb300dev"))
    env = {credentials.PATH_VARIABLE: str(store)}

    default = ClientConfig.from_env(env)
    assert (default.api_base, default.api_key) == (API, "vfk_gb300dev")
    other = ClientConfig.from_env({**env, "GFAAS_API_BASE": OTHER + "/"})
    assert (other.api_base, other.api_key) == (OTHER, "vfk_dev")
    unknown = ClientConfig.from_env({**env, "GFAAS_API_BASE": "https://third.example.com/api"})
    assert unknown.api_key is None
    named = ClientConfig.from_env({**env, credentials.PROFILE_VARIABLE: "dev"})
    assert (named.api_base, named.api_key) == (OTHER, "vfk_dev")
    mismatched = ClientConfig.from_env(
        {**env, credentials.PROFILE_VARIABLE: "dev", "GFAAS_API_BASE": API}
    )
    assert mismatched.api_key is None
    explicit = ClientConfig.from_env({**env, "GFAAS_API_KEY": "from-environment"})
    assert (explicit.api_base, explicit.api_key) == (
        "http://127.0.0.1:8000/api",
        "from-environment",
    )


def test_browser_login_exchanges_the_callback_code(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: dict[str, str] = {}

    def browser(url: str) -> bool:
        query = {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}
        opened.update(query, url=url)

        def approve() -> None:
            # A stray request before the callback is ignored.
            httpx.get(query["redirect_uri"].replace("/callback", "/favicon.ico"))
            httpx.get(query["redirect_uri"], params={"code": "one-time", "state": query["state"]})

        threading.Thread(target=approve).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", browser)

    def server(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/token"
        body = json.loads(request.content)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(body["code_verifier"].encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "one-time"
        assert body["redirect_uri"] == opened["redirect_uri"]
        assert challenge == opened["code_challenge"]
        return httpx.Response(200, json=MINTED)

    assert run(["--api-base", API, "--profile", "gb300dev", "login"], server) == 0

    assert opened["url"].startswith(f"{API}/v1/auth/browser/login?")
    assert opened["code_challenge_method"] == "S256"
    assert opened["redirect_uri"].startswith("http://127.0.0.1:")
    profile = credentials.load(store).profiles["gb300dev"]
    assert (profile.api_base, profile.api_key, profile.email) == (
        API,
        "vfk_minted-key",
        "person@example.com",
    )
    assert "Logged in as person@example.com on gb300dev" in capsys.readouterr().out


def test_device_login_polls_until_the_code_is_approved(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)
    polls: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/v1/auth/device":
            assert body["label"] == "tray shell"
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "BCDF-GHJK",
                    "verification_uri": f"{API}/v1/auth/browser/device",
                    "verification_uri_complete": f"{API}/v1/auth/browser/device?user_code=BCDF-GHJK",
                    "expires_in": 600,
                    "interval": 5,
                },
            )
        assert body == {"grant_type": "device_code", "device_code": "device-secret"}
        polls.append("poll")
        return (
            problem("authorization_pending") if len(polls) < 3 else httpx.Response(200, json=MINTED)
        )

    argv = ["--api-base", API, "login", "--no-browser", "--label", "tray shell"]
    assert run(argv, server) == 0

    assert len(polls) == 3
    output = capsys.readouterr().out
    assert "BCDF-GHJK" in output and "user_code=BCDF-GHJK" in output
    assert credentials.load(store).profiles["api.example.com"].api_key == "vfk_minted-key"


def test_a_remote_shell_uses_a_login_code(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.1 50000 192.0.2.2 22")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)
    monkeypatch.setattr(auth.webbrowser, "open", lambda url: pytest.fail("opened a browser"))
    paths: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v1/auth/device":
            return httpx.Response(
                200,
                json={
                    "device_code": "d",
                    "user_code": "BCDF-GHJK",
                    "verification_uri_complete": "https://x",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        return httpx.Response(200, json=MINTED)

    assert run(["--api-base", API, "login"], server) == 0
    assert paths == ["/api/v1/auth/device", "/api/v1/auth/token"]


def test_an_expired_device_login_fails(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)

    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/device":
            return httpx.Response(
                200,
                json={"device_code": "d", "user_code": "C", "verification_uri_complete": "u"},
            )
        return problem("expired_token")

    with pytest.raises(CliError, match="expired_token"):
        run(["--api-base", API, "login", "--no-browser"], server)
    assert not store.exists()


def test_whoami_and_logout_use_the_stored_profile(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials.save(store, credentials.Profile("gb300dev", API, "vfk_stored"))
    revoked: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "vfk_stored"
        if request.url.path == "/api/v1/auth/whoami":
            return httpx.Response(200, json={"kind": "user", **MINTED})
        revoked.append(request.url.path)
        return httpx.Response(204)

    assert run(["whoami"], server) == 0
    assert "person@example.com" in capsys.readouterr().out
    assert run(["logout"], server) == 0
    assert revoked == ["/api/v1/auth/logout"]
    assert credentials.load(store).profiles == {}
    with pytest.raises(CliError, match="not logged in"):
        run(["whoami"], server)


def test_logout_refuses_an_environment_key(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials.save(store, credentials.Profile("gb300dev", API, "vfk_stored"))
    monkeypatch.setenv("GFAAS_API_KEY", "from-environment")

    with pytest.raises(CliError, match="GFAAS_API_KEY is set"):
        run(["logout"], lambda request: pytest.fail("contacted the service"))
    assert "gb300dev" in credentials.load(store).profiles
