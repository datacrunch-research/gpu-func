"""`vfunc login`, `whoami`, and `logout`: per-person API keys through browser single sign-on.

On a workstation, login opens the browser at the service's sign-on page and receives a one-time
code on a 127.0.0.1 listener, then exchanges it with its PKCE verifier for a key. Without a usable
browser it prints a short code to approve from any browser and polls for the key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from gfaas import credentials
from gfaas.config import ClientConfig

from .client import client_config_from_args
from .errors import CliError

AUTH_COMMANDS = frozenset({"login", "whoami", "logout"})
LOGIN_TIMEOUT_S = 600.0

HttpFactory = Callable[[str], httpx.Client]


def add_commands(commands: Any) -> None:
    login = commands.add_parser("login", help="sign in through the browser and store an API key")
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="approve a short code from any browser instead of opening one here",
    )
    login.add_argument("--label", help="name for this key (default: vfunc CLI on HOST)")
    commands.add_parser("whoami", help="show who the configured API key belongs to")
    commands.add_parser("logout", help="revoke the stored API key and forget it")


def dispatch(args: argparse.Namespace, *, http_factory: HttpFactory | None = None) -> int:
    factory = http_factory or (lambda base: httpx.Client(base_url=base, timeout=30.0))
    if args.command_name == "login":
        return login(args, factory)
    if args.command_name == "whoami":
        return whoami(args, factory)
    return logout(args, factory)


def _problem_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("code") if isinstance(body, dict) else None


def _api_base(args: argparse.Namespace) -> str:
    if args.api_base:
        return str(args.api_base).rstrip("/")
    stored = credentials.select(_environment(args), None)
    if stored is not None:
        return stored.api_base
    raise CliError("pass --api-base, for example --api-base https://api.example.com/api")


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    if getattr(args, "profile", None):
        environment[credentials.PROFILE_VARIABLE] = args.profile
    return environment


def _credentials_path(args: argparse.Namespace) -> Any:
    path = credentials.credentials_path(_environment(args))
    if path is None:
        raise CliError("cannot store credentials without HOME or VFUNC_CREDENTIALS")
    return path


def _browser_available(environment: Mapping[str, str]) -> bool:
    """A browser on this machine can reach a listener on this machine's loopback address."""
    remote = environment.get("SSH_CONNECTION") or environment.get("SSH_TTY")
    display = environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
    return not remote or bool(display)


def login(args: argparse.Namespace, http_factory: HttpFactory) -> int:
    api_base = _api_base(args)
    name = args.profile or urlsplit(api_base).hostname or "default"
    label = args.label or f"vfunc CLI on {socket.gethostname()}"[:64]
    environment = _environment(args)
    with http_factory(api_base) as http:
        minted = None
        if not args.no_browser and _browser_available(environment):
            minted = _loopback_login(http, api_base, label)
        if minted is None:
            minted = _device_login(http, label)
    profile = credentials.Profile(
        name=name,
        api_base=api_base,
        api_key=minted["api_key"],
        email=minted.get("email"),
        key_id=minted.get("key_id"),
        expires_at_ms=minted.get("expires_at_ms"),
    )
    path = _credentials_path(args)
    credentials.save(path, profile)
    print(f"✓ Logged in as {profile.email} on {name}")
    print(f"  Key saved to {path} (profile: {name})")
    return 0


class _Callback:
    """The first `/callback` request that the loopback listener receives."""

    def __init__(self) -> None:
        self.received = threading.Event()
        self.query: dict[str, str] = {}

    def handler(self) -> type[BaseHTTPRequestHandler]:
        callback = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the standard library names it
                target = urlsplit(self.path)
                if target.path != "/callback" or callback.received.is_set():
                    self.send_error(404)
                    return
                query = parse_qs(target.query)
                callback.query = {key: values[0] for key, values in query.items() if values}
                callback.received.set()
                body = (
                    b"<!doctype html><title>vfunc</title><body style='font-family:system-ui;"
                    b"margin:4rem auto;max-width:32rem'><h1>Signed in</h1>"
                    b"<p>You can close this tab and return to your terminal.</p></body>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        return Handler


def _loopback_login(http: httpx.Client, api_base: str, label: str) -> dict[str, Any] | None:
    """Sign in through this machine's browser; None when no browser can be opened."""
    callback = _Callback()
    server = HTTPServer(("127.0.0.1", 0), callback.handler())
    redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    query = urlencode(
        {
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "label": label,
        }
    )
    url = f"{api_base}/v1/auth/browser/login?{query}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f"Opening {api_base}/v1/auth/browser/login in your browser…")
        if not webbrowser.open(url):
            print("No browser could be opened; using a login code instead.", file=sys.stderr)
            return None
        print(f"If it did not open, visit:\n  {url}")
        if not callback.received.wait(LOGIN_TIMEOUT_S):
            raise CliError("login timed out waiting for the browser")
    finally:
        server.shutdown()
        server.server_close()
    result = callback.query
    if result.get("state") != state or "code" not in result:
        raise CliError("the browser returned an unexpected login response; try again")
    response = http.post(
        "/v1/auth/token",
        json={
            "grant_type": "authorization_code",
            "code": result["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
    )
    if response.status_code != 200:
        raise CliError(f"login failed: {_problem_code(response) or response.status_code}")
    return dict(response.json())


def _device_login(http: httpx.Client, label: str) -> dict[str, Any]:
    response = http.post("/v1/auth/device", json={"label": label})
    if response.status_code == 404:
        raise CliError("this service does not offer vfunc login")
    response.raise_for_status()
    device = response.json()
    print(f"Open {device['verification_uri_complete']}")
    print(f"and confirm the code {device['user_code']} to sign in.")
    interval = float(device.get("interval", 5))
    deadline = time.monotonic() + float(device.get("expires_in", LOGIN_TIMEOUT_S))
    while time.monotonic() < deadline:
        time.sleep(interval)
        poll = http.post(
            "/v1/auth/token",
            json={"grant_type": "device_code", "device_code": device["device_code"]},
        )
        if poll.status_code == 200:
            return dict(poll.json())
        code = _problem_code(poll)
        if code == "authorization_pending":
            continue
        if code == "slow_down":
            interval += 5
            continue
        raise CliError(f"login failed: {code or poll.status_code}")
    raise CliError("the login code expired; run `vfunc login` again")


def _key_config(args: argparse.Namespace) -> ClientConfig:
    config = client_config_from_args(args)
    if not config.api_key:
        raise CliError("not logged in; run `vfunc login --api-base URL`")
    return config


def whoami(args: argparse.Namespace, http_factory: HttpFactory) -> int:
    config = _key_config(args)
    with http_factory(config.api_base) as http:
        response = http.get("/v1/auth/whoami", headers={"X-API-Key": config.api_key or ""})
    if response.status_code == 401:
        raise CliError("the API key is invalid, expired, or revoked; run `vfunc login`")
    response.raise_for_status()
    identity = response.json()
    if identity.get("kind") == "user":
        remaining = (int(identity["expires_at_ms"]) / 1000 - time.time()) / 86_400
        print(
            f"{identity['email']} · {config.api_base} · key {identity['key_id']} "
            f"expires in {max(remaining, 0):.0f} days"
        )
    else:
        print(f"deployment key for {identity.get('principal_id')} · {config.api_base}")
    return 0


def logout(args: argparse.Namespace, http_factory: HttpFactory) -> int:
    environment = _environment(args)
    if environment.get("GFAAS_API_KEY"):
        raise CliError("GFAAS_API_KEY is set; unset it to log out of a stored profile")
    base = str(args.api_base).rstrip("/") if args.api_base else None
    profile = credentials.select(environment, base)
    if profile is None:
        raise CliError("not logged in")
    with http_factory(profile.api_base) as http:
        response = http.post("/v1/auth/logout", headers={"X-API-Key": profile.api_key})
    if response.status_code not in {204, 401}:
        raise CliError(f"logout failed: {_problem_code(response) or response.status_code}")
    credentials.remove(_credentials_path(args), profile.name)
    print(f"Logged out of {profile.name}; its key is revoked.")
    return 0
