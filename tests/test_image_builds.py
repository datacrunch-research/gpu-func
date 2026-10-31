from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import gfaas
from gfaas.client import Client
from gfaas.config import ClientConfig
from gfaas.errors import GfaasError
from gfaas_cli import images
from gfaas_cli import main as cli
from gfaas_cli.errors import CliError

FLASHINFER = gfaas.Image.from_container(
    "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
    python_version="3.12",
    pip_packages=["cloudpickle==3.1.2", "flashinfer-python[cu13]==0.7.0.post1"],
)
DIGEST = "sha256:" + "ab" * 32
EXPERIMENT = """
import gfaas
import flashinfer  # not installed locally; the CLI must not import this file

CUDA = "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
PACKAGES = ["cloudpickle==3.1.2", "flashinfer-python[cu13]==0.7.0.post1"]
image = gfaas.Image.from_container(CUDA, python_version="3.12", pip_packages=PACKAGES)
app = gfaas.App("exp", image=image)


@app.function(gpu_count=1)
def bench() -> str:
    return flashinfer.__version__
"""


def _client(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> Client:
    client = Client(
        ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="secret",
            poll_interval_s=0.01,
            request_timeout_s=1,
            **config,
        )
    )
    client._http = httpx.Client(
        base_url="https://gpu.example.com/api",
        headers={"X-API-Key": "secret"},
        transport=httpx.MockTransport(handler),
    )
    return client


def test_container_image_names_end_with_a_hash_of_the_specification() -> None:
    # fast-container's image build service tests pin the same names.
    named = gfaas.Image.from_container(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        python_version="3.12",
        pip_packages=["cloudpickle==3.1.2", "flashinfer-python[cu13]==0.7.0.post1"],
        name="flashinfer-cu13",
    )
    changed = gfaas.Image.from_container(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        python_version="3.12",
        pip_packages=["cloudpickle==3.1.2", "flashinfer-python[cu13]==0.7.0"],
        name="flashinfer-cu13",
    )

    assert FLASHINFER.name == "img-eazw5y"
    assert named.name == "flashinfer-cu13-eazw5y"
    assert named.build_spec == FLASHINFER.build_spec
    assert changed.name.startswith("flashinfer-cu13-") and changed.name != named.name
    for invalid in ("-flashinfer", "flash infer", "x" * 122, "cuda/13"):
        with pytest.raises(ValueError, match="start and end with a letter or digit"):
            gfaas.Image.from_container("python:3.12", name=invalid)


def test_request_image_build_sends_the_specification_with_the_api_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"name": FLASHINFER.name, "state": "queued"})

    with _client(handler) as client:
        build = client.request_image_build(FLASHINFER)

    assert build["state"] == "queued"
    assert str(requests[0].url) == "https://gpu.example.com/api/v1/image-builds/builds"
    assert requests[0].headers["X-API-Key"] == "secret"
    assert json.loads(requests[0].content) == {
        "spec": FLASHINFER.build_spec,
        "name": FLASHINFER.name,
    }


def test_image_build_service_location_can_be_configured() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"state": "building", "offset": 4, "text": "step"})

    with _client(handler, image_builds_url="https://builder.example.com/") as client:
        chunk = client.read_image_build_log(FLASHINFER.name, offset=2)

    assert chunk["offset"] == 4
    assert urls == [f"https://builder.example.com/builds/{FLASHINFER.name}/log?offset=2"]
    assert (
        ClientConfig.from_env({"GFAAS_IMAGE_BUILDS_URL": "https://b.example.com"}).image_builds_url
        == "https://b.example.com"
    )


def test_image_build_requests_reject_unservable_images() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "not your principal"})

    renamed = gfaas.Image(name="custom", build_spec=FLASHINFER.build_spec)
    with _client(handler) as client:
        with pytest.raises(GfaasError, match="does not end with its specification's hash"):
            client.request_image_build(renamed)
        with pytest.raises(GfaasError, match="no container build specification"):
            client.request_image_build(gfaas.Image("pytorch-cu130"))
        with pytest.raises(GfaasError, match="NAME-xxxxxx"):
            client.get_image_build("../../v1/calls")
        with pytest.raises(GfaasError, match="does not publish images for this API key"):
            client.request_image_build(FLASHINFER)


def test_submit_selects_a_container_image_by_its_published_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("def train():\n    return 1\n")
    environments: list[dict[str, Any]] = []
    client = _client(lambda request: httpx.Response(500))
    monkeypatch.setattr(client, "upload_artifact", lambda data, **kwargs: {"id": "art_1"})

    def create_environment(definition: dict[str, Any]) -> dict[str, Any]:
        environments.append(definition)
        return {"id": "env_1"}

    monkeypatch.setattr(client, "create_environment", create_environment)
    monkeypatch.setattr(client, "create_function", lambda definition: {"id": "fn_1"})
    monkeypatch.setattr(client, "create_call", lambda request, **kwargs: {"id": "call_1"})

    with client:
        client.submit(image=FLASHINFER, function=("experiment", "train"), source_file=source)

    assert environments[0]["source"] == {"kind": "registered_image", "name": FLASHINFER.name}


def test_submit_explains_an_unpublished_container_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "experiment.py"
    source.write_text("def train():\n    return 1\n")
    client = _client(lambda request: httpx.Response(404, json={"detail": "not found"}))
    monkeypatch.setattr(client, "upload_artifact", lambda data, **kwargs: {"id": "art_1"})

    with client, pytest.raises(GfaasError, match="vfunc image publish") as captured:
        client.submit(image=FLASHINFER, function=("experiment", "train"), source_file=source)
    assert captured.value.status_code == 404

    plain = _client(lambda request: httpx.Response(404, json={"detail": "not found"}))
    monkeypatch.setattr(plain, "upload_artifact", lambda data, **kwargs: {"id": "art_1"})
    with plain, pytest.raises(GfaasError, match="environment creation failed: 404"):
        plain.submit(image="missing", function=("experiment", "train"), source_file=source)


def test_declared_images_are_read_without_importing_the_file(tmp_path: Path) -> None:
    source = tmp_path / "exp.py"
    source.write_text(EXPERIMENT)

    assert images.declared_images(source) == {"image": FLASHINFER}
    assert images.declared_image(source) == ("image", FLASHINFER)
    assert images.declared_image(source, "image") == ("image", FLASHINFER)
    with pytest.raises(CliError, match="does not assign app = Image.from_container"):
        images.declared_image(source, "app")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("from gfaas import Image\nimage = Image.from_container(base())\n", "literals"),
        ("import gfaas\nimage = gfaas.Image.from_container(*ARGS)\n", r"\* or \*\*"),
        ("import gfaas\nimage = gfaas.Image.from_container('x', gpu=1)\n", "unexpected keyword"),
        (
            "import gfaas\na = gfaas.Image.from_container('x')\nb = gfaas.Image.from_container('y')\n",
            "several container images",
        ),
    ],
)
def test_declared_images_report_unsupported_declarations(
    tmp_path: Path, text: str, message: str
) -> None:
    source = tmp_path / "exp.py"
    source.write_text(text)

    with pytest.raises(CliError, match=message):
        images.declared_image(source)


def test_declared_images_keep_their_custom_name(tmp_path: Path) -> None:
    source = tmp_path / "exp.py"
    source.write_text(
        EXPERIMENT.replace(
            "pip_packages=PACKAGES)", 'pip_packages=PACKAGES, name="flashinfer-cu13")'
        )
    )

    assert images.declared_image(source) == (
        "image",
        gfaas.Image.from_container(
            FLASHINFER.build_spec["base_image"],
            python_version="3.12",
            pip_packages=FLASHINFER.build_spec["pip_packages"],
            name="flashinfer-cu13",
        ),
    )
    assert images.declared_image(source)[1].name == "flashinfer-cu13-eazw5y"


def test_a_reassigned_variable_uses_its_last_module_level_value(tmp_path: Path) -> None:
    source = tmp_path / "exp.py"
    source.write_text(
        "import gfaas\n"
        "BASE = 'python:3.11'\n"
        "image = gfaas.Image.from_container(BASE)\n"
        "BASE = 'python:3.12'\n"
        "image: gfaas.Image = gfaas.Image.from_container(BASE)\n"
        "if True:\n"
        "    image = gfaas.Image.from_container('ignored')\n"
    )

    assert images.declared_image(source) == ("image", gfaas.Image.from_container("python:3.12"))


class FakeBuildClient:
    def __init__(self, states: list[str], *, error: str | None = None) -> None:
        self.cfg = ClientConfig(
            api_base="https://gpu.example.com/api",
            api_key="secret",
            poll_interval_s=0,
            request_timeout_s=1,
        )
        self.states = states
        self.error = error
        self.requests: list[tuple[gfaas.Image, str | None]] = []
        self.offsets: list[int] = []

    def __enter__(self) -> FakeBuildClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def _build(self, state: str) -> dict[str, Any]:
        build: dict[str, Any] = {"name": FLASHINFER.name, "state": state, "error": None}
        if state == "ready":
            build["digest"] = DIGEST
        if state == "failed":
            build["error"] = self.error
        return build

    def request_image_build(
        self, image: gfaas.Image, *, architecture: str | None = None
    ) -> dict[str, Any]:
        self.requests.append((image, architecture))
        return self._build(self.states[0])

    def read_image_build_log(self, name: str, *, offset: int = 0) -> dict[str, Any]:
        self.offsets.append(offset)
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        text = "" if state in {"ready", "failed"} else f"{state}\n"
        return {"state": state, "offset": offset + len(text), "text": text}

    def get_image_build(self, name: str) -> dict[str, Any]:
        return self._build(self.states[-1])


def test_image_publish_follows_the_build_until_it_is_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "exp.py"
    source.write_text(EXPERIMENT)
    client = FakeBuildClient(["queued", "building", "publishing", "ready"])
    monkeypatch.setattr(images.time, "sleep", lambda _seconds: None)

    status = cli.main(["image", "publish", f"{source}:image"], client_factory=lambda: client)

    captured = capsys.readouterr()
    assert status == 0
    assert client.requests == [(FLASHINFER, None)]
    assert captured.out == f"{FLASHINFER.name}\n"
    assert "building\npublishing\n" in captured.err
    assert f"digest={DIGEST}" in captured.err
    assert client.offsets == [0, 7, 16, 27]


def test_image_publish_reports_a_failed_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "exp.py"
    source.write_text(EXPERIMENT)
    client = FakeBuildClient(["building", "failed"], error="build failed with exit status 1")
    monkeypatch.setattr(images.time, "sleep", lambda _seconds: None)

    status = cli.main(
        ["image", "publish", str(source), "--architecture", "arm64"],
        client_factory=lambda: client,
    )

    assert status == 1
    assert client.requests == [(FLASHINFER, "arm64")]
    assert "image build failed: build failed with exit status 1" in capsys.readouterr().err


def test_image_publish_can_return_after_acceptance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "exp.py"
    source.write_text(EXPERIMENT)
    client = FakeBuildClient(["queued"])

    status = cli.main(
        ["image", "publish", str(source), "--no-wait", "--json"], client_factory=lambda: client
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["state"] == "queued"
    assert client.offsets == []


def test_image_status_reads_the_build(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "exp.py"
    source.write_text(EXPERIMENT)

    status = cli.main(
        ["image", "status", f"{source}:image"], client_factory=lambda: FakeBuildClient(["ready"])
    )

    assert status == 0
    assert capsys.readouterr().out == f"{FLASHINFER.name}\n"


class FakeListClient(FakeBuildClient):
    def list_image_builds(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "flashinfer-cu13-eazw5y",
                "state": "ready",
                "updated_ms": 1793361900986,
                "digest": DIGEST,
            },
            {"name": "img-hak5jy", "state": "failed", "updated_ms": 1793361000000},
        ]


@pytest.mark.parametrize("command", ["image", "images"])
def test_image_list_shows_the_principals_builds(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeListClient(["ready"])

    assert cli.main([command, "list"], client_factory=lambda: client) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["NAME", "STATE", "UPDATED", "(UTC)", "DIGEST"]
    assert lines[1].split() == [
        "flashinfer-cu13-eazw5y",
        "ready",
        "2026-10-30",
        time_of(1793361900986),
        DIGEST[:19],
    ]
    assert lines[2].split()[:2] == ["img-hak5jy", "failed"]

    assert cli.main([command, "list", "--json"], client_factory=lambda: client) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def time_of(milliseconds: int) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(milliseconds / 1000))


def test_list_image_builds_reads_the_principals_builds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gpu.example.com/api/v1/image-builds/builds"
        return httpx.Response(200, json={"items": [{"name": "img-eazw5y", "state": "ready"}]})

    with _client(handler) as client:
        assert client.list_image_builds() == [{"name": "img-eazw5y", "state": "ready"}]


class FakeSubmitClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []

    def __enter__(self) -> FakeSubmitClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, **kwargs: Any) -> Any:
        self.submissions.append(kwargs)
        return type("Remote", (), {"call_id": "call_test"})()


@pytest.mark.parametrize(
    ("text", "arguments", "expected"),
    [
        (EXPERIMENT, [], FLASHINFER),
        (EXPERIMENT, ["--image", "research"], "research"),
        ("print('no image')\n", [], "pytorch-cu130"),
        ("match = 1\nprint(f'{match!r:>{10}}' if True else (yield))\n", [], "pytorch-cu130"),
        ("def broken(:\n", [], "pytorch-cu130"),
    ],
)
def test_run_uses_the_files_declared_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    arguments: list[str],
    expected: gfaas.Image | str,
) -> None:
    monkeypatch.delenv("GFAAS_IMAGE", raising=False)
    source = tmp_path / "exp.py"
    source.write_text(text)
    client = FakeSubmitClient()

    for target in (str(source), f"{source}:bench"):
        assert cli.main(["run", target, "--detach", *arguments], client_factory=lambda: client) == 0

    assert [submission["image"] for submission in client.submissions] == [expected, expected]


def no_client() -> Client:
    raise AssertionError("the dashboard command must not need an API client")


@pytest.fixture
def dashboard(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {"get": [], "open": []}

    def get(url: str, **_: Any) -> httpx.Response:
        calls["get"].append(url)
        return httpx.Response(200, text="<h1>Image builds</h1>")

    def open_url(url: str) -> bool:
        calls["open"].append(url)
        return True

    monkeypatch.setattr(images.httpx, "get", get)
    monkeypatch.setattr(images.webbrowser, "open", open_url)
    monkeypatch.delenv("GFAAS_IMAGE_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    return calls


def test_image_dashboard_opens_the_forwarded_dashboard(
    dashboard: dict[str, list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["image-dashboard"], client_factory=no_client) == 0

    assert dashboard == {
        "get": [images.DEFAULT_DASHBOARD_URL],
        "open": [images.DEFAULT_DASHBOARD_URL],
    }
    assert capsys.readouterr().out == f"Opened {images.DEFAULT_DASHBOARD_URL}\n"


def test_image_dashboard_prints_the_address_without_a_browser(
    dashboard: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GFAAS_IMAGE_DASHBOARD_URL", "http://127.0.0.1:9091/")

    assert cli.main(["image-dashboard", "--no-browser"], client_factory=no_client) == 0
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 50000 10.0.0.2 22")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert cli.main(["image-dashboard"], client_factory=no_client) == 0

    assert dashboard["open"] == []
    assert capsys.readouterr().out == "http://127.0.0.1:9091/\n" * 2


def test_image_dashboard_explains_a_missing_forward(
    dashboard: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(url: str, **_: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(images.httpx, "get", refuse)

    assert cli.main(["image-dashboard"], client_factory=no_client) == 1

    assert dashboard["open"] == []
    assert "ssh -N -L 8091:127.0.0.1:8091 BUILD_HOST" in capsys.readouterr().err
