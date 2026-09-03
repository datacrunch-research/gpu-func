from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import gfaas

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]*]\(([^)]+)\)")


def test_documented_public_sdk_surface_is_available() -> None:
    expected = {
        "App",
        "ArtifactCheckpoint",
        "ArtifactOutput",
        "ArtifactRef",
        "ArtifactTreeUploadError",
        "ArtifactUploadProgress",
        "Client",
        "ClientConfig",
        "CudaCompilationError",
        "CudaError",
        "CudaProcessError",
        "CudaSource",
        "Function",
        "Image",
        "RemoteResult",
        "UnsupportedGpuPoolError",
        "compile_and_run",
        "scratch_path",
    }

    assert set(gfaas.__all__) == expected
    for name in expected:
        assert getattr(gfaas, name) is not None


def test_package_contains_only_client_modules() -> None:
    assert importlib.util.find_spec("gfaas.image_release") is None
    assert importlib.util.find_spec("gfaas.wire") is None
    assert importlib.util.find_spec("gfaas.cli") is None
    assert importlib.util.find_spec("gpu_func_cli") is None


def test_project_dependencies_are_self_contained() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = document["project"]["dependencies"]

    assert all("../gfaas" not in dependency for dependency in dependencies)
    assert "sources" not in document.get("tool", {}).get("uv", {})


def test_project_installs_one_gfaas_command() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert document["project"]["scripts"] == {"gfaas": "gfaas_cli.main:entrypoint"}


def test_project_declares_the_apache_license() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert document["project"]["license"] == "Apache-2.0"
    assert document["project"]["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").is_file()


def test_documentation_local_links_resolve() -> None:
    documents = [ROOT / "README.md", ROOT / "GUIDE.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    documents.extend((ROOT / "examples" / "cli").glob("*.md"))

    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = unquote(match.group(1).split("#", 1)[0])
            if not target or "://" in target:
                continue
            assert (document.parent / target).exists(), f"broken link in {document}: {target}"


def test_documentation_uses_public_service_placeholders() -> None:
    text = "\n".join(
        document.read_text(encoding="utf-8") for document in (ROOT / "docs").rglob("*.md")
    )

    assert "export GFAAS_API_BASE=https://gpu.example.com/api" in text
    assert "docs/admin" not in text
