from __future__ import annotations

from gfaas.config import ClientConfig


def test_default_api_base_includes_the_public_api_prefix() -> None:
    config = ClientConfig.from_env({})

    assert config.api_base == "http://127.0.0.1:8000/api"


def test_api_base_removes_a_trailing_slash() -> None:
    config = ClientConfig.from_env({"GFAAS_API_BASE": "https://gpu.example.com/api/"})

    assert config.api_base == "https://gpu.example.com/api"
