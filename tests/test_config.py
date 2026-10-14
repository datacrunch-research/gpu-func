from __future__ import annotations

from gfaas.config import ClientConfig


def test_default_api_base_includes_the_public_api_prefix() -> None:
    config = ClientConfig.from_env({})

    assert config.api_base == "http://127.0.0.1:8000/api"


def test_api_base_removes_a_trailing_slash() -> None:
    config = ClientConfig.from_env({"GFAAS_API_BASE": "https://gpu.example.com/api/"})

    assert config.api_base == "https://gpu.example.com/api"


def test_upload_concurrency_configuration_is_bounded() -> None:
    import pytest

    assert ClientConfig.from_env({"GFAAS_UPLOAD_CONCURRENCY": "3"}).upload_concurrency == 3
    for value in (0, 17, True, 1.5):
        with pytest.raises(ValueError, match="upload_concurrency"):
            ClientConfig("https://example.invalid", None, 1, 30, upload_concurrency=value)
