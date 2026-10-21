"""Keep durable SDK recovery state isolated from the developer's cache."""

import pytest


@pytest.fixture(autouse=True)
def transfer_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("transfer-cache")))
