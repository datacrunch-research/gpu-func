from __future__ import annotations

import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gfaas import Client

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "document",
    ["README.md", "docs/artifacts/outputs.md", "docs/images/qualification.md"],
)
def test_gpu_example_submits_only_from_main_with_a_gpu(document, tmp_path, monkeypatch):
    """Exercise the published examples without contacting a service or allocating a GPU."""
    text = (ROOT / document).read_text()
    snippet = re.search(r"^```python\n(.*?)^```", text, re.M | re.S)
    assert snippet is not None
    source = tmp_path / "example.py"
    source.write_text(snippet[1])
    submitted = []

    def submit(self, **request):
        submitted.append(request)
        return SimpleNamespace(
            call_id="call_example",
            wait=lambda: "example result",
            artifacts=lambda: {"items": []},
            iter_events=lambda: iter(()),
        )

    monkeypatch.setattr(Client, "submit", submit)
    runpy.run_path(str(source), run_name="imported_example")
    assert not submitted, "remote workers import the module before invoking its function"
    runpy.run_path(str(source), run_name="__main__")
    assert len(submitted) == 1
    assert submitted[0]["gpu_count"] == 1
    assert submitted[0]["gpu_type"] == "gb300"
