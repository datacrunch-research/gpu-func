"""Explicit inventories remain previews until apply, and partial cleanup remains visible."""

import json

import httpx
import pytest
from test_artifact_retention import make_client

from gfaas_cli import main as cli


def test_cli_inventory_preview_apply_and_partial_status(tmp_path, capsys):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(["art_keep", "art_scratch"]))
    seen = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/artifact-cleanup"
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "applied": body["apply"],
                "items": [
                    {"artifact_id": "art_keep", "status": "in_use"},
                    {
                        "artifact_id": "art_scratch",
                        "status": "pending" if body["apply"] else "eligible",
                    },
                ],
            },
        )

    args = [
        "artifact",
        "cleanup",
        "--ids-file",
        str(inventory),
        "--maximum-artifacts",
        "2",
        "--maximum-bytes",
        "500",
    ]
    assert cli.main(args, client_factory=lambda: make_client(handler)) == 0
    assert seen == [
        {
            "artifact_ids": ["art_keep", "art_scratch"],
            "apply": False,
            "maximum_artifacts": 2,
            "maximum_bytes": 500,
        }
    ]
    assert cli.main([*args, "--apply"], client_factory=lambda: make_client(handler)) == 1
    assert seen[-1] == {**seen[0], "apply": True}
    assert "pending" in capsys.readouterr().out
    before = len(seen)
    assert cli.main([*args, "art_extra"], client_factory=lambda: make_client(handler)) != 0
    assert len(seen) == before


@pytest.mark.parametrize(
    "ids,limits",
    [
        ("art_one", {}),
        ([], {}),
        (["art_one", "art_one"], {}),
        (["art_one/../two"], {}),
        (["art_one"], {"maximum_artifacts": 1001}),
        (["art_one"], {"maximum_bytes": 0}),
        (["art_one"], {"maximum_artifacts": True}),
    ],
)
def test_invalid_cleanup_selection_sends_no_request(ids, limits):
    def handler(request):
        raise AssertionError("invalid selection must not reach the service")

    with make_client(handler) as client:
        with pytest.raises(ValueError):
            client.cleanup_artifacts(ids, **limits)
        with pytest.raises(ValueError):
            client.preview_artifact_cleanup(ids, **limits)
