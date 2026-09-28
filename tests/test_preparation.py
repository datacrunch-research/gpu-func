"""Preparation stays visible during polling, without changing terminal Call states."""

import pytest

from gfaas import call_status_summary
from gfaas_cli.events import show_event


def test_current_preparation_distinguishes_transfer_and_silence() -> None:
    call = {
        "state": "queued",
        "preparation": {
            "status": "active",
            "phase": "transferring",
            "worker_id": "worker-1",
            "placement_generation": 2,
            "completed_files": 0,
            "completed_bytes": 0,
            "downloaded_bytes": 65536,
            "uploaded_bytes": 0,
            "total_files": 1,
            "total_bytes": 1048576,
            "totals_complete": False,
            "idle_ms": 1000,
        },
    }
    summary = call_status_summary(call)
    assert "state=queued preparation=active phase=transferring" in summary
    assert "files=0 bytes=0 downloaded=65536" in summary
    assert "known-files=1 known-bytes=1048576" in summary
    assert "worker=worker-1 generation=2" in summary
    call["preparation"]["status"] = "stalled"
    call["preparation"]["idle_ms"] = 31000
    assert "preparation=stalled" in call_status_summary(call)
    assert "progress-idle=31s" in call_status_summary(call)


@pytest.mark.parametrize("state", ["starting", "cancelling", "cancelled", "succeeded", "failed"])
def test_lifecycle_decisions_override_old_preparation(state: str) -> None:
    assert call_status_summary({"state": state, "preparation": {"status": "active"}}) == (
        f"state={state}"
    )


def test_old_server_and_unknown_status_remain_readable() -> None:
    assert call_status_summary({"state": "queued"}) == "state=queued"
    assert call_status_summary({}) == "state=unknown"


def test_event_shows_bytes_before_the_first_file_completes(capsys) -> None:
    show_event(
        {
            "type": "preparation",
            "attributes": {
                "phase": "transferring",
                "placement_generation": 3,
                "completed_files": 0,
                "completed_bytes": 0,
                "downloaded_bytes": 65536,
                "uploaded_bytes": 4096,
                "total_files": 1,
                "total_bytes": 1048576,
                "totals_complete": True,
            },
        },
        json_output=False,
    )
    text = capsys.readouterr().err
    assert "files=0 bytes=0B downloaded=64.0KiB uploaded=4.0KiB" in text
    assert "generation=3 total-files=1 total-bytes=1.0MiB" in text
