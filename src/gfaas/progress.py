"""Describe current Call preparation without changing the Call lifecycle contract."""

from __future__ import annotations

from typing import Any


def call_status_summary(call: dict[str, Any]) -> str:
    """Format current status, including preparation when the server reports it.

    Preparation is an observation before assignment. A stalled observation means
    no recent progress; it does not establish that a worker is dead.
    """
    state = str(call.get("state", "unknown"))
    parts = [f"state={state}"]
    preparation = call.get("preparation")
    if state not in {"queued", "running"} or not isinstance(preparation, dict):
        return " ".join(parts)
    parts.extend(
        [
            f"preparation={preparation.get('status', 'unknown')}",
            f"phase={preparation.get('phase', 'unknown')}",
            f"worker={preparation.get('worker_id', 'unknown')}",
            f"generation={preparation.get('placement_generation', 'unknown')}",
            f"files={preparation.get('completed_files', 0)}",
            f"bytes={preparation.get('completed_bytes', 0)}",
            f"downloaded={preparation.get('downloaded_bytes', 0)}",
            f"uploaded={preparation.get('uploaded_bytes', 0)}",
        ]
    )
    if preparation.get("total_files") is not None:
        label = "total" if preparation.get("totals_complete") else "known"
        parts.extend(
            [
                f"{label}-files={preparation['total_files']}",
                f"{label}-bytes={preparation.get('total_bytes', 0)}",
            ]
        )
    if isinstance(preparation.get("idle_ms"), int):
        parts.append(f"progress-idle={max(preparation['idle_ms'], 0) // 1000}s")
    return " ".join(parts)
