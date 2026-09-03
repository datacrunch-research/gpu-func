"""Render durable Call events for all CLI workflows."""

from __future__ import annotations

import json
import sys
from typing import Any


def _format_bytes(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return str(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for candidate in units[1:]:
        if amount < 1024:
            break
        amount /= 1024
        unit = candidate
    if unit == "B":
        return f"{value}B"
    return f"{amount:.1f}{unit}"


def _format_milliseconds(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return str(value)
    if value < 1000:
        return f"{value:g}ms"
    return f"{value / 1000:.2f}s"


def _event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes = event.get("attributes")
    return attributes if isinstance(attributes, dict) else {}


def _human_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    attributes = _event_attributes(event)

    if event_type == "state":
        state = event.get("state") or attributes.get("state") or "unknown"
        parts = [f"state={state}"]
        worker_id = attributes.get("worker_id")
        if worker_id:
            parts.append(f"worker={worker_id}")
        resources = attributes.get("resources")
        if isinstance(resources, dict) and isinstance(resources.get("gpus"), list):
            parts.append(f"gpus={len(resources['gpus'])}")
        timing = attributes.get("timing")
        if isinstance(timing, dict):
            for name in ("total_ms", "queue_ms", "execute_ms"):
                if name in timing:
                    label = name.removesuffix("_ms")
                    parts.append(f"{label}={_format_milliseconds(timing[name])}")
        return " ".join(parts)

    if event_type == "preparation":
        parts = [f"preparation phase={attributes.get('phase', 'unknown')}"]
        if attributes.get("worker_id"):
            parts.append(f"worker={attributes['worker_id']}")
        parts.append(f"files={attributes.get('completed_files', 0)}")
        parts.append(f"bytes={_format_bytes(attributes.get('completed_bytes', 0))}")
        details = attributes.get("details")
        if isinstance(details, dict):
            for name in ("image_name", "image_id", "artifact_id"):
                if details.get(name):
                    label = name.removesuffix("_name").removesuffix("_id")
                    parts.append(f"{label}={details[name]}")
            if "size_bytes" in details:
                parts.append(f"size={_format_bytes(details['size_bytes'])}")
        return " ".join(parts)

    if event_type == "diagnostic":
        diagnostic_type = attributes.get("type", "unknown")
        if diagnostic_type == "placement_rejection":
            return (
                "waiting for capacity"
                f" reason={attributes.get('reason', 'unknown')}"
                f" worker={attributes.get('worker_id', 'unknown')}"
                f" generation={attributes.get('placement_generation', 'unknown')}"
            )
        return f"diagnostic type={diagnostic_type}"

    if event_type == "artifact":
        parts = ["artifact"]
        for name in ("role", "name", "generation", "artifact_id"):
            if name in attributes:
                label = "id" if name == "artifact_id" else name
                parts.append(f"{label}={attributes[name]}")
        return " ".join(parts)

    if event_type == "retention.truncated":
        return "retained event stream is truncated"

    return None


def show_event(
    event: dict[str, Any],
    *,
    json_output: bool,
    json_envelope: bool = True,
) -> None:
    """Write one event as JSON or concise terminal output."""
    if json_output:
        value = {"type": "call_event", "event": event} if json_envelope else event
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return
    event_type = event.get("type")
    if event_type in {"stdout", "stderr"}:
        stream = sys.stdout if event_type == "stdout" else sys.stderr
        print(str(event.get("stream_data", "")), end="", file=stream, flush=True)
        return
    summary = _human_event(event)
    if summary is not None:
        print(f"[gfaas] {summary}", file=sys.stderr)
        return
    print(f"[gfaas] event={json.dumps(event, sort_keys=True)}", file=sys.stderr)
