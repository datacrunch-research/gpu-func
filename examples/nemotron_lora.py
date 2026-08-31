#!/usr/bin/env python3
"""Fine-tune Nemotron with one or more GPUs on one worker through public gfaas APIs.

This file contains both the local submission program and the remote function.
It does not require access to a worker or to the gfaas deployment files.

The local program does these operations:

1. It uploads the model directory and training file as immutable Artifacts.
2. It submits the remote function with references to those Artifacts.
3. It follows the Call logs and waits for a terminal state.
4. It records the Call state and published outputs in a local JSON report.

The remote function receives the model and data as read-only local paths. It
does not receive storage credentials. It runs NeMo AutoModel without network
access and writes its result below declared output paths.

Before you use this example, ask the operator for a GPU pool and the prepared
``nemo-automodel-lora-cu130`` image. Download the pinned model on the client
host, or supply the identity of an existing model Artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gfaas

MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MODEL_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
IMAGE_NAME = "nemo-automodel-lora-cu130"
AUTOMODEL_TRAINING_FORMAT = "nemo-automodel-agent-chat-v1"
TRAINING_CPU_MILLICORES = 16_000
TRAINING_CPU_THREADS = TRAINING_CPU_MILLICORES // 1_000
MINIMUM_TRAINING_MEMORY_GIB = 128
TRAINING_MEMORY_GIB_PER_GPU = 64
DEFAULT_TRAINING_SCRATCH_GIB = 64
MAXIMUM_CHECKPOINT_VERSIONS = 4
STARTUP_DIAGNOSTIC_INTERVAL_SECONDS = 60
DISTRIBUTED_STRATEGIES = ("fsdp2", "ddp")

HISTORICAL_TOOL_SCHEMAS = {
    "write_plan": {
        "type": "function",
        "function": {
            "name": "write_plan",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of concrete steps to carry out during execution.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A one or two sentence summary of the overall approach.",
                    },
                    "title": {
                        "type": "string",
                        "description": "A short title for the plan.",
                    },
                },
                "required": ["steps"],
            },
        },
    },
    "read_plan": {
        "type": "function",
        "function": {
            "name": "read_plan",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "finish_planning": {
        "type": "function",
        "function": {
            "name": "finish_planning",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
}

# The final adapter is small enough to use directly for evaluation or inference.
ADAPTER_OUTPUT = gfaas.ArtifactOutput.directory(
    "adapter",
    "adapter",
    publish_on_failure=True,
)

# The small JSON output makes run metadata easy to inspect without downloading
# the complete checkpoint tree.
TRAINING_REPORT = gfaas.ArtifactOutput(
    "training-report",
    "training-report.json",
    media_type="application/json",
    publish_on_failure=True,
)

# The resumable checkpoint contains optimizer and scheduler state. It is separate
# from the adapter because most consumers do not need the complete training state.
CHECKPOINT_OUTPUT = gfaas.ArtifactCheckpoint(
    "checkpoint",
    "checkpoint",
    maximum_versions=MAXIMUM_CHECKPOINT_VERSIONS,
)

METRIC_PATTERN = re.compile(
    r"\b(loss|grad_norm|gradient_total|num_label_tokens)(?:\s*[=:]\s*|\s+)"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
WORLD_SIZE_PATTERNS = (
    re.compile(r"\bWorld size:\s*(\d+)\b"),
    re.compile(r"\binitializing torch distributed with\s+(\d+)\s+workers?\b"),
)


class UploadReporter:
    """Show bounded progress for one public Artifact upload."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.last_bucket = -1
        self.terminal = sys.stderr.isatty()

    def __call__(self, item: gfaas.ArtifactUploadProgress) -> None:
        percent = 100 if item.total_bytes == 0 else item.completed_bytes * 100 // item.total_bytes
        bucket = percent // 5
        if (
            not self.terminal
            and bucket == self.last_bucket
            and item.completed_files < item.total_files
        ):
            return
        self.last_bucket = bucket
        completed_gib = item.completed_bytes / 1024**3
        total_gib = item.total_bytes / 1024**3
        message = (
            f"[nemotron-lora] {self.label} {percent:3d}% "
            f"{completed_gib:.2f}/{total_gib:.2f} GiB "
            f"files={item.completed_files}/{item.total_files}"
        )
        final = item.completed_files == item.total_files
        print(
            message, file=sys.stderr, end="\n" if final or not self.terminal else "\r", flush=True
        )


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function", tool)
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _called_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = _tool_name(call)
            if name is not None:
                names.add(name)
    return names


def normalize_training_record(
    record: dict[str, Any],
    *,
    reasoning_mode: str,
    tool_selection: str,
    repairs: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Convert one corpus record to the format used by the Nemotron template."""
    if reasoning_mode not in {"include", "omit"}:
        raise ValueError("reasoning_mode must be include or omit")
    if tool_selection not in {"all", "used"}:
        raise ValueError("tool_selection must be all or used")
    normalized = copy.deepcopy(record)
    messages = normalized.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("training record messages must be a non-empty list")
    if not any(isinstance(message, dict) and message.get("role") == "user" for message in messages):
        raise ValueError("training record must contain a user message")
    called_tool_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each training message must be an object")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("training message content must be a string")
        reasoning = message.pop("reasoning_content", None)
        if reasoning_mode == "include" and isinstance(reasoning, str) and reasoning.strip():
            content = f"{reasoning.strip()}</think>\n{content.lstrip()}"
        message["content"] = content
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("each tool call must be an object")
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    called_tool_ids.add(call_id)
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("each tool call must contain a function object")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    decoded = json.loads(arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError("tool-call arguments must decode to an object")
                    function["arguments"] = decoded
                    if repairs is not None:
                        repairs["decoded_tool_arguments"] += 1
                elif arguments is not None and not isinstance(arguments, dict):
                    raise ValueError("tool-call arguments must be an object")
                arguments = function.get("arguments")
                if function.get("name") == "write_plan" and isinstance(arguments, dict):
                    steps = arguments.get("steps")
                    if isinstance(steps, str):
                        try:
                            decoded_steps = json.loads(steps)
                        except json.JSONDecodeError:
                            if "\\'" not in steps:
                                raise
                            decoded_steps = json.loads(steps.replace("\\'", "'"))
                            if repairs is not None:
                                repairs["repaired_plan_escapes"] += 1
                        if not isinstance(decoded_steps, list) or not all(
                            isinstance(step, str) for step in decoded_steps
                        ):
                            raise ValueError("write_plan steps must decode to a string list")
                        arguments["steps"] = decoded_steps
                        if repairs is not None:
                            repairs["decoded_plan_steps"] += 1
        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in called_tool_ids:
                raise ValueError("tool response has no prior matching tool call")
    tools = normalized.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("training record tools must be a list of objects")
    declared = {_tool_name(tool) for tool in tools}
    used = _called_tool_names(messages)
    for name in sorted(used - declared):
        schema = HISTORICAL_TOOL_SCHEMAS.get(name)
        if schema is None:
            raise ValueError(f"tool call uses undeclared tool {name}")
        tools.append(copy.deepcopy(schema))
        declared.add(name)
        if repairs is not None:
            repairs["restored_tool_schemas"] += 1
    if tool_selection == "used":
        tools = [tool for tool in tools if _tool_name(tool) in used]
    normalized["tools"] = tools
    return normalized


def encode_automodel_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI tool calls to the agent-chat schema in AutoModel 0.5.0."""
    encoded = copy.deepcopy(record)
    output: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    for message in encoded["messages"]:
        role = message.get("role")
        if role == "tool":
            if not pending_call_ids:
                raise ValueError("tool response has no adjacent tool call")
            expected_id = pending_call_ids.pop(0)
            if message.get("tool_call_id") != expected_id:
                raise ValueError("tool response order does not match tool calls")
            output.append({"role": "tool_response", "content": message["content"]})
            continue
        if pending_call_ids:
            raise ValueError("tool call has no adjacent response")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported training message role {role}")
        converted = {"role": role, "content": message["content"]}
        if "weight" in message:
            converted["weight"] = message["weight"]
        output.append(converted)
        calls = message.get("tool_calls")
        if not calls:
            continue
        pending_call_ids = [call["id"] for call in calls]
        for call in calls:
            function = call["function"]
            output.append(
                {
                    "role": "tool_call",
                    "content": json.dumps(
                        {
                            "name": function["name"],
                            "arguments": function.get("arguments", {}),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    encoded["messages"] = output
    encoded["tools"] = json.dumps(
        encoded.get("tools", []),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded["gfaas_training_format"] = AUTOMODEL_TRAINING_FORMAT
    return encoded


def _convert_automodel_training_messages(
    messages: list[dict[str, Any]],
    example_id: int | str | None = None,
    drop_history_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    """Convert agent-chat messages without converting argument objects to strings."""
    output: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    call_counter = 0
    id_prefix = f"call_{example_id}" if example_id is not None else "call"
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool_call":
            calls = []
            pending_call_ids = []
            while index < len(messages) and messages[index].get("role") == "tool_call":
                call = json.loads(messages[index].get("content") or "{}")
                name = call.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool_call message does not contain a tool name")
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool-call arguments must be an object")
                call_id = f"{id_prefix}_{call_counter}"
                call_counter += 1
                pending_call_ids.append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                index += 1
            if (
                output
                and output[-1].get("role") == "assistant"
                and not output[-1].get("tool_calls")
            ):
                output[-1]["tool_calls"] = calls
            else:
                output.append({"role": "assistant", "content": "", "tool_calls": calls})
            continue
        if role in {"tool_response", "tool"}:
            response_index = 0
            while index < len(messages) and messages[index].get("role") in {
                "tool_response",
                "tool",
            }:
                if response_index >= len(pending_call_ids):
                    raise ValueError("tool response has no adjacent tool call")
                content = messages[index].get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                output.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": pending_call_ids[response_index],
                    }
                )
                response_index += 1
                index += 1
            if response_index != len(pending_call_ids):
                raise ValueError("tool call has no adjacent response")
            continue
        pending_call_ids = []
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported AutoModel message role {role}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("training message content must be a string")
        converted = {"role": role, "content": content}
        reasoning = message.get("reasoning_content")
        if role == "assistant" and isinstance(reasoning, str) and reasoning:
            converted["reasoning_content"] = reasoning
        output.append(converted)
        index += 1
    if drop_history_reasoning_content:
        last_assistant = max(
            (index for index, message in enumerate(output) if message.get("role") == "assistant"),
            default=-1,
        )
        for index, message in enumerate(output):
            if index != last_assistant and message.get("role") == "assistant":
                message.pop("reasoning_content", None)
    return output


def render_automodel_training_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Render messages with the conversion used by the training dataset."""
    return _convert_automodel_training_messages(
        record["messages"],
        example_id=record.get("id"),
    )


def make_nemotron_agent_chat_dataset(tokenizer: Any, **kwargs: Any) -> Any:
    """Use AutoModel's dataset adapter while preserving structured tool arguments."""
    automodel_version = version("nemo-automodel")
    if automodel_version != "0.5.0":
        raise RuntimeError(
            f"the Nemotron dataset adapter requires nemo-automodel 0.5.0; found {automodel_version}"
        )
    agent_chat = importlib.import_module("nemo_automodel.components.datasets.llm.agent_chat")
    agent_chat._convert_messages = _convert_automodel_training_messages
    return agent_chat.make_agent_chat_dataset(tokenizer, **kwargs)


def _token_ids(encoded: Any) -> list[Any]:
    """Return one token sequence from a tokenizer result."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer result does not contain input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
        raise TypeError("tokenizer result must contain a token sequence")
    if encoded and isinstance(encoded[0], Sequence):
        if len(encoded) != 1:
            raise ValueError("tokenizer result must contain exactly one sequence")
        encoded = encoded[0]
    return list(encoded)


def _token_lengths(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    messages = record["messages"] if messages is None else messages
    if messages[-1].get("role") != "assistant":
        return 0, 0
    template_args: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    tools = record.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if tools:
        template_args["tools"] = tools
    full = _token_ids(tokenizer.apply_chat_template(messages, **template_args))
    prefix = _token_ids(tokenizer.apply_chat_template(messages[:-1], **template_args))
    if full[: len(prefix)] != prefix:
        raise ValueError("assistant training tokens do not follow the conversation prefix")
    return len(full), len(full) - len(prefix)


def _record_weight(value: Any) -> Fraction:
    """Return one finite positive corpus weight with a bounded denominator."""
    if isinstance(value, bool):
        raise ValueError("record weight must be a number")
    try:
        weight = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("record weight must be a number") from error
    if weight <= 0:
        raise ValueError("record weight must be positive")
    if weight.denominator > 100:
        raise ValueError("record weight denominator exceeds 100")
    return weight


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_training_data(
    source: Path,
    destination: Path,
    *,
    tokenizer_path: Path,
    sequence_length: int,
    maximum_samples: int | None,
    reasoning_mode: str,
    tool_selection: str,
) -> dict[str, Any]:
    """Write weighted records that fit and contain supervised tokens.

    AutoModel does not accept a per-record loss weight. The output therefore
    represents rational source weights with the smallest exact integer number
    of record copies. A second qualification pass sees unit weights and does
    not expand the records again.
    """
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    counts = {
        "scanned": 0,
        "selected": 0,
        "emitted_samples": 0,
        "invalid": 0,
        "too_long": 0,
        "without_labels": 0,
        "decoded_tool_arguments": 0,
        "decoded_plan_steps": 0,
        "repaired_plan_escapes": 0,
        "restored_tool_schemas": 0,
        "weight_scale": 0,
        "record_weights": {},
        "invalid_reasons": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    spool_path: Path | None = None
    output_path: Path | None = None
    weights: set[Fraction] = set()
    weight_counts: dict[str, int] = {}
    invalid_reasons: dict[str, int] = counts["invalid_reasons"]
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".records",
            delete=False,
        ) as spool_file:
            spool_path = Path(spool_file.name)
            with source.open(encoding="utf-8") as input_file:
                for line in input_file:
                    if maximum_samples is not None and counts["selected"] >= maximum_samples:
                        break
                    if not line.strip():
                        continue
                    counts["scanned"] += 1
                    record_repairs = {
                        "decoded_tool_arguments": 0,
                        "decoded_plan_steps": 0,
                        "repaired_plan_escapes": 0,
                        "restored_tool_schemas": 0,
                    }
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            raise ValueError("record must be an object")
                        weight = _record_weight(raw.get("weight", 1))
                        if raw.get("gfaas_training_format") == AUTOMODEL_TRAINING_FORMAT:
                            record = copy.deepcopy(raw)
                        else:
                            record = normalize_training_record(
                                raw,
                                reasoning_mode=reasoning_mode,
                                tool_selection=tool_selection,
                                repairs=record_repairs,
                            )
                            if record["messages"][-1].get("weight", 1) != 1:
                                raise ValueError("final assistant message must have weight 1")
                            record = encode_automodel_training_record(record)
                        rendered_messages = render_automodel_training_messages(record)
                        if (
                            not rendered_messages
                            or rendered_messages[-1].get("role") != "assistant"
                        ):
                            raise ValueError("training record must end with an assistant message")
                        total_tokens, label_tokens = _token_lengths(
                            tokenizer,
                            record,
                            messages=rendered_messages,
                        )
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        for name, value in record_repairs.items():
                            counts[name] += value
                        reason = str(error)
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        counts["invalid"] += 1
                        continue
                    for name, value in record_repairs.items():
                        counts[name] += value
                    if label_tokens == 0:
                        counts["without_labels"] += 1
                        continue
                    if total_tokens > sequence_length:
                        counts["too_long"] += 1
                        continue
                    record["weight"] = 1
                    weight_text = str(weight)
                    weights.add(weight)
                    weight_counts[weight_text] = weight_counts.get(weight_text, 0) + 1
                    spool_file.write(
                        json.dumps(
                            {
                                "weight_numerator": weight.numerator,
                                "weight_denominator": weight.denominator,
                                "record": record,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts["selected"] += 1

        if counts["selected"] == 0:
            raise ValueError(
                f"no training records fit the sequence length with supervised labels: {counts}"
            )

        weight_scale = math.lcm(*(weight.denominator for weight in weights))
        counts["weight_scale"] = weight_scale
        counts["record_weights"] = dict(sorted(weight_counts.items()))
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".output",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)
            with spool_path.open(encoding="utf-8") as spool_file:
                for line in spool_file:
                    item = json.loads(line)
                    repetitions = (
                        item["weight_numerator"] * weight_scale // item["weight_denominator"]
                    )
                    encoded = json.dumps(item["record"], separators=(",", ":")) + "\n"
                    for _ in range(repetitions):
                        output_file.write(encoded)
                    counts["emitted_samples"] += repetitions
        os.replace(output_path, destination)
        output_path = None
        return counts
    finally:
        if spool_path is not None:
            spool_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def prepare_checkpoint_outputs(checkpoint_root: Path, adapter_root: Path) -> list[dict[str, Any]]:
    """Replace checkpoint links and create a compact final adapter package."""
    latest_checkpoint = resolve_latest_checkpoint(checkpoint_root)
    files = []
    for path in sorted(checkpoint_root.rglob("*")):
        if path.is_symlink():
            target = os.readlink(path)
            path.unlink()
            path.write_text(target + "\n", encoding="utf-8")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(checkpoint_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
    adapter_weights = (
        next(
            (
                path
                for path in latest_checkpoint.rglob("adapter_model.safetensors")
                if path.is_file()
            ),
            None,
        )
        if latest_checkpoint is not None
        else None
    )
    if adapter_weights is None:
        return files
    adapter_files = [adapter_weights]
    adapter_files.extend(path for path in adapter_weights.parent.glob("*.json") if path.is_file())
    for source in adapter_files:
        destination = adapter_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return files


def resolve_latest_checkpoint(checkpoint_root: Path) -> Path | None:
    """Resolve AutoModel's completed checkpoint without leaving its root."""
    latest = checkpoint_root / "LATEST"
    try:
        target = os.readlink(latest) if latest.is_symlink() else latest.read_text().strip()
    except FileNotFoundError:
        return None
    relative = Path(target)
    if not target or relative.is_absolute() or len(relative.parts) != 1 or relative.name != target:
        raise RuntimeError(f"AutoModel wrote an invalid LATEST checkpoint target: {target!r}")
    resolved = checkpoint_root / relative
    if not resolved.is_dir():
        raise RuntimeError(f"AutoModel's latest checkpoint is unavailable: {resolved}")
    return resolved


def publish_latest_checkpoint(
    checkpoint_root: Path,
    published: dict[str, int],
) -> None:
    """Publish one completed AutoModel checkpoint version at most once."""
    latest = resolve_latest_checkpoint(checkpoint_root)
    if latest is None or latest.name in published:
        return
    generation = CHECKPOINT_OUTPUT.publish(latest.name)
    published[latest.name] = generation
    print(
        "[nemotron-lora] checkpoint"
        f" directory={latest.name} generation={generation} publication=requested",
        flush=True,
    )


def restore_checkpoint(source: Path, checkpoint_root: Path) -> None:
    """Copy one immutable checkpoint Artifact into AutoModel's writable root."""
    if not source.is_dir():
        raise ValueError(f"resume checkpoint is not a directory: {source}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_root.iterdir()):
        raise RuntimeError(f"checkpoint directory is not empty: {checkpoint_root}")
    restored = checkpoint_root / "resume"
    shutil.copytree(source, restored)
    (checkpoint_root / "LATEST").symlink_to(restored.name)


def validate_training_metrics(metrics: dict[str, float | int]) -> None:
    """Reject a process that completed without a useful optimizer step."""
    label_tokens = metrics.get("num_label_tokens")
    loss = metrics.get("loss")
    gradient = metrics.get("grad_norm", metrics.get("gradient_total"))
    if not isinstance(label_tokens, int) or label_tokens <= 0:
        raise RuntimeError("AutoModel did not report positive supervised label tokens")
    if not isinstance(loss, float) or not math.isfinite(loss) or loss <= 0:
        raise RuntimeError("AutoModel did not report a finite positive loss")
    if not isinstance(gradient, float) or not math.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("AutoModel did not report a finite positive gradient norm")


def parse_training_metrics(line: str) -> dict[str, float | int]:
    """Read the metrics that AutoModel writes in one training log line."""
    return {
        name: int(float(value)) if name == "num_label_tokens" else float(value)
        for name, value in METRIC_PATTERN.findall(line)
    }


def parse_world_size(line: str) -> int | None:
    """Read AutoModel's distributed process count from one log line."""
    for pattern in WORLD_SIZE_PATTERNS:
        match = pattern.search(line)
        if match is not None:
            return int(match.group(1))
    return None


def training_threads_per_rank(total_threads: int, gpu_count: int) -> int:
    """Divide the Call's native CPU threads across its GPU ranks."""
    if total_threads < 1:
        raise ValueError("total training threads must be positive")
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    return max(1, total_threads // gpu_count)


def checkpoint_interval(max_steps: int, requested_steps: int | None) -> int:
    """Select an interval that fits the bounded checkpoint publication budget."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    interval = math.ceil(max_steps / MAXIMUM_CHECKPOINT_VERSIONS)
    if requested_steps is not None:
        if requested_steps < 1:
            raise ValueError("checkpoint_every_steps must be positive")
        interval = requested_steps
    if interval > max_steps:
        raise ValueError("checkpoint_every_steps must not exceed max_steps")
    versions = math.ceil(max_steps / interval)
    if versions > MAXIMUM_CHECKPOINT_VERSIONS:
        raise ValueError(
            "checkpoint_every_steps creates too many checkpoint versions: "
            f"maximum {MAXIMUM_CHECKPOINT_VERSIONS}, requested {versions}"
        )
    return interval


def _process_tree(root_pid: int) -> list[int]:
    """Return the live Linux process tree rooted at one PID."""
    pending = [root_pid]
    found: list[int] = []
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        found.append(pid)
        pending.extend(int(child) for child in children)
    return sorted(found)


def _process_usage(pids: Sequence[int]) -> tuple[float, int, int]:
    """Read aggregate CPU, memory, and thread counters from procfs."""
    clock_ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    cpu_ticks = 0
    resident_pages = 0
    threads = 0
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            cpu_ticks += int(fields[11]) + int(fields[12])
            resident_pages += int(Path(f"/proc/{pid}/statm").read_text().split()[1])
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("Threads:"):
                    threads += int(line.split()[1])
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return cpu_ticks / clock_ticks, resident_pages * page_size, threads


def _gpu_usage() -> str:
    """Read one bounded GPU usage summary for startup diagnostics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    values = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            values.append(f"{fields[0]}:{fields[1]}%/{fields[2]}MiB")
    return ",".join(values) or "unavailable"


def monitor_training_start(
    process: subprocess.Popen[str],
    *,
    started_at: float,
    first_step: threading.Event,
    stop: threading.Event,
    interval_seconds: float = STARTUP_DIAGNOSTIC_INTERVAL_SECONDS,
) -> None:
    """Report bounded process and GPU counters until the first optimizer step."""
    while not first_step.is_set() and not stop.wait(interval_seconds):
        if process.poll() is not None:
            return
        pids = _process_tree(process.pid)
        cpu_seconds, resident_bytes, threads = _process_usage(pids)
        print(
            "[nemotron-lora] startup-diagnostic"
            f" elapsed={int(time.monotonic() - started_at)}s"
            f" processes={len(pids)}"
            f" cpu_seconds={cpu_seconds:.1f}"
            f" rss_gib={resident_bytes / 1024**3:.2f}"
            f" threads={threads}"
            f" gpu={_gpu_usage()}",
            flush=True,
        )


def configure_distributed_diagnostics(environment: dict[str, str]) -> None:
    """Enable bounded initialization logs without logging every collective."""
    environment["NCCL_DEBUG"] = "INFO"
    environment["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH"
    environment["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def stop_subprocess(process: subprocess.Popen[str]) -> None:
    """Stop an owned AutoModel subprocess after a wrapper error."""
    if process.poll() is not None:
        return
    process_tree = _process_tree(process.pid)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        for pid in reversed(process_tree):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        process.wait(timeout=30)


def automodel_command(recipe_path: Path, gpu_count: int) -> list[str]:
    """Create the same-node distributed AutoModel command."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    return ["automodel", str(recipe_path), "--nproc-per-node", str(gpu_count)]


def configure_local_rendezvous(environment: dict[str, str], gpu_count: int) -> None:
    """Use an IP-literal torchrun rendezvous for a local multi-rank job."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if gpu_count == 1:
        return
    # PyTorch otherwise selects localhost:0 for a one-node c10d rendezvous.
    # Minimal container roots do not always include a localhost NSS entry.
    environment["PET_RDZV_BACKEND"] = "c10d"
    environment["PET_RDZV_ENDPOINT"] = "127.0.0.1:0"
    environment["PET_LOCAL_ADDR"] = "127.0.0.1"


def make_recipe(
    *,
    model_path: Path,
    training_data_path: Path,
    checkpoint_path: Path,
    max_steps: int,
    sequence_length: int,
    maximum_samples: int | None,
    gpu_count: int,
    checkpoint_every_steps: int,
    use_triton: bool,
    distributed_strategy: str,
    shuffle: bool,
    seed: int,
) -> dict[str, Any]:
    """Create a bounded, same-worker NeMo AutoModel recipe.

    The recipe uses only paths that gfaas stages inside the function. It does
    not use a Hugging Face model name or a remote dataset name.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if sequence_length < 128:
        raise ValueError("sequence_length must be at least 128")
    if maximum_samples is not None and maximum_samples < 1:
        raise ValueError("maximum_samples must be positive")
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if checkpoint_every_steps < 1 or checkpoint_every_steps > max_steps:
        raise ValueError("checkpoint_every_steps must be between 1 and max_steps")
    if distributed_strategy not in DISTRIBUTED_STRATEGIES:
        raise ValueError(f"unsupported distributed strategy: {distributed_strategy}")
    if seed < 0:
        raise ValueError("seed must not be negative")

    # The bundled adapter preserves tool argument objects for templates that
    # iterate their keys. AutoModel 0.5.0 otherwise converts these objects to
    # JSON strings before it invokes the model's chat template.
    dataset: dict[str, Any] = {
        "_target_": f"{Path(__file__).stem}.make_nemotron_agent_chat_dataset",
        "path": str(training_data_path),
        "seq_length": sequence_length,
        # The qualification pass rejects long records before AutoModel starts.
        # Do not truncate a supervised assistant turn during training.
        "truncation": False,
        "train_on_last_turn_only": True,
        "mask_reasoning_content": False,
        "drop_history_reasoning_content": False,
        "truncate_history": False,
        "tokenizer": {
            "pretrained_model_name_or_path": str(model_path),
            "trust_remote_code": True,
            "pad_token_id": 0,
        },
    }
    if maximum_samples is not None:
        dataset["limit_dataset_samples"] = maximum_samples

    return {
        # Use AutoModel's standard next-token fine-tuning loop.
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        # Each data-parallel rank receives one local sample. The global batch
        # therefore grows with the number of GPUs in this same-worker Call.
        "step_scheduler": {
            "global_batch_size": gpu_count,
            "local_batch_size": 1,
            "ckpt_every_steps": checkpoint_every_steps,
            "val_every_steps": max_steps + 1,
            "max_steps": max_steps,
        },
        # AutoModel initializes one NCCL process group on this worker.
        "dist_env": {"backend": "nccl", "timeout_minutes": 20},
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": seed,
            "ranked": True,
        },
        # Load the exact local model tree. ``force_hf`` selects the qualified
        # Transformers implementation used by the official Nemotron recipe.
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": str(model_path),
            "trust_remote_code": True,
            "force_hf": True,
            "attn_implementation": "sdpa",
        },
        # Compilation can make a one-step qualification slower and less clear.
        # A later throughput run can qualify compilation separately.
        "compile": {
            "enabled": False,
            "mode": "default",
            "fullgraph": False,
            "dynamic": True,
            "backend": None,
        },
        # Save a final consolidated adapter in the declared output directory.
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(checkpoint_path),
            "model_save_format": "safetensors",
            "save_consolidated": "final",
        },
        # Apply a small LoRA adapter. Nemotron's output projections are excluded
        # because the official AutoModel recipe excludes them.
        "peft": {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "exclude_modules": ["*.out_proj"],
            "dim": 8,
            "alpha": 32,
            "use_triton": use_triton,
        },
        "distributed": distributed_config(distributed_strategy),
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
        "dataset": dataset,
        "packed_sequence": {"packed_sequence_size": 0},
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "shuffle": shuffle,
        },
        # These conservative values are for a functional smoke test. They are
        # not a recommended quality recipe for a complete corpus.
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {"lr_decay_style": "cosine", "min_lr": 1e-6},
        "nvtx": False,
    }


def validate_model_directory(path: Path) -> None:
    """Make sure that a local directory contains the expected pinned model."""
    if not path.is_dir():
        raise ValueError(f"model directory does not exist: {path}")
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    tokenizer_path = path / "tokenizer.json"
    for required in (config_path, index_path, tokenizer_path):
        if not required.is_file():
            raise ValueError(f"model directory is missing {required.name}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "nemotron_h":
        raise ValueError("model config does not identify Nemotron-H")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index is empty")
    missing = sorted(
        filename
        for filename in set(weight_map.values())
        if not isinstance(filename, str) or not (path / filename).is_file()
    )
    if missing:
        raise ValueError(f"model directory is missing weight files: {missing}")


def configure_cache_paths(scratch: Path) -> dict[str, str]:
    """Put library caches below the SDK-provided scratch directory."""
    cache_root = scratch / "cache"
    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_MODULES_CACHE": cache_root / "huggingface" / "modules",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TMPDIR": scratch / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return {name: str(path) for name, path in paths.items()}


def configure_training_threads(thread_count: int) -> dict[str, str]:
    """Bound native thread pools to the CPUs requested by this function."""
    if thread_count < 1:
        raise ValueError("training thread count must be positive")
    value = str(thread_count)
    limits = {
        "OMP_NUM_THREADS": value,
        "OMP_THREAD_LIMIT": value,
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_DYNAMIC": "false",
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "BLIS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }
    os.environ.update(limits)
    return limits


def training_runtime_metadata(torch_module: Any) -> dict[str, str | int]:
    """Return runtime identity as plain strings that any SDK client can decode."""
    return {
        "device": str(torch_module.cuda.get_device_name(0)),
        "visible_gpu_count": int(torch_module.cuda.device_count()),
        # torch.__version__ is a torch.torch_version.TorchVersion subclass.
        # Keep that implementation type out of the pickled Call result.
        "torch_version": str(torch_module.__version__),
        "nemo_automodel_version": str(version("nemo-automodel")),
        "transformers_version": str(version("transformers")),
    }


def train_nemotron(
    model: gfaas.ArtifactRef,
    training_data: gfaas.ArtifactRef,
    resume_checkpoint: gfaas.ArtifactRef | None,
    model_name: str,
    model_revision: str,
    gpu_count: int,
    max_steps: int,
    sequence_length: int,
    maximum_samples: int | None,
    reasoning_mode: str,
    tool_selection: str,
    checkpoint_every_steps: int,
    use_triton: bool,
    startup_diagnostics: bool,
    distributed_strategy: str,
    shuffle: bool,
    seed: int,
) -> dict[str, Any]:
    """Run one offline AutoModel LoRA job inside a gfaas function."""
    # Ask the SDK for the real writable path. Do not assume that the worker
    # mounts scratch at a fixed container path.
    scratch = gfaas.scratch_path()
    configure_cache_paths(scratch)
    cpu_threads_per_rank = training_threads_per_rank(TRAINING_CPU_THREADS, gpu_count)
    configure_training_threads(cpu_threads_per_rank)

    # These packages come from the prepared operator image. The client does not
    # install packages when it submits the Call.
    import torch  # type: ignore[import-not-found]
    import yaml  # type: ignore[import-not-found]

    started = time.monotonic()

    # Scratch space is private and temporary. Output paths are separate because
    # gfaas publishes only files below declared output roots.
    recipe_path = scratch / "nemotron-lora.yaml"
    qualified_data_path = scratch / "qualified-training.jsonl"
    try:
        qualification = qualify_training_data(
            training_data.path,
            qualified_data_path,
            tokenizer_path=model.path,
            sequence_length=sequence_length,
            maximum_samples=maximum_samples,
            reasoning_mode=reasoning_mode,
            tool_selection=tool_selection,
        )
    except Exception as error:
        with TRAINING_REPORT.open("w", encoding="utf-8") as output:
            json.dump(
                {
                    "model": model_name,
                    "model_revision": model_revision,
                    "model_artifact_id": model.artifact_id,
                    "training_data_artifact_id": training_data.artifact_id,
                    "resume_checkpoint_artifact_id": (
                        resume_checkpoint.artifact_id if resume_checkpoint is not None else None
                    ),
                    "gpu_count": gpu_count,
                    "phase": "qualification",
                    "error": str(error),
                },
                output,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
        raise
    print(f"[nemotron-lora] qualification={json.dumps(qualification, sort_keys=True)}", flush=True)

    checkpoint_path = CHECKPOINT_OUTPUT.path
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is not None:
        restore_checkpoint(resume_checkpoint.path, checkpoint_path)
        print(
            f"[nemotron-lora] restored checkpoint_artifact={resume_checkpoint.artifact_id}",
            flush=True,
        )

    # ArtifactRef.path resolves to a read-only file or directory. AutoModel can
    # use these paths like normal local inputs.
    recipe = make_recipe(
        model_path=model.path,
        training_data_path=qualified_data_path,
        checkpoint_path=checkpoint_path,
        max_steps=max_steps,
        sequence_length=sequence_length,
        maximum_samples=maximum_samples,
        gpu_count=gpu_count,
        checkpoint_every_steps=checkpoint_every_steps,
        use_triton=use_triton,
        distributed_strategy=distributed_strategy,
        shuffle=shuffle,
        seed=seed,
    )
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    command = automodel_command(recipe_path, gpu_count)
    print(
        f"[nemotron-lora] start model={model_name} steps={max_steps} "
        f"sequence_length={sequence_length} gpu_count={gpu_count} "
        f"checkpoint_every={checkpoint_every_steps} triton={str(use_triton).lower()} "
        f"startup_diagnostics={str(startup_diagnostics).lower()} "
        f"distributed_strategy={distributed_strategy}",
        flush=True,
    )
    # Do not raise here. The code first writes a diagnostic report so gfaas can
    # publish it when AutoModel returns a failure status.
    metrics: dict[str, float | int] = {}
    observed_world_sizes: set[int] = set()
    published_checkpoints: dict[str, int] = {}
    process_env = os.environ.copy()
    configure_local_rendezvous(process_env, gpu_count)
    if startup_diagnostics:
        configure_distributed_diagnostics(process_env)
    source_directory = str(Path(__file__).resolve().parent)
    python_path = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        source_directory if not python_path else f"{source_directory}{os.pathsep}{python_path}"
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )
    first_step = threading.Event()
    diagnostic_stop = threading.Event()
    diagnostic_thread = None
    if startup_diagnostics:
        diagnostic_thread = threading.Thread(
            target=monitor_training_start,
            kwargs={
                "process": process,
                "started_at": started,
                "first_step": first_step,
                "stop": diagnostic_stop,
            },
            name="nemotron-startup-diagnostics",
            daemon=True,
        )
        diagnostic_thread.start()
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            line_metrics = parse_training_metrics(line)
            metrics.update(line_metrics)
            if "loss" in line_metrics and not first_step.is_set():
                first_step.set()
                print(
                    "[nemotron-lora] phase=first-optimizer-step"
                    f" elapsed={time.monotonic() - started:.3f}s",
                    flush=True,
                )
            world_size = parse_world_size(line)
            if world_size is not None:
                observed_world_sizes.add(world_size)
            publish_latest_checkpoint(checkpoint_path, published_checkpoints)
        returncode = process.wait()
        publish_latest_checkpoint(checkpoint_path, published_checkpoints)
    except BaseException:
        stop_subprocess(process)
        raise
    finally:
        diagnostic_stop.set()
        if diagnostic_thread is not None:
            diagnostic_thread.join(timeout=5)

    # Record every produced file. This inventory helps a caller check the
    # output shape before it downloads the directory Artifact.
    files = prepare_checkpoint_outputs(CHECKPOINT_OUTPUT.path, ADAPTER_OUTPUT.path)
    report = {
        "model": model_name,
        "model_revision": model_revision,
        "model_artifact_id": model.artifact_id,
        "training_data_artifact_id": training_data.artifact_id,
        "resume_checkpoint_artifact_id": (
            resume_checkpoint.artifact_id if resume_checkpoint is not None else None
        ),
        "gpu_count": gpu_count,
        "cpu_threads_per_rank": cpu_threads_per_rank,
        "observed_world_sizes": sorted(observed_world_sizes),
        "max_steps": max_steps,
        "checkpoint_every_steps": checkpoint_every_steps,
        "checkpoint_versions": published_checkpoints,
        "sequence_length": sequence_length,
        "maximum_samples": maximum_samples,
        "shuffle": shuffle,
        "seed": seed,
        "use_triton": use_triton,
        "startup_diagnostics": startup_diagnostics,
        "distributed_strategy": distributed_strategy,
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        **training_runtime_metadata(torch),
        "qualification": qualification,
        "training_metrics": metrics,
        "files": files,
    }
    # This report is a separate declared Artifact. It remains small even when
    # the checkpoint directory contains large files.
    with TRAINING_REPORT.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")

    # A successful process is not sufficient. Require the adapter file that a
    # later inference or evaluation Call needs.
    if returncode != 0:
        raise RuntimeError(f"AutoModel stopped with status {returncode}")
    if not (ADAPTER_OUTPUT.path / "adapter_model.safetensors").is_file():
        raise RuntimeError("AutoModel did not save adapter_model.safetensors")
    if observed_world_sizes != {gpu_count}:
        raise RuntimeError(
            "AutoModel did not report the requested distributed world size: "
            f"expected {gpu_count}, observed {sorted(observed_world_sizes)}"
        )
    if report["visible_gpu_count"] != gpu_count:
        raise RuntimeError(
            "the training container did not expose the requested GPU count: "
            f"expected {gpu_count}, observed {report['visible_gpu_count']}"
        )
    validate_training_metrics(metrics)

    print(f"[nemotron-lora] complete duration_seconds={report['duration_seconds']}", flush=True)
    return report


def positive_integer(value: str) -> int:
    """Parse one positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def training_memory_bytes(gpu_count: int, memory_gib: int | None) -> int:
    """Resolve the Call memory request for local multi-GPU training."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if memory_gib is not None:
        if memory_gib < 1:
            raise ValueError("memory_gib must be positive")
        return memory_gib * 1024**3
    default_gib = max(MINIMUM_TRAINING_MEMORY_GIB, TRAINING_MEMORY_GIB_PER_GPU * gpu_count)
    return default_gib * 1024**3


def training_scratch_bytes(scratch_gib: int) -> int:
    """Resolve the Call limit for writable ephemeral storage."""
    if scratch_gib < 1:
        raise ValueError("scratch_gib must be positive")
    return scratch_gib * 1024**3


def distributed_config(strategy: str) -> dict[str, Any]:
    """Create one supported AutoModel distributed strategy section."""
    if strategy == "fsdp2":
        return {
            "strategy": "fsdp2",
            "dp_size": None,
            "tp_size": 1,
            "cp_size": 1,
            "sequence_parallel": False,
        }
    if strategy == "ddp":
        return {
            "strategy": "ddp",
            "dp_size": None,
            "find_unused_parameters": True,
        }
    raise ValueError(f"unsupported distributed strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # A caller can upload local inputs once, then reuse the printed Artifact
    # identities for every later experiment.
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="local model directory to upload")
    model.add_argument("--model-artifact", help="existing model tree Artifact ID")
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="model repository name to record in the training report",
    )
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION,
        help="pinned model revision to record in the training report",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        help="local tokenizer directory for client-side corpus qualification",
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--training-data", type=Path, help="local training JSONL file to upload")
    data.add_argument("--training-data-artifact", help="existing training Artifact ID")
    parser.add_argument(
        "--resume-checkpoint-artifact",
        help="durable checkpoint Artifact to restore before training",
    )
    # Pool and image names are operator-provided public names. They do not grant
    # direct access to a worker.
    parser.add_argument("--gpu-type", required=True, help="operator GPU-pool name")
    parser.add_argument(
        "--gpu-count",
        type=positive_integer,
        default=1,
        help="GPUs to lease on one worker and use as AutoModel ranks (default: 1)",
    )
    parser.add_argument(
        "--memory-gib",
        type=positive_integer,
        help="Call memory limit in GiB (default: 128 GiB or 64 GiB per GPU, whichever is larger)",
    )
    parser.add_argument(
        "--scratch-gib",
        type=positive_integer,
        default=DEFAULT_TRAINING_SCRATCH_GIB,
        help="writable ephemeral-storage limit in GiB (default: 64)",
    )
    parser.add_argument("--image", default=IMAGE_NAME, help="operator image name")
    parser.add_argument("--max-steps", type=positive_integer, default=1)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=positive_integer,
        help=("checkpoint interval (default: divide the run into at most four durable versions)"),
    )
    parser.add_argument(
        "--triton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use AutoModel's Triton LoRA implementation (default: enabled)",
    )
    parser.add_argument(
        "--startup-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="report process, GPU, NCCL, and distributed initialization details before step 0",
    )
    parser.add_argument(
        "--distributed-strategy",
        choices=DISTRIBUTED_STRATEGIES,
        default="fsdp2",
        help="multi-GPU parameter strategy (default: fsdp2)",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle qualified records before training (default: enabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1111,
        help="reproducible training and shuffle seed (default: 1111)",
    )
    sample_limit = parser.add_mutually_exclusive_group()
    sample_limit.add_argument(
        "--maximum-samples",
        type=int,
        help="maximum qualified records for this bounded pilot (default: 8)",
    )
    sample_limit.add_argument(
        "--full-corpus",
        action="store_const",
        const=None,
        dest="maximum_samples",
        help="qualify every source record instead of using the bounded pilot",
    )
    parser.set_defaults(maximum_samples=8)
    parser.add_argument(
        "--reasoning-mode",
        choices=("include", "omit"),
        default="include",
        help="include reasoning_content in assistant content or omit it",
    )
    parser.add_argument(
        "--tool-selection",
        choices=("used", "all"),
        default="used",
        help="retain only called tools or retain every tool definition",
    )
    # Capacity wait covers all time before worker acceptance. This period
    # includes image preparation and Artifact staging. Timeout covers execution.
    # The local log follower allows both periods plus a short final margin.
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--capacity-wait",
        type=int,
        default=3600,
        help="maximum pre-assignment time, including Artifact staging, in seconds",
    )
    parser.add_argument(
        "--detach-on-interrupt",
        action="store_true",
        help="leave the remote Call active when the local command receives Ctrl-C",
    )
    parser.add_argument("--report", type=Path, help="write the local submission report here")
    return parser.parse_args()


def _input_ref(
    client: gfaas.Client,
    *,
    artifact_id: str | None,
    path: Path | None,
    directory: bool,
    label: str,
) -> gfaas.ArtifactRef:
    """Resolve an existing Artifact or upload one local input."""
    if artifact_id is not None:
        # Fail before submission if the caller cannot read this Artifact.
        client.get_artifact(artifact_id)
        return gfaas.ArtifactRef(artifact_id)
    if path is None:
        raise ValueError(f"{label} path is required")
    if directory:
        validate_model_directory(path)
    print(f"[nemotron-lora] uploading {label}: {path}", flush=True)
    reporter = UploadReporter(label)
    if directory:
        uploaded = client.upload_artifact_directory(path, kind="input", progress=reporter)
    else:
        uploaded = client.upload_artifact_file(path, kind="input", progress=reporter)
    print(f"[nemotron-lora] {label}_artifact={uploaded['id']}", flush=True)
    return gfaas.ArtifactRef(uploaded["id"])


def follow_call(
    job: gfaas.RemoteResult,
    *,
    timeout_s: float,
    capacity_wait_s: int,
    status_interval_s: float = 30,
) -> None:
    """Print bounded preparation progress and remote process output."""
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    cursor = None
    last_event = "submission"
    last_event_at = started_at
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(f"call {job.call_id} event stream exceeded {timeout_s}s")
        try:
            events = job.iter_events(
                after=cursor,
                timeout_s=min(status_interval_s, remaining_s),
            )
            for event in events:
                event_cursor = event.get("cursor")
                if event_cursor is not None:
                    cursor = str(event_cursor)
                last_event_at = time.monotonic()
                event_type = event.get("type")
                last_event = str(event_type or "unknown")
                if event_type in {"stdout", "stderr"}:
                    target = sys.stderr if event_type == "stderr" else sys.stdout
                    print(event.get("stream_data", ""), end="", file=target, flush=True)
                    continue
                attributes = event.get("attributes")
                if event_type == "artifact" and isinstance(attributes, dict):
                    generation = attributes.get("generation")
                    artifact_id = attributes.get("artifact_id")
                    if isinstance(generation, int) and isinstance(artifact_id, str):
                        last_event = "artifact:checkpoint"
                        print(
                            "[nemotron-lora] checkpoint"
                            f" generation={generation} artifact={artifact_id} durable=true",
                            flush=True,
                        )
                    continue
                if event_type == "diagnostic" and isinstance(attributes, dict):
                    if attributes.get("type") != "placement_rejection":
                        continue
                    last_event = "diagnostic:placement_rejection"
                    elapsed_s = min(int(last_event_at - started_at), capacity_wait_s)
                    capacity_remaining_s = max(capacity_wait_s - elapsed_s, 0)
                    print(
                        "[nemotron-lora] waiting for capacity"
                        f" reason={attributes.get('reason', 'unknown')}"
                        f" worker={attributes.get('worker_id', 'unknown')}"
                        f" generation={attributes.get('placement_generation', 'unknown')}"
                        f" elapsed={elapsed_s}s"
                        f" remaining={capacity_remaining_s}s",
                        flush=True,
                    )
                    continue
                if event_type != "preparation":
                    continue
                if not isinstance(attributes, dict):
                    continue
                phase = attributes.get("phase", "unknown")
                last_event = f"preparation:{phase}"
                print(
                    "[nemotron-lora] preparation"
                    f" phase={phase}"
                    f" files={attributes.get('completed_files', 0)}"
                    f" bytes={attributes.get('completed_bytes', 0)}"
                    f" worker={attributes.get('worker_id', 'unknown')}",
                    flush=True,
                )
            # The durable event iterator returns normally only after the Call
            # reaches a terminal state.
            return
        except TimeoutError:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"call {job.call_id} event stream exceeded {timeout_s}s"
                ) from None
            try:
                call_state = job.status().get("state", "unknown")
            except Exception:
                call_state = "unknown"
            print(
                "[nemotron-lora] still waiting"
                f" state={call_state}"
                f" last={last_event}"
                f" idle={int(now - last_event_at)}s"
                f" elapsed={int(now - started_at)}s",
                flush=True,
            )


def interrupt_message(job: gfaas.RemoteResult, *, detach: bool) -> str:
    """Cancel an interrupted foreground Call unless the caller requested detachment."""
    if detach:
        return "local wait stopped; the remote Call remains active"
    try:
        cancellation = job.cancel(reason="client interrupted while waiting")
    except Exception as error:
        return (
            "local wait stopped; remote Call cancellation failed and the Call may "
            f"still be active: {error}"
        )
    return (
        "local wait stopped; remote Call cancellation requested"
        f" (state={cancellation.get('state', 'unknown')})"
    )


def main() -> int:
    args = parse_args()
    memory_bytes = training_memory_bytes(args.gpu_count, args.memory_gib)
    scratch_bytes = training_scratch_bytes(args.scratch_gib)
    checkpoint_every_steps = checkpoint_interval(
        args.max_steps,
        args.checkpoint_every_steps,
    )
    temporary = tempfile.TemporaryDirectory(prefix="gfaas-nemotron-")
    local_qualification = None
    training_data_path = args.training_data
    tokenizer_path = args.tokenizer_model or args.model
    if training_data_path is not None and tokenizer_path is not None:
        qualified_path = Path(temporary.name) / "qualified-training.jsonl"
        local_qualification = qualify_training_data(
            training_data_path,
            qualified_path,
            tokenizer_path=tokenizer_path,
            sequence_length=args.sequence_length,
            maximum_samples=args.maximum_samples,
            reasoning_mode=args.reasoning_mode,
            tool_selection=args.tool_selection,
        )
        local_qualification.update(
            {
                "source_sha256": file_sha256(args.training_data),
                "output_sha256": file_sha256(qualified_path),
                "output_size_bytes": qualified_path.stat().st_size,
            }
        )
        training_data_path = qualified_path
        print(
            f"[nemotron-lora] local_qualification="
            f"{json.dumps(local_qualification, sort_keys=True)}",
            flush=True,
        )
    elif training_data_path is not None:
        print(
            "[nemotron-lora] client tokenizer is unavailable; "
            "the remote function will qualify the corpus",
            file=sys.stderr,
        )

    # The client uses only GFAAS_API_BASE and GFAAS_API_KEY. All following
    # operations use the same public API that an SDK user can access.
    with gfaas.Client() as client:
        model = _input_ref(
            client,
            artifact_id=args.model_artifact,
            path=args.model,
            directory=True,
            label="model",
        )
        training_data = _input_ref(
            client,
            artifact_id=args.training_data_artifact,
            path=training_data_path,
            directory=False,
            label="training_data",
        )
        resume_checkpoint = None
        if args.resume_checkpoint_artifact is not None:
            client.get_artifact(args.resume_checkpoint_artifact)
            resume_checkpoint = gfaas.ArtifactRef(args.resume_checkpoint_artifact)
        # The Call requests resources, not a named worker. The operator policy
        # validates each value and selects a compatible worker from the pool.
        job = client.submit(
            image=gfaas.Image(args.image),
            function=train_nemotron,
            args=(
                model,
                training_data,
                resume_checkpoint,
                args.model_name,
                args.model_revision,
                args.gpu_count,
                args.max_steps,
                args.sequence_length,
                args.maximum_samples,
                args.reasoning_mode,
                args.tool_selection,
                checkpoint_every_steps,
                args.triton,
                args.startup_diagnostics,
                args.distributed_strategy,
                args.shuffle,
                args.seed,
            ),
            gpu_count=args.gpu_count,
            gpu_type=args.gpu_type,
            app_name="nemotron-nano-lora",
            timeout_s=args.timeout,
            capacity_wait_s=args.capacity_wait,
            cpu_millicores=TRAINING_CPU_MILLICORES,
            memory_bytes=memory_bytes,
            ephemeral_storage_bytes=scratch_bytes,
            shared_memory_bytes=8 * 1024**3,
            max_log_bytes=64 * 1024**2,
            max_output_bytes=8 * 1024**3,
            # Disable network-backed model and dataset lookup. Cache paths use
            # writable scratch space instead of the read-only image filesystem.
            env={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "GLOO_SOCKET_IFNAME": "lo",
            },
            # The function can publish only the outputs declared here.
            outputs=(ADAPTER_OUTPUT, TRAINING_REPORT, CHECKPOINT_OUTPUT),
        )
        print(f"[nemotron-lora] call={job.call_id}", flush=True)

        # A bounded event follower includes queue and execution time. It does
        # not change either server-side limit.
        total_wait = args.capacity_wait + args.timeout + 60
        result = None
        failure = None
        exit_status = 0
        try:
            follow_call(
                job,
                timeout_s=total_wait,
                capacity_wait_s=args.capacity_wait,
            )
            result = job.wait(timeout_s=args.timeout + 60)
        except KeyboardInterrupt:
            message = interrupt_message(job, detach=args.detach_on_interrupt)
            failure = {"type": "KeyboardInterrupt", "message": message}
            exit_status = 130
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            exit_status = 1
        # Keep useful public metadata even when the function fails. These
        # lookups also find outputs that were published on a failure path.
        try:
            status = job.status()
        except Exception as error:
            status = {"lookup_error": str(error)}
        try:
            artifacts = job.artifacts()
        except Exception as error:
            artifacts = {"items": [], "lookup_error": str(error)}
        try:
            attempts = client.list_attempts(job.call_id)
        except Exception as error:
            attempts = {"items": [], "lookup_error": str(error)}

    # The local report is not a remote output. It lets an operator or tutorial
    # reader reproduce the exact Call with retained Artifact identities.
    local_report = {
        "call_id": job.call_id,
        "model": args.model_name,
        "model_revision": args.model_revision,
        "model_artifact_id": model.artifact_id,
        "training_data_artifact_id": training_data.artifact_id,
        "resume_checkpoint_artifact_id": (
            resume_checkpoint.artifact_id if resume_checkpoint is not None else None
        ),
        "resources": {
            "cpu_millicores": TRAINING_CPU_MILLICORES,
            "gpu_count": args.gpu_count,
            "memory_bytes": memory_bytes,
            "ephemeral_storage_bytes": scratch_bytes,
        },
        "checkpoint_every_steps": checkpoint_every_steps,
        "triton": args.triton,
        "startup_diagnostics": args.startup_diagnostics,
        "distributed_strategy": args.distributed_strategy,
        "local_qualification": local_qualification,
        "status": status,
        "result": result,
        "error": failure,
        "artifacts": artifacts,
        "attempts": attempts,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(local_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for item in artifacts.get("items", []):
        artifact = item.get("artifact") or {}
        print(f"[nemotron-lora] output={item.get('name')} artifact={artifact.get('id')}")
    if failure is not None:
        summary = failure["message"].splitlines()[0]
        print(f"[nemotron-lora] ERROR: {summary}", file=sys.stderr)
        if args.report is not None:
            print(f"[nemotron-lora] report={args.report}", file=sys.stderr)
    temporary.cleanup()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
