"""Adapter from gpu-func workflows to the current gfaas Python SDK."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Self

from gfaas import ArtifactRef, Client, ClientConfig, RemoteResult
from gfaas.errors import GfaasError

from .errors import CliError
from .worker_job import PROFILE_OUTPUT
from .worker_job import run as run_worker_job

_TERMINAL_CALL_STATES = {"succeeded", "failed", "timed_out", "cancelled"}


class GfaasClient:
    """Small domain adapter; transport and durability stay owned by gfaas."""

    def __init__(self, client: Client, *, poll_interval: float) -> None:
        self._client = client
        self.poll_interval = poll_interval

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        defaults = ClientConfig.from_env()
        config = ClientConfig(
            api_base=args.api_base or defaults.api_base,
            api_key=defaults.api_key,
            poll_interval_s=(
                args.poll_interval if args.poll_interval is not None else defaults.poll_interval_s
            ),
            request_timeout_s=(
                args.request_timeout
                if args.request_timeout is not None
                else defaults.request_timeout_s
            ),
        )
        return cls(Client(config), poll_interval=config.poll_interval_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def capabilities(self) -> dict[str, Any]:
        try:
            return self._client.get_capabilities()
        except GfaasError as exc:
            raise CliError(str(exc)) from exc

    def resolve_gpu_type(self, requested: str | None) -> str:
        if requested and requested != "any":
            return requested
        capabilities = self.capabilities()
        pools = capabilities.get("gpu_pools", [])
        names = [
            pool.get("name") or pool.get("gpu_type") for pool in pools if isinstance(pool, dict)
        ]
        configured = [name for name in names if isinstance(name, str) and name]
        if len(configured) == 1:
            return configured[0]
        if not configured:
            raise CliError("the coordinator reports no configured GPU pools")
        raise CliError(
            "more than one GPU pool is configured; select one with --gpu-type "
            f"({', '.join(sorted(configured))})"
        )

    def submit_job(
        self,
        *,
        job: dict[str, Any],
        workspace: Path,
        args: argparse.Namespace,
        app_name: str,
    ) -> RemoteResult:
        try:
            uploaded = self._client.upload_artifact_directory(
                workspace,
                kind="input",
                filename=f"{app_name}-workspace",
            )
            print(
                f"[gpu-func] workspace artifact={uploaded['id']} "
                f"files={len(uploaded.get('child_artifact_ids', []))}",
                file=sys.stderr,
            )
            return self._client.submit(
                image=args.image,
                function=run_worker_job,
                kwargs={
                    "job": job,
                    "workspace": ArtifactRef(str(uploaded["id"])),
                    "profile_output": PROFILE_OUTPUT,
                },
                gpu_count=args.gpu_count,
                gpu_type=args.gpu_type,
                app_name=app_name,
                timeout_s=args.timeout,
                capacity_wait_s=args.capacity_wait,
                cpu_millicores=args.cpu_millicores,
                memory_bytes=args.memory,
                ephemeral_storage_bytes=args.storage,
                shared_memory_bytes=args.shared_memory,
                max_log_bytes=args.max_log,
                max_output_bytes=args.max_output,
                env=dict(args.env),
                idempotency_key=args.idempotency_key,
                outputs=(PROFILE_OUTPUT,),
            )
        except (GfaasError, OSError, ValueError) as exc:
            raise CliError(str(exc)) from exc

    def wait_for_result(
        self,
        remote: RemoteResult,
        *,
        timeout_s: float | None,
        json_events: bool = False,
    ) -> dict[str, Any]:
        """Follow durable events by cursor, then fetch the terminal result.

        Polling retained event pages deliberately avoids coupling this CLI to a
        single long-lived HTTP response. A transient stream disconnect cannot
        lose the cursor or detach the remote Call from its local identity.
        """
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        cursor: str | None = None
        terminal = False
        try:
            while not terminal:
                page = self._client.list_events(remote.call_id, after=cursor, limit=1000)
                for event in page.get("items", []):
                    self._show_event(event, json_output=json_events)
                    event_cursor = event.get("cursor")
                    if event_cursor is not None:
                        cursor = str(event_cursor)
                    if event.get("type") == "state" and event.get("state") in _TERMINAL_CALL_STATES:
                        terminal = True
                next_cursor = page.get("next_cursor")
                if next_cursor is not None:
                    cursor = str(next_cursor)
                if terminal:
                    break
                call = self._client.get_call(remote.call_id)
                if call.get("state") in _TERMINAL_CALL_STATES:
                    terminal = True
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Call {remote.call_id} did not finish within {timeout_s}s")
                time.sleep(self.poll_interval)
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            result = remote.wait(timeout_s=remaining)
        except KeyboardInterrupt:
            try:
                remote.cancel(reason="gpu-func client interrupted")
                print(
                    f"[gpu-func] cancellation requested call={remote.call_id}",
                    file=sys.stderr,
                )
            except Exception as cancel_error:
                print(
                    f"[gpu-func] cancellation failed call={remote.call_id}: {cancel_error}",
                    file=sys.stderr,
                )
            raise
        except (GfaasError, TimeoutError) as exc:
            raise CliError(str(exc)) from exc
        if not isinstance(result, dict):
            raise CliError("worker returned a non-object result")
        return result

    def profile_publications(self, call_id: str) -> list[dict[str, Any]]:
        try:
            page = self._client.list_call_artifacts(call_id)
        except GfaasError as exc:
            raise CliError(str(exc)) from exc
        return [
            item
            for item in page.get("items", [])
            if isinstance(item, dict) and item.get("name") == PROFILE_OUTPUT.name
        ]

    def download_profiles(self, call_id: str, destination: Path) -> list[Path]:
        publications = self.profile_publications(call_id)
        if not publications:
            return []
        destination.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="gpu-func-profiles-") as temporary:
            temporary_root = Path(temporary)
            for index, publication in enumerate(publications):
                artifact = publication.get("artifact")
                artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
                if not isinstance(artifact_id, str):
                    raise CliError("profile publication has no Artifact identity")
                tree = temporary_root / f"publication-{index}"
                try:
                    self._client.download_artifact_directory(artifact_id, tree)
                except (GfaasError, OSError, ValueError) as exc:
                    raise CliError(str(exc)) from exc
                for source in sorted(path for path in tree.rglob("*") if path.is_file()):
                    relative = source.relative_to(tree)
                    target = destination / relative
                    if target.exists() or target.is_symlink():
                        raise CliError(f"refusing to replace existing profile: {target}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    downloaded.append(target)
        return downloaded

    @staticmethod
    def _show_event(event: dict[str, Any], *, json_output: bool) -> None:
        if json_output:
            print(json.dumps(event, separators=(",", ":"), sort_keys=True))
            return
        event_type = event.get("type")
        if event_type in {"stdout", "stderr"}:
            stream = sys.stdout if event_type == "stdout" else sys.stderr
            print(str(event.get("stream_data", "")), end="", file=stream, flush=True)
            return
        attributes = event.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        if event_type == "state":
            state = event.get("state") or attributes.get("state") or "unknown"
            worker = attributes.get("worker_id")
            suffix = f" worker={worker}" if worker else ""
            print(f"[gpu-func] state={state}{suffix}", file=sys.stderr)
        elif event_type == "preparation":
            print(
                "[gpu-func] preparation"
                f" phase={attributes.get('phase', 'unknown')}"
                f" files={attributes.get('completed_files', 0)}"
                f" bytes={attributes.get('completed_bytes', 0)}",
                file=sys.stderr,
            )
        elif event_type == "diagnostic" and attributes.get("type") == "placement_rejection":
            print(
                "[gpu-func] waiting for capacity"
                f" reason={attributes.get('reason', 'unknown')}"
                f" worker={attributes.get('worker_id', 'unknown')}",
                file=sys.stderr,
            )
