#!/usr/bin/env python3
"""Compare upload concurrency through the public API; emit one JSON record per upload.

Uses GFAAS_API_BASE and GFAAS_API_KEY. Uploads random content and deletes the resulting
Artifacts. No Environments, Functions, or Calls are created. A failed upload stops the
benchmark and leaves its session identity in the error for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import httpx

from gfaas import Client, ClientConfig


def write_random(path: Path, size: int) -> None:
    with path.open("wb") as stream:
        while size:
            chunk = os.urandom(min(size, 1024 * 1024))
            stream.write(chunk)
            size -= len(chunk)


def remove_artifact(client: Client, artifact_id: str) -> None:
    """Reconcile an ambiguous DELETE before a bounded retry of this benchmark's object."""
    path = f"/v1/artifacts/{artifact_id}"
    for attempt in range(3):
        try:
            response = client._http.delete(path)
            if response.status_code in {204, 404}:
                return
            response.raise_for_status()
        except httpx.TransportError:
            receipt = client._http.get(path)
            if receipt.status_code == 404:
                return
            receipt.raise_for_status()
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--mib", type=int, default=64, help="Bytes per file case and per tree case, in MiB"
    )
    parser.add_argument("--files", type=int, default=32)
    parser.add_argument("--client-revision", required=True)
    parser.add_argument("--service-revision", required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Append publication receipts here before deleting Artifacts",
    )
    args = parser.parse_args()
    if min(args.repetitions, args.mib, args.files, *args.concurrency) < 1:
        parser.error("sizes, repetitions, and concurrency must be positive")
    config = ClientConfig.from_env()
    for repetition in range(args.repetitions):
        # Alternate order to reduce systematic cache and service-load bias.
        order = args.concurrency if repetition % 2 == 0 else args.concurrency[::-1]
        for concurrency in order:
            if hasattr(config, "upload_concurrency"):
                config = replace(config, upload_concurrency=concurrency)
            elif concurrency != 1:
                parser.error("this SDK revision only supports serial uploads")
            with Client(config) as client, tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                size = args.mib * 1024 * 1024
                file = root / "payload.bin"
                tree = root / "tree"
                tree.mkdir()
                write_random(file, size)
                quotient, remainder = divmod(size, args.files)
                for index in range(args.files):
                    write_random(tree / f"{index:04}.bin", quotient + (index < remainder))
                for kind, path in [("file", file), ("tree", tree)]:
                    started = time.perf_counter()
                    artifact = (
                        client.upload_artifact_file
                        if kind == "file"
                        else client.upload_artifact_directory
                    )(path)
                    elapsed = time.perf_counter() - started
                    record = {
                        "schema": "gfaas.upload-benchmark/v1",
                        "kind": kind,
                        "repetition": repetition,
                        "concurrency": concurrency,
                        "size_bytes": size,
                        "files": 1 if kind == "file" else args.files,
                        "seconds": elapsed,
                        "mib_per_second": args.mib / elapsed,
                        "client_revision": args.client_revision,
                        "service_revision": args.service_revision,
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "artifact_id": artifact["id"],
                        "digest": artifact.get("digest"),
                        "content": "fresh random bytes per case; no intentional deduplication",
                    }
                    print(
                        f"{kind}: upload finished in {elapsed:.3f}s; removing benchmark Artifacts",
                        file=sys.stderr,
                        flush=True,
                    )
                    with args.inventory.open("a", encoding="utf-8") as inventory:
                        inventory.write(json.dumps({**record, "artifact": artifact}) + "\n")
                        inventory.flush()
                        os.fsync(inventory.fileno())
                    # Tree references must disappear before deleting its children.
                    try:
                        remove_artifact(client, artifact["id"])
                        for child in artifact.get("child_artifact_ids", []):
                            remove_artifact(client, child)
                    except Exception:
                        record["cleanup"] = "failed"
                        print(json.dumps(record), flush=True)
                        raise
                    record["cleanup"] = "passed"
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
