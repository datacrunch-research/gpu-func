"""Bounded, fail-fast compilation shard dispatch shared by kernel backends."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any


def compile_shards(
    owner: Any,
    source: str,
    kernel_name: str,
    library_version: str,
    variants: list[dict[str, Any]],
    *,
    backend: str,
) -> tuple[dict[str, Any], str | None]:
    started = time.perf_counter()
    call_id: str | None
    job_count = owner._job_count(len(variants))
    if job_count == 1:
        call_id, report = owner._submit_shard(source, kernel_name, library_version, variants)
    else:
        # Hash order spreads neighboring tile/configuration families across
        # jobs. Final results still follow the user's original order.
        ordered = sorted(variants, key=lambda v: v["id"])
        chunks = [ordered[i::job_count] for i in range(job_count)]

        def submit(chunk: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
            return owner._submit_shard(source, kernel_name, library_version, chunk)

        outcomes: list[tuple[str | None, dict[str, Any]]] = []
        completed: dict[int, tuple[str | None, dict[str, Any]]] = {}
        limit = min(owner.max_concurrent_jobs, job_count)
        next_index = 0
        stopped = False
        with ThreadPoolExecutor(max_workers=limit) as pool:
            pending = {}
            while next_index < limit:
                pending[pool.submit(submit, chunks[next_index])] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    outcome: tuple[str | None, dict[str, Any]]
                    try:
                        outcome = future.result()
                    except Exception as error:
                        outcome = (
                            getattr(error, "call_id", None),
                            {
                                "job_failed": True,
                                "results": [
                                    {
                                        "id": v["id"],
                                        "status": "failed",
                                        "diagnostics": f"Compiler job could not complete: {type(error).__name__}: {error}",
                                    }
                                    for v in chunks[index]
                                ],
                            },
                        )
                    completed[index] = outcome
                    stopped = stopped or bool(outcome[1].get("job_failed"))
                # Structural job failures stop new submissions; already
                # running jobs are awaited and their results remain visible.
                while not stopped and next_index < job_count and len(pending) < limit:
                    pending[pool.submit(submit, chunks[next_index])] = next_index
                    next_index += 1
        for index in range(job_count):
            if index in completed:
                outcomes.append(completed[index])
            else:
                outcomes.append(
                    (
                        None,
                        {
                            "not_submitted": True,
                            "results": [
                                {
                                    "id": v["id"],
                                    "status": "failed",
                                    "diagnostics": "Not submitted after another compiler job failed",
                                }
                                for v in chunks[index]
                            ],
                        },
                    )
                )
        by_id = {row["id"]: row for _, shard in outcomes for row in shard["results"]}
        call_id = next((identity for identity, _ in outcomes if identity), None)
        report = {
            "schema": f"vfunc.{backend}-compilation-batch/v1",
            "variants": variants,
            "results": [by_id[v["id"]] for v in variants],
            "call_ids": [identity for identity, _ in outcomes if identity],
            "shards": [
                {"call_id": identity, "output_name": f"compiled-{backend}", "report": shard}
                for identity, shard in outcomes
            ],
            "batch_wall_seconds": time.perf_counter() - started,
            "max_concurrent_jobs": owner.max_concurrent_jobs,
            "workers_per_job": owner._workers,
        }
    return report, call_id
