"""Bounded GPU shard dispatch, verified-device replication and incremental reduction."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from . import triton_quick_runner
from .app import Function
from .errors import GfaasError
from .triton_policy import TritonTuning


class TritonBenchmarkError(GfaasError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.call_ids = report["call_ids"]
        super().__init__(report.get("diagnostics", "Triton benchmarking failed"))


def benchmark_shards(
    function: Any,
    variants: list[dict[str, Any]],
    request: dict[str, Any],
    policy: TritonTuning,
    call_ids: list[str],
    *,
    single_job: bool = False,
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    started_ids = list(call_ids)
    report: dict[str, Any] = {
        "schema": "vfunc.triton-replicated-benchmark/v1",
        "call_ids": call_ids,
        "shards": history,
        "replication_factor": policy.replication_factor,
    }

    def fail(message: str) -> None:
        report.update(status="failed", diagnostics=message, results=list(states.values()))
        raise TritonBenchmarkError(report)

    def reduce() -> None:
        active = [s for s in states.values() if s["status"] == "measured"]
        if not active:
            return
        for state in active:
            state["runtime_us"] = sum(r["runtime_us"] for r in state["replicas"]) / len(
                state["replicas"]
            )
        best = min(s["runtime_us"] for s in active)
        p = policy.refined_pruning
        limit = best + max(best * p.relative_delta, p.absolute_us) if p else math.inf
        for state in active:
            if state["runtime_us"] > limit:
                state["status"] = "globally_pruned"

    def dispatch(
        chunk: list[dict[str, Any]], excluded: tuple[str, ...], final_only: bool
    ) -> dict[str, Any]:
        identity = None
        try:
            worker = function
            extra = {}
            cuda_backend = request.get("backend") == "cuda"
            from . import cuda_kernel_runner

            if single_job and isinstance(function, Function):
                worker = replace(
                    function,
                    handler=cuda_kernel_runner.benchmark_replicas
                    if cuda_backend
                    else triton_quick_runner.benchmark_selected_replicas,
                    gpu=None,
                    gpu_count=policy.replication_factor,
                )
                extra = {"device_count": policy.replication_factor}
            elif final_only and isinstance(function, Function) and policy.replication_factor > 1:
                worker = replace(
                    function,
                    handler=cuda_kernel_runner.benchmark_replicas
                    if cuda_backend
                    else triton_quick_runner.benchmark_replicas,
                    gpu=None,
                    gpu_count=policy.replication_factor,
                )
                extra = {"device_count": policy.replication_factor}
            job = worker.spawn(
                **request,
                **extra,
                variants=json.dumps(chunk),
                final_only=final_only or single_job,
                excluded_gpu_uuids=list(excluded),
                estimates={v["id"]: states[v["id"]]["refined_us"] for v in chunk}
                if final_only
                else None,
            )
            identity = job.call_id
            output = job.wait()
            if not isinstance(output, dict):
                raise GfaasError("Invalid benchmark shard report")
            outputs = output.get("replica_reports", [output])
            if not isinstance(outputs, list) or not outputs:
                raise GfaasError("Empty replica bundle")
            for result in outputs:
                if not result.get("gpu_uuid"):
                    raise GfaasError("Benchmark shard did not report a GPU UUID")
                if result.get("status") == "duplicate_gpu":
                    if result["gpu_uuid"] not in excluded:
                        raise GfaasError("Unexpected duplicate GPU report")
                else:
                    rows = result.get("results", [])
                    if len(rows) != len(chunk) or {r["id"] for r in rows} != {
                        v["id"] for v in chunk
                    }:
                        raise GfaasError("Incomplete benchmark shard results")
            return {
                "call_id": identity,
                "report": output,
                "requested_ids": [v["id"] for v in chunk],
            }
        except Exception as error:
            return {
                "call_id": identity,
                "error": f"{type(error).__name__}: {error}",
                "requested_ids": [v["id"] for v in chunk],
            }

    def wave(jobs: list[tuple[list[dict[str, Any]], tuple[str, ...]]], final_only: bool) -> None:
        failures = []
        with ThreadPoolExecutor(max_workers=policy.quick_benchmark_max_concurrent_jobs) as pool:
            futures = [
                pool.submit(dispatch, chunk, excluded, final_only) for chunk, excluded in jobs
            ]
            for future in as_completed(futures):
                shard = future.result()
                history.append(shard)
                if shard["call_id"]:
                    call_ids.append(shard["call_id"])
                if "error" in shard:
                    failures.append(shard["error"])
                    continue
                outputs = shard["report"].get("replica_reports", [shard["report"]])
                for output in outputs:
                    if output["status"] == "duplicate_gpu":
                        continue
                    for row in output["results"]:
                        identity = row["id"]
                        if not final_only and identity not in states:
                            states[identity] = {**row, "replicas": [], "attempts": 0}
                        state = states[identity]
                        if row.get("status") == "invalid" and state["replicas"]:
                            failures.append(
                                f"Configuration {identity} failed evaluation on one GPU after passing on another"
                            )
                            continue
                        if row.get("status") == "measured" and state.get("status") == "invalid":
                            failures.append(
                                f"Configuration {identity} passed evaluation on one GPU after failing on another"
                            )
                            continue
                        if len(state["replicas"]) >= policy.replication_factor:
                            continue
                        if row["status"] != "measured":
                            state.update(status=row["status"])
                            continue
                        runtime = row.get("runtime_us")
                        if (
                            not isinstance(runtime, (int, float))
                            or not math.isfinite(runtime)
                            or runtime <= 0
                        ):
                            failures.append("Invalid final benchmark timing")
                            continue
                        if output["gpu_uuid"] in {r["gpu_uuid"] for r in state["replicas"]}:
                            failures.append("A replica reused an already measured GPU UUID")
                            continue
                        state["replicas"].append(
                            {
                                "gpu_uuid": output["gpu_uuid"],
                                "runtime_us": runtime,
                                "call_id": shard["call_id"],
                                "final": row.get("final"),
                                "evaluation": row.get("evaluation"),
                            }
                        )
                    if final_only:
                        reduce()
        if failures:
            fail("; ".join(failures))

    size = policy.quick_benchmark_variants_per_job
    wave([(variants[i : i + size], ()) for i in range(0, len(variants), size)], False)
    # Wait for all initial shards before using a global best for cross-shard pruning.
    reduce()
    by_id = {v["id"]: v for v in variants}
    while True:
        pending = [
            s
            for s in states.values()
            if s["status"] == "measured" and len(s["replicas"]) < policy.replication_factor
        ]
        if not pending:
            break
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for state in pending:
            if state["attempts"] >= policy.replication_max_attempts:
                fail(
                    "Could not obtain the requested number of distinct GPU replicas within the placement retry limit"
                )
            state["attempts"] += 1
            excluded = tuple(sorted(r["gpu_uuid"] for r in state["replicas"]))
            groups.setdefault(excluded, []).append(by_id[state["id"]])
        jobs = [
            (group[i : i + size], excluded)
            for excluded, group in groups.items()
            for i in range(0, len(group), size)
        ]
        wave(jobs, True)
    complete = [
        s
        for s in states.values()
        if s["status"] == "measured" and len(s["replicas"]) == policy.replication_factor
    ]
    if not complete:
        fail("No configuration passed evaluation and completed all requested replicas")
    winner = min(complete, key=lambda s: s["runtime_us"])
    report.update(
        status="passed",
        results=[states[v["id"]] for v in variants],
        best_id=winner["id"],
        best_runtime_us=winner["runtime_us"],
        best_configuration=by_id[winner["id"]],
        benchmark_call_ids=[i for i in call_ids if i not in started_ids],
        gpu_identity_verified=True,
    )
    return report
