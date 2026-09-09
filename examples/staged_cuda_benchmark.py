#!/usr/bin/env python3
"""Validate and benchmark staged CUDA Calls on an isolated GPU pool.

This program creates real remote Calls. Use a quiet test pool so unrelated work does not affect the
comparison. The baseline deliberately uses the old single-stage compile-and-run path.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from gfaas import Client, RemoteResult, cuda_runner
from gfaas import cuda as cuda_api

SOURCE = r"""
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void hold_gpu(unsigned long long ticks) {
    const unsigned long long start = clock64();
    while (clock64() - start < ticks) {}
}

int main(int argc, char **argv) {
    const int milliseconds = argc > 1 ? std::atoi(argv[1]) : 500;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 2;
    const auto ticks = static_cast<unsigned long long>(milliseconds) * properties.clockRate;
    hold_gpu<<<1, 1>>>(ticks);
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        return 3;
    }
    std::printf("held GPU for %d ms\n", milliseconds);
    return 0;
}
"""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", required=True, help="isolated GPU pool to test")
    parser.add_argument("--calls", type=int, default=4, help="Calls in each comparison batch")
    parser.add_argument("--gpu-hold-ms", type=int, default=1500)
    parser.add_argument("--overlap-hold-ms", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--image", default="cuda-nvcc")
    parser.add_argument("--output", type=Path, required=True, help="new JSON result file")
    args = parser.parse_args()
    if args.calls < 2 or args.calls > 64:
        parser.error("--calls must be between 2 and 64")
    if args.gpu_hold_ms < 1 or args.overlap_hold_ms < 1:
        parser.error("GPU hold times must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def _submit(
    client: Client,
    *,
    staged: bool,
    gpu_type: str,
    image: str,
    hold_ms: int,
) -> RemoteResult:
    kwargs = {
        "source": SOURCE,
        "profile": False,
        "ncu_args": [],
        "nvcc_flags": [],
        "program_args": [str(hold_ms)],
    }
    if staged:
        return cuda_api.spawn(
            SOURCE,
            gpu_count=1,
            gpu_type=gpu_type,
            program_args=[str(hold_ms)],
            timeout_s=600,
            image=image,
            client=client,
        )
    return client.submit(
        image=image,
        function=cuda_runner.run,
        kwargs=kwargs,
        gpu_count=1,
        gpu_type=gpu_type,
        timeout_s=600,
        app_name="cuda-full-reservation-baseline",
    )


def _wait_until_gpu_running(client: Client, job: RemoteResult, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        call = client.get_call(job.call_id)
        stage = call.get("current_stage")
        if (
            isinstance(stage, dict)
            and stage.get("state") == "running"
            and stage.get("effective_resources", {}).get("gpu", {}).get("count", 0) > 0
        ):
            return
        if call.get("state") in {"succeeded", "failed", "timed_out", "cancelled"}:
            raise RuntimeError(f"Call {job.call_id} ended before its GPU stage was observed")
        time.sleep(0.05)
    raise TimeoutError(f"Call {job.call_id} did not start GPU execution")


def _events(client: Client, call_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page = client.list_events(call_id, after=cursor, limit=1000)
        events.extend(page.get("items", []))
        cursor = page.get("next_cursor")
        if cursor is None:
            return events


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _stage_event_time(
    events: list[dict[str, Any]],
    *,
    stage: str,
    stage_state: str,
) -> float:
    for event in events:
        attributes = event.get("attributes")
        if (
            isinstance(attributes, dict)
            and attributes.get("stage") == stage
            and attributes.get("stage_state") == stage_state
        ):
            return _timestamp(str(event["occurred_at"]))
    raise RuntimeError(f"retained events do not contain {stage}:{stage_state}")


def _call_timing(client: Client, call_id: str) -> dict[str, Any]:
    events = _events(client, call_id)
    call = client.get_call(call_id)
    terminal = _timestamp(str(call["terminal_at"]))
    reserved = _stage_event_time(events, stage="execute", stage_state="accepted")
    compile_started: float | None = None
    compile_finished: float | None = None
    try:
        compile_started = _stage_event_time(events, stage="compile", stage_state="running")
        compile_finished = _stage_event_time(events, stage="compile", stage_state="succeeded")
    except RuntimeError:
        pass
    return {
        "call_id": call_id,
        "gpu_reserved_seconds": terminal - reserved,
        "compile_seconds": (
            compile_finished - compile_started
            if compile_started is not None and compile_finished is not None
            else None
        ),
        "terminal_at": terminal,
        "events": events,
    }


def _overlap_probe(client: Client, args: argparse.Namespace) -> dict[str, Any]:
    blocker = _submit(
        client,
        staged=False,
        gpu_type=args.gpu_type,
        image=args.image,
        hold_ms=args.overlap_hold_ms,
    )
    _wait_until_gpu_running(client, blocker, args.timeout)
    candidate = _submit(
        client,
        staged=True,
        gpu_type=args.gpu_type,
        image=args.image,
        hold_ms=args.gpu_hold_ms,
    )
    candidate.wait(timeout_s=args.timeout)
    blocker.wait(timeout_s=args.timeout)
    blocker_timing = _call_timing(client, blocker.call_id)
    candidate_timing = _call_timing(client, candidate.call_id)
    compile_finished = _stage_event_time(
        candidate_timing["events"], stage="compile", stage_state="succeeded"
    )
    blocker_started = _stage_event_time(
        blocker_timing["events"], stage="execute", stage_state="running"
    )
    overlapped = blocker_started <= compile_finished < blocker_timing["terminal_at"]
    if not overlapped:
        raise RuntimeError("the staged compilation did not overlap the baseline GPU execution")
    return {
        "passed": True,
        "blocker_call_id": blocker.call_id,
        "staged_call_id": candidate.call_id,
        "blocker_gpu_started_at": blocker_started,
        "staged_compile_finished_at": compile_finished,
        "blocker_terminal_at": blocker_timing["terminal_at"],
    }


def _batch(client: Client, args: argparse.Namespace, *, staged: bool) -> dict[str, Any]:
    started = time.monotonic()
    jobs = [
        _submit(
            client,
            staged=staged,
            gpu_type=args.gpu_type,
            image=args.image,
            hold_ms=args.gpu_hold_ms,
        )
        for _ in range(args.calls)
    ]
    for job in jobs:
        job.wait(timeout_s=args.timeout)
    elapsed = time.monotonic() - started
    timings = [_call_timing(client, job.call_id) for job in jobs]
    reserved_seconds = sum(float(item["gpu_reserved_seconds"]) for item in timings)
    return {
        "mode": "staged" if staged else "full_reservation_baseline",
        "calls": args.calls,
        "elapsed_seconds": elapsed,
        "throughput_calls_per_second": args.calls / elapsed,
        "reserved_gpu_seconds": reserved_seconds,
        "average_reserved_gpus": reserved_seconds / elapsed,
        "call_ids": [job.call_id for job in jobs],
    }


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    with Client() as client:
        overlap = _overlap_probe(client, args)
        baseline = _batch(client, args, staged=False)
        staged = _batch(client, args, staged=True)
    result = {
        "schema": "vfunc.staged-cuda-benchmark/v1",
        "gpu_type": args.gpu_type,
        "image": args.image,
        "gpu_hold_ms": args.gpu_hold_ms,
        "overlap": overlap,
        "baseline": baseline,
        "staged": staged,
        "throughput_ratio": (
            staged["throughput_calls_per_second"] / baseline["throughput_calls_per_second"]
        ),
        "note": (
            "average_reserved_gpus measures lease occupancy. Use an isolated one-GPU pool if it "
            "must be interpreted as a utilization fraction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
