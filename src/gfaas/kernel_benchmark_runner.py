"""Shared staged measurement for precompiled kernel callables."""

from __future__ import annotations

import math
from typing import Any

from . import triton_quick_runner as benchmark_runtime


def measure_entries(
    entries: list[tuple[dict[str, Any], Any]],
    rows: list[dict[str, Any]],
    ring: Any,
    torch: Any,
    flush: Any,
    policy: dict[str, Any],
    evaluate: Any,
    *,
    final_only: bool = False,
    estimates: dict[str, float] | None = None,
) -> None:
    def check(row: dict[str, Any], candidate: Any) -> bool:
        if evaluate is None:
            row["evaluation"] = "skipped"
            return True
        args, kwargs = ring.fresh()
        verdict = evaluate(candidate, *args, **kwargs)
        torch.cuda.synchronize()
        if type(verdict) is not bool:
            raise TypeError("evaluate must return a bool")
        row["evaluation"] = "passed" if verdict else "failed"
        if not verdict:
            row["status"] = "invalid"
        return verdict

    settings = policy.get("benchmark", {})
    if not final_only:
        best = None
        for offset in range(0, len(entries), policy["group_size"]):
            group = entries[offset : offset + policy["group_size"]]
            values = benchmark_runtime.ring_trials(
                [c for _, c in group],
                [settings.get("pilot_trials", 3)] * len(group),
                ring,
                torch,
                flush,
                settings.get("l2_flush_iterations", 100),
            )
            for (row, candidate), trials in zip(group, values, strict=True):
                runtime = min(trials)
                if not math.isfinite(runtime) or runtime <= 0:
                    raise RuntimeError("Invalid pilot timing")
                row.update(status="measured", pilot_us=runtime, pilot_trial_us=trials)
                if (best is None or runtime < best) and check(row, candidate):
                    best = runtime
        keep = set(benchmark_runtime.prune_timings(rows, "pilot_us", policy["pilot_pruning"]))
        survivors = [(r, c) for r, c in entries if r["id"] in keep]
        for row, _ in entries:
            if row["status"] == "measured" and row["id"] not in keep:
                row["status"] = "pilot_pruned"
        for offset in range(0, len(survivors), policy["group_size"]):
            group = survivors[offset : offset + policy["group_size"]]
            counts = [
                min(
                    settings.get("max_refinement_trials", 250),
                    math.ceil(settings.get("refinement_duration_ms", 1.0) * 1000 / r["pilot_us"]),
                )
                for r, _ in group
            ]
            active = [
                (r, c, n)
                for (r, c), n in zip(group, counts, strict=True)
                if n >= settings.get("min_refinement_trials", 10)
            ]
            for (row, _), n in zip(group, counts, strict=True):
                if n < settings.get("min_refinement_trials", 10):
                    row.update(refined_us=row["pilot_us"], refined_trial_us=row["pilot_trial_us"])
            if active:
                values = benchmark_runtime.ring_trials(
                    [c for _, c, _ in active],
                    [n for _, _, n in active],
                    ring,
                    torch,
                    flush,
                    settings.get("l2_flush_iterations", 100),
                )
                for (row, _, n), trials in zip(active, values, strict=True):
                    row.update(
                        refined_us=min(trials), refined_trial_us=trials, refinement_iterations=n
                    )
        # Validate the proposed refined best before it can prune any peers.
        refined_evaluated = set()
        for row, candidate in sorted(survivors, key=lambda pair: pair[0]["refined_us"]):
            refined_evaluated.add(row["id"])
            if check(row, candidate):
                break
        keep = set(benchmark_runtime.prune_timings(rows, "refined_us", policy["refined_pruning"]))
        for row, candidate in survivors:
            if row["status"] != "measured":
                continue
            if row["id"] not in keep:
                row["status"] = "refined_pruned"
            elif row["id"] not in refined_evaluated and not check(row, candidate):
                continue
    for row, candidate in entries:
        if final_only:
            row["status"] = "measured"
            if not check(row, candidate):
                continue
            if estimates is None:
                samples = benchmark_runtime.ring_trials(
                    [candidate],
                    [settings.get("pilot_trials", 3)],
                    ring,
                    torch,
                    flush,
                    settings.get("l2_flush_iterations", 100),
                )[0]
                estimate = min(samples)
                row.update(refined_us=estimate, estimate_trial_us=samples)
            else:
                estimate = estimates[row["id"]]
                row["refined_us"] = estimate
        elif row.get("status") == "measured":
            estimate = row["refined_us"]
        else:
            continue
        if not math.isfinite(estimate) or estimate <= 0:
            raise RuntimeError("Invalid refined runtime")
        final = benchmark_runtime.final_benchmark(candidate, estimate, ring, torch, flush, settings)
        if not math.isfinite(final["runtime_us"]) or final["runtime_us"] <= 0:
            raise RuntimeError("Invalid final runtime")
        row.update(runtime_us=final["runtime_us"], final=final)
