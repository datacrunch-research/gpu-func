"""Bounded upload scheduling. Workers never submit more work to their own executor."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from threading import Event
from typing import TypeVar

from .errors import GfaasError

T = TypeVar("T")
R = TypeVar("R")


def check_cancelled(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise GfaasError("Artifact upload cancelled")


def run_uploads(
    items: Iterable[T],
    action: Callable[[T], R],
    completed: Callable[[R], None],
    concurrency: int,
) -> None:
    """Keep at most concurrency tasks outstanding; drain them before propagating errors.

    Successful in-flight work still reaches completed after a sibling fails. This lets a
    directory report every published child, including children that finish during shutdown.
    """
    iterator = iter(items)
    pending: set[Future[R]] = set()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="artifact-upload") as pool:
        try:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < concurrency:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                    else:
                        pending.add(pool.submit(action, item))
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Observe all errors before submitting more work.
                for future in done:
                    result = future.result()
                    pending.remove(future)
                    completed(result)
        except BaseException:
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    # Keep the original error; the caller records per-file failures.
                    with suppress(BaseException):
                        completed(future.result())
            raise
