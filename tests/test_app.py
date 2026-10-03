from __future__ import annotations

from typing import Any

import gfaas


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return object()


def test_function_decorator_forwards_an_explicit_gpu_count() -> None:
    client = FakeClient()
    app = gfaas.App("multi-gpu", image=gfaas.Image("pytorch"), client=client)

    @app.function(gpu_count=4, gpu_type="gb300")
    def train() -> None:
        pass

    train.spawn()

    assert client.calls[0]["gpu"] is None
    assert client.calls[0]["gpu_count"] == 4
    assert client.calls[0]["gpu_type"] == "gb300"


def test_function_qualify_forwards_the_immutable_image_digest() -> None:
    client = FakeClient()
    app = gfaas.App("image-check", image=gfaas.Image("candidate"), client=client)

    @app.function(gpu_type="gb300")
    def check() -> None:
        pass

    digest = f"sha256:{'a' * 64}"
    check.qualify(digest)

    assert client.calls[0]["qualification_image_digest"] == digest


def test_context_nested_inheritance_clear_and_exception_restore():
    import pytest

    from gfaas.app import active_function_scope

    app = gfaas.App("nested", image=gfaas.Image("default"))
    with app.function(gpu="gb300", timeout=91, env={"A": "B"}) as outer:
        with app.function(cpu_millicores=2000, env=None) as inner:
            assert inner.options["gpu"] == "gb300" and inner.options["timeout"] == 91
            assert inner.options["env"] == {}
            assert active_function_scope() is inner
        assert active_function_scope() is outer
        with pytest.raises(ValueError), app.function(timeout=5):
            raise ValueError("test")
        assert active_function_scope() is outer

        @app.function()
        def saved():
            pass

    assert saved.timeout_s == 91 and saved.gpu == "gb300"
    with pytest.raises(RuntimeError):
        active_function_scope()


def test_context_isolation_across_tasks_and_reentrant_scope():
    import asyncio

    from gfaas.app import active_function_scope

    app = gfaas.App("tasks")
    scope = app.function(timeout=7)

    async def worker(timeout):
        with app.function(timeout=timeout):
            await asyncio.sleep(0)
            assert active_function_scope().options["timeout"] == timeout
            with scope:
                with scope:
                    assert active_function_scope().options["timeout"] == 7
                assert active_function_scope().options["timeout"] == 7
            return active_function_scope().options["timeout"]

    async def run():
        return await asyncio.gather(worker(11), worker(22))

    assert asyncio.run(run()) == [11, 22]


def test_context_isolation_across_threads_and_apps():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from gfaas.app import active_function_scope

    gate = Barrier(2)
    app = gfaas.App("threads")

    def worker(timeout):
        with app.function(timeout=timeout):
            gate.wait(timeout=5)
            assert active_function_scope().options["timeout"] == timeout
            with gfaas.App("other").function() as other:
                assert other.options["timeout"] == 300
            return active_function_scope().options["timeout"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(worker, [11, 22])) == [11, 22]
