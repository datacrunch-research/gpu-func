"""``App`` and ``@app.function`` decorator for remote GPU functions.

We deliberately keep the surface tiny:

    image = gfaas.Image("cuda-nvcc")
    app = gfaas.App("hello", image=image)

    @app.function(gpu="any", timeout=600)
    def my_fn(x): ...

    my_fn.remote(42)        # blocking, returns the value
    my_fn.spawn(42).wait()  # non-blocking; same result

No daemon process, no lazy state, no metaclass magic. The decorator just
captures the function + per-call config; ``remote()`` packages and submits.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactCheckpoint, ArtifactOutput
from .image import Image

if TYPE_CHECKING:
    from .client import Client, RemoteResult


@dataclass
class Function:
    app: App
    handler: Callable[..., Any]
    image: Image | None = None
    gpu: str | None = None
    gpu_count: int | None = None
    gpu_type: str = "any"
    timeout_s: int = 300
    capacity_wait_s: int | None = None
    cpu_millicores: int | None = None
    memory_bytes: int | None = None
    ephemeral_storage_bytes: int | None = None
    shared_memory_bytes: int | None = None
    max_log_bytes: int | None = None
    max_output_bytes: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        self.name = self.handler.__name__
        source = inspect.getsourcefile(self.handler)
        if source is None:
            raise ValueError(f"cannot resolve source file for {self.handler.__name__!r}")
        self.source_file = Path(source).resolve()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.handler(*args, **kwargs)

    def _resolve_image(self) -> Image:
        image = self.image or self.app.image
        if image is None:
            raise ValueError(
                f"function {self.name!r} has no image; pass image=... to App or @app.function"
            )
        return image

    def spawn(self, *args: Any, **kwargs: Any) -> RemoteResult:
        return self.app.client.submit(
            image=self._resolve_image(),
            function=self.handler,
            args=args,
            kwargs=kwargs,
            gpu=self.gpu,
            gpu_count=self.gpu_count,
            gpu_type=self.gpu_type,
            app_name=self.app.name,
            timeout_s=self.timeout_s,
            capacity_wait_s=self.capacity_wait_s,
            cpu_millicores=self.cpu_millicores,
            memory_bytes=self.memory_bytes,
            ephemeral_storage_bytes=self.ephemeral_storage_bytes,
            shared_memory_bytes=self.shared_memory_bytes,
            max_log_bytes=self.max_log_bytes,
            max_output_bytes=self.max_output_bytes,
            env=self.env,
            source_file=self.source_file,
            outputs=self.outputs,
        )

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.spawn(*args, **kwargs).wait()


class App:
    def __init__(
        self,
        name: str,
        *,
        image: Image | None = None,
        client: Client | None = None,
    ) -> None:
        self.name = name
        self.image = image
        self._client = client
        self._functions: dict[str, Function] = {}

    @property
    def client(self) -> Client:
        if self._client is None:
            from .client import Client

            self._client = Client()
        return self._client

    def function(
        self,
        *,
        image: Image | None = None,
        gpu: str | None = None,
        gpu_count: int | None = None,
        gpu_type: str = "any",
        timeout: int = 300,
        capacity_wait: int | None = None,
        cpu_millicores: int | None = None,
        memory_bytes: int | None = None,
        ephemeral_storage_bytes: int | None = None,
        shared_memory_bytes: int | None = None,
        max_log_bytes: int | None = None,
        max_output_bytes: int | None = None,
        env: dict[str, str] | None = None,
        outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] = (),
    ) -> Callable[[Callable[..., Any]], Function]:
        def decorator(handler: Callable[..., Any]) -> Function:
            fn = Function(
                app=self,
                handler=handler,
                image=image,
                gpu=gpu,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                timeout_s=timeout,
                capacity_wait_s=capacity_wait,
                cpu_millicores=cpu_millicores,
                memory_bytes=memory_bytes,
                ephemeral_storage_bytes=ephemeral_storage_bytes,
                shared_memory_bytes=shared_memory_bytes,
                max_log_bytes=max_log_bytes,
                max_output_bytes=max_output_bytes,
                env=dict(env or {}),
                outputs=tuple(outputs),
            )
            self._functions[fn.name] = fn
            return fn

        return decorator
