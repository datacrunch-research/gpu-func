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
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactCheckpoint, ArtifactOutput
from .image import Image

if TYPE_CHECKING:
    from .client import Client, RemoteResult


class _Unset:
    pass


_UNSET = _Unset()
_ACTIVE_FUNCTION: ContextVar[FunctionScope | None] = ContextVar("vfunc_function", default=None)


def active_function_scope() -> FunctionScope:
    scope = _ACTIVE_FUNCTION.get()
    if scope is None:
        raise RuntimeError("TritonKernel must be called inside with app.function(...)")
    return scope


class FunctionScope:
    """Existing function configuration, usable as a decorator or scoped context."""

    def __init__(self, app: App, options: dict[str, Any]) -> None:
        self.app = app
        self.options = dict(options)
        self.options["env"] = dict(options.get("env") or {})
        self.options["outputs"] = tuple(options.get("outputs") or ())
        self._tokens: ContextVar[tuple[Token[FunctionScope | None], ...]] = ContextVar(
            "vfunc_scope_tokens", default=()
        )

    def bind(self, handler: Callable[..., Any]) -> Function:
        options = dict(self.options)
        options["timeout_s"] = options.pop("timeout")
        options["capacity_wait_s"] = options.pop("capacity_wait")
        options["env"] = dict(options["env"])
        return Function(app=self.app, handler=handler, **options)

    def __call__(self, handler: Callable[..., Any]) -> Function:
        fn = self.bind(handler)
        self.app._functions[fn.name] = fn
        return fn

    def __enter__(self) -> FunctionScope:
        token = _ACTIVE_FUNCTION.set(self)
        self._tokens.set((*self._tokens.get(), token))
        return self

    def __exit__(self, *exception: Any) -> None:
        tokens = self._tokens.get()
        _ACTIVE_FUNCTION.reset(tokens[-1])
        self._tokens.set(tokens[:-1])


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
        return self._spawn(args, kwargs, qualification_image_digest=None)

    def qualify(self, image_digest: str, *args: Any, **kwargs: Any) -> RemoteResult:
        """Run this Function as the qualification Call for an immutable image digest."""
        return self._spawn(args, kwargs, qualification_image_digest=image_digest)

    def _spawn(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        qualification_image_digest: str | None,
    ) -> RemoteResult:
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
            qualification_image_digest=qualification_image_digest,
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
        image: Image | None | _Unset = _UNSET,
        gpu: str | None | _Unset = _UNSET,
        gpu_count: int | None | _Unset = _UNSET,
        gpu_type: str | _Unset = _UNSET,
        timeout: int | _Unset = _UNSET,
        capacity_wait: int | None | _Unset = _UNSET,
        cpu_millicores: int | None | _Unset = _UNSET,
        memory_bytes: int | None | _Unset = _UNSET,
        ephemeral_storage_bytes: int | None | _Unset = _UNSET,
        shared_memory_bytes: int | None | _Unset = _UNSET,
        max_log_bytes: int | None | _Unset = _UNSET,
        max_output_bytes: int | None | _Unset = _UNSET,
        env: dict[str, str] | None | _Unset = _UNSET,
        outputs: tuple[ArtifactOutput | ArtifactCheckpoint, ...] | _Unset = _UNSET,
    ) -> FunctionScope:
        supplied = {k: v for k, v in locals().items() if k != "self" and v is not _UNSET}
        options: dict[str, Any] = {
            "image": None,
            "gpu": None,
            "gpu_count": None,
            "gpu_type": "any",
            "timeout": 300,
            "capacity_wait": None,
            "cpu_millicores": None,
            "memory_bytes": None,
            "ephemeral_storage_bytes": None,
            "shared_memory_bytes": None,
            "max_log_bytes": None,
            "max_output_bytes": None,
            "env": {},
            "outputs": (),
        }
        parent = _ACTIVE_FUNCTION.get()
        if parent is not None and parent.app is self:
            options.update(parent.options)
        if "gpu" in supplied and "gpu_count" not in supplied:
            options["gpu_count"] = None
        if ("gpu_count" in supplied or "gpu_type" in supplied) and "gpu" not in supplied:
            options["gpu"] = None
        options.update(supplied)
        return FunctionScope(self, options)
