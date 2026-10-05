"""Meta TLX kernels using the shared vFunc compilation and tuning pipeline."""

from __future__ import annotations

import importlib
from typing import Any

from .triton_compat import UnsupportedTritonKernelError
from .triton_kernel import TritonKernel


class TLXKernel(TritonKernel):
    """Wrap TLX @triton.jit or vanilla @triton.autotune kernels.

    The client and app.function image must provide the same TLX-enabled Triton.
    Image/resources, tuning controls, tensor writes, and specialization reports
    follow TritonKernel. Host tensor descriptors and custom hooks are unsupported.
    """

    def _validate_kernel(self, kernel: Any) -> tuple[Any, list[dict[str, Any]] | None]:
        try:
            importlib.import_module("triton.language.extra.tlx")
        except ImportError as error:
            raise UnsupportedTritonKernelError(
                "TLXKernel requires TLX-enabled Triton on the client and in the image; "
                "install Meta's fbtriton distribution in place of upstream Triton"
            ) from error
        return super()._validate_kernel(kernel)

    def _source_bundle(self, kernel: Any) -> str:
        # Also verify TLX in remote environments for kernels whose body only
        # reaches TLX indirectly through a dependency.
        return "import triton.language.extra.tlx as vfunc_tlx\n" + super()._source_bundle(kernel)
