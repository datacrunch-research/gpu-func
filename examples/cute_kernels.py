"""CuTe DSL host entries usable with vfunc.CuteDSLKernel.

Install matching CuTe DSL versions locally and in the selected vFunc image.
Kernels use a typed stream parameter so vFunc can capture launches in CUDA graphs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.kernel
def add_device(a: cute.Tensor, b: cute.Tensor, BLOCK: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * BLOCK + tx
    if i < a.shape[0]:
        b[i] = a[i] + 1


@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor, BLOCK: cutlass.Constexpr, stream: cuda.CUstream):
    add_device(a, b, BLOCK).launch(
        grid=(cute.ceil_div(a.shape[0], BLOCK), 1, 1), block=(BLOCK, 1, 1), stream=stream
    )


@cute.kernel
def copy_device(a: cute.Tensor, b: cute.Tensor, BLOCK: cutlass.Constexpr, SCALE: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * BLOCK + tx
    if i < a.shape[0] * a.shape[1]:
        row, col = i // a.shape[1], i % a.shape[1]
        b[row, col] = (a[row, col] * SCALE).to(b.element_type)


class ScaledCopy:
    def __init__(self, scale=2):
        self.scale = scale

    @cute.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, BLOCK: cutlass.Constexpr, stream: cuda.CUstream
    ):
        copy_device(a, b, BLOCK, self.scale).launch(
            grid=(cute.ceil_div(a.shape[0] * a.shape[1], BLOCK), 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )


@cute.kernel
def matmul_device(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, BLOCK: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * BLOCK + tx
    if i < c.shape[0] * c.shape[1]:
        row, col = i // c.shape[1], i % c.shape[1]
        accumulator = cutlass.Float32(0)
        for k in range(a.shape[1]):
            accumulator += cutlass.Float32(a[row, k]) * cutlass.Float32(b[k, col])
        c[row, col] = accumulator


@cute.jit
def matmul(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, BLOCK: cutlass.Constexpr, stream: cuda.CUstream
):
    # This small reference-style kernel demonstrates the API, not a tuned GEMM algorithm.
    matmul_device(a, b, c, BLOCK).launch(
        grid=(cute.ceil_div(c.shape[0] * c.shape[1], BLOCK), 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
    )


def evaluate_add(candidate, a, b):
    import torch

    candidate(a, b)
    return bool(torch.equal(b, a + 1))


def evaluate_copy(candidate, a, b):
    import torch

    candidate(a, b)
    return bool(torch.equal(b, a * 2))


def evaluate_matmul(candidate, a, b, c):
    import torch

    candidate(a, b, c)
    return bool(torch.allclose(c.float(), a.float() @ b.float(), atol=1e-4, rtol=1e-3))


@cute.jit
def add_tiled(a: cute.Tensor, b: cute.Tensor, TILE: cutlass.Constexpr, stream: cuda.CUstream):
    block = TILE[0] * TILE[1]
    add_device(a, b, block).launch(
        grid=(cute.ceil_div(a.shape[0], block), 1, 1), block=(block, 1, 1), stream=stream
    )
