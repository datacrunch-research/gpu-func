"""Helion examples for vFunc compilation, replicated tuning and execution.

Use an image containing exactly the client's Helion, PyTorch and Triton versions.
Input tensors may be on CPU: vFunc transports their contents to the assigned GPU.
"""

from __future__ import annotations

import argparse

import helion
import helion.language as hl
import torch

import gfaas as vfunc


@helion.kernel(configs=[helion.Config(block_sizes=[n], indexing="pointer") for n in (64, 128, 256)])
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        output[tile] = x[tile] + y[tile]
    return output


@helion.kernel(configs=[helion.Config(block_sizes=[n], indexing="pointer") for n in (1, 4, 16)])
def row_sum(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        output[tile] = x[tile, :].sum(-1)
    return output


@helion.kernel(
    configs=[
        helion.Config(block_sizes=list(shape), indexing="pointer", num_warps=4)
        for shape in ((32, 32, 32), (64, 64, 32), (64, 128, 32))
    ]
)
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    output = torch.empty((m, b.size(1)), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, b.size(1))):
        accumulator = hl.zeros((tile_m, tile_n), dtype=torch.float32)
        for tile_k in hl.tile(k):
            accumulator = torch.addmm(accumulator, a[tile_m, tile_k], b[tile_k, tile_n])
        output[tile_m, tile_n] = accumulator.to(a.dtype)
    return output


@helion.kernel(configs=[helion.Config(block_sizes=[n], indexing="pointer") for n in (128, 256)])
def increment(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1
    return x


@helion.kernel
def square_and_sum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    squared = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row, column in hl.tile(x.shape):
        squared[row, column] = x[row, column] * x[row, column]
    # The grid-wide barrier makes the squared values visible to the reduction.
    hl.barrier()
    for row in hl.tile(x.size(0)):
        output[row] = squared[row, :].sum(-1)
    return squared, output


def evaluate_square_and_sum(candidate, x):
    squared, total = candidate(x)
    return bool(
        torch.equal(squared, x.square())
        and torch.allclose(total, x.square().sum(-1), atol=1e-3, rtol=1e-4)
    )


def evaluate_add(candidate, x, y):
    return bool(torch.equal(candidate(x, y), x + y))


def evaluate_sum(candidate, x):
    return bool(torch.allclose(candidate(x), x.sum(-1), atol=1e-4, rtol=1e-4))


def evaluate_matmul(candidate, a, b):
    return bool(
        torch.allclose(candidate(a, b).float(), a.float() @ b.float(), atol=0.03, rtol=0.02)
    )


def evaluate_increment(candidate, x):
    expected = x + 1
    return bool(torch.equal(candidate(x), expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    app = vfunc.App("helion-examples", image=vfunc.Image(args.image))
    generator = torch.Generator().manual_seed(0)
    cases = [
        (
            add,
            evaluate_add,
            (torch.randn(2**20, generator=generator), torch.randn(2**20, generator=generator)),
        ),
        (row_sum, evaluate_sum, (torch.randn((1024, 1024), generator=generator),)),
        (
            matmul,
            evaluate_matmul,
            (
                torch.randn((512, 512), generator=generator).to(torch.bfloat16) * 0.1,
                torch.randn((512, 512), generator=generator).to(torch.bfloat16) * 0.1,
            ),
        ),
        (increment, evaluate_increment, (torch.zeros(2**20),)),
        (
            square_and_sum,
            evaluate_square_and_sum,
            (torch.randn((1024, 1024), generator=generator),),
        ),
    ]
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        for native, evaluate, inputs in cases:
            kernel = vfunc.HelionKernel(
                native,
                variants_per_job=16,
                tuning=vfunc.KernelTuning(evaluate=evaluate, replication_factor=3),
            )
            # A normal call tunes if necessary, executes once, and returns the
            # Helion result. Writes into supplied tensors are preserved too.
            output = kernel(*inputs)
            # Fresh measurements reuse the cached configuration without changing
            # supplied tensors. Durations and limits use the shared benchmark API.
            measured = vfunc.benchmark(
                kernel, *inputs, options=vfunc.KernelBenchmark(replication_factor=3)
            )
            shapes = (
                tuple(output.shape)
                if isinstance(output, torch.Tensor)
                else [tuple(tensor.shape) for tensor in output]
            )
            print(native.fn.__name__, shapes, measured["runtime_us"], "us")
            print("Selected configuration:", measured["configuration"]["configuration"])
            print("Replicas:", [(r["gpu_uuid"], r["runtime_us"]) for r in measured["replicas"]])


if __name__ == "__main__":
    main()
