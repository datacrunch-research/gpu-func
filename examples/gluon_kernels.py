"""Gluon examples for vFunc: strided copy, accumulating add, and row softmax.

CPU tensors are transported with their strides, offsets, and aliases intact.
The image must contain matching PyTorch/Triton versions and Gluon support.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

import gfaas as vfunc


@gluon.jit
def copy_strided(X, Y, N: gl.constexpr, SX: gl.constexpr, SY: gl.constexpr, BLOCK: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=BlockedLayout([1], [32], [4], [0])
    )
    values = gl.load(X + offsets * SX, offsets < N, other=0)
    gl.store(Y + offsets * SY, values, offsets < N)


@gluon.jit
def add_values(x, y):
    return x + y


@gluon.jit
def accumulating_add(X, Y, Z, N: gl.constexpr, BLOCK: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    x = gl.load(X + offsets, offsets < N, other=0)
    y = gl.load(Y + offsets, offsets < N, other=0)
    z = gl.load(Z + offsets, offsets < N, other=0)
    gl.store(Z + offsets, add_values(x, y) + z, offsets < N)


@gluon.jit
def row_softmax(X, Y, ROWS: gl.constexpr, COLS: gl.constexpr, BLOCK: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
    x = gl.load(X + row * COLS + cols, cols < COLS, other=-float("inf"))
    e = gl.exp(x - gl.max(x, 0))
    gl.store(Y + row * COLS + cols, e / gl.sum(e, 0), cols < COLS)


def grid_1d(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"]),)


def grid_rows(meta):
    return (meta["ROWS"],)


def evaluate_add(candidate, x, y, z, *, N):
    candidate(x, y, z, N=N)
    return bool(torch.allclose(z, x + y, atol=1e-6, rtol=1e-6))


def evaluate_softmax(candidate, x, y, *, ROWS, COLS):
    candidate(x, y, ROWS=ROWS, COLS=COLS)
    return bool(torch.allclose(y, torch.softmax(x, dim=1), atol=1e-6, rtol=1e-5))


def samples():
    generator = torch.Generator().manual_seed(0)
    backing = torch.randn(2 * 262145, generator=generator)
    output_backing = torch.full_like(backing, -11)
    x, y = backing[1::2], output_backing[1::2]
    yield (
        "strided_copy",
        vfunc.GluonKernel(copy_strided),
        grid_1d,
        (x, y),
        {"N": x.numel(), "SX": 2, "SY": 2, "BLOCK": 256},
        x.clone(),
    )
    x = torch.randn(65537, generator=generator)
    y = torch.randn(65537, generator=generator)
    z = torch.zeros_like(x)
    native = triton.autotune(
        configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in (128, 256, 512)],
        key=["N"],
        reset_to_zero=["Z"],
    )(accumulating_add)
    yield (
        "accumulating_add",
        vfunc.GluonKernel(native, tuning=vfunc.TritonTuning(evaluate=evaluate_add)),
        grid_1d,
        (x, y, z),
        {"N": x.numel()},
        x + y,
    )
    x = torch.randn((257, 511), generator=generator)
    y = torch.zeros_like(x)
    native = triton.autotune(
        configs=[triton.Config({"BLOCK": b}, num_warps=4) for b in (512, 1024, 2048)],
        key=["COLS"],
    )(row_softmax)
    yield (
        "row_softmax",
        vfunc.GluonKernel(native, tuning=vfunc.TritonTuning(evaluate=evaluate_softmax)),
        grid_rows,
        (x, y),
        {"ROWS": 257, "COLS": 511},
        torch.softmax(x, dim=1),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    app = vfunc.App("gluon-examples", image=vfunc.Image(args.image))
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        for name, kernel, grid, inputs, kwargs, expected in samples():
            first = vfunc.benchmark(kernel[grid], *inputs, **kwargs)
            repeat = vfunc.benchmark(kernel[grid], *inputs, **kwargs)
            kernel[grid](*inputs, **kwargs)
            torch.testing.assert_close(inputs[-1], expected, atol=1e-6, rtol=1e-5)
            print(name, first["runtime_us"], repeat["runtime_us"], repeat["reused_specialization"])


if __name__ == "__main__":
    main()
