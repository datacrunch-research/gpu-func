"""TLX warp specialization, shared-memory reduction, and shared-memory matmul.

Install matching fbtriton in the client and the selected vFunc image. This
example uses tensor/scalar arguments and native vanilla autotune configurations.
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

import gfaas as vfunc


@triton.jit
def dual_add(X, Y, A, B, OUT1, OUT2, N: tl.constexpr, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offsets = start + tl.arange(0, BLOCK)
            x = tl.load(X + offsets, offsets < N, other=0)
            y = tl.load(Y + offsets, offsets < N, other=0)
            tl.store(OUT1 + offsets, x + y, offsets < N)
        with tlx.async_task(num_warps=4):
            offsets = start + tl.arange(0, BLOCK)
            a = tl.load(A + offsets, offsets < N, other=0)
            b = tl.load(B + offsets, offsets < N, other=0)
            tl.store(OUT2 + offsets, a + b, offsets < N)


@triton.jit
def row_sum(X, OUT, ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(X + row * COLS + col, col < COLS, other=0)
    shared = tlx.local_alloc((BLOCK,), tl.float32, 1)
    tlx.local_store(shared[0], values)
    restored = tlx.local_load(shared[0])
    tl.store(OUT + row, tl.sum(restored, 0))


@triton.jit
def shared_matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    shared_a = tlx.local_alloc((BM, BK), tl.float16, 1)
    shared_b = tlx.local_alloc((BK, BN), tl.float16, 1)
    total = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + k
        a = tl.load(
            A + rows[:, None] * K + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0
        )
        b = tl.load(
            B + ks[:, None] * N + cols[None, :], (ks[:, None] < K) & (cols[None, :] < N), other=0
        )
        tlx.local_store(shared_a[0], a)
        tlx.local_store(shared_b[0], b)
        a = tlx.local_load(shared_a[0])
        b = tlx.local_load(shared_b[0])
        total += tl.dot(a, b)
    tl.store(
        C + rows[:, None] * N + cols[None, :], total, (rows[:, None] < M) & (cols[None, :] < N)
    )


def dual_grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"]),)


def sum_grid(meta):
    return (meta["ROWS"],)


def matmul_grid(meta):
    return (triton.cdiv(meta["M"], meta["BM"]), triton.cdiv(meta["N"], meta["BN"]))


def evaluate_dual(candidate, x, y, a, b, out1, out2, *, N):
    candidate(x, y, a, b, out1, out2, N=N)
    return bool(torch.equal(out1, x + y) and torch.equal(out2, a + b))


def evaluate_sum(candidate, x, out, *, ROWS, COLS):
    candidate(x, out, ROWS=ROWS, COLS=COLS)
    return bool(torch.allclose(out, x.sum(1), atol=1e-4, rtol=1e-4))


def evaluate_matmul(candidate, a, b, c, *, M, N, K):
    candidate(a, b, c, M=M, N=N, K=K)
    return bool(torch.allclose(c.float(), a.float() @ b.float(), atol=0.02, rtol=0.02))


def samples():
    generator = torch.Generator().manual_seed(17)
    vectors = [torch.randn(98432, generator=generator) for _ in range(4)]
    matrix = torch.randn((37, 257), generator=generator)
    a = torch.randn((128, 96), dtype=torch.float16, generator=generator) * 0.1
    b = torch.randn((96, 160), dtype=torch.float16, generator=generator) * 0.1
    return [
        (
            "dual_add",
            dual_add,
            [triton.Config({"BLOCK": block}, num_warps=4) for block in (256, 512, 1024)],
            dual_grid,
            evaluate_dual,
            (*vectors, torch.full_like(vectors[0], -11), torch.full_like(vectors[0], -11)),
            {"N": vectors[0].numel()},
        ),
        (
            "row_sum",
            row_sum,
            [triton.Config({"BLOCK": 512}, num_warps=warps) for warps in (4, 8)],
            sum_grid,
            evaluate_sum,
            (matrix, torch.full((37,), -11.0)),
            {"ROWS": 37, "COLS": 257},
        ),
        (
            "shared_matmul",
            shared_matmul,
            [triton.Config({"BM": bm, "BN": 32, "BK": 32}, num_warps=4) for bm in (16, 32)],
            matmul_grid,
            evaluate_matmul,
            (a, b, torch.full((128, 160), -11, dtype=torch.float16)),
            {"M": 128, "N": 160, "K": 96},
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Registered TLX-enabled image")
    args = parser.parse_args()
    app = vfunc.App("tlx-kernels", image=vfunc.Image(args.image))
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        for name, jit, configs, grid, evaluate, inputs, kwargs in samples():
            native = triton.autotune(configs=configs, key=list(kwargs))(jit)
            kernel = vfunc.TLXKernel(native, tuning=vfunc.TritonTuning(evaluate=evaluate))
            report = vfunc.benchmark(kernel[grid], *inputs, **kwargs)
            assert report["autotuned"]
            kernel[grid](*inputs, **kwargs)
            assert evaluate(kernel[grid], *inputs, **kwargs)
            print(name, report["runtime_us"], "us", report["configuration"])


if __name__ == "__main__":
    main()
