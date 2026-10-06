"""Compile, autotune, benchmark and execute four CUDA C++ kernels on vFunc.

Run: python examples/cuda_kernel.py --image REGISTERED_CUDA_DEVEL_IMAGE
The image needs nvcc and PyTorch. Local CPU tensors work; no local nvcc/GPU needed.
"""

from __future__ import annotations

import argparse

import torch

import gfaas as vfunc

ADD = r"""
#ifndef SCALE
#define SCALE 1
#endif
#if BROKEN
#error Deliberately unsupported configuration, retained as compilation evidence
#endif
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = SCALE * (a[i] + b[i]);
}
"""
STRIDED = r"""
extern "C" __global__ void update(float* x, float* y, int n, int sx, int sy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i*sx] *= 2; y[i*sy] += x[i*sx]; }
}
"""
MATMUL = r"""
extern "C" __global__ void matmul(const float* a, const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float value = 0;
        for (int k=0; k<n; ++k) value += a[row*n+k] * b[k*n+col];
        c[row*n+col] = value;
    }
}
"""


SHARED = r"""
extern "C" __global__ void scale(float* x, long long n, double alpha) {
    extern __shared__ float values[];
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[threadIdx.x] = (float)(x[i] * alpha);
    __syncthreads();
    if (i < n) x[i] = values[threadIdx.x];
}
"""


def vector_grid(meta):
    return ((meta["n"] + meta["block"][0] - 1) // meta["block"][0],)


def matrix_grid(meta):
    return (
        (meta["n"] + meta["block"][0] - 1) // meta["block"][0],
        (meta["n"] + meta["block"][1] - 1) // meta["block"][1],
    )


def evaluate_add(candidate, a, b, c, n):
    expected = a + b
    candidate(a, b, c, n)
    return bool(torch.allclose(c, expected, atol=1e-5, rtol=1e-5))


def evaluate_update(candidate, x, y, n, sx, sy):
    expected = x.clone() * 2
    candidate(x, y, n, sx, sy)
    return bool(torch.allclose(x, expected) and torch.allclose(y, expected))


def evaluate_matmul(candidate, a, b, c, n):
    candidate(a, b, c, n)
    return bool(torch.allclose(c, a @ b, atol=1e-3, rtol=1e-3))


def evaluate_scale(candidate, x, n, alpha):
    expected = x.clone() * alpha
    candidate(x, n, alpha)
    return bool(torch.allclose(x, expected))


def samples():
    torch.manual_seed(0)
    n = 65536
    a, b = torch.randn(n), torch.randn(n)
    add = vfunc.CUDAKernel(
        ADD,
        name="add",
        signature={"a": "pointer", "b": "pointer", "c": "pointer", "n": "int32"},
        configs=[
            vfunc.CUDAConfig((64,), {"SCALE": 2}),
            vfunc.CUDAConfig((128,)),
            vfunc.CUDAConfig((256,)),
            vfunc.CUDAConfig((256,), {"BROKEN": 1}),
        ],
        restore_value=("a",),
        tuning=vfunc.KernelTuning(evaluate=evaluate_add),
    )
    # c aliases a: writes must preserve client object identity and storage aliases.
    yield "add-alias", add, vector_grid, (a, b, a, n), [(a, a.clone() + b)]
    xbase, ybase = torch.randn(n * 2 + 8), torch.zeros(n * 3 + 16)
    x, y = xbase[3 : 3 + 2 * n : 2], ybase[7 : 7 + 3 * n : 3]
    update = vfunc.CUDAKernel(
        STRIDED,
        name="update",
        signature={"x": "pointer", "y": "pointer", "n": "int32", "sx": "int32", "sy": "int32"},
        configs=[vfunc.CUDAConfig((128,)), vfunc.CUDAConfig((256,))],
        restore_value=("x",),
        reset_to_zero=("y",),
        tuning=vfunc.KernelTuning(evaluate=evaluate_update),
    )
    expected = x.clone() * 2
    yield (
        "strided-reset-restore",
        update,
        vector_grid,
        (x, y, n, 2, 3),
        [(x, expected), (y, expected)],
    )
    n = 256
    a, b, c = torch.randn(n, n) * 0.1, torch.randn(n, n) * 0.1, torch.zeros(n, n)
    matmul = vfunc.CUDAKernel(
        MATMUL,
        name="matmul",
        signature={"a": "pointer", "b": "pointer", "c": "pointer", "n": "int32"},
        configs=[vfunc.CUDAConfig((8, 8)), vfunc.CUDAConfig((16, 16)), vfunc.CUDAConfig((32, 8))],
        tuning=vfunc.KernelTuning(evaluate=evaluate_matmul),
    )
    yield "matmul", matmul, matrix_grid, (a, b, c, n), [(c, a @ b)]

    n = 65536
    x = torch.randn(n)
    scale = vfunc.CUDAKernel(
        SHARED,
        name="scale",
        signature={"x": "pointer", "n": "int64", "alpha": "float64"},
        configs=[vfunc.CUDAConfig((128,), shared_memory=512)],
        nvcc_flags=("--std=c++17",),
        restore_value=("x",),
        tuning=vfunc.KernelTuning(
            evaluate=evaluate_scale,
            benchmark=vfunc.BenchmarkSettings(graph_duration_ms=0.01),
        ),
    )
    yield "shared-scalars", scale, vector_grid, (x, n, 1.25), [(x, x.clone() * 1.25)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    app = vfunc.App("cuda-kernel-examples", image=vfunc.Image(args.image))
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        for name, kernel, grid, inputs, expected in samples():
            # Exercise direct events on the shared-memory sample as well.
            options = (
                vfunc.KernelBenchmark(graph_duration_ms=0.01)
                if name == "shared-scalars"
                else vfunc.KernelBenchmark()
            )
            cold = vfunc.benchmark(kernel[grid], *inputs, options=options)
            warm = vfunc.benchmark(kernel[grid], *inputs, options=options)
            kernel[grid](*inputs)
            assert all(
                torch.allclose(actual, reference, atol=1e-3, rtol=1e-3)
                for actual, reference in expected
            )
            print(
                name,
                "us:",
                warm["runtime_us"],
                "configuration:",
                warm["configuration"],
                "cold autotuned:",
                cold["autotuned"],
                "warm reused:",
                warm["reused_specialization"],
            )


if __name__ == "__main__":
    main()
