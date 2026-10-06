"""ThunderKittens kernels: vector add, in-place exp, BF16 matmul, and raw CUDA scaling.

Clone HazyResearch/ThunderKittens and pass its include directory via --headers.
The selected vFunc image must provide compatible CUDA/nvcc, C++20 and PyTorch.
"""

from __future__ import annotations

import argparse

import torch

import gfaas as vfunc

VECTOR_ADD = r"""
#include "kittens.cuh"
using namespace kittens;
using layout = gl<float, 1, 1, 1, -1>;
__global__ void add_tiles(layout a, layout b, layout c, int n) {
    int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int offset = warp * TILE;
    if (offset + TILE > n) {
        for (int i = offset + threadIdx.x % 32; i < n && i < offset + TILE; i += 32)
            c.raw_ptr[i] = a.raw_ptr[i] + b.raw_ptr[i];
        return;
    }
    rv_fl<TILE> av, bv;
    warp::load(av, a, {0, 0, 0, warp});
    warp::load(bv, b, {0, 0, 0, warp});
    warp::add(av, av, bv);
    warp::store(c, av, {0, 0, 0, warp});
}
extern "C" int launch(float* a, float* b, float* c, int n, VFUNC_LAUNCH_ARGS) {
    layout ag(a, nullptr, nullptr, nullptr, n);
    layout bg(b, nullptr, nullptr, nullptr, n);
    layout cg(c, nullptr, nullptr, nullptr, n);
    add_tiles<<<VFUNC_GRID, VFUNC_BLOCK, VFUNC_SHARED_BYTES, VFUNC_STREAM>>>(ag, bg, cg, n);
    return static_cast<int>(cudaGetLastError());
}
"""

INPLACE_EXP = r"""
#include "kittens.cuh"
using namespace kittens;
using layout = gl<float, 1, 1, 1, -1>;
__global__ void exp_tiles(layout x, int n) {
    int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int offset = warp * TILE;
    if (offset + TILE > n) {
        for (int i = offset + threadIdx.x % 32; i < n && i < offset + TILE; i += 32)
            x.raw_ptr[i] = expf(x.raw_ptr[i]);
        return;
    }
    rv_fl<TILE> value;
    warp::load(value, x, {0, 0, 0, warp});
    warp::exp(value, value);
    warp::store(x, value, {0, 0, 0, warp});
}
extern "C" int launch(float* x, int n, VFUNC_LAUNCH_ARGS) {
    layout input(x, nullptr, nullptr, nullptr, n);
    exp_tiles<<<VFUNC_GRID, VFUNC_BLOCK, VFUNC_SHARED_BYTES, VFUNC_STREAM>>>(input, n);
    return static_cast<int>(cudaGetLastError());
}
"""

MATMUL = r"""
#include "kittens.cuh"
using namespace kittens;
using bf_layout = gl<bf16, 1, 1, -1, -1>;
using fl_layout = gl<float, 1, 1, -1, -1>;
__global__ void matmul_tiles(bf_layout a, bf_layout b, fl_layout c, int k) {
    int row = blockIdx.y, col = blockIdx.x;
    rt_bf<BM, BK> av;
    rt_bf<BK, BN, ducks::rt_layout::col> bv;
    rt_fl<BM, BN> acc;
    warp::zero(acc);
    for (int ki = 0; ki < k / BK; ++ki) {
        warp::load(av, a, {0, 0, row, ki});
        warp::load(bv, b, {0, 0, ki, col});
        warp::mma_AB(acc, av, bv, acc);
    }
    warp::store(c, acc, {0, 0, row, col});
}
extern "C" int launch(bf16* a, bf16* b, float* c, int m, int n, int k, VFUNC_LAUNCH_ARGS) {
    if (m % BM || n % BN || k % BK || block_x != 32) return int(cudaErrorInvalidValue);
    bf_layout ag(a, nullptr, nullptr, m, k);
    bf_layout bg(b, nullptr, nullptr, k, n);
    fl_layout cg(c, nullptr, nullptr, m, n);
    matmul_tiles<<<VFUNC_GRID, VFUNC_BLOCK, VFUNC_SHARED_BYTES, VFUNC_STREAM>>>(ag, bg, cg, k);
    return static_cast<int>(cudaGetLastError());
}
"""


RAW_SCALE = r"""
#include "kittens.cuh"
extern "C" __global__ __launch_bounds__(BLOCK) void scale(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    kittens::rv_fl<32> value;
    value[0][0] = i < n ? x[i] : 0.f;
    kittens::warp::mul(value, value, 2.f);
    if (i < n) y[i] = value[0][0];
}
"""


def raw_grid(meta):
    return ((meta["N"] + meta["BLOCK"] - 1) // meta["BLOCK"],)


def evaluate_scale(candidate, x, y, N):
    original = x.clone()
    candidate(x, y, N)
    return bool(torch.equal(y, original * 2))


def vector_grid(meta):
    return (
        (meta["N"] + meta["TILE"] * (meta["block"][0] // 32) - 1)
        // (meta["TILE"] * (meta["block"][0] // 32)),
    )


def matmul_grid(meta):
    return (meta["N"] // meta["BN"], meta["M"] // meta["BM"])


def evaluate_add(candidate, a, b, c, N):
    candidate(a, b, c, N)
    return bool(torch.allclose(c, a + b, rtol=1e-5, atol=1e-6))


def evaluate_exp(candidate, x, N):
    original = x.clone()
    candidate(x, N)
    return bool(torch.allclose(x, original.exp(), rtol=1e-5, atol=1e-6))


def evaluate_matmul(candidate, a, b, c, M, N, K):
    candidate(a, b, c, M, N, K)
    return bool(torch.allclose(c, a.float() @ b.float(), rtol=0.02, atol=0.03))


def make_examples(headers):
    vector_configs = [
        vfunc.ThunderKittensConfig({"TILE": tile}, block=(warps * 32,))
        for tile in (64, 128, 256)
        for warps in (2, 4)
    ]
    return {
        "raw": (
            vfunc.ThunderKittensKernel(
                RAW_SCALE,
                "scale",
                entrypoint_kind="kernel",
                signature={"X": "pointer", "Y": "pointer", "N": "int32"},
                headers=headers,
                configs=[
                    vfunc.ThunderKittensConfig({"BLOCK": block}, block=(block,))
                    for block in (128, 256)
                ],
                tuning=vfunc.TritonTuning(evaluate=evaluate_scale),
                restore_value=("X",),
            ),
            raw_grid,
        ),
        "add": (
            vfunc.ThunderKittensKernel(
                VECTOR_ADD,
                "launch",
                signature={"A": "pointer", "B": "pointer", "C": "pointer", "N": "int32"},
                headers=headers,
                configs=vector_configs,
                tuning=vfunc.TritonTuning(evaluate=evaluate_add),
                reset_to_zero=("C",),
            ),
            vector_grid,
        ),
        "exp": (
            vfunc.ThunderKittensKernel(
                INPLACE_EXP,
                "launch",
                signature={"X": "pointer", "N": "int32"},
                headers=headers,
                configs=vector_configs,
                tuning=vfunc.TritonTuning(evaluate=evaluate_exp),
                restore_value=("X",),
            ),
            vector_grid,
        ),
        "matmul": (
            vfunc.ThunderKittensKernel(
                MATMUL,
                "launch",
                signature={
                    "A": "pointer",
                    "B": "pointer",
                    "C": "pointer",
                    "M": "int32",
                    "N": "int32",
                    "K": "int32",
                },
                headers=headers,
                configs=[
                    vfunc.ThunderKittensConfig({"BM": bm, "BN": bn, "BK": bk}, block=(32,))
                    for bm in (16, 32)
                    for bn in (16, 32)
                    for bk in (16, 32)
                ],
                tuning=vfunc.TritonTuning(evaluate=evaluate_matmul),
            ),
            matmul_grid,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    app = vfunc.App("thunderkittens-examples", image=vfunc.Image(args.image))
    examples = make_examples(args.headers)
    torch.manual_seed(0)
    n = 1_048_576
    a, b = torch.randn(n), torch.randn(n)
    x = torch.randn(n) * 0.1
    size = 256
    ma = torch.randn(size, size, dtype=torch.bfloat16) * 0.1
    mb = torch.randn(size, size, dtype=torch.bfloat16) * 0.1
    raw = torch.randn(n)
    expected_raw = raw.clone() * 2
    expected_exp = x.exp()
    calls = {
        "raw": (raw, raw, n),
        "add": (a, b, torch.zeros(n), n),
        "exp": (x, n),
        "matmul": (ma, mb, torch.zeros(size, size), size, size, size),
    }
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        for name, (kernel, grid) in examples.items():
            values = calls[name]
            first = vfunc.benchmark(kernel[grid], *values)
            cached = vfunc.benchmark(kernel[grid], *values)
            kernel[grid](*values)
            print(name, first["runtime_us"], cached["runtime_us"], cached["configuration"])
    assert torch.equal(raw, expected_raw)
    assert torch.allclose(calls["add"][2], a + b, rtol=1e-5, atol=1e-6)
    assert torch.allclose(x, expected_exp, rtol=1e-5, atol=1e-6)
    assert torch.allclose(calls["matmul"][2], ma.float() @ mb.float(), rtol=0.02, atol=0.03)


if __name__ == "__main__":
    main()
