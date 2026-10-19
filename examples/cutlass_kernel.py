"""Tune C++ CUTLASS matmul, benchmark its winner, then apply tensor writes.

The image must contain nvcc, PyTorch, and CUTLASS headers; use a header tar
ArtifactRef instead when the headers are not in the image. See README for ABI.
"""

from pathlib import Path

import torch

import gfaas as vfunc


def evaluate(candidate, a, b, out):
    expected = a.float() @ b.float()
    candidate(a, b, out)
    return bool(torch.allclose(out.float(), expected, atol=0.03, rtol=0.02))


def make_kernel(headers=None):
    return vfunc.CutlassKernel(
        (Path(__file__).parent / "cutlass" / "gemm.cu").read_text(),
        argument_names=("a", "b", "out"),
        configurations=[
            {"VF_TILE_M": m, "VF_TILE_N": n, "VF_BF16": True, "VF_STAGES": stages}
            for m in (64, 128)
            for n in (64, 128)
            for stages in (2, 3)
        ],
        headers=headers,
        tuning=vfunc.CutlassTuning(evaluate=evaluate, replication_factor=3),
    )


def run(app, headers=None):
    kernel = make_kernel(headers)
    generator = torch.Generator().manual_seed(0)
    a = torch.randn((256, 256), generator=generator, dtype=torch.bfloat16) * 0.1
    b = torch.randn((256, 256), generator=generator, dtype=torch.bfloat16).t() * 0.1
    out = torch.zeros((256, 256), dtype=torch.bfloat16)
    with app.function(gpu="gb300", timeout=600, capacity_wait=600):
        cold = vfunc.benchmark(kernel, a, b, out)
        warm = vfunc.benchmark(
            kernel, a, b, out, options=vfunc.KernelBenchmark(replication_factor=3)
        )
        kernel(a, b, out)
    print(cold["configuration"], warm["runtime_us"], warm["replicas"])
    return kernel, out
