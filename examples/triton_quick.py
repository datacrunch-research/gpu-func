"""Quick-benchmark an ordinary autotune configuration set with correctness gating."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

import gfaas as vfunc


@triton.autotune(configs=[triton.Config({"BLOCK": block}) for block in (64, 128, 256)], key=["N"])
@triton.jit
def add(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, values + 1, offsets < N)


def evaluate(candidate, x, out, *, N) -> bool:
    candidate(x, out, N=N)
    return bool(torch.allclose(out, x + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", default="gb300")
    args = parser.parse_args()
    app = vfunc.App("quick-benchmark", image=vfunc.Image(args.image))
    kernel = vfunc.TritonKernel(
        add,
        tuning=vfunc.TritonTuning(
            quick_benchmark_delta=0.10,
            evaluate=evaluate,
            pruning_min_runtime_us=100,
        ),
    )
    x = torch.arange(4096, dtype=torch.float32)
    out = torch.empty_like(x)
    try:
        with app.function(gpu=args.gpu, cpu_millicores=4000):
            kernel[lambda meta: (triton.cdiv(meta["N"], meta["BLOCK"]),)](x, out, N=x.numel())
    except vfunc.TritonExecutionNotImplementedError as result:
        print(result.report["quick_benchmark"])
        # Full benchmarking and ordinary execution will follow in a later phase.


if __name__ == "__main__":
    main()
