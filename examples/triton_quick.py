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


def make_inputs(metadata):
    spec = metadata["args"][0]
    x = torch.randn(spec["shape"], dtype=getattr(torch, spec["dtype"]), device="cuda")
    return (x, torch.empty_like(x)), {"N": metadata["kwargs"]["N"]["value"]}


def reset_inputs(x, out, *, N):
    out.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", default="gb300")
    args = parser.parse_args()
    app = vfunc.App("quick-benchmark", image=vfunc.Image(args.image))
    kernel = vfunc.TritonKernel(
        add,
        make_inputs=make_inputs,
        reset_inputs=reset_inputs,
        tuning=vfunc.TritonTuning(
            pilot_pruning=vfunc.TritonPruning(relative_delta=1.0, absolute_us=1.0),
            refined_pruning=vfunc.TritonPruning(relative_delta=0.10, absolute_us=1.0),
            evaluate=evaluate,
            replication_factor=3,
        ),
    )
    x = torch.arange(4096, dtype=torch.float32)
    out = torch.empty_like(x)
    with app.function(gpu=args.gpu, cpu_millicores=4000):
        report = kernel[lambda meta: (triton.cdiv(meta["N"], meta["BLOCK"]),)](x, out, N=x.numel())
    print(report["benchmark"])


if __name__ == "__main__":
    main()
