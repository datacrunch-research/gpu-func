"""Compile an ordinary Triton autotuned kernel through vFunc without launching."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

import gfaas as vfunc


@triton.autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["N"])
@triton.jit
def add(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, x + 1, offsets < N)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="registered image matching local Triton")
    parser.add_argument("--gpu", default="gb300")
    args = parser.parse_args()
    app = vfunc.App("triton-compile", image=vfunc.Image(args.image))
    kernel = vfunc.TritonKernel(add)
    x = torch.empty(1024)  # CPU tensors suffice: only argument metadata is sent.
    out = torch.empty_like(x)
    try:
        with app.function(gpu=args.gpu):
            kernel[lambda meta: (triton.cdiv(1024, meta["BLOCK"]),)](x, out, N=1024)
    except vfunc.TritonExecutionNotImplementedError as error:
        print(f"Calls: {error.call_ids}")
        for variant in error.report["results"]:
            print(variant)


if __name__ == "__main__":
    main()
