"""Tune a small Triton vector kernel through one staged Call."""

from __future__ import annotations

import argparse
import json

from gfaas import TritonCandidate, TritonCase, spawn_triton_tuning

WORKLOAD = r"""
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    left = tl.load(X + offsets, offsets < N, other=0)
    right = tl.load(Y + offsets, offsets < N, other=0)
    tl.store(OUT + offsets, left + right, offsets < N)

def make_inputs(case):
    n = case["params"]["n"]
    return {
        "X": torch.ones(n, device="cuda"),
        "Y": torch.full((n,), 2.0, device="cuda"),
        "OUT": torch.empty(n, device="cuda"),
    }

def grid(case, candidate):
    n = case["params"]["n"]
    block = candidate["constexprs"]["BLOCK"]
    return ((n + block - 1) // block,)

def reset_inputs(case, inputs):
    inputs["OUT"].zero_()

def validate(case, inputs):
    return bool(torch.all(inputs["OUT"] == 3).item())

def suggest_candidates(case, results):
    return [{"name": "block-256", "constexprs": {"BLOCK": 256}, "num_warps": 4, "num_stages": 2}]
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="registered image with Triton 3.6.0 and torch"
    )
    parser.add_argument("--gpu", required=True, help="GPU pool name")
    parser.add_argument("--target-arch", type=int, required=True, help="CUDA SM, such as 103")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    result = spawn_triton_tuning(
        source=WORKLOAD,
        kernel_name="vector_add",
        signature={
            "X": "*fp32",
            "Y": "*fp32",
            "OUT": "*fp32",
            "N": "constexpr",
            "BLOCK": "constexpr",
        },
        candidates=[
            TritonCandidate("block-64", {"BLOCK": 64}),
            TritonCandidate("block-128", {"BLOCK": 128}),
        ],
        cases=[
            TritonCase("n=256", {"n": 256}, {"N": 256}),
            TritonCase("n=1024", {"n": 1024}, {"N": 1024}),
        ],
        target_arch=args.target_arch,
        image=args.image,
        gpu=args.gpu,
        max_adaptive_candidates=1 if args.adaptive else 0,
    )
    print(f"Call: {result.call_id}")
    print(json.dumps(result.wait(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
