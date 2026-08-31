"""Run one matrix multiplication on a GPU with PyTorch."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected image")

    device = torch.device("cuda")
    left = torch.randn((args.size, args.size), device=device)
    right = torch.randn((args.size, args.size), device=device)
    result = left @ right
    torch.cuda.synchronize()

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"matrix shape: {tuple(result.shape)}")
    print(f"checksum: {result.sum().item():.6f}")


if __name__ == "__main__":
    main()
