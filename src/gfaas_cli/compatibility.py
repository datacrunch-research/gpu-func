"""Shared CLI options for the SDK's hardware requirements."""

from __future__ import annotations

import argparse

from gfaas import Compatibility


def add_compatibility_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--isolation",
        choices=("container", "sandbox"),
        help="require Environment execution isolation; does not reserve a dedicated GPU or host",
    )
    parser.add_argument(
        "--cpu-architecture",
        choices=("amd64", "arm64"),
        help="require this worker CPU architecture; the image must also match",
    )
    parser.add_argument(
        "--gpu-compute-capability",
        action="append",
        default=[],
        metavar="MAJOR.MINOR",
        help="accepted exact GPU compute capability, e.g. 10.3; repeat for alternatives",
    )


def compatibility_from_args(args: argparse.Namespace) -> Compatibility | None:
    architecture = getattr(args, "cpu_architecture", None)
    capabilities = tuple(getattr(args, "gpu_compute_capability", ()))
    if architecture is None and not capabilities:
        return None
    return Compatibility(
        cpu_architecture=architecture,
        gpu_compute_capabilities=capabilities,
    )
