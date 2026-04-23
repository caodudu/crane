"""Minimal CLI placeholder for the new CRANE package."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crane",
        description="CLI placeholder for the new CRANE package.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run CRANE on an input h5ad file.")
    run_parser.add_argument("--input-h5ad", required=True, help="Path to input .h5ad")
    run_parser.add_argument(
        "--perturbation-key",
        required=True,
        help="obs column containing perturbation labels",
    )
    run_parser.add_argument(
        "--control-value",
        required=True,
        help="label used as the control condition",
    )
    run_parser.add_argument("--output", help="Optional output path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        raise NotImplementedError(
            "The CRANE CLI shape is reserved, but the runnable pipeline is not wired yet. "
            "Use the demo script as the current interface reference."
        )

    parser.print_help()
    return 0
