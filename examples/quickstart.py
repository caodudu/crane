"""Run a minimal CRANE gene-response demo on a local .h5ad file."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import crane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True, help="Path to an input .h5ad file.")
    parser.add_argument(
        "--perturbation-key",
        required=True,
        help="Column in adata.obs containing perturbation labels.",
    )
    parser.add_argument("--control-value", required=True, help="Control label value.")
    parser.add_argument(
        "--case-value",
        default=None,
        help="Perturbation label to analyze. Optional only when there is one non-control label.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Expression layer to use. Omit to use adata.X.",
    )
    parser.add_argument(
        "--cell-response",
        action="store_true",
        help="Also run cell-level perturbation response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    if not input_h5ad.exists():
        raise SystemExit(f"Input file does not exist: {input_h5ad}")

    adata = sc.read_h5ad(input_h5ad)

    result = crane.tl.gene_response(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        layer=args.layer,
        inplace=False,
    )

    print("\nTop gene-response results:")
    print(result.summary().head(20))

    if args.cell_response:
        cell_result = crane.tl.cell_response(
            adata,
            perturbation_key=args.perturbation_key,
            control_value=args.control_value,
            case_value=args.case_value,
            layer=args.layer,
            inplace=False,
        )
        print("\nTop cell-response results:")
        print(cell_result.summary().head(20))


if __name__ == "__main__":
    main()
