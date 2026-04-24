"""Evaluate GSC CRISPR demo cell covariates with CRANE extension response."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc

import crane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="demo_workspace/data/demo_gsc.h5ad",
        help="Rebuilt GSC CRISPR demo .h5ad file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    if not input_h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {input_h5ad}\n"
            "Run: python examples/rebuild_demo_data.py"
        )

    adata = sc.read_h5ad(input_h5ad)
    gene_result = crane.tl.gene_response(
        adata,
        perturbation_key="perturbation_targets",
        control_value="control",
        case_value="GSC",
        layer="log1p_norm",
        inplace=False,
    )

    cell_vector = adata.obs.loc[:, ["n_genes", "S_score", "G2M_score", "phase", "rep_batch"]].copy()
    ext_result = crane.tl.extension_response(
        adata,
        result=gene_result,
        cell_vector=cell_vector,
    )

    print("\nCell-covariate extension-response results:")
    print(ext_result.summary().head(20))


if __name__ == "__main__":
    main()
