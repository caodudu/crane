"""Run CRANE functional response on the Erlotinib drug-data demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scanpy as sc

import crane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="demo_workspace/data/demo_erlotinib_drug.h5ad",
        help="Rebuilt Erlotinib drug-data demo .h5ad file.",
    )
    parser.add_argument(
        "--gene-sets",
        default="demo_data/erlotinib_mapk_egfr_genesets.json",
        help="MAPK and EGFR test gene-set JSON.",
    )
    return parser.parse_args()


def load_gene_sets(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gene_sets = payload.get("gene_sets", payload)
    return {str(key): [str(gene) for gene in value] for key, value in gene_sets.items()}


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    gene_set_path = Path(args.gene_sets)
    if not input_h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {input_h5ad}\n"
            "Run: python examples/rebuild_demo_data.py"
        )

    adata = sc.read_h5ad(input_h5ad)
    gene_sets = load_gene_sets(gene_set_path)

    gene_result = crane.tl.gene_response(
        adata,
        perturbation_key="perturbation_targets",
        control_value="nc",
        case_value="sensi",
        layer="log1p_norm",
        inplace=False,
    )

    function_result = crane.tl.function_response(
        adata,
        result=gene_result,
        gene_set=gene_sets,
        layer="log1p_norm",
        set_min_genes_count=10,
    )

    print("\nFunctional-response results:")
    print(function_result.summary().head(20))


if __name__ == "__main__":
    main()
