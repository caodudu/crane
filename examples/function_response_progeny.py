"""Run CRANE function response on the drug Trace-seq PROGENy demo."""

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
        default="data/demo_drug_trace_progeny.h5ad",
        help="Rebuilt drug Trace-seq demo .h5ad file.",
    )
    parser.add_argument(
        "--gene-sets",
        default="demo_data/progeny_model_human_top200_genesets.json",
        help="PROGENy top-200 gene-set JSON.",
    )
    parser.add_argument(
        "--pathways",
        nargs="*",
        default=["MAPK", "EGFR"],
        help="Pathways to show in the simple demo. Use --all-pathways to evaluate all.",
    )
    parser.add_argument(
        "--all-pathways",
        action="store_true",
        help="Evaluate all pathways in the PROGENy JSON.",
    )
    return parser.parse_args()


def load_gene_sets(path: Path, pathways: list[str] | None) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gene_sets = payload.get("gene_sets", payload)
    gene_sets = {str(key): [str(gene) for gene in value] for key, value in gene_sets.items()}
    if pathways is None:
        return gene_sets
    missing = [pathway for pathway in pathways if pathway not in gene_sets]
    if missing:
        raise SystemExit(f"Pathway(s) not found in {path}: {', '.join(missing)}")
    return {pathway: gene_sets[pathway] for pathway in pathways}


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    gene_set_path = Path(args.gene_sets)
    if not input_h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {input_h5ad}\n"
            "Run: python examples/rebuild_demo_data.py --dataset drug-trace-progeny"
        )

    adata = sc.read_h5ad(input_h5ad)
    gene_sets = load_gene_sets(gene_set_path, None if args.all_pathways else args.pathways)

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

    print("\nFunction-response results:")
    print(function_result.summary().head(20))


if __name__ == "__main__":
    main()
