"""Run CRANE functional response on the Erlotinib drug-data demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def run_standard_scanpy_view(adata, layer: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    view = adata.copy()
    sc.pp.highly_variable_genes(
        view,
        layer=layer,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
    )
    baseline = view[:, view.var["highly_variable"].to_numpy()].copy()
    baseline.X = baseline.layers[layer].copy()
    sc.pp.pca(baseline, n_comps=min(30, baseline.n_vars - 1), random_state=0)
    sc.pp.neighbors(baseline, n_neighbors=20, n_pcs=min(30, baseline.n_vars - 1), metric="cosine")
    sc.tl.umap(baseline, random_state=0)
    baseline.obs[["UMAP_1", "UMAP_2"]] = baseline.obsm["X_umap"]
    baseline.obs[["UMAP_1", "UMAP_2"]].to_csv(output_dir / "umap_coordinates.csv")
    for color in ["perturbation_targets", "n_genes", "S_score", "G2M_score", "phase"]:
        if color not in baseline.obs:
            continue
        sc.pl.umap(
            baseline,
            color=color,
            show=False,
            frameon=False,
            title=f"Standard Scanpy HVG UMAP: {color}",
        )
        plt.savefig(output_dir / f"umap_{color}.png", dpi=160, bbox_inches="tight")
        plt.close("all")
    print(f"\nStandard Scanpy baseline saved to: {output_dir}")


def main() -> None:
    np.random.seed(0)
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    gene_set_path = Path(args.gene_sets)
    if not input_h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {input_h5ad}\n"
            "Run: python examples/rebuild_demo_data.py"
        )

    adata = sc.read_h5ad(input_h5ad)
    run_standard_scanpy_view(
        adata,
        layer="log1p_norm",
        output_dir=Path("demo_workspace/scanpy/erlotinib_drug"),
    )
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
    summary = function_result.summary(normalized=True)
    best_per_function = (
        summary.assign(function=summary.index.to_series().str.replace(r"_pc\d+$", "", regex=True))
        .sort_values("response_score", ascending=False)
        .groupby("function", sort=False)
        .head(1)
    )
    print(
        best_per_function.loc[
            :,
            ["function", "response_score", "gene_self_cor", "gene_label_cor", "gene_call"],
        ]
        .head(20)
        .round(3)
    )


if __name__ == "__main__":
    main()
