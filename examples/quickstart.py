"""Run a minimal CRANE gene-response demo on a local .h5ad file."""

from __future__ import annotations

import argparse
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
        default="demo_workspace/data/demo_gsc.h5ad",
        help="Path to an input .h5ad file.",
    )
    parser.add_argument(
        "--perturbation-key",
        default="perturbation_targets",
        help="Column in adata.obs containing perturbation labels.",
    )
    parser.add_argument("--control-value", default="control", help="Control label value.")
    parser.add_argument(
        "--case-value",
        default="GSC",
        help="Perturbation label to analyze. Omit only when there is one non-control label.",
    )
    parser.add_argument(
        "--layer",
        default="log1p_norm",
        help="Expression layer to use. Omit to use adata.X.",
    )
    parser.add_argument(
        "--cell-response",
        action="store_true",
        help="Also run cell-level perturbation response.",
    )
    return parser.parse_args()


def run_standard_scanpy_view(adata, layer: str, output_dir: Path) -> None:
    """Run a standard HVG/PCA/neighbors/UMAP view without changing CRANE input."""

    output_dir.mkdir(parents=True, exist_ok=True)
    view = adata.copy()
    sc.pp.highly_variable_genes(
        view,
        layer=layer,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
    )
    hvg = view.var["highly_variable"].to_numpy()
    baseline = view[:, hvg].copy()
    baseline.X = baseline.layers[layer].copy()
    sc.pp.pca(baseline, n_comps=min(30, baseline.n_vars - 1), random_state=0)
    sc.pp.neighbors(baseline, n_neighbors=20, n_pcs=min(30, baseline.n_vars - 1), metric="cosine")
    sc.tl.umap(baseline, random_state=0)

    baseline.obs[["UMAP_1", "UMAP_2"]] = baseline.obsm["X_umap"]
    baseline.obs[["UMAP_1", "UMAP_2"]].to_csv(output_dir / "umap_coordinates.csv")
    for color in ["perturbation_targets", "n_genes", "S_score", "G2M_score", "phase", "rep_batch"]:
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
    if not input_h5ad.exists():
        raise SystemExit(f"Input file does not exist: {input_h5ad}")

    adata = sc.read_h5ad(input_h5ad)
    run_standard_scanpy_view(
        adata,
        layer=args.layer,
        output_dir=Path("demo_workspace/scanpy/gsc_crispr"),
    )

    result = crane.tl.gene_response(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        layer=args.layer,
        inplace=False,
    )

    print("\nTop gene-response results:")
    print(result.summary(normalized=True).head(20))
    print("\nGene-pair response matrix:")
    pair = result.gene_pair()
    print(pair.iloc[:5, :5])
    print("\nGene modules:")
    print(result.gene_module()[["module_label"]].head(20))

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
