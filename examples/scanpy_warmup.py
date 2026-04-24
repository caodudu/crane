"""Run a standard Scanpy HVG/PCA/neighbors/UMAP workflow on the demo data.

This script is intentionally separate from CRANE. It warms up Scanpy/UMAP
runtime dependencies and saves baseline UMAP plots for quick visual inspection.
The HVG-filtered object created here is not used as CRANE input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scanpy as sc


DEMOS = {
    "gsc_crispr": {
        "h5ad": "demo_workspace/data/demo_gsc.h5ad",
        "layer": "log1p_norm",
        "colors": ["perturbation_targets", "n_genes", "S_score", "G2M_score", "phase", "rep_batch"],
    },
    "erlotinib_drug": {
        "h5ad": "demo_workspace/data/demo_erlotinib_drug.h5ad",
        "layer": "log1p_norm",
        "colors": ["perturbation_targets", "n_genes", "S_score", "G2M_score", "phase"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_workspace/scanpy",
        help="Directory for baseline Scanpy outputs.",
    )
    return parser.parse_args()


def run_scanpy_baseline(name: str, config: dict[str, object], output_dir: Path) -> None:
    h5ad = Path(str(config["h5ad"]))
    if not h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {h5ad}\n"
            "Run: python examples/rebuild_demo_data.py"
        )

    layer = str(config["layer"])
    adata = sc.read_h5ad(h5ad)
    if layer not in adata.layers:
        raise KeyError(f"{layer!r} is not present in {h5ad}.")

    sc.pp.highly_variable_genes(
        adata,
        layer=layer,
        n_top_genes=min(2000, adata.n_vars),
        flavor="seurat",
    )
    hvg = adata.var["highly_variable"].to_numpy()
    baseline = adata[:, hvg].copy()
    baseline.X = baseline.layers[layer].copy()

    sc.pp.pca(baseline, n_comps=min(30, baseline.n_vars - 1), random_state=0)
    sc.pp.neighbors(baseline, n_neighbors=20, n_pcs=min(30, baseline.n_vars - 1), metric="cosine")
    sc.tl.umap(baseline, random_state=0)

    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    baseline.obs[["UMAP_1", "UMAP_2"]] = baseline.obsm["X_umap"]
    baseline.obs[["UMAP_1", "UMAP_2"]].to_csv(dataset_dir / "umap_coordinates.csv")

    colors = [color for color in config["colors"] if color in baseline.obs]
    for color in colors:
        sc.pl.umap(
            baseline,
            color=color,
            show=False,
            frameon=False,
            title=f"{name}: Scanpy HVG UMAP colored by {color}",
        )
        plt.savefig(dataset_dir / f"umap_{color}.png", dpi=160, bbox_inches="tight")
        plt.close("all")

    print(f"Wrote Scanpy baseline outputs to {dataset_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for name, config in DEMOS.items():
        run_scanpy_baseline(name, config, output_dir)


if __name__ == "__main__":
    main()
