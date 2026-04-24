"""Evaluate GSC CRISPR demo cell covariates with CRANE extension response."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import crane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        default="demo_workspace/data/demo_gsc.h5ad",
        help="Rebuilt GSC CRISPR demo .h5ad file.",
    )
    return parser.parse_args()


def morans_i(values: np.ndarray, graph) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x - np.nanmean(x)
    denominator = float(np.sum(x * x))
    if denominator <= 0:
        return float("nan")
    if sparse.issparse(graph):
        numerator = float(x @ (graph @ x))
        weight_sum = float(graph.sum())
    else:
        numerator = float(x @ graph @ x)
        weight_sum = float(np.sum(graph))
    if weight_sum <= 0:
        return float("nan")
    return float((len(x) / weight_sum) * numerator / denominator)


def covariate_vectors(obs: pd.DataFrame) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for col in ["n_genes", "S_score", "G2M_score", "phase", "rep_batch"]:
        values = obs[col]
        if pd.api.types.is_numeric_dtype(values):
            vectors[col] = values.astype(float).to_numpy()
        else:
            as_str = values.astype(str)
            for category in sorted(as_str.unique()):
                vectors[f"{col}_{category}"] = (as_str == category).astype(float).to_numpy()
    return vectors


def run_standard_scanpy_graph(adata, layer: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    view = adata.copy()
    sc.pp.highly_variable_genes(
        view,
        layer=layer,
        n_top_genes=min(2000, view.n_vars),
        flavor="seurat",
    )
    baseline = view[:, view.var["highly_variable"].to_numpy()].copy()
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
    return baseline.obsp["connectivities"]


def main() -> None:
    np.random.seed(0)
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    if not input_h5ad.exists():
        raise SystemExit(
            f"Input file does not exist: {input_h5ad}\n"
            "Run: python examples/rebuild_demo_data.py"
        )

    adata = sc.read_h5ad(input_h5ad)
    scanpy_graph = run_standard_scanpy_graph(
        adata,
        layer="log1p_norm",
        output_dir=Path("demo_workspace/scanpy/gsc_crispr"),
    )
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

    ext_summary = ext_result.summary()
    rows = []
    for feature, values in covariate_vectors(adata.obs).items():
        if feature not in ext_summary.index:
            continue
        scanpy_strength = abs(morans_i(values, scanpy_graph))
        crane_strength = abs(float(ext_summary.loc[feature, "gene_self_cor"]))
        reduction = (
            100.0 * (1.0 - crane_strength / scanpy_strength)
            if scanpy_strength > 0
            else float("nan")
        )
        rows.append(
            {
                "feature": feature,
                "scanpy_hvg_graph": scanpy_strength,
                "crane_graph": crane_strength,
                "reduction_percent": reduction,
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("scanpy_hvg_graph", ascending=False)
        .set_index("feature")
    )
    print("\nCovariate structure on standard Scanpy graph vs CRANE graph:")
    print(comparison.head(20).round(3))


if __name__ == "__main__":
    main()
