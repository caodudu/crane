"""Rebuild CRANE demo .h5ad files from compressed repository components."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse


DATASETS = {
    "gsc": "gsc",
    "drug-trace-progeny": "drug_trace_progeny",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="Demo dataset to rebuild.",
    )
    parser.add_argument(
        "--demo-data-dir",
        default="demo_data",
        help="Directory containing compressed demo components.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where rebuilt .h5ad files will be written.",
    )
    return parser.parse_args()


def _matrix_path_from_manifest(dataset_dir: Path, manifest: dict) -> Path:
    matrix_file = manifest.get("matrix_file")
    if matrix_file:
        return dataset_dir / str(matrix_file)

    chunks = manifest.get("matrix_chunks")
    if not chunks:
        raise ValueError(f"No matrix_file or matrix_chunks in {dataset_dir / 'manifest.json'}")

    tmp = tempfile.NamedTemporaryFile(prefix="crane_demo_matrix_", suffix=".npz", delete=False)
    tmp_path = Path(tmp.name)
    with tmp:
        for chunk_name in chunks:
            with (dataset_dir / str(chunk_name)).open("rb") as handle:
                tmp.write(handle.read())
    return tmp_path


def rebuild_one(dataset_dir: Path, output_dir: Path) -> Path:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    matrix_path = _matrix_path_from_manifest(dataset_dir, manifest)
    try:
        matrix = sparse.load_npz(matrix_path).tocsr()
    finally:
        if matrix_path.parent == Path(tempfile.gettempdir()):
            matrix_path.unlink(missing_ok=True)

    obs = pd.read_csv(dataset_dir / manifest["obs_file"], index_col=0)
    var_names = pd.read_csv(
        dataset_dir / manifest["var_names_file"],
        header=None,
    )[0].astype(str)
    var = pd.DataFrame(index=var_names)
    var.index.name = "gene"

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    layer = manifest.get("expression_layer")
    if layer:
        adata.layers[str(layer)] = matrix.copy()
    adata.uns["crane_demo"] = manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / manifest["output_h5ad"]
    adata.write_h5ad(output_path, compression="gzip")
    return output_path


def main() -> None:
    args = parse_args()
    demo_data_dir = Path(args.demo_data_dir)
    output_dir = Path(args.output_dir)

    selected = DATASETS.values() if args.dataset == "all" else [DATASETS[args.dataset]]
    for dataset_name in selected:
        output_path = rebuild_one(demo_data_dir / dataset_name, output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
