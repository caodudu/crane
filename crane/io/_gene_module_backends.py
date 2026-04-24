"""Optional gene-module backends for CRANE result-level extensions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

BackendRunner = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray | None]]

_BACKEND_RUNNERS: dict[str, BackendRunner] = {}
_COMPAT_METHODS = {"wgcna_leiden", "knn_leiden", "eva", "dynamictreecut"}
_BUILTIN_METHODS = _COMPAT_METHODS | {"correlation_components"}
_SUPPORTED_METHODS = _BUILTIN_METHODS | {"auto"}


def register_gene_module_backend(name: str, runner: BackendRunner) -> None:
    normalized = str(name).strip().lower()
    if not normalized or normalized == "auto":
        raise ValueError("Custom gene-module backend names must be non-empty and cannot be 'auto'.")
    _BACKEND_RUNNERS[normalized] = runner


def available_gene_module_methods() -> tuple[str, ...]:
    return tuple(sorted(_SUPPORTED_METHODS | set(_BACKEND_RUNNERS)))


def resolve_gene_module_method(method: str | None, gene_count: int) -> str:
    normalized = "auto" if method is None else str(method).strip().lower()
    if normalized not in (_SUPPORTED_METHODS | set(_BACKEND_RUNNERS)):
        supported = ", ".join(available_gene_module_methods())
        raise ValueError(f"Unsupported gene_module() method {method!r}. Supported methods: {supported}.")
    if normalized == "auto":
        return "wgcna_leiden" if int(gene_count) > 300 else "knn_leiden"
    return normalized


def run_gene_module_backend(
    *,
    method: str | None,
    gene_cor: np.ndarray,
    fs_ind: pd.Series | np.ndarray | None,
    min_correlation: float,
) -> tuple[str, np.ndarray, np.ndarray | None]:
    actual_method = resolve_gene_module_method(method, gene_cor.shape[0])
    fs_values = _coerce_fs_indicator(fs_ind, gene_cor.shape[0])

    runner = _BACKEND_RUNNERS.get(actual_method)
    if runner is None:
        raise ValueError(f"Unsupported gene_module backend {actual_method!r}.")
    return actual_method, *runner(gene_cor, fs_values, float(min_correlation))


def _coerce_fs_indicator(fs_ind: pd.Series | np.ndarray | None, gene_count: int) -> np.ndarray:
    if fs_ind is None:
        return np.ones(int(gene_count), dtype=np.int8)
    if isinstance(fs_ind, pd.Series):
        return fs_ind.fillna(0).astype(np.int8).to_numpy()
    values = np.asarray(fs_ind, dtype=np.int8).reshape(-1)
    if values.shape[0] != int(gene_count):
        raise ValueError("Gene-module backend received an fs indicator with inconsistent length.")
    return values


def _warn_optional_dependency(method: str, packages: str, exc: Exception) -> None:
    warnings.warn(
        (
            f"gene_module(method='{method}') requires optional extension dependencies that are not part of the "
            f"default CRANE install. Install the optional extension set before retrying. "
            f"Suggested install: pip install {packages}. Original import error: {exc}"
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def _normalize_gene_cor(gene_cor: np.ndarray) -> np.ndarray:
    min_val = float(gene_cor.min())
    max_val = float(gene_cor.max())
    if max_val <= min_val:
        return np.zeros_like(gene_cor, dtype=np.float32)
    return ((gene_cor - min_val) / (max_val - min_val)).astype(np.float32, copy=False)


def _sparsify_soft_distance(
    filtered: np.ndarray,
    *,
    max_soft_thres: float,
    sparsity_threshold: int,
) -> np.ndarray:
    if filtered.shape[0] > sparsity_threshold:
        thresholds = np.percentile(filtered, 99, axis=1, keepdims=True)
        sparse_mat = np.where(filtered >= thresholds, filtered, 0)
    else:
        sparse_mat = np.where(filtered < max_soft_thres * 0.5, 0, filtered)
    return np.asarray(sparse_mat, dtype=np.float32)


def _prepare_soft_affinity(
    gene_cor: np.ndarray,
    *,
    soft_beta: int,
    sparsity_threshold: int = 1800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalize_gene_cor(gene_cor)
    soft_distance = np.power(normalized, soft_beta).astype(np.float32, copy=False)
    max_soft = np.max(soft_distance, axis=1)
    max_soft_thres = float(np.mean(max_soft)) if max_soft.size else 0.0
    mask = max_soft >= max_soft_thres
    filtered = soft_distance[np.ix_(mask, mask)]
    filtered_sparse = _sparsify_soft_distance(
        filtered,
        max_soft_thres=max_soft_thres,
        sparsity_threshold=sparsity_threshold,
    )
    filtered_sparse = (filtered_sparse * filtered_sparse.T).astype(np.float32, copy=False)
    return mask, filtered_sparse, np.where(~mask)[0]


def _compress_module_labels(
    gene_df: pd.DataFrame,
    *,
    min_module_thres: int,
) -> np.ndarray:
    module_values = gene_df["module_backend_label"].astype(str).to_numpy()
    module_counts = Counter(module_values[module_values != "0"])
    small_modules = {module for module, count in module_counts.items() if int(count) < int(min_module_thres)}
    gene_df["module_backend_label"] = gene_df["module_backend_label"].apply(
        lambda value: "0" if str(value) in small_modules else str(value)
    )
    unique_labels = [label for label in gene_df["module_backend_label"].astype(str).unique().tolist() if label != "0"]
    relabel = {old: str(idx + 1) for idx, old in enumerate(unique_labels)}
    gene_df["module_backend_label"] = gene_df["module_backend_label"].map(lambda value: relabel.get(str(value), "0"))
    return gene_df["module_backend_label"].astype(str).to_numpy()


def _run_knn_leiden(gene_cor: np.ndarray, fs_ind: np.ndarray, min_correlation: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - environment dependent
        _warn_optional_dependency("knn_leiden", "igraph leidenalg", exc)
        raise ImportError("Optional dependencies for gene_module(method='knn_leiden') are missing.") from exc

    gene_count = int(gene_cor.shape[0])
    if gene_count < 2:
        labels = np.array(["1"] * gene_count, dtype=object)
        return labels, np.eye(gene_count, dtype=np.float32)

    cor_mat = np.asarray(gene_cor, dtype=np.float32)
    n_pcs = max(1, min(30, gene_count - 1))
    n_neighbors = max(1, min(5, gene_count - 1))
    embedding = PCA(n_components=n_pcs, svd_solver="auto", random_state=0).fit_transform(cor_mat)
    connectivities = kneighbors_graph(
        embedding,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
        metric="euclidean",
    )
    connectivities = connectivities.maximum(connectivities.T).astype(np.float32)
    cor_ad = ad.AnnData(embedding.astype(np.float32, copy=False))
    cor_ad.obsp["connectivities"] = connectivities
    cor_ad.uns["neighbors"] = {"connectivities_key": "connectivities"}
    sc.tl.leiden(cor_ad, key_added="module_backend_label", resolution=0.1)
    return (
        cor_ad.obs["module_backend_label"].astype(str).to_numpy(),
        np.asarray(cor_ad.obsp["connectivities"].toarray(), dtype=np.float32),
    )


def _run_wgcna_leiden(gene_cor: np.ndarray, fs_ind: np.ndarray, min_correlation: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - environment dependent
        _warn_optional_dependency("wgcna_leiden", "igraph leidenalg dynamicTreeCut", exc)
        raise ImportError("Optional dependencies for gene_module(method='wgcna_leiden') are missing.") from exc

    gene_df = pd.DataFrame({"module_backend_label": "-1"}, index=np.arange(gene_cor.shape[0]))
    num_genes = int(gene_cor.shape[0])
    leiden_resolution = np.log10(max(num_genes, 10)) / 3
    min_module_thres = max(5, int(np.ceil(num_genes * 0.003)))
    soft_beta = 4 if num_genes < 1800 else 6
    mask, filtered_sparse, discarded = _prepare_soft_affinity(gene_cor, soft_beta=soft_beta)
    gene_df.loc[discarded, "module_backend_label"] = "0"

    if filtered_sparse.shape[0] == 0:
        return gene_df["module_backend_label"].astype(str).to_numpy(), np.zeros_like(gene_cor, dtype=np.float32)

    ad_leiden = ad.AnnData(filtered_sparse)
    ad_leiden.obsp["connectivities"] = sparse.csr_matrix(filtered_sparse)
    ad_leiden.uns["neighbors"] = {"connectivities_key": "connectivities"}
    sc.tl.leiden(ad_leiden, key_added="module_backend_label", resolution=leiden_resolution)
    module_label = (ad_leiden.obs["module_backend_label"].astype(int) + 1).astype(str).to_numpy()

    kept_idx = np.where(mask)[0]
    gene_df.loc[kept_idx, "module_backend_label"] = module_label
    _compress_module_labels(gene_df, min_module_thres=min_module_thres)

    full_affinity = np.zeros((num_genes, num_genes), dtype=np.float32)
    full_affinity[np.ix_(mask, mask)] = filtered_sparse
    return gene_df["module_backend_label"].astype(str).to_numpy(), full_affinity


def _run_dynamictreecut(gene_cor: np.ndarray, fs_ind: np.ndarray, min_correlation: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        from dynamicTreeCut import cutreeHybrid
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist
    except Exception as exc:  # pragma: no cover - environment dependent
        _warn_optional_dependency("dynamictreecut", "dynamicTreeCut", exc)
        raise ImportError("Optional dependencies for gene_module(method='dynamictreecut') are missing.") from exc

    gene_df = pd.DataFrame({"module_backend_label": "-1"}, index=np.arange(gene_cor.shape[0]))
    num_genes = int(gene_cor.shape[0])
    min_module_thres = max(5, int(np.ceil(num_genes * 0.003)))
    soft_beta = 4 if num_genes < 1800 else 6
    mask, filtered_sparse, discarded = _prepare_soft_affinity(gene_cor, soft_beta=soft_beta)
    gene_df.loc[discarded, "module_backend_label"] = "0"

    if filtered_sparse.shape[0] == 0:
        return gene_df["module_backend_label"].astype(str).to_numpy(), np.zeros_like(gene_cor, dtype=np.float32)
    if filtered_sparse.shape[0] == 1:
        kept_idx = np.where(mask)[0]
        gene_df.loc[kept_idx, "module_backend_label"] = "1"
        return gene_df["module_backend_label"].astype(str).to_numpy(), np.asarray(filtered_sparse, dtype=np.float32)

    distances = pdist(filtered_sparse, metric="euclidean")
    link = linkage(distances, method="average")
    clusters = cutreeHybrid(link, distances)
    module_label = np.asarray(clusters["labels"]).astype(str)

    kept_idx = np.where(mask)[0]
    gene_df.loc[kept_idx, "module_backend_label"] = module_label
    _compress_module_labels(gene_df, min_module_thres=min_module_thres)

    full_affinity = np.zeros((num_genes, num_genes), dtype=np.float32)
    full_affinity[np.ix_(mask, mask)] = filtered_sparse
    return gene_df["module_backend_label"].astype(str).to_numpy(), full_affinity


def _run_eva(gene_cor: np.ndarray, fs_ind: np.ndarray, min_correlation: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import networkx as nx
        from Eva import eva_best_partition
    except Exception as exc:  # pragma: no cover - environment dependent
        _warn_optional_dependency("eva", "networkx eva-lcd", exc)
        raise ImportError("Optional dependencies for gene_module(method='eva') are missing.") from exc

    gene_df = pd.DataFrame({"module_backend_label": "-1"}, index=np.arange(gene_cor.shape[0]))
    num_genes = int(gene_cor.shape[0])
    soft_beta = int(np.ceil(np.log10(max(num_genes, 10))) + 2)
    mask, filtered_sparse, discarded = _prepare_soft_affinity(gene_cor, soft_beta=soft_beta)
    filtered_fs = fs_ind[mask]
    gene_df.loc[discarded, "module_backend_label"] = "0"

    graph = nx.from_numpy_array(filtered_sparse)
    for idx, gene_fs in enumerate(filtered_fs):
        graph.nodes[idx]["fs_ind"] = int(gene_fs)

    part, _ = eva_best_partition(graph, alpha=0.5)
    module_label = np.array([str(int(part[node]) + 1) for node in range(filtered_sparse.shape[0])], dtype=object)

    kept_idx = np.where(mask)[0]
    gene_df.loc[kept_idx, "module_backend_label"] = module_label

    min_module_thres = max(7, int(np.ceil(num_genes * 0.005)))
    _compress_module_labels(gene_df, min_module_thres=min_module_thres)

    full_affinity = np.zeros((num_genes, num_genes), dtype=np.float32)
    full_affinity[np.ix_(mask, mask)] = filtered_sparse
    return gene_df["module_backend_label"].astype(str).to_numpy(), full_affinity


def _run_correlation_components(
    gene_cor: np.ndarray,
    fs_ind: np.ndarray,
    min_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = (gene_cor >= float(min_correlation)).astype(np.int8, copy=False)
    if adjacency.size:
        np.fill_diagonal(adjacency, 0)
    component_count, component_labels = connected_components(
        sparse.csr_matrix(adjacency),
        directed=False,
        return_labels=True,
    )
    labels = np.array([str(int(v) + 1) for v in component_labels], dtype=object)
    if int(component_count) == 0:
        labels = np.array(["0"] * gene_cor.shape[0], dtype=object)
    return labels, adjacency.astype(np.float32, copy=False)


register_gene_module_backend("correlation_components", _run_correlation_components)
register_gene_module_backend("wgcna_leiden", _run_wgcna_leiden)
register_gene_module_backend("dynamictreecut", _run_dynamictreecut)
register_gene_module_backend("knn_leiden", _run_knn_leiden)
register_gene_module_backend("eva", _run_eva)
