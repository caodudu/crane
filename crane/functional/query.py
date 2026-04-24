"""Functional and extension response evaluation on a CRANE result space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from ..step1.step1 import _compute_sp_moran_between
from ..internal.logger import CRANELogger
from ..io.schema import LoggerConfig
from ..step2.kernels import heat_kernel_smoothing, overlay_raw_exp


@dataclass
class CRANEExtensionResult:
    """Result object for graph-backed CRANE extension/function evaluation."""

    extension_ad: ad.AnnData
    source_result_ad: ad.AnnData
    metadata: dict[str, Any] = field(default_factory=dict)
    logger_config: LoggerConfig | None = None

    @property
    def result_ad(self) -> ad.AnnData:
        return self.extension_ad

    def __repr__(self) -> str:
        mode = self.metadata.get("mode", "extension_response")
        return f"CRANE {mode} completed with {self.extension_ad.n_vars} evaluated feature(s)."

    __str__ = __repr__

    def summary(
        self,
        *,
        responsive_only: bool = False,
        sort_by: str = "response_score",
        ascending: bool = False,
        represent: str = "mean",
        normalized: bool = False,
        centered: bool = False,
    ) -> pd.DataFrame:
        """Return the feature-level extension summary table.

        The extra keyword arguments mirror ``CRANEResult.summary()`` so demo
        code can switch between gene and extension/function results without
        tripping over a different signature.
        """

        frame = self.extension_ad.var.copy()
        if normalized and {"gene_self_cor", "gene_label_cor"}.issubset(frame.columns):
            score = np.sqrt(
                np.square(frame["gene_self_cor"].astype(np.float32))
                + np.square(frame["gene_label_cor"].astype(np.float32))
            )
            frame["response_score"] = np.asarray(score / np.sqrt(2.0), dtype=np.float32)
        if centered and "response_score" in frame.columns:
            frame["response_score"] = frame["response_score"] - float(frame["response_score"].mean())
        frame.index.name = "feature"
        if responsive_only and "response_identity" in frame.columns:
            frame = frame.loc[frame["response_identity"] == 1].copy()
        if sort_by in frame.columns:
            frame = frame.sort_values(by=sort_by, ascending=ascending)
        return frame

    def to_anndata(self, copy: bool = True) -> ad.AnnData:
        return self.extension_ad.copy() if copy else self.extension_ad


def _as_result_ad(result: Any) -> ad.AnnData:
    result_ad = getattr(result, "result_ad", result)
    if result_ad is None:
        raise ValueError("extension/function response requires result=CRANEResult or result_ad=AnnData.")
    if not hasattr(result_ad, "obs") or not hasattr(result_ad, "var") or not hasattr(result_ad, "obsp"):
        raise TypeError("result must be a CRANEResult or a result-space AnnData.")
    return result_ad


def _dense_matrix(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _resolve_affinity(result_ad: ad.AnnData, affinity_key: str = "affinity") -> np.ndarray:
    if affinity_key not in result_ad.obsp:
        raise ValueError(f"result.result_ad.obsp[{affinity_key!r}] is required.")
    affinity = _dense_matrix(result_ad.obsp[affinity_key])
    if affinity.ndim != 2 or affinity.shape != (result_ad.n_obs, result_ad.n_obs):
        raise ValueError("result.result_ad affinity graph must be square and obs-aligned.")
    if float(np.sum(affinity)) == 0.0:
        raise ValueError("result.result_ad affinity graph has zero total weight.")
    return affinity


def _resolve_label(result_ad: ad.AnnData, label_key: str | None = None) -> tuple[str, np.ndarray]:
    candidates = [label_key] if label_key is not None else ["label_last", "label_mixed", "label_raw"]
    for key in candidates:
        if key is not None and key in result_ad.obs:
            return key, result_ad.obs[key].to_numpy(dtype=np.float32)
    raise ValueError("result.result_ad.obs must contain label_last, label_mixed, or label_raw.")


def _result_alignment_labels(result_ad: ad.AnnData) -> pd.Index:
    if "original_cell_id" in result_ad.obs.columns:
        return pd.Index(result_ad.obs["original_cell_id"].astype(str), name="original_cell_id")
    return pd.Index(result_ad.obs_names.astype(str), name="cell_id")


def _anndata_expression_frame(
    adata: Any,
    *,
    layer: str | None = None,
    genes: set[str] | None = None,
) -> pd.DataFrame:
    if adata is None:
        raise ValueError("adata is required for missing-gene supplementation.")

    var_names = adata.var_names.astype(str)
    selected = list(var_names) if genes is None else [gene for gene in var_names if gene in genes]
    adata_view = adata[:, selected] if genes is not None else adata
    matrix = adata_view.X if layer is None or layer == "X" else adata_view.layers[layer]
    return pd.DataFrame(
        _dense_matrix(matrix),
        index=adata_view.obs_names.astype(str),
        columns=adata_view.var_names.astype(str),
    )


def _result_expression_frame(
    result_ad: ad.AnnData,
    *,
    layer: str | None = None,
    genes: set[str] | None = None,
) -> pd.DataFrame:
    result_var_names = result_ad.var_names.astype(str)
    selected = list(result_var_names) if genes is None else [gene for gene in result_var_names if gene in genes]
    if layer is None or layer == "X":
        matrix = result_ad[:, selected].X
    else:
        if layer not in result_ad.layers:
            raise ValueError(f"Requested result-space layer {layer!r} is not available.")
        matrix = result_ad[:, selected].layers[layer]
    return pd.DataFrame(
        _dense_matrix(matrix),
        index=result_ad.obs_names.astype(str),
        columns=selected,
    )


def _align_input_features_to_result(
    frame: pd.DataFrame,
    result_ad: ad.AnnData,
) -> pd.DataFrame:
    source = frame.copy()
    source.index = source.index.astype(str)
    target_labels = _result_alignment_labels(result_ad)
    aligned = source.reindex(target_labels, fill_value=0.0)
    aligned.index = result_ad.obs_names.astype(str)
    return aligned.astype(np.float32)


def _resolve_extension_iterations(result_ad: ad.AnnData) -> int:
    crane_info = dict(result_ad.uns.get("crane_info", {}))
    run_meta = dict(crane_info.get("run", {}))
    for key in ("iterations", "step2_iterations"):
        value = run_meta.get(key)
        if value is not None:
            return max(int(value), 0)
    return 0


def _adapt_extra_features(
    extra: pd.DataFrame,
    *,
    affinity: np.ndarray,
    iter_rounds: int,
    kernel_alpha: float = 0.1,
    overlap_ratio: float = 0.2,
) -> pd.DataFrame:
    if extra.empty:
        return extra.copy()

    centered = extra.astype(np.float32) - extra.astype(np.float32).mean(axis=0)
    if iter_rounds <= 0:
        return centered.astype(np.float32)

    exp_raw = centered.to_numpy(dtype=np.float32)
    exp_last = exp_raw.copy()
    for round_idx in range(1, iter_rounds + 1):
        exp_denoised = heat_kernel_smoothing(
            adj_matrix=affinity,
            property_matrix=exp_last,
            beta=kernel_alpha,
        ).astype(np.float32, copy=False)
        exp_last = overlay_raw_exp(
            exp_raw=exp_raw,
            exp_last=exp_last,
            f_exp_last=exp_denoised,
            iter_round=round_idx,
            weight=1 - overlap_ratio,
            mode="raw",
        ).astype(np.float32, copy=False)
    return pd.DataFrame(exp_last, index=extra.index.copy(), columns=extra.columns.copy())


def _build_extension_expression(
    adata: Any,
    result_ad: ad.AnnData,
    *,
    affinity: np.ndarray,
    layer: str | None = None,
    genes: set[str] | None = None,
) -> pd.DataFrame:
    core = _result_expression_frame(result_ad, genes=genes)
    if genes is None:
        return core.astype(np.float32)

    missing = [gene for gene in genes if gene not in core.columns]
    if not missing:
        return core.astype(np.float32)

    extra_input = _anndata_expression_frame(adata, layer=layer, genes=set(missing))
    aligned_extra = _align_input_features_to_result(extra_input, result_ad)
    adapted_extra = _adapt_extra_features(
        aligned_extra,
        affinity=affinity,
        iter_rounds=_resolve_extension_iterations(result_ad),
    )
    supplement_cols = [gene for gene in missing if gene in adapted_extra.columns]
    if supplement_cols:
        core = pd.concat([core, adapted_extra.loc[:, supplement_cols]], axis=1)
    return core.astype(np.float32)


def _coerce_cell_vectors(vectors: Any, result_ad: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(vectors, pd.Series):
        frame = vectors.to_frame(name=vectors.name or "extension_1")
    elif isinstance(vectors, pd.DataFrame):
        frame = vectors.copy()
    else:
        array = np.asarray(vectors)
        if array.ndim == 1:
            frame = pd.DataFrame({"extension_1": array}, index=result_ad.obs_names)
        elif array.ndim == 2:
            frame = pd.DataFrame(
                array,
                index=result_ad.obs_names,
                columns=[f"extension_{idx + 1}" for idx in range(array.shape[1])],
            )
        else:
            raise TypeError("cell_vector must be a Series, DataFrame, or 1D/2D array.")

    frame.index = frame.index.astype(str)
    target_labels = _result_alignment_labels(result_ad)
    missing = target_labels.difference(frame.index)
    if len(missing):
        raise ValueError("cell_vector must cover all result-space cells after original-cell alignment.")
    aligned = frame.reindex(target_labels)
    aligned.index = result_ad.obs_names.astype(str)

    value_blocks: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for column in aligned.columns:
        series = aligned[column]
        if pd.api.types.is_bool_dtype(series):
            name = str(column)
            value_blocks.append(pd.DataFrame({name: series.astype(np.int8)}, index=aligned.index))
            meta_rows.append(
                {
                    "feature": name,
                    "function_label": name,
                    "function_class": "cell_vector",
                    "parent": str(column),
                    "value_dtype": "bool",
                }
            )
        elif pd.api.types.is_numeric_dtype(series):
            name = str(column)
            value_blocks.append(pd.DataFrame({name: pd.to_numeric(series)}, index=aligned.index))
            meta_rows.append(
                {
                    "feature": name,
                    "function_label": name,
                    "function_class": "cell_vector",
                    "parent": str(column),
                    "value_dtype": "numeric",
                }
            )
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=str(column)).astype(np.int8)
            value_blocks.append(dummies)
            for dummy_name in dummies.columns:
                meta_rows.append(
                    {
                        "feature": str(dummy_name),
                        "function_label": str(dummy_name),
                        "function_class": "cell_vector",
                        "parent": str(column),
                        "value_dtype": "categorical",
                    }
                )

    if not value_blocks:
        raise ValueError("cell_vector did not produce any usable feature columns.")

    values = pd.concat(value_blocks, axis=1).astype(np.float32)
    metadata = pd.DataFrame(meta_rows).set_index("feature")
    return values, metadata


def _coerce_gene_set(gene_set: Any) -> dict[str, list[str]]:
    if isinstance(gene_set, Mapping):
        return {
            str(key): [str(gene) for gene in value]
            for key, value in gene_set.items()
        }
    if isinstance(gene_set, (list, tuple, set, pd.Index, pd.Series)):
        return {"gene_set_1": [str(gene) for gene in list(gene_set)]}
    raise TypeError("gene_set must be a mapping or a one-dimensional gene collection.")


def _summarize_vectors(
    vectors: pd.DataFrame,
    *,
    label: np.ndarray,
    affinity: np.ndarray,
) -> pd.DataFrame:
    values = vectors.to_numpy(dtype=np.float32)
    label_matrix = label.reshape(-1, 1).astype(np.float32)
    self_cor = np.diag(_compute_sp_moran_between(values, values, affinity.copy())).astype(np.float32)
    label_cor = _compute_sp_moran_between(values, label_matrix, affinity.copy()).reshape(-1).astype(np.float32)
    return pd.DataFrame(
        {
            "response_score": np.sqrt(np.square(self_cor) + np.square(label_cor)).astype(np.float32),
            "gene_self_cor": self_cor,
            "gene_label_cor": label_cor,
            "gene_direction": np.sign(label_cor).astype(np.int8),
        },
        index=vectors.columns.copy(),
    )


def _gene_set_to_cell_vectors(
    exp: pd.DataFrame,
    gene_set: Mapping[str, Any],
    *,
    min_genes_count: int = 10,
    loading_threshold: float = 0.5,
    embedding_threshold: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_embeddings: list[pd.DataFrame] = []
    gene_loadings = pd.DataFrame(0.0, index=exp.columns.copy(), columns=[])
    meta_frames: list[pd.DataFrame] = []

    for set_name, genes in gene_set.items():
        requested = [str(gene) for gene in genes]
        common = [gene for gene in requested if gene in exp.columns]
        if len(common) < min_genes_count:
            warnings.warn(
                f"Gene set {set_name!r} has {len(common)} common genes; returning a zero score.",
                stacklevel=2,
            )
            column = str(set_name)
            cell_embeddings.append(pd.DataFrame({column: np.zeros(exp.shape[0], dtype=np.float32)}, index=exp.index))
            meta_frames.append(
                pd.DataFrame(
                    {
                        "function_label": [column],
                        "function_class": ["gene_set"],
                        "parent": [str(set_name)],
                        "variance": [0.0],
                        "norm_variance": [0.0],
                        "overlap_ratio": [0.0],
                        "gene_call": [0],
                    },
                    index=[column],
                )
            )
            continue

        expr_subset = exp.loc[:, common]
        n_components = max(2, int(np.log(len(common)) / np.log(3)))
        n_components = min(n_components, expr_subset.shape[0], expr_subset.shape[1])
        pca = PCA(n_components=n_components, random_state=123)
        pca_result = pca.fit_transform(expr_subset)
        loading_squared = pca.components_ ** 2
        loading_norm = loading_squared / np.sum(loading_squared, axis=1, keepdims=True)
        valid_indices: list[int] = []
        for idx in range(n_components):
            if np.max(loading_norm[idx]) > loading_threshold:
                continue
            projection = pca_result[:, idx]
            above_median = np.sum(projection > np.median(projection)) / len(projection)
            if above_median > embedding_threshold:
                valid_indices.append(idx)
        if not valid_indices:
            valid_indices = [0]

        columns = [f"{set_name}_pc{idx + 1}" for idx in valid_indices]
        cell_embeddings.append(
            pd.DataFrame(pca_result[:, valid_indices].astype(np.float32), index=exp.index, columns=columns)
        )
        loading_block = pd.DataFrame(
            pca.components_[valid_indices].T.astype(np.float32),
            index=common,
            columns=columns,
        )
        gene_loadings = pd.concat([gene_loadings, pd.DataFrame(0.0, index=gene_loadings.index, columns=columns)], axis=1)
        gene_loadings.loc[common, columns] = loading_block
        variance = pca.explained_variance_ratio_[valid_indices]
        variance_sum = float(np.sum(variance)) or 1.0
        gene_call = [int(np.sum(loading_block[col].pow(2) > loading_block[col].pow(2).mean())) for col in columns]
        meta_frames.append(
            pd.DataFrame(
                {
                    "function_label": columns,
                    "function_class": ["gene_set"] * len(columns),
                    "parent": [str(set_name)] * len(columns),
                    "variance": variance.astype(np.float32),
                    "norm_variance": (variance / variance_sum).astype(np.float32),
                    "overlap_ratio": [count / len(common) for count in gene_call],
                    "gene_call": gene_call,
                },
                index=columns,
            )
        )

    if not cell_embeddings:
        return pd.DataFrame(index=exp.index), pd.DataFrame(), pd.DataFrame()
    return pd.concat(cell_embeddings, axis=1), pd.concat(meta_frames, axis=0), gene_loadings.fillna(0.0)


def _gene_vector_to_cell_vectors(
    exp: pd.DataFrame,
    gene_vector: pd.DataFrame | pd.Series,
    *,
    min_genes_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vector_df = (
        gene_vector.to_frame(name=gene_vector.name or "gene_vector_1")
        if isinstance(gene_vector, pd.Series)
        else gene_vector.copy()
    )
    common = [gene for gene in vector_df.index.astype(str) if gene in exp.columns]
    if len(common) < min_genes_count:
        warnings.warn(
            f"gene_vector has {len(common)} common genes; returning zero scores.",
            stacklevel=2,
        )
        cell_vectors = pd.DataFrame(0.0, index=exp.index, columns=vector_df.columns)
    else:
        exp_sub = exp.loc[:, common].T
        mean = exp_sub.mean(axis=0)
        std = exp_sub.std(axis=0, ddof=0).replace(0, 1)
        exp_sub = (exp_sub - mean) / std
        weights = vector_df.loc[common, :].fillna(0).astype(np.float32)
        cell_vectors = pd.DataFrame(
            np.dot(exp_sub.values.T, weights.values).astype(np.float32),
            index=exp.index,
            columns=weights.columns,
        )
    metadata = pd.DataFrame(
        {
            "function_label": cell_vectors.columns.astype(str),
            "function_class": "gene_vector",
            "parent": cell_vectors.columns.astype(str),
            "common_gene_count": len(common),
        },
        index=cell_vectors.columns,
    )
    return cell_vectors, metadata


def evaluate_extension(
    adata: Any,
    *,
    result: Any,
    gene_set: Mapping[str, Any] | None = None,
    gene_vector: pd.DataFrame | pd.Series | None = None,
    cell_vector: Any = None,
    logger: CRANELogger,
    layer: str | None = None,
    label_key: str | None = None,
    affinity_key: str = "affinity",
    set_min_genes_count: int = 10,
    set_loading_threshold: float = 0.5,
    set_embedding_threshold: float = 0.1,
    vector_min_genes_count: int = 50,
    mode: str = "extension_response",
    metadata: Mapping[str, Any] | None = None,
) -> CRANEExtensionResult:
    """Evaluate additional gene-set/gene-vector/cell-vector inputs on a CRANE result graph."""

    supplied = [gene_set is not None, gene_vector is not None, cell_vector is not None]
    if sum(supplied) != 1:
        raise ValueError("extension_response requires exactly one of gene_set, gene_vector, or cell_vector.")

    result_ad = _as_result_ad(result)
    affinity = _resolve_affinity(result_ad, affinity_key=affinity_key)
    resolved_label_key, label = _resolve_label(result_ad, label_key=label_key)
    logger.event(
        f"{mode}.entry",
        f"CRANE {mode} entry initialized.",
        audience="reviewer",
    )

    gene_loadings = None
    if gene_set is not None:
        normalized_gene_set = _coerce_gene_set(gene_set)
        potential_genes = {str(gene) for genes in normalized_gene_set.values() for gene in genes}
        expression_frame = _build_extension_expression(
            adata,
            result_ad,
            affinity=affinity,
            layer=layer,
            genes=potential_genes,
        )
        vectors, feature_metadata, gene_loadings = _gene_set_to_cell_vectors(
            expression_frame,
            normalized_gene_set,
            min_genes_count=set_min_genes_count,
            loading_threshold=set_loading_threshold,
            embedding_threshold=set_embedding_threshold,
        )
        feature_class = "gene_set"
    elif gene_vector is not None:
        potential_genes = set(map(str, gene_vector.index))
        expression_frame = _build_extension_expression(
            adata,
            result_ad,
            affinity=affinity,
            layer=layer,
            genes=potential_genes,
        )
        vectors, feature_metadata = _gene_vector_to_cell_vectors(
            expression_frame,
            gene_vector,
            min_genes_count=vector_min_genes_count,
        )
        feature_class = "gene_vector"
    else:
        vectors, feature_metadata = _coerce_cell_vectors(cell_vector, result_ad)
        feature_class = "cell_vector"

    summary_var = _summarize_vectors(vectors, label=label, affinity=affinity)
    summary_var["gene_important"] = (summary_var["response_score"] > 0).astype(np.int8)
    summary_var = pd.concat([summary_var, feature_metadata.reindex(summary_var.index)], axis=1)
    extension_ad = ad.AnnData(
        X=vectors.to_numpy(dtype=np.float32),
        obs=result_ad.obs.copy(),
        var=summary_var,
    )
    extension_ad.obsp[affinity_key] = sparse.csr_matrix(affinity.astype(np.float32, copy=False))
    extension_ad.uns["crane_info"] = {
        "kind": "crane_extension_result",
        "result_space": mode,
        "function_class": feature_class,
        "source_result_space": dict(result_ad.uns.get("crane_info", {})),
        "label_key": resolved_label_key,
        "affinity_key": affinity_key,
        "expression_layer": layer or "X",
        "core_expression_source": "result.result_ad.X",
        "supplement_missing_genes_from_adata": bool(adata is not None and feature_class in {"gene_set", "gene_vector"}),
        "run": dict(metadata or {}),
    }
    if gene_loadings is not None:
        extension_ad.uns["gene_X_set_loading"] = gene_loadings

    logger.event(
        f"{mode}.complete",
        f"CRANE {mode} completed.",
        audience="reviewer",
        n_features=int(extension_ad.n_vars),
        function_class=feature_class,
    )
    return CRANEExtensionResult(
        extension_ad=extension_ad,
        source_result_ad=result_ad,
        metadata={"mode": mode, "function_class": feature_class, **dict(metadata or {})},
        logger_config=logger.config,
    )


def evaluate_function(
    adata: Any,
    *,
    result: Any,
    gene_set: Mapping[str, Any],
    logger: CRANELogger,
    layer: str | None = None,
    label_key: str | None = None,
    affinity_key: str = "affinity",
    set_min_genes_count: int = 10,
    set_loading_threshold: float = 0.5,
    set_embedding_threshold: float = 0.1,
    metadata: Mapping[str, Any] | None = None,
) -> CRANEExtensionResult:
    """Evaluate functional gene sets through the unified extension evaluator."""

    return evaluate_extension(
        adata,
        result=result,
        gene_set=gene_set,
        logger=logger,
        layer=layer,
        label_key=label_key,
        affinity_key=affinity_key,
        set_min_genes_count=set_min_genes_count,
        set_loading_threshold=set_loading_threshold,
        set_embedding_threshold=set_embedding_threshold,
        mode="function_response",
        metadata=metadata,
    )
