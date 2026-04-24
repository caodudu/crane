"""Default Step 1 feature selection and tendency evaluation for CRANE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .feature_screen import (
    KSFeatureScreenOptions,
    KSFeatureScreenResult,
    screen_ks_features,
    select_top_by_stable_raw_pvalues,
)
from .preprocess import PreparedInput
from .sampling import PerturbationTendency, SamplingOptions, SamplingPlan, build_sampling_plan


@dataclass(frozen=True)
class Step1Options:
    """Internal Step 1 controls kept low-visibility in the public API."""

    screen_layer: str = "log1p_norm"
    n_top: int | None = None
    n_bottom: int | None = None
    extra_genes_dict: Mapping[str, Sequence[str]] | None = None
    init_ratio: float | None = None
    init_ratio_case_cap: int | None = 320
    init_ratio_case_cap_min_cells: int = 640
    init_ratio_case_cap_random_state: int = 20260420
    fix_point_factor: float = 0.3
    weak_ratio: float = 0.03
    n_neighbors: int = 20
    graph_method: str = "umap"
    n_bg_permutations: int = 0
    compute_tendency_diagnostics: bool = False
    feature_screen_options: KSFeatureScreenOptions = field(default_factory=KSFeatureScreenOptions)


@dataclass(frozen=True)
class Step1FeatureSelectionResult:
    """Step 1 feature-selection outputs passed to the next stage."""

    default_fs: pd.DataFrame
    working_adata: Any
    aux_fs: pd.Series
    margin_eval: pd.DataFrame
    screen_result: KSFeatureScreenResult
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step1Result:
    """Complete Step 1 output before Step 2 closed-loop refinement."""

    prepared_input: PreparedInput
    feature_selection: Step1FeatureSelectionResult
    tendency: PerturbationTendency
    sampling_plan: SamplingPlan
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _compute_variability(working_adata: ad.AnnData) -> pd.Series:
    import scanpy as sc

    matrix = np.asarray(working_adata.X, dtype=np.float32)
    neg_value = np.sum(matrix < 0) / matrix.shape[0] / matrix.shape[1]
    if neg_value < 0.01:
        hvg_adata = ad.AnnData(matrix.copy(), obs=working_adata.obs.copy(), var=working_adata.var.copy())
        sc.pp.highly_variable_genes(hvg_adata, subset=False)
        variability = hvg_adata.var["dispersions_norm"].copy()
        variability.index = working_adata.var_names.copy()
        return variability

    median_values = np.median(matrix, axis=0)
    absolute_deviation = np.abs(matrix - median_values)
    return pd.Series(np.median(absolute_deviation, axis=0), index=working_adata.var_names.copy())


def _find_stable_curve_index(y: np.ndarray, window_size: int = 2, relative_epsilon_factor: float = 0.3) -> int:
    df = pd.DataFrame({"y": y})
    df["y"] = df["y"].replace({None: np.nan}).bfill()
    df["y_smooth"] = df["y"].rolling(window=window_size, center=True).mean()
    df["slope"] = df["y_smooth"].diff()

    total_slope = abs((y[-1] - y[0]) / (len(y) - 1))
    relative_epsilon = total_slope * relative_epsilon_factor
    found_index = None
    for i in range(len(df)):
        if abs(df["slope"].iloc[i]) < relative_epsilon:
            found_index = i
            break
    if found_index is None:
        min_slope = abs(df["slope"]).min()
        adjusted_epsilon = abs(min_slope) + relative_epsilon
        for i in range(len(df)):
            if abs(df["slope"].iloc[i]) < adjusted_epsilon:
                found_index = i
                break
    return int(found_index if found_index is not None else len(df) - 1)


def _build_margin_table(screen_result: KSFeatureScreenResult, feature_names: np.ndarray) -> pd.DataFrame:
    rank = np.empty_like(screen_result.ranking)
    rank[screen_result.ranking] = np.arange(screen_result.ranking.shape[0], dtype=int)
    return pd.DataFrame(
        {
            "Feature": feature_names,
            "rank": rank,
            "statistic": screen_result.statistic,
            "p-value": screen_result.window_raw_pvalues,
            "adj_p-value": screen_result.adj_pvalues,
        }
    )


def _coerce_extra_gene_set(
    extra_genes_dict: Mapping[str, Sequence[str]] | None,
    feature_index: pd.Index,
) -> set[str] | None:
    if extra_genes_dict is None:
        return None
    extra_genes_set: set[str] = set()
    for genes in extra_genes_dict.values():
        extra_genes_set.update(str(gene) for gene in genes)
    return extra_genes_set & set(feature_index)


def _build_lightweight_adata(
    exp_matrix: np.ndarray,
    obs_frame: pd.DataFrame,
    var_frame: pd.DataFrame,
) -> ad.AnnData:
    adata_obj = ad.AnnData(np.asarray(exp_matrix, dtype=np.float32, order="C"))
    adata_obj.obs = obs_frame.copy()
    adata_obj.var = var_frame.copy()
    adata_obj.uns["crane_setting"] = {"preprocess": "baseline"}
    return adata_obj


def _build_feature_subset_adata(source_adata: ad.AnnData, feature_names: Sequence[str]) -> ad.AnnData:
    subset = source_adata[:, list(feature_names)].copy()
    subset.X = np.asarray(subset.X, dtype=np.float32)
    if "log1p_norm" in subset.layers:
        subset.layers["log1p_norm"] = np.asarray(subset.layers["log1p_norm"], dtype=np.float32)
    subset.uns["crane_setting"] = {"preprocess": "baseline"}
    return subset


def _ranking_subset(ranked_features: list[str], keep_n: int) -> list[str]:
    if keep_n <= 0:
        return []
    return ranked_features[: min(keep_n, len(ranked_features))]


def _compute_init_ratio_distance_curve(
    case_expression: np.ndarray,
    init_ratio_set: Sequence[float],
) -> list[tuple[float, float]]:
    """Compute the init-ratio distance curve without repeated prefixes.

    This compares full-feature and prefix-feature normalized Euclidean
    distance matrices. Squared Euclidean distances are additive over feature
    blocks, so prefix distances can be accumulated exactly in condensed form.
    """

    case_expression = np.asarray(case_expression, dtype=np.float32)
    if case_expression.ndim != 2:
        raise ValueError("case_expression must be a 2D array.")
    n_features = case_expression.shape[1]
    if n_features == 0:
        return [(float(ratio), float("nan")) for ratio in init_ratio_set]

    full_dist = np.sqrt(pdist(case_expression, metric="sqeuclidean") / float(n_features))
    prefix_sq_dist = np.zeros_like(full_dist)
    previous_keep_n = 0
    curve: list[tuple[float, float]] = []

    for init_ratio in init_ratio_set:
        keep_n = min(int(init_ratio * n_features), n_features)
        if keep_n > previous_keep_n:
            block = case_expression[:, previous_keep_n:keep_n]
            prefix_sq_dist += pdist(block, metric="sqeuclidean")
            previous_keep_n = keep_n
        if keep_n == 0:
            norm_value = float("nan")
        else:
            prefix_dist = np.sqrt(prefix_sq_dist / float(keep_n))
            diff = full_dist - prefix_dist
            norm_value = float(np.sqrt(2.0 * np.dot(diff, diff)))
        curve.append((float(init_ratio), norm_value))
    return curve


def _cap_init_ratio_case_expression(
    case_expression: np.ndarray,
    options: Step1Options,
) -> tuple[np.ndarray, dict[str, Any]]:
    case_expression = np.asarray(case_expression, dtype=np.float32)
    n_case = int(case_expression.shape[0])
    cap = options.init_ratio_case_cap
    cap_metadata: dict[str, Any] = {
        "enabled": False,
        "case_cells_total": n_case,
        "case_cells_used": n_case,
        "cap": None if cap is None else int(cap),
        "min_case_cells": int(options.init_ratio_case_cap_min_cells),
        "random_state": int(options.init_ratio_case_cap_random_state),
    }
    if cap is None or cap <= 0:
        return case_expression, cap_metadata
    if n_case <= options.init_ratio_case_cap_min_cells or n_case <= cap:
        return case_expression, cap_metadata

    rng = np.random.default_rng(options.init_ratio_case_cap_random_state)
    row_indices = np.sort(rng.choice(np.arange(n_case), size=int(cap), replace=False))
    cap_metadata["enabled"] = True
    cap_metadata["case_cells_used"] = int(cap)
    return case_expression[row_indices, :], cap_metadata


def run_feature_selection(
    prepared_input: PreparedInput,
    options: Step1Options | None = None,
) -> Step1FeatureSelectionResult:
    options = options or Step1Options()
    step1_adata = prepared_input.step1_adata
    screen_matrix = np.asarray(step1_adata.layers[options.screen_layer], dtype=np.float32)
    feature_names = np.asarray(step1_adata.var_names, dtype=object)
    case_expr = screen_matrix[prepared_input.case_mask, :]
    control_expr = screen_matrix[prepared_input.control_mask, :]

    screen_result = screen_ks_features(
        case_expr=case_expr,
        control_expr=control_expr,
        feature_names=feature_names,
        options=options.feature_screen_options,
    )
    margin_table = _build_margin_table(screen_result=screen_result, feature_names=feature_names)
    margin_table.index = margin_table["Feature"]

    ranked_features = feature_names[screen_result.ranking].tolist()
    coarse_features = feature_names[screen_result.coarse_mask].tolist()
    core_features = feature_names[screen_result.core_mask].tolist()
    aux_features = feature_names[screen_result.window_mask].tolist()

    full_feature_frame = pd.DataFrame({"gene": feature_names}, index=feature_names)
    full_feature_frame["i_0"] = full_feature_frame.index.isin(coarse_features).astype(np.int8)
    variability = _compute_variability(working_adata=step1_adata)
    extra_genes_set = _coerce_extra_gene_set(options.extra_genes_dict, full_feature_frame.index)

    n_top = 2700 if options.n_top is None else options.n_top
    n_bottom = 500 if options.n_bottom is None else options.n_bottom
    n_coarse = len(coarse_features)
    strong_selection_metadata: dict[str, Any] = {"strategy": "all_features"}

    if len(step1_adata.var_names) > (n_top + n_bottom):
        top_fs = _ranking_subset(ranked_features, n_top)
        if n_top < n_coarse:
            bottom_fs = list(variability[full_feature_frame["i_0"] == 0].nlargest(n_bottom).index)
            strong_features = top_fs + bottom_fs
            strong_selection_metadata = {
                "strategy": "coarse_plus_variability",
                "evaluated_raw_pvalue_count": 0,
            }
        else:
            strong_gene_indices, strong_selection_metadata = select_top_by_stable_raw_pvalues(
                case_expr=case_expr,
                control_expr=control_expr,
                ranking=screen_result.ranking,
                keep_n=n_top + n_bottom,
                method=options.feature_screen_options.pvalue_method,
            )
            strong_features = feature_names[strong_gene_indices].tolist()
            strong_selection_metadata["strategy"] = "stable_raw_pvalue_boundary"

        if extra_genes_set is not None:
            extra_features = extra_genes_set - set(strong_features)
            if len(extra_features) <= 300:
                strong_features = strong_features + list(extra_features)
            else:
                extra_features = extra_genes_set - set(top_fs)
                if len(extra_features) <= (n_bottom + 300):
                    strong_features = top_fs + list(extra_features)
                else:
                    rng = np.random.default_rng(0)
                    sampled = rng.choice(sorted(extra_features), size=n_bottom + 300, replace=False)
                    strong_features = top_fs + list(sampled)
        adata_strong = _build_feature_subset_adata(step1_adata, strong_features)
    else:
        strong_features = ranked_features.copy()
        adata_strong = _build_feature_subset_adata(step1_adata, strong_features)

    strong_ranking = [gene for gene in ranked_features if gene in set(strong_features)]
    strong_margin = margin_table.loc[strong_features, :]
    coarse_features_in_strong_set = [gene for gene in strong_ranking if gene in coarse_features]
    core_features_in_strong_set = [gene for gene in strong_ranking if gene in core_features]
    init_ratio_metadata: dict[str, Any] = {"mode": "fixed" if options.init_ratio is not None else "auto"}

    if options.init_ratio is not None:
        if options.init_ratio < 0.7:
            init_ratio_features = _ranking_subset(strong_ranking, int(options.init_ratio * len(strong_ranking)))
        else:
            init_ratio_features = strong_margin.loc[strong_margin["p-value"] < 0.7, :].index.tolist()
    else:
        init_ratio_set = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        case_rows = adata_strong.obs[prepared_input.contract.perturbation_key] == prepared_input.case_value
        ranking_cols = adata_strong.var_names.get_indexer(strong_ranking)
        ranked_case_expression = np.asarray(adata_strong.layers[options.screen_layer], dtype=np.float32)[case_rows.values, :][
            :, ranking_cols
        ]
        capped_case_expression, cap_metadata = _cap_init_ratio_case_expression(ranked_case_expression, options=options)
        init_ratio_metadata["case_cap"] = cap_metadata
        init_ratio_results = _compute_init_ratio_distance_curve(
            case_expression=capped_case_expression,
            init_ratio_set=init_ratio_set,
        )
        init_ratio_results_df = pd.DataFrame(init_ratio_results, columns=["init_ratio", "avg_norm_all_vs_0"])
        fix_index = _find_stable_curve_index(
            init_ratio_results_df["avg_norm_all_vs_0"].values,
            window_size=3,
            relative_epsilon_factor=options.fix_point_factor,
        )
        fix_init_ratio = float(init_ratio_results_df["init_ratio"].iloc[fix_index])
        init_ratio_metadata["fix_index"] = int(fix_index)
        init_ratio_metadata["fix_init_ratio"] = fix_init_ratio
        init_ratio_features = _ranking_subset(strong_ranking, int(fix_init_ratio * len(strong_ranking)))

    if (len(core_features_in_strong_set) <= int(options.weak_ratio * len(adata_strong.var_names))) and (
        len(adata_strong.var_names) > 1e3
    ):
        retained_features = coarse_features_in_strong_set
    else:
        retained_features = (
            init_ratio_features if len(init_ratio_features) > len(coarse_features_in_strong_set) else coarse_features_in_strong_set
        )
    if not retained_features:
        raise ValueError("Step 1 did not retain any features for downstream refinement.")

    gene_input_strong = adata_strong.var_names
    default_fs = pd.DataFrame({"gene": gene_input_strong}, index=gene_input_strong)
    default_fs["i_0"] = default_fs.index.isin(retained_features).astype(np.int8)
    aux_fs = pd.Series(default_fs.index.isin(aux_features).astype(np.int8), index=gene_input_strong, name="i_0")

    metadata = {
        "coarse_deg_count": len(coarse_features),
        "guide_deg_count": len(aux_features),
        "strong_feature_count": int(len(strong_features)),
        "reserve_deg_count": int(len(retained_features)),
        "raw_pvalue_eval_count": int(screen_result.evaluated_raw_pvalue_count),
        "adj_pvalue_eval_count": int(screen_result.evaluated_adj_pvalue_count),
        "strong_fs_selection": strong_selection_metadata,
        "init_ratio": init_ratio_metadata,
    }
    return Step1FeatureSelectionResult(
        default_fs=default_fs,
        working_adata=adata_strong,
        aux_fs=aux_fs,
        margin_eval=margin_table.reset_index(drop=True),
        screen_result=screen_result,
        metadata=metadata,
    )


def _compute_sp_moran_between(exp_df1: np.ndarray, exp_df2: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
    if adj_matrix[0, 0] != 0:
        np.fill_diagonal(adj_matrix, 0)
    n = exp_df1.shape[0]
    constant = n / np.sum(adj_matrix)
    std_exp_df1 = exp_df1 - exp_df1.mean(axis=0)
    std_exp_df2 = exp_df2 - exp_df2.mean(axis=0)
    l2_norms1 = np.sqrt(np.sum(std_exp_df1 ** 2, axis=0))
    l2_norms2 = np.sqrt(np.sum(std_exp_df2 ** 2, axis=0))
    weighted_std_exp_df = std_exp_df1.T @ adj_matrix @ std_exp_df2
    result_matrix = constant * (weighted_std_exp_df / np.outer(l2_norms1, l2_norms2))
    result_matrix = np.nan_to_num(result_matrix)
    result_matrix[result_matrix > 1] = 1
    result_matrix[result_matrix < -1] = -1
    return result_matrix


def _step1_neighbors_kwargs(graph_method: str) -> dict[str, Any]:
    graph_method = str(graph_method).strip().lower()
    if graph_method not in {"umap", "gauss"}:
        raise ValueError("Step1 graph_method must be 'umap' or 'gauss'.")
    return {"method": graph_method}


def _compute_sp_moran_between_col(exp1: np.ndarray, exp2: np.ndarray, adj_matrix: np.ndarray) -> float:
    if adj_matrix[0, 0] != 0:
        np.fill_diagonal(adj_matrix, 0)
    n = len(exp1)
    constant = n / np.sum(adj_matrix)
    std_exp1 = exp1 - np.mean(exp1)
    std_exp2 = exp2 - np.mean(exp2)
    weighted_std_exp = std_exp1.T @ adj_matrix @ std_exp2
    return float(constant * (weighted_std_exp / (np.sqrt(np.sum(std_exp1 ** 2)) * np.sqrt(np.sum(std_exp2 ** 2)))))


def _compute_neighbourhood_label_proportion(adj_mat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    neighbourhood_label_sum = adj_mat @ labels
    neighbourhood_size = adj_mat.sum(axis=1)
    return neighbourhood_label_sum / neighbourhood_size


def _detect_elbow(
    iterations: Sequence[int],
    fitness_values: Sequence[float],
    threshold: float = 0.01,
    consecutive: int = 4,
) -> int:
    deltas = pd.Series(fitness_values).diff().abs()
    count = 0
    for idx in range(1, len(deltas)):
        if deltas.iloc[idx] < threshold:
            count += 1
            if count >= consecutive:
                return int(iterations[idx])
        else:
            count = 0
    return -1


def _evaluate_init_moran(adj_mat: Any, cell_ident: pd.Series) -> tuple[float, float]:
    if hasattr(adj_mat, "toarray"):
        adj_dense = adj_mat.toarray()
    else:
        adj_dense = np.asarray(adj_mat, dtype=np.float32)
    adj_dense = np.asarray(adj_dense, dtype=np.float32)
    if adj_dense.shape[0] != adj_dense.shape[1]:
        raise ValueError("adj_mat must be a square adjacency matrix.")
    values = np.asarray(cell_ident.values, dtype=np.float32).reshape(-1)
    if adj_dense.shape[0] != values.shape[0]:
        raise ValueError("cell_ident length must match adj_mat size.")
    moran_i = _compute_sp_moran_between_col(values, values, adj_dense)
    return round(float(moran_i), 6), 1.0


def _evaluate_init_score(
    adata: ad.AnnData,
    layer: str,
    group_obs: str,
    control_label: Any,
    case_label: Any,
    fs_slice_ref: pd.Series | None,
    ident_method: str,
    voting: bool,
    nn_k: int,
    graph_method: str = "umap",
) -> pd.Series:
    import scanpy as sc
    from scipy.sparse import issparse as sparse_issparse
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics.pairwise import cosine_similarity

    obs_frame = adata.obs.copy()
    bool_ca = obs_frame[group_obs].astype(str) == str(case_label)
    bool_co = obs_frame[group_obs].astype(str) == str(control_label)
    num_ca = int(bool_ca.sum())
    num_co = int(bool_co.sum())
    cells_bc_raw = adata.obs_names.copy()
    selected_cells = cells_bc_raw

    if num_co > num_ca * 3:
        num_co_down = num_ca * 3 if num_ca >= 100 else min(num_co, 200)
        co_select = np.random.choice(cells_bc_raw[bool_co.values], num_co_down, replace=False)
        ca_select = cells_bc_raw[bool_ca.values]
        selected_cells = pd.Index(list(ca_select) + list(co_select))
        obs_frame = obs_frame.loc[selected_cells].copy()

    base_matrix = adata.X if layer == "X" else adata.layers[layer]
    if sparse_issparse(base_matrix):
        base_matrix = base_matrix.toarray()
    else:
        base_matrix = np.asarray(base_matrix, dtype=np.float32)
    row_ids = adata.obs_names.get_indexer(selected_cells)
    base_matrix = np.asarray(base_matrix[row_ids, :], dtype=np.float32)

    if fs_slice_ref is not None:
        fs_mask = fs_slice_ref.reindex(adata.var_names, fill_value=0).to_numpy() == 1
        base_matrix = base_matrix[:, fs_mask]
        var_frame = adata.var.loc[fs_mask].copy()
    else:
        var_frame = adata.var.copy()

    adata_ = _build_lightweight_adata(base_matrix, obs_frame=obs_frame, var_frame=var_frame)
    label_raw = (obs_frame[group_obs].astype(str) != str(control_label)).astype(int)

    exp_mat = np.asarray(base_matrix, dtype=np.float32)
    nn_k_select = min(nn_k, num_ca)
    try:
        sc.tl.pca(adata_, svd_solver="arpack")
        sc.pp.neighbors(
            adata_,
            n_neighbors=nn_k_select,
            metric="cosine",
            n_pcs=50,
            **_step1_neighbors_kwargs(graph_method),
        )
    except Exception:
        adata_.obsm["X_pca"] = exp_mat
        sc.pp.neighbors(
            adata_,
            n_neighbors=nn_k_select,
            metric="cosine",
            **_step1_neighbors_kwargs(graph_method),
        )

    adj_mat = adata_.obsp["connectivities"].copy()
    adj_mat.setdiag(0)
    if sparse_issparse(adj_mat):
        adj_mat = adj_mat.toarray()

    co_indices = np.where(label_raw.values.flatten() == 0)[0]
    ca_indices = np.where(label_raw.values.flatten() == 1)[0]
    if ident_method == "mean_delta":
        ca_similarities = cosine_similarity(exp_mat, exp_mat[ca_indices])
        co_similarities = cosine_similarity(exp_mat, exp_mat[co_indices])
        avg_co_similarity = np.mean(co_similarities, axis=1)
    else:
        cell_similarity = cosine_similarity(exp_mat, exp_mat)

    label_last_dict = {0: label_raw.values}
    label_binary_dict = {0: label_raw.values}
    label_last_cor_list: list[float] = []
    label_last_kappa_list: list[float] = []

    for i in range(1, 6):
        label_last = label_last_dict[i - 1].copy()
        ca_weights = label_last.flatten() * label_last_dict[0].flatten()
        if ident_method == "mean_delta":
            try:
                avg_ca_similarity = np.average(ca_similarities, axis=1, weights=ca_weights[ca_indices])
            except Exception:
                avg_ca_similarity = np.mean(ca_similarities, axis=1)
            similarity_diff = avg_ca_similarity - avg_co_similarity
            bool_ca_iter = (similarity_diff > 0).astype(np.float32)
            bool_ca_inter = bool_ca_iter * label_binary_dict[0]
            crane_score_i = _compute_neighbourhood_label_proportion(adj_mat, bool_ca_inter)
        else:
            cell_with_ca_moran = _compute_sp_moran_between(cell_similarity, ca_weights[:, np.newaxis], adj_mat).flatten()
            crane_score_i = np.clip(cell_with_ca_moran, -1, 1)
            co_bc_score = np.mean(cell_with_ca_moran[label_last_dict[0] == 0]) + np.std(
                cell_with_ca_moran[label_last_dict[0] == 0]
            )
            bool_ca_iter = (cell_with_ca_moran.flatten() > co_bc_score).astype(np.float32)
            bool_ca_inter = bool_ca_iter * label_binary_dict[0]
            if voting:
                crane_score_i = _compute_neighbourhood_label_proportion(adj_mat, bool_ca_inter)

        crane_score_i_norm = crane_score_i / max(float(np.max(crane_score_i)), 0.1)
        label_last_dict[i] = crane_score_i_norm.copy()
        label_binary_dict[i] = bool_ca_inter.copy()

        label_last_cor = min(_compute_sp_moran_between_col(crane_score_i_norm, label_last, adj_mat), 1.0)
        label_last_cor_list.append(label_last_cor)
        label_binary_kappa = min(float(cohen_kappa_score(label_binary_dict[i], label_binary_dict[i - 1])), 1.0)
        label_last_kappa_list.append(label_binary_kappa)
        iter_elbow = _detect_elbow(
            iterations=list(range(1, len(label_last_cor_list) + 1)),
            fitness_values=label_last_cor_list,
        )
        if label_last_cor >= 0.90 or label_binary_kappa >= 0.95 or iter_elbow != -1 or i >= 5:
            label_output_clean = label_last_dict[i]
            break

    label_output_clean = pd.Series(np.asarray(label_output_clean, dtype=np.float32), index=label_raw.index)
    crane_score = pd.Series(np.float32(0.0), index=cells_bc_raw, dtype=np.float32)
    crane_score.update(label_output_clean)
    return crane_score


def run_tendency_evaluation(
    prepared_input: PreparedInput,
    feature_selection: Step1FeatureSelectionResult,
    options: Step1Options | None = None,
) -> PerturbationTendency:
    import scanpy as sc

    options = options or Step1Options()
    adata_strong = feature_selection.working_adata
    eval_mask = feature_selection.default_fs["i_0"].to_numpy() == 1
    eval_matrix = np.asarray(adata_strong.X[:, eval_mask], dtype=np.float32)
    adata_eval = _build_lightweight_adata(
        eval_matrix,
        obs_frame=adata_strong.obs,
        var_frame=adata_strong.var.loc[eval_mask],
    )
    n_neighbors = min(options.n_neighbors, len(adata_eval.obs_names))
    try:
        sc.tl.pca(adata_eval, svd_solver="arpack")
        sc.pp.neighbors(
            adata_eval,
            n_neighbors=n_neighbors,
            metric="cosine",
            n_pcs=50,
            **_step1_neighbors_kwargs(options.graph_method),
        )
    except Exception:
        sc.pp.neighbors(
            adata_eval,
            n_neighbors=n_neighbors,
            metric="cosine",
            use_rep="X",
            **_step1_neighbors_kwargs(options.graph_method),
        )
    connectivities = adata_eval.obsp["connectivities"]
    label_raw = adata_eval.obs[prepared_input.contract.perturbation_key].map(
        {prepared_input.contract.control_value: 0, prepared_input.case_value: 1}
    )
    moran_i_raw, moran_p_raw = _evaluate_init_moran(connectivities, label_raw)
    ident_method = "moran" if moran_i_raw > 0.2 else "mean_delta"
    voting = moran_i_raw <= 0.2
    init_adj_score = _evaluate_init_score(
        adata=adata_strong,
        layer="X",
        group_obs=prepared_input.contract.perturbation_key,
        control_label=prepared_input.contract.control_value,
        case_label=prepared_input.case_value,
        fs_slice_ref=feature_selection.aux_fs,
        ident_method=ident_method,
        voting=voting,
        nn_k=options.n_neighbors,
        graph_method=options.graph_method,
    )
    metadata = {
        "moran_I": moran_i_raw,
        "p_value": moran_p_raw,
        "ident_method": ident_method,
        "voting": voting,
        "graph_method": str(options.graph_method).strip().lower(),
    }
    if options.compute_tendency_diagnostics:
        moran_i_adj, moran_p_adj = _evaluate_init_moran(connectivities, init_adj_score.loc[adata_eval.obs_names])
        metadata["moran_adj_I"] = moran_i_adj
        metadata["p_adj_value"] = moran_p_adj
    return PerturbationTendency(
        values=init_adj_score.copy(),
        source="graph_neighbor_proportion",
        clipped=False,
        metadata=metadata,
    )


def run_step1(
    prepared_input: PreparedInput,
    step1_options: Step1Options | None = None,
    sampling_options: SamplingOptions | None = None,
) -> Step1Result:
    step1_options = step1_options or Step1Options()
    sampling_options = sampling_options or SamplingOptions()
    feature_selection = run_feature_selection(prepared_input=prepared_input, options=step1_options)
    tendency = run_tendency_evaluation(
        prepared_input=prepared_input,
        feature_selection=feature_selection,
        options=step1_options,
    )
    sampling_plan = build_sampling_plan(
        prepared_input=prepared_input,
        tendency=tendency,
        options=sampling_options,
        working_adata=feature_selection.working_adata,
        init_feature_selection=feature_selection.default_fs["i_0"].copy(),
        aux_feature_selection=feature_selection.aux_fs.copy(),
        margin_evaluation=feature_selection.margin_eval.copy(),
    )
    metadata = {
        "strong_feature_count": int(feature_selection.working_adata.n_vars),
        "sample_count": int(len(sampling_plan.control_case_samples)),
    }
    return Step1Result(
        prepared_input=prepared_input,
        feature_selection=feature_selection,
        tendency=tendency,
        sampling_plan=sampling_plan,
        metadata=metadata,
    )
