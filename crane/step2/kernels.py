"""Numerical kernels for the default Step 2 path.

Heavy optional dependencies are imported inside the functions that need them so
importing the Step 2 package itself stays light.
"""

from __future__ import annotations

import numpy as np

from .contracts import Step2Options, Step2SampleOutput, Step2SamplePack


def ensure_float_type(data: np.ndarray, target_dtype: str = "float32") -> np.ndarray:
    """Coerce arrays to a target floating dtype when needed."""

    if target_dtype not in {"float16", "float32", "float64"}:
        raise ValueError("target_dtype must be 'float16', 'float32', or 'float64'")
    if not np.issubdtype(data.dtype, np.floating) or str(data.dtype) != target_dtype:
        return data.astype(target_dtype)
    return data


def pca_select(mat: np.ndarray, n_pcs: int = 50) -> np.ndarray:
    """Run PCA with scanpy when enough dimensions are available."""

    if n_pcs == -1 or min(mat.shape) < n_pcs:
        return mat
    from anndata import AnnData
    from scanpy.preprocessing._pca import pca

    adata = AnnData(mat)
    pca(adata, n_comps=n_pcs)
    return adata.obsm["X_pca"]


def compute_distance_cosine(mat: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute cosine distance, optionally divided by 2."""

    from scipy.spatial.distance import pdist, squareform

    mat = ensure_float_type(np.asarray(mat), target_dtype="float32")
    distances = squareform(pdist(mat, metric="cosine"))
    if normalized:
        distances = distances / 2
    return ensure_float_type(distances, target_dtype="float32")


def adaptive_knn(distance_matrix: np.ndarray, n_neighbors: int = 10, delta: float = -0.5) -> np.ndarray:
    """Build the adaptive symmetric KNN graph and force-connect by MST."""

    total_neighbors = n_neighbors + 1
    inds = np.argsort(distance_matrix, axis=1)[:, :total_neighbors]
    dists = np.take_along_axis(distance_matrix, inds, axis=1)
    adaptive_threshold = ((np.sum(np.sqrt(dists), axis=1) / (total_neighbors - 2 - delta))[:, None]) ** 2
    adaptive_mask = dists < adaptive_threshold
    adaptive_mask = np.delete(adaptive_mask, -1, axis=1)
    inds = np.delete(inds, -1, axis=1)

    adj = np.zeros(distance_matrix.shape, dtype=np.int8)
    row_ids = np.repeat(np.arange(inds.shape[0]), adaptive_mask.sum(axis=1))
    col_ids = inds[adaptive_mask]
    adj[row_ids, col_ids] = 1
    np.fill_diagonal(adj, 0)
    adj = ((adj + adj.T) > 0).astype(np.int8)

    if not is_connected(adj):
        adj = connect_disconnected_mst(adj, distance_matrix).astype(np.int8)
    return adj


def is_connected(adj: np.ndarray) -> bool:
    """Return whether an adjacency matrix is connected."""

    import igraph as ig

    graph = ig.Graph.Adjacency((adj > 0).tolist(), mode="UNDIRECTED")
    return bool(graph.is_connected())


def connect_disconnected_mst(adj: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """Union the current graph with the distance MST."""

    import igraph as ig

    graph = ig.Graph.Adjacency((adj > 0).tolist(), mode="UNDIRECTED")
    if not graph.is_directed():
        graph.to_undirected()
    mst = ig.Graph.Weighted_Adjacency(distance_matrix.tolist(), mode="UNDIRECTED").spanning_tree()
    combined_edges = set(graph.get_edgelist()).union(mst.get_edgelist())
    combined_graph = ig.Graph(edges=list(combined_edges), directed=False)
    return np.array(combined_graph.get_adjacency().data, dtype=np.int8)


def scanpy_gaussian_weighting(dist_matrix: np.ndarray, adj_matrix: np.ndarray, k: int) -> np.ndarray:
    """Apply scanpy's gaussian weighting and keep only existing graph edges."""

    from scipy.sparse import csr_matrix
    from scanpy.neighbors._connectivity import gauss

    weighted = gauss(csr_matrix(dist_matrix), k, knn=False)
    weighted = weighted.multiply(csr_matrix(adj_matrix))
    return weighted.toarray()


def contextual_smoothing(
    adj_matrix: np.ndarray,
    property_matrix: np.ndarray,
    beta: float = 0.1,
) -> np.ndarray:
    """Context-aware signal smoothing for Step 2's small local graphs."""

    from scipy.linalg import expm

    adj = np.asarray(adj_matrix)
    laplacian = np.diag(np.sum(adj, axis=1)) - adj
    heat_kernel = expm(-beta * laplacian)
    return heat_kernel.dot(property_matrix)


def response_signal_blending(
    exp_raw: np.ndarray,
    exp_last: np.ndarray,
    f_exp_last: np.ndarray,
    iter_round: int,
    weight: float,
    mode: str = "raw",
) -> np.ndarray:
    """Blend observed and context-adjusted expression or label signals."""

    if iter_round == 0:
        return exp_raw
    if mode == "raw":
        return weight * exp_raw + (1 - weight) * f_exp_last
    if mode == "last":
        return weight * exp_last + (1 - weight) * f_exp_last
    if mode == "decrease":
        round_weight = min(0.9, weight * iter_round)
        return (1 - round_weight) * exp_raw + round_weight * f_exp_last
    if mode == "increase":
        round_weight = min(0.9, weight * iter_round)
        return round_weight * exp_raw + (1 - round_weight) * f_exp_last
    if mode == "alternate":
        if iter_round % 2 == 1:
            return weight * exp_raw + (1 - weight) * f_exp_last
        return weight * exp_last + (1 - weight) * f_exp_last
    raise ValueError("Invalid overlay mode.")


def bounded_deviation(values: np.ndarray, floor: float = 0.05) -> float:
    """Robust deviation estimate for small-sample thresholding."""

    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return floor
    data_mad = float(np.median(np.abs(data - np.median(data))))
    data_std = float(np.std(data))
    if data_mad < floor:
        if floor < data_std < floor * 3:
            return data_std
        return floor
    if data_mad < floor * 3:
        return data_mad
    if data_std > floor * 3:
        return min(data_mad, 0.2)
    return data_std


def ratio_upper_limit(active_count: int) -> int:
    """Limit per-round backward elimination to keep RI updates monotone and smooth."""

    drop_count = int(max(0.25 * (active_count - 1200), 0.1 * active_count))
    return max(drop_count, 1)


def _prepare_spatial_weights(adj_matrix: np.ndarray, *, row_standardize: bool = False) -> np.ndarray:
    """Prepare dense spatial weights for the internal Moran-style helpers."""

    weights = np.asarray(adj_matrix, dtype=np.float32).copy()
    if weights.ndim != 2:
        raise ValueError("adj_matrix must be a 2D matrix.")
    if weights.shape[0] != weights.shape[1]:
        raise ValueError("adj_matrix must be square.")
    if weights.size == 0:
        return weights
    if weights[0, 0] != 0:
        np.fill_diagonal(weights, 0.0)
    if row_standardize:
        row_sums = np.sum(weights, axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weights = weights / safe_row_sums
    return weights


def compute_sp_moran_between(
    exp_df1: np.ndarray,
    exp_df2: np.ndarray,
    adj_matrix: np.ndarray,
    *,
    row_standardize: bool = False,
) -> np.ndarray:
    """Bivariate Moran-style statistic used by Step2, with optional row standardization."""

    weights = _prepare_spatial_weights(adj_matrix, row_standardize=row_standardize)
    n = exp_df1.shape[0]
    constant = n / np.sum(weights)
    std_exp_df1 = exp_df1 - exp_df1.mean(axis=0)
    std_exp_df2 = exp_df2 - exp_df2.mean(axis=0)
    l2_norms1 = np.sqrt(np.sum(std_exp_df1**2, axis=0))
    l2_norms2 = np.sqrt(np.sum(std_exp_df2**2, axis=0))
    weighted_std_exp_df = std_exp_df1.T @ weights @ std_exp_df2
    result_matrix = constant * (weighted_std_exp_df / np.outer(l2_norms1, l2_norms2))
    result_matrix = np.nan_to_num(result_matrix)
    result_matrix[result_matrix > 1] = 1
    result_matrix[result_matrix < -1] = -1
    return result_matrix


def compute_sp_moran_between_col(
    exp1: np.ndarray,
    exp2: np.ndarray,
    adj_matrix: np.ndarray,
    *,
    row_standardize: bool = False,
) -> float:
    """Scalar Moran-style statistic between two 1D vectors."""

    exp1_arr = np.asarray(exp1, dtype=np.float32).reshape(-1, 1)
    exp2_arr = np.asarray(exp2, dtype=np.float32).reshape(-1, 1)
    return float(
        compute_sp_moran_between(
            exp1_arr,
            exp2_arr,
            np.asarray(adj_matrix, dtype=np.float32),
            row_standardize=row_standardize,
        )[0, 0]
    )


def _prefer_active_graph(reference_label: np.ndarray, active_adj_mat: np.ndarray, aux_adj_mat: np.ndarray) -> bool:
    """Compare two graph paths and keep the active path when it clears the frozen rule."""

    # Keep the frozen comparison rule while avoiding any extra dependency path here.
    active_score = round(
        compute_sp_moran_between_col(reference_label, reference_label, active_adj_mat, row_standardize=True),
        6,
    )
    aux_score = round(
        compute_sp_moran_between_col(reference_label, reference_label, aux_adj_mat, row_standardize=True),
        6,
    )
    return bool(active_score >= (aux_score**2))


def compute_gene_moran_scores(
    exp_denoised: np.ndarray,
    label_smooth: np.ndarray,
    label_raw: np.ndarray,
    affinity: np.ndarray,
    score_mode: str = "self_n_label",
    binary_delta_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute default Step 2 gene response scores without materializing gene-pair matrices."""

    exp_input = np.asarray(exp_denoised, dtype=np.float32).copy()
    label_raw = np.asarray(label_raw).reshape(-1)
    label_smooth = np.asarray(label_smooth, dtype=np.float32).reshape(-1)
    case_cells = label_raw == 1
    control_cells = label_raw == 0

    if np.any(case_cells) and np.any(control_cells):
        exp_median = np.median(exp_input, axis=0)
        case_exp = exp_input[case_cells]
        case_weights = label_smooth[case_cells]
        weight_sum = float(np.sum(case_weights))
        if weight_sum == 0:
            case_avg = np.mean(case_exp, axis=0)
        else:
            case_avg = np.average(case_exp, axis=0, weights=case_weights)
        control_avg = np.mean(exp_input[control_cells], axis=0)
        low_delta = np.abs(case_avg - control_avg) < binary_delta_threshold
        if np.any(low_delta):
            exp_input[:, low_delta] = (exp_input[:, low_delta] >= exp_median[low_delta]).astype(np.float32)

    affinity = np.asarray(affinity, dtype=np.float32)
    if affinity[0, 0] != 0:
        affinity = affinity.copy()
        np.fill_diagonal(affinity, 0)
    n_cells = exp_input.shape[0]
    weight_sum = float(np.sum(affinity))
    if weight_sum == 0:
        zeros = np.zeros(exp_input.shape[1], dtype=np.float32)
        return zeros, zeros, zeros

    constant = n_cells / weight_sum
    x = exp_input - exp_input.mean(axis=0)
    y = label_smooth - label_smooth.mean()
    wx = affinity @ x
    wy = affinity @ y
    x_norm_sq = np.sum(x * x, axis=0)
    y_norm = float(np.sqrt(np.sum(y * y)))

    with np.errstate(divide="ignore", invalid="ignore"):
        gene_self = constant * (np.sum(x * wx, axis=0) / x_norm_sq)
        gene_label = constant * (x.T @ wy / (np.sqrt(x_norm_sq) * y_norm))
    gene_self = np.nan_to_num(gene_self, nan=0.0, posinf=0.0, neginf=0.0)
    gene_label = np.nan_to_num(gene_label, nan=0.0, posinf=0.0, neginf=0.0)
    gene_self = np.clip(gene_self, -1, 1).astype(np.float32)
    gene_label = np.clip(gene_label, -1, 1).astype(np.float32)

    if score_mode == "self":
        combined = np.abs(gene_self)
    elif score_mode == "label":
        combined = np.abs(gene_label)
    elif score_mode == "self_n_label":
        combined = np.sqrt(gene_self**2 + gene_label**2) / np.sqrt(2)
    else:
        raise ValueError("score_mode must be one of: 'self', 'label', 'self_n_label'.")
    return gene_self, gene_label, combined.astype(np.float32)


def normalize_sample_score(combined_score: np.ndarray, ri_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center each sample's score against genes already outside RI."""

    active = np.asarray(ri_mask, dtype=bool)
    background = np.asarray(combined_score)[~active]
    retained = np.asarray(combined_score)[active]
    if background.size == 0:
        background_center = 0.0
    else:
        background_center = float(np.median(background))
    retained_center = 0.0 if retained.size == 0 else float(np.median(retained))
    return (combined_score - background_center).astype(np.float32), retained_center, background_center


def update_response_identity(
    current_ri: np.ndarray,
    response_score: np.ndarray,
    threshold_k: float = 2.0,
    drop_limit: bool = True,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Monotone RI update for the default Step 2 path."""

    current = np.asarray(current_ri, dtype=bool)
    score = np.asarray(response_score, dtype=np.float32)
    background = score[~current]
    if background.size == 0:
        threshold = 0.0
    else:
        threshold = float(np.median(background) + threshold_k * bounded_deviation(background))

    candidate = current & (score >= threshold)
    potential_drop = np.flatnonzero(current & ~candidate)
    active_count = int(np.sum(current))
    max_drop = ratio_upper_limit(active_count) if drop_limit else max(active_count - 1, 1)
    if potential_drop.size > max_drop:
        order = np.argsort(score[potential_drop], kind="stable")
        drop_ids = potential_drop[order[:max_drop]]
        next_ri = current.copy()
        next_ri[drop_ids] = False
    else:
        next_ri = candidate
    meta = {
        "threshold": threshold,
        "active_count": active_count,
        "next_active_count": int(np.sum(next_ri)),
        "potential_drop": int(potential_drop.size),
        "max_drop": int(max_drop),
        "actual_drop": int(active_count - np.sum(next_ri)),
    }
    return next_ri, meta


def update_response_identity_with_stage(
    current_ri: np.ndarray,
    response_score: np.ndarray,
    threshold_k: float = 2.0,
    drop_limit: bool = True,
    *,
    update_stage: str = "strict",
    fs_strict_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """RI update with strict/wave stage semantics."""

    current = np.asarray(current_ri, dtype=bool)
    score = np.asarray(response_score, dtype=np.float32)
    background = score[~current]
    if background.size == 0:
        threshold = 0.0
    else:
        threshold = float(np.median(background) + threshold_k * bounded_deviation(background))

    binary_indicator = score >= threshold
    active_count = int(np.sum(current))
    if update_stage == "wave" and fs_strict_mask is not None:
        strict_mask = np.asarray(fs_strict_mask, dtype=bool)
        next_ri = binary_indicator & strict_mask
        meta = {
            "stage": "wave",
            "threshold": threshold,
            "active_count": active_count,
            "next_active_count": int(np.sum(next_ri)),
            "potential_drop": int(np.sum(current & ~next_ri)),
            "max_drop": int(max(active_count - 1, 1)),
            "actual_drop": int(active_count - np.sum(next_ri)),
            "restore_count": int(np.sum((~current) & next_ri)),
            "fs_strict_active": int(np.sum(strict_mask)),
        }
        return next_ri, meta

    candidate = current & binary_indicator
    potential_drop = np.flatnonzero(current & ~candidate)
    max_drop = ratio_upper_limit(active_count) if drop_limit else max(active_count - 1, 1)
    if potential_drop.size > max_drop:
        order = np.argsort(score[potential_drop], kind="stable")
        drop_ids = potential_drop[order[:max_drop]]
        next_ri = current.copy()
        next_ri[drop_ids] = False
    else:
        next_ri = candidate
    meta = {
        "stage": "strict",
        "threshold": threshold,
        "active_count": active_count,
        "next_active_count": int(np.sum(next_ri)),
        "potential_drop": int(potential_drop.size),
        "max_drop": int(max_drop),
        "actual_drop": int(active_count - np.sum(next_ri)),
        "restore_count": 0,
        "fs_strict_active": int(np.sum(fs_strict_mask)) if fs_strict_mask is not None else 0,
    }
    return next_ri, meta


def label_nh_prop_moran(
    exp_mat: np.ndarray,
    adj_mat: np.ndarray,
    label_raw: np.ndarray,
    label_last: np.ndarray,
    fs_mask: np.ndarray,
) -> np.ndarray:
    """Default Step 2 label continuization path."""

    from sklearn.metrics.pairwise import cosine_similarity

    selected_genes = np.where(fs_mask == 1)[0]
    filtered_exp_mat = exp_mat[:, selected_genes]
    case_weights = label_last.astype(np.float32)
    cell_similarity = cosine_similarity(filtered_exp_mat, filtered_exp_mat)
    cell_with_ca_moran = compute_sp_moran_between(
        cell_similarity,
        case_weights[:, np.newaxis],
        adj_mat,
    ).flatten()
    scale = np.max(cell_with_ca_moran)
    if scale > 0.1:
        cell_with_ca_moran = cell_with_ca_moran / scale
    else:
        cell_with_ca_moran = cell_with_ca_moran / 0.1
    score = np.clip(cell_with_ca_moran, 0, 1)
    score[label_raw == 0] = 0
    return score.astype(np.float32)


def _run_graph_path(
    *,
    exp_raw: np.ndarray,
    exp_last_arr: np.ndarray,
    label_raw: np.ndarray,
    label_last_arr: np.ndarray,
    fs_mask: np.ndarray,
    iter_round: int,
    options: Step2Options,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_dc = pca_select(exp_last_arr[:, fs_mask == 1], n_pcs=options.n_pcs)
    dist = compute_distance_cosine(cell_dc, normalized=True)
    adjacency = adaptive_knn(dist, n_neighbors=options.cell_k, delta=-0.5)
    affinity = ensure_float_type(
        scanpy_gaussian_weighting(dist, adjacency, k=options.cell_k),
        target_dtype=options.dtype,
    )
    exp_denoised = ensure_float_type(
        contextual_smoothing(affinity, exp_last_arr, beta=options._smooth_alpha),
        target_dtype=options.dtype,
    )
    exp_mixed = ensure_float_type(
        response_signal_blending(
            exp_raw=exp_raw,
            exp_last=exp_last_arr,
            f_exp_last=exp_denoised,
            iter_round=iter_round,
            weight=1 - options._smooth_weight,
            mode="raw",
        ),
        target_dtype=options.dtype,
    )
    label_smooth = label_nh_prop_moran(
        exp_mat=exp_mixed,
        adj_mat=affinity,
        label_raw=label_raw,
        label_last=label_last_arr,
        fs_mask=fs_mask,
    )
    label_mixed = ensure_float_type(
        response_signal_blending(
            exp_raw=label_raw,
            exp_last=label_last_arr,
            f_exp_last=label_smooth,
            iter_round=iter_round,
            weight=options._smooth_weight,
            mode="decrease",
        ),
        target_dtype=options.dtype,
    )
    return affinity, exp_denoised, exp_mixed, label_smooth, label_mixed


def run_sample_core(
    pack: Step2SamplePack,
    exp_last: np.ndarray | None,
    label_last: np.ndarray | None,
    branch_ready: bool = False,
    options: Step2Options | None = None,
) -> Step2SampleOutput:
    """Run one sample through the default Step 2 path."""

    options = options or Step2Options()
    exp_raw = ensure_float_type(pack.exp_raw, target_dtype=options.dtype)
    exp_last_arr = exp_raw if exp_last is None else ensure_float_type(np.asarray(exp_last), target_dtype=options.dtype)
    label_raw = ensure_float_type(pack.label_raw, target_dtype=options.dtype)
    label_last_arr = label_raw if label_last is None else ensure_float_type(np.asarray(label_last), target_dtype=options.dtype)
    fs_mask = np.asarray(pack.fs_mask)
    iter_round = int(options.extras.get("iteration", 1 if exp_last is None else 2))
    affinity, exp_denoised, exp_mixed, label_smooth, label_mixed = _run_graph_path(
        exp_raw=exp_raw,
        exp_last_arr=exp_last_arr,
        label_raw=label_raw,
        label_last_arr=label_last_arr,
        fs_mask=fs_mask,
        iter_round=iter_round,
        options=options,
    )
    next_branch_ready = True
    aux_fs_mask = pack.aux_fs_mask
    compare_round_cap = options._guide_compare_rounds
    within_compare_window = True
    if compare_round_cap is not None:
        compare_round_cap = int(compare_round_cap)
        if compare_round_cap > 0:
            within_compare_window = iter_round <= compare_round_cap
    aux_path_enabled = bool(
        within_compare_window
        and aux_fs_mask is not None
        and np.any(aux_fs_mask)
        and not branch_ready
    )
    if aux_path_enabled:
        aux_affinity, aux_exp_denoised, aux_exp_mixed, _, _ = _run_graph_path(
            exp_raw=exp_raw,
            exp_last_arr=exp_last_arr,
            label_raw=label_raw,
            label_last_arr=label_last_arr,
            fs_mask=np.asarray(aux_fs_mask),
            iter_round=iter_round,
            options=options,
        )
        exp_denoised = aux_exp_denoised
        exp_mixed = aux_exp_mixed
        label_smooth = label_nh_prop_moran(
            exp_mat=aux_exp_mixed,
            adj_mat=affinity,
            label_raw=label_raw,
            label_last=label_last_arr,
            fs_mask=fs_mask,
        )
        label_mixed = ensure_float_type(
            response_signal_blending(
                exp_raw=label_raw,
                exp_last=label_last_arr,
                f_exp_last=label_smooth,
                iter_round=iter_round,
                weight=options._smooth_weight,
                mode="decrease",
            ),
            target_dtype=options.dtype,
        )
        aux_label_smooth = label_nh_prop_moran(
            exp_mat=aux_exp_mixed,
            adj_mat=affinity,
            label_raw=label_raw,
            label_last=label_last_arr,
            fs_mask=np.asarray(aux_fs_mask),
        )
        aux_label_mixed = ensure_float_type(
            response_signal_blending(
                exp_raw=label_raw,
                exp_last=label_last_arr,
                f_exp_last=aux_label_smooth,
                iter_round=iter_round,
                weight=options._smooth_weight,
                mode="decrease",
            ),
            target_dtype=options.dtype,
        )
        next_branch_ready = _prefer_active_graph(aux_label_smooth, affinity, aux_affinity)
        if not next_branch_ready:
            affinity = aux_affinity
            label_smooth = aux_label_smooth
            label_mixed = aux_label_mixed
    gene_self, gene_label, combined_score = compute_gene_moran_scores(
        exp_denoised=exp_denoised,
        label_smooth=label_smooth,
        label_raw=label_raw,
        affinity=affinity,
        score_mode=options.score_mode,
        binary_delta_threshold=options._binary_delta_threshold,
    )
    norm_score, retained_center, background_center = normalize_sample_score(combined_score, fs_mask)
    return Step2SampleOutput(
        exp_last_next=exp_mixed,
        label_last_next=label_mixed,
        affinity=affinity,
        gene_self_cor=gene_self,
        gene_label_cor=gene_label,
        combined_score=combined_score,
        norm_combined_score=norm_score,
        branch_ready_next=next_branch_ready,
        metadata={
            "sample_id": pack.sample_id,
            "cell_count": int(exp_raw.shape[0]),
            "gene_count": int(exp_raw.shape[1]),
            "selected_gene_count": int(np.sum(fs_mask == 1)),
            "deg_total_summary": retained_center,
            "nondeg_total_summary": background_center,
            "guide_active_genes": int(np.sum(aux_fs_mask == 1)) if aux_fs_mask is not None else 0,
            "guide_used": bool(aux_path_enabled),
            "guide_pass_next": bool(next_branch_ready),
        },
    )
