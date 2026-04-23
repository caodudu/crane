"""Step 1 KS-based feature screening specialized for CRANE.

This module keeps the paper-facing KS semantics while replacing the heavy
general-purpose legacy call pattern with a Step1-specific strategy:

- full-gene KS statistic ranking
- raw-p boundary search instead of full-gene p-value evaluation
- local adj-p protection only around the guide-set window

It does not import the legacy ``script/couture`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class KSFeatureScreenOptions:
    """Paper-facing options for Step 1 KS feature screening."""

    raw_p_alpha: float = 0.05
    guide_adj_p_alpha: float = 0.05
    guide_min_features: int = 20
    guide_max_features: int = 100
    guide_min_significant: int = 5
    pvalue_method: str = "auto"
    guide_eval_window: int = 256


@dataclass(frozen=True)
class KSFeatureScreenResult:
    """Compact result package for Step 1 feature filtering."""

    statistic: np.ndarray
    guide_raw_pvalues: np.ndarray
    adj_pvalues: np.ndarray
    ranking: np.ndarray
    coarse_mask: np.ndarray
    core_mask: np.ndarray
    guide_mask: np.ndarray
    feature_names: np.ndarray | None
    evaluated_raw_pvalue_count: int
    evaluated_adj_pvalue_count: int
    raw_boundary_rank: int | None
    guide_eval_stop_rank: int | None


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return arr


def _validate_case_control(case_expr: np.ndarray, control_expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    case_arr = _as_2d_float("case_expr", case_expr)
    control_arr = _as_2d_float("control_expr", control_expr)
    if case_arr.shape[1] != control_arr.shape[1]:
        raise ValueError("case_expr and control_expr must have the same number of genes.")
    if case_arr.shape[0] == 0 or control_arr.shape[0] == 0:
        raise ValueError("case_expr and control_expr must both contain cells.")
    return case_arr, control_arr


def compute_ks_statistics(case_expr: np.ndarray, control_expr: np.ndarray) -> np.ndarray:
    """Compute two-sample KS statistics for every gene.

    This keeps only the minimum path needed for Step 1 ranking:
    sort -> concatenate -> searchsorted -> max CDF difference.
    """

    case_arr, control_arr = _validate_case_control(case_expr, control_expr)
    n_case = case_arr.shape[0]
    n_control = control_arr.shape[0]
    n_genes = case_arr.shape[1]
    out = np.empty(n_genes, dtype=float)

    for gene_idx in range(n_genes):
        sorted_case = np.sort(case_arr[:, gene_idx])
        sorted_control = np.sort(control_arr[:, gene_idx])
        values = np.concatenate([sorted_case, sorted_control])
        cdf_case = np.searchsorted(sorted_case, values, side="right") / n_case
        cdf_control = np.searchsorted(sorted_control, values, side="right") / n_control
        cdf_diff = cdf_case - cdf_control
        min_diff = np.clip(-cdf_diff[np.argmin(cdf_diff)], 0, 1)
        max_diff = cdf_diff[np.argmax(cdf_diff)]
        out[gene_idx] = min_diff if min_diff > max_diff else max_diff

    return out


def _bh_adjust_sorted_pvalues(sorted_pvalues: np.ndarray, total_tests: int) -> np.ndarray:
    """BH adjustment for already-sorted raw p-values.

    Tail p-values that were never evaluated are allowed to be fixed at 1.0.
    """

    ranks = np.arange(1, sorted_pvalues.shape[0] + 1, dtype=float)
    adjusted = sorted_pvalues * float(total_tests) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    return np.clip(adjusted, 0.0, 1.0)


def _compute_raw_pvalue(
    case_arr: np.ndarray,
    control_arr: np.ndarray,
    gene_idx: int,
    method: str,
) -> float:
    result = ks_2samp(case_arr[:, gene_idx], control_arr[:, gene_idx], method=method)
    return float(result.pvalue)


def find_raw_pvalue_boundary(
    case_expr: np.ndarray,
    control_expr: np.ndarray,
    ranking: np.ndarray,
    raw_p_alpha: float = 0.05,
    method: str = "auto",
) -> tuple[int, dict[int, float], int]:
    """Locate the last rank with ``raw p <= raw_p_alpha`` by binary search.

    Because all genes share the same case/control sample sizes, raw p-values are
    monotonic in the KS statistic ranking.
    """

    case_arr, control_arr = _validate_case_control(case_expr, control_expr)
    n_genes = case_arr.shape[1]
    cache: dict[int, float] = {}
    evaluated = 0

    def _p_at_rank(rank_idx: int) -> float:
        nonlocal evaluated
        if rank_idx not in cache:
            cache[rank_idx] = _compute_raw_pvalue(case_arr, control_arr, int(ranking[rank_idx]), method)
            evaluated += 1
        return cache[rank_idx]

    if _p_at_rank(0) > raw_p_alpha:
        return -1, cache, evaluated
    if _p_at_rank(n_genes - 1) <= raw_p_alpha:
        return n_genes - 1, cache, evaluated

    left = 0
    right = n_genes - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if _p_at_rank(mid) <= raw_p_alpha:
            left = mid
        else:
            right = mid
    return left, cache, evaluated


def compute_guide_prefix_raw_pvalues(
    case_expr: np.ndarray,
    control_expr: np.ndarray,
    ranking: np.ndarray,
    stop_rank: int,
    method: str = "auto",
) -> tuple[np.ndarray, int]:
    """Evaluate raw p-values only for the guide-protection prefix."""

    case_arr, control_arr = _validate_case_control(case_expr, control_expr)
    n_genes = case_arr.shape[1]
    raw_pvalues = np.ones(n_genes, dtype=float)
    if stop_rank < 0:
        return raw_pvalues, 0

    evaluated = 0
    for rank_idx in range(stop_rank + 1):
        gene_idx = int(ranking[rank_idx])
        raw_pvalues[gene_idx] = _compute_raw_pvalue(case_arr, control_arr, gene_idx, method)
        evaluated += 1
    return raw_pvalues, evaluated


def select_top_by_stable_raw_pvalues(
    case_expr: np.ndarray,
    control_expr: np.ndarray,
    ranking: np.ndarray,
    keep_n: int,
    method: str = "auto",
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Select top genes by raw p-value with legacy-stable tie-break.

    Raw p-values are monotonic in the KS-statistic ranking for fixed sample
    sizes, so only the cutoff bucket needs explicit resolution. Within that
    bucket we follow ``np.argsort(raw_pvalues, kind="stable")`` semantics,
    which keeps original gene order for equal p-values and matches legacy
    ``DataFrame.nsmallest(..., 'p-value')`` on the full result table.
    """

    case_arr, control_arr = _validate_case_control(case_expr, control_expr)
    n_genes = case_arr.shape[1]
    if keep_n <= 0:
        return np.empty(0, dtype=int), {
            "evaluated_raw_pvalue_count": 0,
            "boundary_left_rank": -1,
            "boundary_right_rank": -1,
            "boundary_pvalue": 1.0,
            "strictly_lower_count": 0,
        }
    if keep_n >= n_genes:
        return np.arange(n_genes, dtype=int), {
            "evaluated_raw_pvalue_count": 0,
            "boundary_left_rank": 0,
            "boundary_right_rank": n_genes - 1,
            "boundary_pvalue": 1.0,
            "strictly_lower_count": 0,
        }

    cache: dict[int, float] = {}
    evaluated = 0

    def _p_at_rank(rank_idx: int) -> float:
        nonlocal evaluated
        if rank_idx not in cache:
            cache[rank_idx] = _compute_raw_pvalue(case_arr, control_arr, int(ranking[rank_idx]), method)
            evaluated += 1
        return cache[rank_idx]

    cutoff_rank = keep_n - 1
    boundary_p = _p_at_rank(cutoff_rank)

    left = -1
    right = cutoff_rank
    while left + 1 < right:
        mid = (left + right) // 2
        if _p_at_rank(mid) < boundary_p:
            left = mid
        else:
            right = mid
    strictly_lower_count = left + 1
    boundary_left_rank = strictly_lower_count

    left = cutoff_rank
    right = n_genes
    while left + 1 < right:
        mid = (left + right) // 2
        if _p_at_rank(mid) <= boundary_p:
            left = mid
        else:
            right = mid
    boundary_right_rank = left

    selected = np.empty(keep_n, dtype=int)
    if strictly_lower_count > 0:
        selected[:strictly_lower_count] = ranking[:strictly_lower_count]

    boundary_gene_indices = np.sort(ranking[boundary_left_rank : boundary_right_rank + 1])
    boundary_take_n = keep_n - strictly_lower_count
    selected[strictly_lower_count:] = boundary_gene_indices[:boundary_take_n]
    return selected, {
        "evaluated_raw_pvalue_count": evaluated,
        "boundary_left_rank": boundary_left_rank,
        "boundary_right_rank": boundary_right_rank,
        "boundary_pvalue": float(boundary_p),
        "strictly_lower_count": strictly_lower_count,
    }


def _build_guide_mask(
    ranking: np.ndarray,
    raw_pvalues: np.ndarray,
    adj_pvalues: np.ndarray,
    options: KSFeatureScreenOptions,
) -> np.ndarray:
    raw_pvalue_order = np.argsort(raw_pvalues, kind="stable")
    guide_candidates = np.flatnonzero(adj_pvalues <= options.guide_adj_p_alpha)

    if guide_candidates.shape[0] < options.guide_min_significant:
        for threshold in (options.raw_p_alpha, 1.0):
            guide_candidates = np.flatnonzero(raw_pvalues <= threshold)
            if guide_candidates.shape[0] >= options.guide_min_significant:
                break

    if guide_candidates.shape[0] > options.guide_max_features:
        guide_candidates = raw_pvalue_order[: options.guide_max_features]
    elif guide_candidates.shape[0] < options.guide_min_features:
        guide_candidates = raw_pvalue_order[: options.guide_min_features]

    guide_mask = np.zeros_like(raw_pvalues, dtype=bool)
    guide_mask[guide_candidates] = True
    return guide_mask


def screen_ks_features(
    case_expr: np.ndarray,
    control_expr: np.ndarray,
    feature_names: Sequence[str] | None = None,
    options: KSFeatureScreenOptions | None = None,
) -> KSFeatureScreenResult:
    """Run the CRANE Step 1 KS screening path.

    The result preserves the current Step 1 feature-filter semantics:
    - full-gene KS ranking
    - ``raw_p <= 0.05`` coarse set
    - local ``adj_p <= 0.05`` protection around the guide window
    - fallback-clipped guide set in the 20-100 range
    """

    options = options or KSFeatureScreenOptions()
    case_arr, control_arr = _validate_case_control(case_expr, control_expr)
    n_genes = case_arr.shape[1]

    if feature_names is not None and len(feature_names) != n_genes:
        raise ValueError("feature_names length must match the number of genes.")

    statistic = compute_ks_statistics(case_arr, control_arr)
    ranking = np.argsort(-statistic, kind="stable")

    raw_boundary_rank, boundary_cache, raw_evaluated = find_raw_pvalue_boundary(
        case_arr,
        control_arr,
        ranking,
        raw_p_alpha=options.raw_p_alpha,
        method=options.pvalue_method,
    )
    coarse_mask = np.zeros(n_genes, dtype=bool)
    if raw_boundary_rank >= 0:
        coarse_mask[ranking[: raw_boundary_rank + 1]] = True

    guide_rank_target = max(
        options.guide_min_features - 1,
        options.guide_max_features + options.guide_eval_window - 1,
    )
    if raw_boundary_rank >= 0:
        guide_stop_rank = min(n_genes - 1, min(raw_boundary_rank, guide_rank_target))
    else:
        guide_stop_rank = min(n_genes - 1, guide_rank_target)
    guide_raw_pvalues, guide_evaluated = compute_guide_prefix_raw_pvalues(
        case_arr,
        control_arr,
        ranking,
        stop_rank=guide_stop_rank,
        method=options.pvalue_method,
    )
    for rank_idx, raw_p in boundary_cache.items():
        guide_raw_pvalues[int(ranking[rank_idx])] = raw_p

    sorted_raw = guide_raw_pvalues[ranking]
    sorted_adj = _bh_adjust_sorted_pvalues(sorted_raw, total_tests=n_genes)
    adj_pvalues = np.empty_like(sorted_adj)
    adj_pvalues[ranking] = sorted_adj

    core_mask = adj_pvalues < options.guide_adj_p_alpha
    guide_mask = _build_guide_mask(ranking, guide_raw_pvalues, adj_pvalues, options)

    feature_name_array = None if feature_names is None else np.asarray(feature_names, dtype=object)
    return KSFeatureScreenResult(
        statistic=statistic,
        guide_raw_pvalues=guide_raw_pvalues,
        adj_pvalues=adj_pvalues,
        ranking=ranking,
        coarse_mask=coarse_mask,
        core_mask=core_mask,
        guide_mask=guide_mask,
        feature_names=feature_name_array,
        evaluated_raw_pvalue_count=raw_evaluated,
        evaluated_adj_pvalue_count=guide_evaluated,
        raw_boundary_rank=raw_boundary_rank,
        guide_eval_stop_rank=guide_stop_rank,
    )
