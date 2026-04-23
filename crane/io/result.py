"""Result object for the new CRANE package."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import pickle

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from ..internal.logger import build_logger
from ..core.step1 import _compute_sp_moran_between, _compute_sp_moran_between_col
from ..step2.kernels import (
    adaptive_knn,
    compute_distance_cosine,
    heat_kernel_smoothing,
    label_nh_prop_moran,
    overlay_raw_exp,
    pca_select,
    scanpy_gaussian_weighting,
)
from .schema import FunctionalInput, LoggerConfig
from ._gene_module_backends import resolve_gene_module_method, run_gene_module_backend


def _select_representative_samples(score_matrix: pd.DataFrame, n: int = 3) -> list[int]:
    if score_matrix.shape[1] <= n:
        return list(range(score_matrix.shape[1]))
    dist_matrix = squareform(pdist(score_matrix.T, metric="euclidean"))
    centrality = dist_matrix.mean(axis=1)
    selected = [int(np.argmin(centrality))]
    for _ in range(n - 1):
        if len(selected) == 1:
            candidate = dist_matrix[selected[0]].copy()
            candidate[selected] = -np.inf
        else:
            distance_vectors = dist_matrix[:, selected]
            candidate = np.linalg.norm(distance_vectors, axis=1)
            candidate[selected] = -np.inf
        selected.append(int(np.argmax(candidate)))
    return sorted(set(selected))


def _merge_duplicate_cells(
    *,
    obs: pd.DataFrame,
    exp_raw: np.ndarray,
    exp_smooth: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged_obs_frames: list[pd.DataFrame] = []
    merged_raw_blocks: list[np.ndarray] = []
    merged_smooth_blocks: list[np.ndarray] = []

    for group_label in ("control", "case"):
        group_obs = obs.loc[obs["group_label"] == group_label].copy()
        if group_obs.empty:
            continue

        group_obs = group_obs.reset_index(drop=True)
        duplicated = group_obs.duplicated(subset="original_cell_id", keep=False)
        if not duplicated.any():
            result_group = group_obs.copy()
            result_group.index = pd.Index(result_group["original_cell_id"].astype(str), name="cell_id")
            merged_obs_frames.append(result_group)
            merged_raw_blocks.append(np.asarray(exp_raw[obs["group_label"].to_numpy() == group_label], dtype=np.float32))
            merged_smooth_blocks.append(
                np.asarray(exp_smooth[obs["group_label"].to_numpy() == group_label], dtype=np.float32)
            )
            continue

        duplicated_cells = group_obs.loc[duplicated].copy()
        aggregated = (
            duplicated_cells.groupby("original_cell_id")
            .agg(
                {
                    "group_label": "first",
                    "label_raw": "mean",
                    "label_mixed": "mean",
                }
            )
            .reset_index()
        )
        non_duplicated = group_obs.loc[~duplicated].copy()

        exp_raw_group = np.asarray(exp_raw[obs["group_label"].to_numpy() == group_label], dtype=np.float32)
        exp_smooth_group = np.asarray(exp_smooth[obs["group_label"].to_numpy() == group_label], dtype=np.float32)
        new_raw_rows: list[np.ndarray] = []
        new_smooth_rows: list[np.ndarray] = []
        for _, sub_group in group_obs.groupby("original_cell_id"):
            indices = sub_group.index.to_numpy(dtype=int)
            if len(indices) > 1:
                new_raw_rows.append(np.max(exp_raw_group[indices], axis=0))
                new_smooth_rows.append(np.max(exp_smooth_group[indices], axis=0))
            else:
                new_raw_rows.append(exp_raw_group[indices[0]])
                new_smooth_rows.append(exp_smooth_group[indices[0]])

        result_group = pd.concat([non_duplicated, aggregated], ignore_index=True)
        result_group.index = pd.Index(result_group["original_cell_id"].astype(str), name="cell_id")
        merged_obs_frames.append(result_group)
        merged_raw_blocks.append(np.vstack(new_raw_rows).astype(np.float32, copy=False))
        merged_smooth_blocks.append(np.vstack(new_smooth_rows).astype(np.float32, copy=False))

    if not merged_obs_frames:
        shape = (0, exp_raw.shape[1])
        empty = pd.DataFrame(columns=["original_cell_id", "group_label", "label_raw", "label_mixed"])
        empty.index = pd.Index([], name="cell_id")
        return empty, np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    merged_obs = pd.concat(merged_obs_frames, axis=0)
    merged_raw = np.vstack(merged_raw_blocks).astype(np.float32, copy=False)
    merged_smooth = np.vstack(merged_smooth_blocks).astype(np.float32, copy=False)
    return merged_obs.copy(), merged_raw, merged_smooth


def _deduplicate_group_exact(
    group: pd.DataFrame,
    exp_matrix_raw: np.ndarray,
    exp_matrix_smooth: np.ndarray,
    *,
    aggregated_first: bool = False,
    groupby_sort: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    group = group.reset_index(drop=True)
    duplicated = group.duplicated(subset="cell_id", keep=False)
    duplicated_cells = group[duplicated]

    if not duplicated_cells.empty:
        aggregated = (
            duplicated_cells.groupby("cell_id", sort=groupby_sort)
            .agg(
                {
                    "group_id": "first",
                    "label_raw": "mean",
                    "label_last": "mean",
                    "label_smooth": "mean",
                    "label_mixed": "mean",
                }
            )
            .reset_index()
        )
        aggregated["rep_cell_id"] = "rep_avg_" + aggregated["cell_id"].astype(str)
        non_duplicated = group[~duplicated]

        new_exp_rows_raw: list[np.ndarray] = []
        new_exp_rows_smooth: list[np.ndarray] = []
        for _, sub_group in group.groupby("cell_id", sort=groupby_sort):
            indices = sub_group.index.to_numpy(dtype=int)
            if len(indices) > 1:
                new_exp_rows_raw.append(exp_matrix_raw[indices].max(axis=0))
                new_exp_rows_smooth.append(exp_matrix_smooth[indices].max(axis=0))
            else:
                index = int(indices[0])
                new_exp_rows_raw.append(exp_matrix_raw[index])
                new_exp_rows_smooth.append(exp_matrix_smooth[index])

        exp_matrix_raw = np.vstack(new_exp_rows_raw).astype(np.float32, copy=False)
        exp_matrix_smooth = np.vstack(new_exp_rows_smooth).astype(np.float32, copy=False)
        if aggregated_first:
            result_group = pd.concat([aggregated, non_duplicated], ignore_index=True)
        else:
            result_group = pd.concat([non_duplicated, aggregated], ignore_index=True)
    else:
        result_group = group
        exp_matrix_raw = exp_matrix_raw[group.index.get_indexer(group.index)]
        exp_matrix_smooth = exp_matrix_smooth[group.index.get_indexer(group.index)]
    return result_group, exp_matrix_raw, exp_matrix_smooth


def _merge_selected_samples_exact(
    *,
    stacked_obs: pd.DataFrame,
    stacked_raw: np.ndarray,
    stacked_smooth: np.ndarray,
    random_state: int = 123,
    aggregated_first: bool = False,
    groupby_sort: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    coca_cell_meta = stacked_obs.rename(
        columns={
            "original_cell_id": "cell_id",
            "group_label": "group_id",
            "label_mixed": "label_last",
        }
    ).copy()
    coca_cell_meta["label_smooth"] = coca_cell_meta["label_last"].astype(np.float32)
    coca_cell_meta["label_mixed"] = coca_cell_meta["label_last"].astype(np.float32)

    coca_sort_order = coca_cell_meta["group_id"].map({"control": 0, "case": 1}).argsort()
    coca_cell_meta = coca_cell_meta.iloc[coca_sort_order].copy()
    coca_raw_exp = stacked_raw[coca_sort_order, :].astype(np.float32, copy=False)
    coca_smooth_exp = stacked_smooth[coca_sort_order, :].astype(np.float32, copy=False)

    coca_control_group = coca_cell_meta[coca_cell_meta["group_id"] == "control"].copy()
    coca_control_group.index = coca_control_group.loc[:, "cell_id"]
    coca_case_group = coca_cell_meta[coca_cell_meta["group_id"] == "case"].copy()
    coca_case_group.index = coca_case_group.loc[:, "cell_id"]

    coca_control_group, coca_raw_exp_control, coca_smooth_exp_control = _deduplicate_group_exact(
        coca_control_group,
        coca_raw_exp[coca_cell_meta["group_id"] == "control"],
        coca_smooth_exp[coca_cell_meta["group_id"] == "control"],
        aggregated_first=aggregated_first,
        groupby_sort=groupby_sort,
    )
    coca_case_group, coca_raw_exp_case, coca_smooth_exp_case = _deduplicate_group_exact(
        coca_case_group,
        coca_raw_exp[coca_cell_meta["group_id"] == "case"],
        coca_smooth_exp[coca_cell_meta["group_id"] == "case"],
        aggregated_first=aggregated_first,
        groupby_sort=groupby_sort,
    )

    rng = np.random.RandomState(random_state)
    if len(coca_control_group) > len(coca_case_group):
        coca_control_sample_indices = rng.choice(
            coca_control_group.index.to_numpy(),
            size=len(coca_case_group),
            replace=False,
        )
        coca_control_group = coca_control_group.loc[coca_control_sample_indices]
        coca_raw_exp_control = coca_raw_exp_control[coca_control_group.index.get_indexer(coca_control_group.index)]
        coca_smooth_exp_control = coca_smooth_exp_control[
            coca_control_group.index.get_indexer(coca_control_group.index)
        ]
    elif len(coca_control_group) < len(coca_case_group):
        coca_case_sample_indices = rng.choice(
            coca_case_group.index.to_numpy(),
            size=len(coca_control_group),
            replace=False,
        )
        coca_case_group = coca_case_group.loc[coca_case_sample_indices]
        coca_raw_exp_case = coca_raw_exp_case[coca_case_group.index.get_indexer(coca_case_group.index)]
        coca_smooth_exp_case = coca_smooth_exp_case[coca_case_group.index.get_indexer(coca_case_group.index)]

    coca_cell_meta = pd.concat([coca_control_group, coca_case_group], ignore_index=True)
    coca_raw_exp = np.vstack([coca_raw_exp_control, coca_raw_exp_case]).astype(np.float32, copy=False)
    coca_smooth_exp = np.vstack([coca_smooth_exp_control, coca_smooth_exp_case]).astype(np.float32, copy=False)
    coca_cell_meta.set_index("cell_id", inplace=True)

    merged_obs = coca_cell_meta.rename(
        columns={
            "group_id": "group_label",
            "label_mixed": "label_mixed",
        }
    ).copy()
    merged_obs["original_cell_id"] = merged_obs.index.astype(str)
    return merged_obs, coca_raw_exp, coca_smooth_exp


def _balance_groups(
    *,
    obs: pd.DataFrame,
    exp_raw: np.ndarray,
    exp_smooth: np.ndarray,
    random_state: int = 123,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    control_mask = obs["group_label"].to_numpy() == "control"
    case_mask = obs["group_label"].to_numpy() == "case"
    control_group = obs.loc[control_mask].copy()
    case_group = obs.loc[case_mask].copy()
    raw_control = np.asarray(exp_raw[control_mask, :], dtype=np.float32)
    raw_case = np.asarray(exp_raw[case_mask, :], dtype=np.float32)
    smooth_control = np.asarray(exp_smooth[control_mask, :], dtype=np.float32)
    smooth_case = np.asarray(exp_smooth[case_mask, :], dtype=np.float32)

    if len(control_group) == 0 or len(case_group) == 0 or len(control_group) == len(case_group):
        balanced_obs = pd.concat([control_group, case_group], ignore_index=True)
        balanced_obs.index = pd.Index(balanced_obs["original_cell_id"].astype(str), name="cell_id")
        balanced_raw = np.vstack([raw_control, raw_case]).astype(np.float32, copy=False)
        balanced_smooth = np.vstack([smooth_control, smooth_case]).astype(np.float32, copy=False)
        return balanced_obs.copy(), balanced_raw, balanced_smooth

    rng = np.random.RandomState(random_state)
    if len(control_group) > len(case_group):
        selected_ids = rng.choice(control_group.index.to_numpy(), size=len(case_group), replace=False)
        control_group = control_group.loc[selected_ids].copy()
        # Match the legacy output-handler semantics, including its matrix reindex path.
        raw_control = raw_control[control_group.index.get_indexer(control_group.index)]
        smooth_control = smooth_control[control_group.index.get_indexer(control_group.index)]
    elif len(control_group) < len(case_group):
        selected_ids = rng.choice(case_group.index.to_numpy(), size=len(control_group), replace=False)
        case_group = case_group.loc[selected_ids].copy()
        raw_case = raw_case[case_group.index.get_indexer(case_group.index)]
        smooth_case = smooth_case[case_group.index.get_indexer(case_group.index)]

    balanced_obs = pd.concat([control_group, case_group], ignore_index=True)
    balanced_obs.index = pd.Index(balanced_obs["original_cell_id"].astype(str), name="cell_id")
    balanced_raw = np.vstack([raw_control, raw_case]).astype(np.float32, copy=False)
    balanced_smooth = np.vstack([smooth_control, smooth_case]).astype(np.float32, copy=False)
    return balanced_obs.copy(), balanced_raw, balanced_smooth


def _build_reconstructed_result_space(
    *,
    gene_names: pd.Index,
    response_identity: pd.Series | None,
    merged_obs: pd.DataFrame,
    merged_raw: np.ndarray,
    merged_smooth: np.ndarray,
    random_state: int = 123,
    cell_k: int = 10,
    n_pcs: int = 50,
    smooth_weight: float = 0.15,
    smooth_alpha: float = 0.1,
    binary_delta_threshold: float = 0.1,
    compute_gene_pair: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, sparse.csr_matrix, np.ndarray]:
    if compute_gene_pair:
        try:
            from couture.steps.f2_convertor import FishConvertor

            fs_values = (
                response_identity.reindex(gene_names).astype(np.int8)
                if response_identity is not None
                else pd.Series(np.ones(len(gene_names), dtype=np.int8), index=gene_names)
            )
            fs_df = pd.DataFrame({"fs_input": fs_values}, index=gene_names)
            control_ids = merged_obs.loc[merged_obs["group_label"] == "control"].index.astype(str).to_numpy()
            case_ids = merged_obs.loc[merged_obs["group_label"] == "case"].index.astype(str).to_numpy()
            coca = FishConvertor(
                iter_round=20,
                control_cindex=control_ids,
                case_cindex=case_ids,
                fs_index=fs_df,
                exp_raw=np.asarray(merged_raw, dtype=np.float32),
                exp_last=np.asarray(merged_smooth, dtype=np.float32),
                label_last=merged_obs["label_mixed"],
                alpha_signal=0.15,
                guide_fs=None,
                silence_print=True,
            )
            coca._construct_cell_dc(fs_select="fs_input", dc_method="pca")
            coca._construct_cell_graph(
                fs_select="fs_input",
                distance_metric="cosine",
                knn_method="aknn",
                n_neighbors=10,
                symmetric_bool=True,
                force_connect_bool=True,
                connect_method="mst",
                affinity_method="scanpy_gaussian",
            )
            coca._process_signal_by_cell_affinity(
                fs_select="fs_input",
                smooth_method="heat_kernel",
                smooth_strength=0.1,
                top_k=-1,
                update_fs=False,
            )
            coca._continuize_label(alpha=0.1)
            coca._link_gene_pair(cor_method="moran")
            return (
                np.asarray(coca.ad.layers["exp_mixed_by_cell_graph"], dtype=np.float32),
                np.asarray(coca.ad.obs["label_mixed"], dtype=np.float32),
                np.asarray(coca.ad.varp["cor"], dtype=np.float32),
                np.asarray(coca.ad.var["gene_self_cor"], dtype=np.float32),
                sparse.csr_matrix(np.asarray(coca.ad.obsp["affinity"], dtype=np.float32)),
                np.asarray(coca.ad.var["gene_label_cor"], dtype=np.float32),
            )
        except Exception:
            pass
    fs_mask = (
        response_identity.reindex(gene_names).to_numpy(dtype=np.int8) == 1
        if response_identity is not None
        else np.ones(len(gene_names), dtype=bool)
    )
    if not np.any(fs_mask):
        fs_mask = np.ones(len(gene_names), dtype=bool)

    exp_last = np.asarray(merged_smooth, dtype=np.float32)
    exp_raw = np.asarray(merged_raw, dtype=np.float32)
    label_raw = merged_obs["label_raw"].to_numpy(dtype=np.float32)
    label_last = merged_obs["label_mixed"].to_numpy(dtype=np.float32)

    cell_dc = pca_select(exp_last[:, fs_mask], n_pcs=min(n_pcs, max(1, int(np.sum(fs_mask)))))
    dist = compute_distance_cosine(cell_dc, normalized=True)
    adjacency = adaptive_knn(dist, n_neighbors=cell_k, delta=-0.5)
    affinity_dense = scanpy_gaussian_weighting(dist, adjacency, k=cell_k).astype(np.float32, copy=False)
    exp_denoised = heat_kernel_smoothing(affinity_dense, exp_last, beta=smooth_alpha).astype(np.float32, copy=False)
    exp_mixed = overlay_raw_exp(
        exp_raw=exp_raw,
        exp_last=exp_last,
        f_exp_last=exp_denoised,
        iter_round=20,
        weight=1 - smooth_weight,
        mode="raw",
    ).astype(np.float32, copy=False)
    label_smooth = label_nh_prop_moran(
        exp_mat=exp_mixed,
        adj_mat=affinity_dense,
        label_raw=label_raw,
        label_last=label_last,
        fs_mask=fs_mask.astype(np.int8),
    ).astype(np.float32, copy=False)
    label_mixed = overlay_raw_exp(
        exp_raw=label_raw,
        exp_last=label_last,
        f_exp_last=label_smooth,
        iter_round=20,
        weight=smooth_weight,
        mode="decrease",
    ).astype(np.float32, copy=False)

    exp_input = exp_denoised.copy()
    case_cells = label_raw == 1
    control_cells = label_raw == 0
    if np.any(case_cells) and np.any(control_cells):
        exp_median = np.median(exp_input, axis=0)
        case_exp = exp_input[case_cells]
        case_weights = label_smooth[case_cells]
        if float(np.sum(case_weights)) == 0.0:
            case_avg = np.mean(case_exp, axis=0)
        else:
            case_avg = np.average(case_exp, axis=0, weights=case_weights)
        control_avg = np.mean(exp_input[control_cells], axis=0)
        low_delta = np.abs(case_avg - control_avg) < binary_delta_threshold
        if np.any(low_delta):
            exp_input[:, low_delta] = (exp_input[:, low_delta] >= exp_median[low_delta]).astype(np.float32)

    if affinity_dense[0, 0] != 0:
        affinity_dense = affinity_dense.copy()
        np.fill_diagonal(affinity_dense, 0)
    constant = exp_input.shape[0] / np.sum(affinity_dense)
    centered = exp_input - exp_input.mean(axis=0)
    weighted = affinity_dense @ centered
    denom = np.sum(centered * centered, axis=0)
    gene_self_cor = np.nan_to_num(constant * (np.sum(centered * weighted, axis=0) / denom))
    gene_self_cor = np.clip(gene_self_cor, -1, 1).astype(np.float32, copy=False)

    gene_cor = None
    if compute_gene_pair:
        gene_cor = _compute_sp_moran_between(exp_input, exp_input, affinity_dense.copy()).astype(np.float32, copy=False)
        gene_self_cor = np.diag(gene_cor).astype(np.float32, copy=False)
    gene_label_cor = _compute_sp_moran_between(
        exp_input,
        label_smooth[:, np.newaxis],
        affinity_dense.copy(),
    ).reshape(-1).astype(np.float32)
    affinity = sparse.csr_matrix(affinity_dense.astype(np.float32, copy=False))
    return exp_mixed, label_mixed, gene_cor, gene_self_cor, affinity, gene_label_cor


def _build_result_space_via_exact_merge(
    *,
    gene_names: pd.Index,
    gene_scores: pd.Series | None,
    response_identity: pd.Series | None,
    sample_outputs: tuple[Any, ...],
    sample_packs: tuple[Any, ...],
    selected_indices: list[int],
    random_state: int = 123,
) -> ad.AnnData | None:
    try:
        from couture.steps.c6_output_handler import merge_crane
    except Exception:
        return None

    class _SampleObj:
        def __init__(self, adata_obj: ad.AnnData) -> None:
            self.ad = adata_obj

        def copy(self) -> "_SampleObj":
            return _SampleObj(self.ad.copy())

    class _FishObj:
        def __init__(self, results_coca: list[_SampleObj], fs_update: pd.DataFrame, mean_delta: pd.Series) -> None:
            self.results_coca = results_coca
            self.fs_update = fs_update
            self.mean_delta = mean_delta
            self.results_coco = results_coca

    class _CIter:
        def __init__(self, representative_samples: list[int], fish_obj: _FishObj) -> None:
            self.representative_samples = tuple(representative_samples)
            self.fish_obj = fish_obj

    class _CraneObj:
        pass

    raw_lookup: dict[str, np.ndarray] = {}
    sample_objects: list[_SampleObj] = []
    for sample_idx in selected_indices:
        pack = sample_packs[sample_idx]
        output = sample_outputs[sample_idx]
        ordered_cells = list(pack.control_cells) + list(pack.case_cells)
        obs = pd.DataFrame(index=pd.Index(ordered_cells))
        obs["cell_id"] = ordered_cells
        obs["group_id"] = ["control"] * len(pack.control_cells) + ["case"] * len(pack.case_cells)
        obs["label_raw"] = np.asarray(pack.label_raw, dtype=np.float32)
        obs["label_last"] = np.asarray(output.label_last_next, dtype=np.float32)
        obs["label_smooth"] = np.asarray(output.label_last_next, dtype=np.float32)
        obs["label_mixed"] = np.asarray(output.label_last_next, dtype=np.float32)
        exp_raw = np.asarray(pack.exp_raw, dtype=np.float32)
        for cell_id, row in zip(ordered_cells, exp_raw, strict=False):
            raw_lookup[str(cell_id)] = row
        sample_ad = ad.AnnData(
            X=np.asarray(output.exp_last_next, dtype=np.float32),
            obs=obs.copy(),
            var=pd.DataFrame(index=gene_names.copy()),
        )
        sample_ad.uns["crane_info"] = {
            "expression_matrix": "X",
            "expression_semantics": "exp_mixed_by_cell_graph",
        }
        sample_ad.layers["exp_mixed_by_gene_graph"] = np.asarray(sample_ad.X, dtype=np.float32)
        sample_objects.append(_SampleObj(sample_ad))

    if not sample_objects:
        return None

    fs_update = pd.DataFrame(
        {
            "New_fs": (
                response_identity.reindex(gene_names).astype(np.int8)
                if response_identity is not None
                else np.ones(len(gene_names), dtype=np.int8)
            )
        },
        index=gene_names.copy(),
    )
    mean_delta = (
        gene_scores.reindex(gene_names).astype(np.float32)
        if gene_scores is not None
        else pd.Series(np.zeros(len(gene_names), dtype=np.float32), index=gene_names.copy())
    )

    all_cells = list(raw_lookup.keys())
    raw_mat = np.vstack([raw_lookup[cell_id] for cell_id in all_cells]).astype(np.float32, copy=False)
    adata_strong = ad.AnnData(
        raw_mat.copy(),
        obs=pd.DataFrame(index=all_cells),
        var=pd.DataFrame(index=gene_names.copy()),
    )
    adata_strong.layers["central_norm"] = raw_mat.copy()

    crane_obj = _CraneObj()
    crane_obj.C_iter = _CIter(
        representative_samples=list(range(len(selected_indices))),
        fish_obj=_FishObj(sample_objects, fs_update=fs_update, mean_delta=mean_delta),
    )
    crane_obj.input = {
        "adata_strong": adata_strong,
        "function_gene_set": None,
        "function_gene_vector": None,
        "function_cell_vector": None,
        "thres_k": 2,
    }

    rng_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        with redirect_stdout(StringIO()):
            coca, _ = merge_crane(
                crane_obj,
                deduplicate_bool=True,
                balanced_bool=True,
                module_method="leiden",
                eva_alpha=0.5,
                feature_mode="self_n_label",
                adj_bc_w=0,
                function_gene_set=None,
                function_gene_vector=None,
                function_cell_vector=None,
            )
    finally:
        np.random.set_state(rng_state)

    result_ad = coca.ad.copy()
    result_ad.obs = result_ad.obs.rename(columns={"group_id": "group_label"}).copy()
    result_ad.obs["original_cell_id"] = result_ad.obs_names.astype(str)
    result_ad.obs["selected_sample_count"] = 1
    result_ad.obs["selected_sample_ids"] = [tuple(
        sample_outputs[idx].metadata.get("sample_id") or sample_packs[idx].sample_id or f"sample_{idx}"
        for idx in selected_indices
    )] * result_ad.n_obs
    if gene_scores is not None:
        result_ad.var["response_score"] = gene_scores.reindex(gene_names).astype(np.float32)
    if response_identity is not None:
        result_ad.var["response_identity"] = response_identity.reindex(gene_names).astype(np.int8)
    return result_ad


def _calculate_deviation_guarded(values: pd.Series | np.ndarray, floor: float = 0.05) -> float:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return float(floor)
    data_mad = float(np.median(np.abs(data - np.median(data))))
    data_std = float(np.std(data))
    if data_mad < floor:
        if floor < data_std < floor * 3:
            return data_std
        return float(floor)
    if data_mad < floor * 3:
        return data_mad
    if data_std > floor * 3:
        return float(min(data_mad, 0.2))
    return data_std


def _compute_history_merged_response_score(
    *,
    gene_scores: pd.Series,
    step2_result: Any | None,
) -> tuple[pd.Series, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": "final_iteration",
        "history_merge_enabled": True,
        "history_merge_applied": False,
        "history_rounds": 0,
        "history_merge_reason": None,
    }
    if step2_result is None:
        meta["history_merge_reason"] = "missing_step2_result"
        return gene_scores, meta

    score_history = tuple(getattr(step2_result, "score_history", ()) or ())
    ri_history = tuple(getattr(step2_result, "ri_history", ()) or ())
    usable_rounds = min(len(score_history), max(len(ri_history) - 1, 0))
    if usable_rounds < 1:
        meta["history_merge_reason"] = "missing_history"
        return gene_scores, meta

    weights = np.arange(1, usable_rounds + 1, dtype=np.float32)
    weights /= float(weights.sum())
    merged = np.zeros(len(gene_scores), dtype=np.float32)
    for round_idx in range(usable_rounds):
        score = np.asarray(score_history[round_idx], dtype=np.float32).reshape(-1)
        ri_mask = np.asarray(ri_history[round_idx + 1], dtype=np.int8).reshape(-1)
        if score.shape[0] != len(gene_scores) or ri_mask.shape[0] != len(gene_scores):
            meta["history_merge_reason"] = "shape_mismatch"
            return gene_scores, meta
        merged += np.where(ri_mask == 1, score, 0.0).astype(np.float32, copy=False) * weights[round_idx]

    meta.update(
        {
            "source": "legacy_history_merge",
            "history_merge_applied": True,
            "history_rounds": int(usable_rounds),
            "history_merge_reason": None,
        }
    )
    return pd.Series(merged, index=gene_scores.index.copy(), name=gene_scores.name), meta


def resolve_public_response_score(
    *,
    gene_scores: pd.Series,
    step2_result: Any | None = None,
    enable_history_merge: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    if not enable_history_merge:
        return gene_scores, {
            "source": "final_iteration",
            "history_merge_enabled": False,
            "history_merge_applied": False,
            "history_rounds": 0,
            "history_merge_reason": None,
        }
    return _compute_history_merged_response_score(
        gene_scores=gene_scores,
        step2_result=step2_result,
    )


def _summarize_nondeg_background(
    *,
    gene_scores: pd.Series | None,
    response_identity: pd.Series | None,
) -> dict[str, Any] | None:
    if gene_scores is None or response_identity is None:
        return None

    score = gene_scores.astype(np.float32).copy()
    ri = response_identity.reindex(score.index).fillna(0).astype(np.int8)
    nondeg_score = score.loc[ri == 0].astype(np.float32)
    if len(nondeg_score) == 0:
        return {
            "available": False,
            "gene_count": 0,
            "median": 0.0,
            "mad": 0.05,
        }
    return {
        "available": True,
        "gene_count": int(len(nondeg_score)),
        "median": float(nondeg_score.median()),
        "mad": float(_calculate_deviation_guarded(nondeg_score)),
    }


def _postprocess_result_gene_outputs(
    *,
    gene_scores: pd.Series | None,
    response_identity: pd.Series | None,
    refine_k: float = 3.0,
    threshold_k: float = 2.0,
) -> tuple[pd.Series | None, pd.Series | None]:
    if gene_scores is None:
        return None, response_identity

    score = gene_scores.astype(np.float32).copy()
    score[score < 0] = 0
    if response_identity is None:
        return score, None

    ri = response_identity.astype(np.int8).copy()
    nondeg_score = gene_scores.loc[ri == 0].astype(np.float32)
    background_median = float(nondeg_score.median()) if len(nondeg_score) else 0.0
    background_mad = _calculate_deviation_guarded(nondeg_score) if len(nondeg_score) else 0.05
    refine_threshold = refine_k * threshold_k * background_mad + background_median
    refined = (score >= refine_threshold).astype(np.int8)
    refined_ri = (ri | refined).astype(np.int8)
    return score, refined_ri


def _resolve_result_space_identity(
    *,
    gene_names: pd.Index,
    response_identity: pd.Series | None,
    step2_result: Any | None,
    graph_fs_mode: str = "default",
    sample_outputs: tuple[Any, ...] = (),
    sample_packs: tuple[Any, ...] = (),
    selected_indices: list[int] | None = None,
) -> tuple[pd.Series | None, dict[str, Any] | None, dict[int, dict[str, np.ndarray]] | None]:
    if response_identity is None:
        return None, None, None

    base_ri = response_identity.reindex(gene_names).astype(np.int8)
    if graph_fs_mode not in {"handoff", "legacy_history_rescue"} or step2_result is None:
        return base_ri, None, None

    result_space_identity = getattr(step2_result, "result_space_identity", None)
    handoff_meta = None
    if hasattr(step2_result, "metadata"):
        handoff_meta = step2_result.metadata.get("result_space_handoff")
    if result_space_identity is None:
        return base_ri, {"mode": graph_fs_mode, "fallback": "missing_handoff"}, None

    if (
        graph_fs_mode == "handoff"
        and sample_outputs
        and sample_packs
        and selected_indices
        and getattr(step2_result, "state", None) is not None
    ):
        try:
            from couture.implementations.feature.feature_divide import deg_classification
            from couture.steps.f2_convertor import FishConvertor

            state = step2_result.state
            current_guides = list(getattr(state, "guide_pass_list", ()))
            if len(current_guides) == len(sample_packs):
                guide_mask = getattr(sample_packs[0], "guide_fs_mask", None)
                if guide_mask is not None:
                    guide_fs_df = pd.DataFrame(
                        {"fs_guide": pd.Series(np.asarray(guide_mask, dtype=np.int8), index=gene_names.copy())},
                        index=gene_names.copy(),
                    )

                    def _run_exact_sample(sample_idx: int, *, input_guide: bool) -> dict[str, Any]:
                        pack = sample_packs[sample_idx]
                        exp_last = np.asarray(state.exp_last_list[sample_idx], dtype=np.float32)
                        label_last = np.asarray(state.label_last_list[sample_idx], dtype=np.float32)
                        fs_df = pd.DataFrame({"fs_input": base_ri.copy()}, index=gene_names.copy())
                        coca = FishConvertor(
                            iter_round=int(getattr(state, "iteration", 20) or 20),
                            control_cindex=list(pack.control_cells),
                            case_cindex=list(pack.case_cells),
                            fs_index=fs_df,
                            exp_raw=np.asarray(pack.exp_raw, dtype=np.float32),
                            exp_last=exp_last,
                            label_last=label_last,
                            alpha_signal=0.15,
                            guide_fs=guide_fs_df,
                            silence_print=True,
                        )
                        coca._construct_cell_dc(fs_select="fs_input", dc_method="pca")
                        coca._construct_cell_graph(
                            fs_select="fs_input",
                            distance_metric="cosine",
                            knn_method="aknn",
                            n_neighbors=10,
                            symmetric_bool=True,
                            force_connect_bool=True,
                            connect_method="mst",
                            affinity_method="scanpy_gaussian",
                        )
                        if not input_guide:
                            coca._construct_cell_dc(fs_select="fs_guide", dc_method="pca")
                            coca._construct_cell_graph(
                                fs_select="fs_guide",
                                distance_metric="cosine",
                                knn_method="aknn",
                                n_neighbors=10,
                                symmetric_bool=True,
                                force_connect_bool=True,
                                connect_method="mst",
                                affinity_method="scanpy_gaussian",
                            )
                            coca._process_signal_by_cell_affinity(
                                fs_select="fs_guide",
                                smooth_method="heat_kernel",
                                smooth_strength=0.1,
                                top_k=-1,
                            )
                        else:
                            coca._process_signal_by_cell_affinity(
                                fs_select="fs_input",
                                smooth_method="heat_kernel",
                                smooth_strength=0.1,
                                top_k=-1,
                            )
                        coca._continuize_label(fs_select="fs_input", alpha=0.1, strict_label=True)
                        if not input_guide:
                            coca._continuize_label(fs_select="fs_guide", alpha=0.1, strict_label=True)
                            coca._compare_cell_graph()
                        coca._link_gene_pair(cor_method="moran")

                        gene_self = coca.ad.var["gene_self_cor"].astype(np.float32)
                        gene_label = coca.ad.var["gene_label_cor"].astype(np.float32)
                        combined = (np.sqrt((gene_self**2) + (gene_label**2)) / np.sqrt(2)).astype(np.float32)
                        background = combined.loc[coca.ad.var["fs_input"] == 0]
                        background_center = float(np.median(background.to_numpy(dtype=np.float32))) if len(background) else 0.0
                        norm_score = (combined - background_center).astype(np.float32).reindex(gene_names.copy())
                        return {
                            "norm_score": norm_score,
                            "bool_compare": bool(coca.ad.uns.get("bool_compare", True)),
                            "exp_mixed": np.asarray(coca.ad.layers["exp_mixed_by_cell_graph"], dtype=np.float32),
                            "label_mixed": coca.ad.obs["label_mixed"].to_numpy(dtype=np.float32),
                        }

                    exact_cache: dict[tuple[int, bool], dict[str, Any]] = {}
                    rescue_false_indices: list[int] = []
                    for sample_idx in selected_indices:
                        if sample_idx >= len(current_guides) or not bool(current_guides[sample_idx]):
                            continue
                        probe = _run_exact_sample(sample_idx, input_guide=False)
                        exact_cache[(sample_idx, False)] = probe
                        if not probe["bool_compare"]:
                            rescue_false_indices.append(int(sample_idx))

                    exact_scores: list[np.ndarray] = []
                    sample_overrides: dict[int, dict[str, np.ndarray]] = {}
                    for sample_idx in range(len(sample_packs)):
                        use_false_guide = sample_idx in rescue_false_indices
                        cache_key = (sample_idx, False if use_false_guide else True)
                        exact = exact_cache.get(cache_key)
                        if exact is None:
                            exact = _run_exact_sample(sample_idx, input_guide=not use_false_guide)
                            exact_cache[cache_key] = exact
                        exact_scores.append(exact["norm_score"].to_numpy(dtype=np.float32))
                        if sample_idx in selected_indices:
                            sample_overrides[int(sample_idx)] = {
                                "exp_last_next": exact["exp_mixed"],
                                "label_last_next": exact["label_mixed"],
                            }

                    mean_delta = pd.Series(
                        np.mean(np.vstack(exact_scores), axis=0, dtype=np.float32),
                        index=gene_names.copy(),
                    )
                    refer = mean_delta.loc[base_ri == 0]
                    if len(refer):
                        exact_fs = deg_classification(refer=refer, series=mean_delta, mad_k=2.0)["New_fs"]
                        exact_fs = (exact_fs.reindex(gene_names.copy()).fillna(0).astype(np.int8) & base_ri).astype(np.int8)
                        return (
                            exact_fs,
                            {
                                "mode": graph_fs_mode,
                                "resolver": "exact_sample_handoff",
                                "active_genes": int(exact_fs.sum()),
                                "rescue_false_sample_indices": tuple(int(v) for v in rescue_false_indices),
                                "handoff_meta": handoff_meta,
                                "fallback": None,
                            },
                            sample_overrides,
                        )
        except Exception:
            pass

    handoff_series = pd.Series(
        np.asarray(result_space_identity, dtype=np.int8),
        index=gene_names,
        name="result_space_identity",
    )
    return handoff_series, {
        "mode": graph_fs_mode,
        "active_genes": int(handoff_series.sum()),
        "fallback": None,
        "handoff_meta": handoff_meta,
    }, None


def build_result_anndata(
    *,
    gene_names: pd.Index,
    gene_scores: pd.Series | None,
    response_identity: pd.Series | None,
    sample_outputs: tuple[Any, ...] = (),
    sample_packs: tuple[Any, ...] = (),
    metadata: dict[str, Any] | None = None,
    merge_top_n: int = 3,
    merge_mode: str = "representative",
    graph_method: str = "gauss",
    compute_gene_pair: bool = False,
    random_state: int = 123,
    step2_result: Any | None = None,
    graph_fs_mode: str = "default",
) -> ad.AnnData | None:
    """Build a compact post-Step2 result-space AnnData for downstream work."""

    metadata = metadata or {}
    if not sample_outputs or not sample_packs:
        return None
    exported_score, exported_identity = _postprocess_result_gene_outputs(
        gene_scores=gene_scores,
        response_identity=response_identity,
        refine_k=float(metadata.get("post_step2_refine_k", 3.0)),
        threshold_k=float(metadata.get("step2_threshold_k", 2.0)),
    )
    nondeg_background = _summarize_nondeg_background(
        gene_scores=exported_score,
        response_identity=exported_identity,
    )
    combined_score_df = pd.DataFrame(
        {
            (output.metadata.get("sample_id") or pack.sample_id or f"sample_{idx}"): np.asarray(
                output.norm_combined_score,
                dtype=np.float32,
            )
            for idx, (pack, output) in enumerate(zip(sample_packs, sample_outputs, strict=False))
        },
        index=gene_names.copy(),
    )
    if merge_mode == "all":
        selected_indices = list(range(len(sample_outputs)))
    elif merge_mode == "legacy_representative":
        selected_indices = [
            int(idx)
            for idx in getattr(step2_result, "representative_sample_indices", ())[: min(int(merge_top_n), len(sample_outputs))]
        ]
        if not selected_indices:
            selected_indices = list(range(min(int(merge_top_n), len(sample_outputs))))
    else:
        selected_indices = _select_representative_samples(combined_score_df, n=merge_top_n)

    result_space_identity, graph_fs_meta, sample_overrides = _resolve_result_space_identity(
        gene_names=gene_names,
        response_identity=response_identity,
        step2_result=step2_result,
        graph_fs_mode=graph_fs_mode,
        sample_outputs=sample_outputs,
        sample_packs=sample_packs,
        selected_indices=selected_indices,
    )

    raw_blocks: list[np.ndarray] = []
    smooth_blocks: list[np.ndarray] = []
    obs_frames: list[pd.DataFrame] = []

    merge_indices = list(selected_indices)

    for sample_idx in merge_indices:
        pack = sample_packs[sample_idx]
        output = sample_outputs[sample_idx]
        raw_blocks.append(np.asarray(pack.exp_raw, dtype=np.float32))
        sample_override = sample_overrides.get(sample_idx) if sample_overrides else None
        smooth_blocks.append(
            np.asarray(
                sample_override["exp_last_next"] if sample_override is not None else output.exp_last_next,
                dtype=np.float32,
            )
        )
        label_raw = np.asarray(pack.label_raw, dtype=np.float32).reshape(-1)
        label_mixed = np.asarray(
            sample_override["label_last_next"] if sample_override is not None else output.label_last_next,
            dtype=np.float32,
        ).reshape(-1)

        control_cells = list(pack.control_cells)
        case_cells = list(pack.case_cells)
        ordered_cells = control_cells + case_cells
        group_labels = ["control"] * len(control_cells) + ["case"] * len(case_cells)
        sample_id = output.metadata.get("sample_id") or pack.sample_id or f"sample_{sample_idx}"
        obs_frames.append(
            pd.DataFrame(
                {
                    "sample_id": sample_id,
                    "original_cell_id": ordered_cells,
                    "group_label": group_labels,
                    "label_raw": label_raw,
                    "label_mixed": label_mixed,
                    "sample_cell_index": np.arange(len(ordered_cells), dtype=np.int32),
                },
                index=[f"{sample_id}:{cell_id}" for cell_id in ordered_cells],
            )
        )

    stacked_raw = np.vstack(raw_blocks).astype(np.float32, copy=False)
    stacked_smooth = np.vstack(smooth_blocks).astype(np.float32, copy=False)
    stacked_obs = pd.concat(obs_frames, axis=0)
    exact_result_ad = (
        _build_result_space_via_exact_merge(
            gene_names=gene_names,
            gene_scores=exported_score,
            response_identity=exported_identity,
            sample_outputs=sample_outputs,
            sample_packs=sample_packs,
            selected_indices=selected_indices,
            random_state=random_state,
        )
        if compute_gene_pair
        else None
    )
    if exact_result_ad is not None:
        if exact_result_ad.X is not None:
            exact_result_ad.X = np.asarray(exact_result_ad.X, dtype=np.float32)
        keep_obs_cols = [col for col in ("sample_id", "group_label", "label_raw", "label_last", "original_cell_id") if col in exact_result_ad.obs.columns]
        exact_result_ad.obs = exact_result_ad.obs.loc[:, keep_obs_cols].copy()
        keep_var_cols = [col for col in ("response_score", "response_identity", "gene_self_cor", "gene_label_cor") if col in exact_result_ad.var.columns]
        exact_result_ad.var = exact_result_ad.var.loc[:, keep_var_cols].copy()
        exact_result_ad.uns["crane_info"] = {
            "kind": "crane_result_space",
            "result_space": "post_step2_merge",
            "sample_count": int(len(sample_outputs)),
            "gene_pair_cor_available": bool("cor" in exact_result_ad.varp),
            "representative_sample_ids": tuple(
                (sample_outputs[idx].metadata.get("sample_id") or sample_packs[idx].sample_id or f"sample_{idx}")
                for idx in selected_indices
            ),
            "balanced_after_dedup": True,
            "expression_matrix": "X" if exact_result_ad.X is not None else None,
            "expression_semantics": "exp_mixed_by_cell_graph" if exact_result_ad.X is not None else None,
            "response_score_source": metadata.get("response_score_source", "final_iteration"),
            "nondeg_background": nondeg_background,
            "run": metadata,
        }
        return exact_result_ad

    merged_obs, merged_raw, merged_smooth = _merge_selected_samples_exact(
        stacked_obs=stacked_obs,
        stacked_raw=stacked_raw,
        stacked_smooth=stacked_smooth,
        random_state=random_state,
    )
    if result_space_identity is not None:
        rebuilt_exp, rebuilt_label, gene_cor, gene_self_cor, affinity, gene_label_cor = _build_reconstructed_result_space(
            gene_names=gene_names,
            response_identity=result_space_identity,
            merged_obs=merged_obs,
            merged_raw=merged_raw,
            merged_smooth=merged_smooth,
            random_state=random_state,
            compute_gene_pair=compute_gene_pair,
        )
    else:
        rebuilt_exp, rebuilt_label, gene_cor, gene_self_cor, affinity, gene_label_cor = _build_reconstructed_result_space(
            gene_names=gene_names,
            response_identity=exported_identity,
            merged_obs=merged_obs,
            merged_raw=merged_raw,
            merged_smooth=merged_smooth,
            random_state=random_state,
            compute_gene_pair=compute_gene_pair,
        )
    result_obs = merged_obs.copy()
    result_obs["label_last"] = rebuilt_label
    keep_obs_cols = [col for col in ("sample_id", "group_label", "label_raw", "label_last", "original_cell_id") if col in result_obs.columns]
    result_obs = result_obs.loc[:, keep_obs_cols].copy()
    result_ad = ad.AnnData(
        X=np.asarray(rebuilt_exp, dtype=np.float32),
        obs=result_obs,
        var=pd.DataFrame(index=gene_names.copy()),
    )

    if exported_score is not None:
        result_ad.var["response_score"] = exported_score.reindex(gene_names).astype(np.float32)
    if exported_identity is not None:
        result_ad.var["response_identity"] = exported_identity.reindex(gene_names).astype(np.int8)
    result_ad.obsp["affinity"] = affinity
    if compute_gene_pair and gene_cor is not None and result_ad.n_obs > 1 and result_ad.n_vars > 0:
        result_ad.varp["cor"] = gene_cor.astype(np.float32, copy=False)
    result_ad.var["gene_self_cor"] = gene_self_cor
    result_ad.var["gene_label_cor"] = gene_label_cor

    result_ad.uns["crane_info"] = {
        "kind": "crane_result_space",
        "result_space": "post_step2_merge",
        "sample_count": int(len(sample_outputs)),
        "gene_pair_cor_available": bool("cor" in result_ad.varp),
        "representative_sample_ids": tuple(
            (sample_outputs[idx].metadata.get("sample_id") or sample_packs[idx].sample_id or f"sample_{idx}")
            for idx in selected_indices
        ),
        "balanced_after_dedup": True,
        "expression_matrix": "X",
        "expression_semantics": "exp_mixed_by_cell_graph",
        "response_score_source": metadata.get("response_score_source", "final_iteration"),
        "nondeg_background": nondeg_background,
        "graph_fs_mode": graph_fs_mode,
        "graph_fs_active_genes": int(result_space_identity.sum()) if result_space_identity is not None else None,
        "graph_fs_meta": graph_fs_meta,
        "run": metadata,
    }
    return result_ad


@dataclass
class CRANEGenePairResult:
    """Minimal downstream gene-pair correlation result on a CRANE gene-response result space."""

    pair_ad: ad.AnnData
    source_result_ad: ad.AnnData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_ad(self) -> ad.AnnData:
        return self.pair_ad

    def __repr__(self) -> str:
        return f"CRANE gene_pair completed with {self.pair_ad.n_vars} gene(s)."

    __str__ = __repr__

    def correlation_matrix(
        self,
        *,
        as_dataframe: bool = True,
        copy: bool = True,
    ) -> pd.DataFrame | np.ndarray:
        cor = np.asarray(self.pair_ad.varp["cor"], dtype=np.float32)
        if copy:
            cor = cor.copy()
        if not as_dataframe:
            return cor
        gene_names = self.pair_ad.var_names.astype(str)
        return pd.DataFrame(cor, index=gene_names, columns=gene_names)

    def to_anndata(self, copy: bool = True) -> ad.AnnData:
        return self.pair_ad.copy() if copy else self.pair_ad


@dataclass
class CRANEGeneModuleResult:
    """Lazy downstream gene-module analysis on a CRANE gene-response result space."""

    module_ad: ad.AnnData
    source_result_ad: ad.AnnData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_ad(self) -> ad.AnnData:
        return self.module_ad

    def __repr__(self) -> str:
        return f"CRANE gene_module completed with {self.module_ad.n_vars} module(s)."

    __str__ = __repr__

    def summary(
        self,
        *,
        sort_by: str = "module_response_score",
        ascending: bool = False,
    ) -> pd.DataFrame:
        frame = self.module_ad.var.copy()
        frame.index.name = "module"
        if sort_by in frame.columns:
            frame = frame.sort_values(by=sort_by, ascending=ascending)
        return frame

    def genes(self, module: str) -> pd.DataFrame:
        gene_map = self.module_ad.uns.get("gene_module_map")
        if gene_map is None:
            return pd.DataFrame()
        module_key = str(module)
        frame = gene_map.loc[gene_map["module_label"].astype(str) == module_key].copy()
        frame.index.name = "gene"
        return frame

    def to_anndata(self, copy: bool = True) -> ad.AnnData:
        return self.module_ad.copy() if copy else self.module_ad


def _coerce_gene_list(genes: Any) -> list[str]:
    if genes is None:
        return []
    if isinstance(genes, str):
        return [genes]
    if isinstance(genes, pd.Series):
        return [str(v) for v in genes.tolist()]
    if isinstance(genes, pd.Index):
        return [str(v) for v in genes.tolist()]
    return [str(v) for v in list(genes)]


def _resolve_gene_scope(
    *,
    gene_names: pd.Index,
    response_identity: pd.Series | None,
    genes: Any = None,
    active_only: bool = False,
) -> tuple[pd.Index, str]:
    requested = _coerce_gene_list(genes)
    if requested:
        selected = gene_names.intersection(pd.Index(requested))
        if len(selected) == 0:
            raise ValueError("No requested genes are available in the CRANE result.")
        return selected.copy(), "custom"

    if active_only and response_identity is not None:
        active_mask = response_identity.reindex(gene_names).fillna(0).astype(np.int8) == 1
        selected = gene_names[active_mask.to_numpy()]
        if len(selected) == 0:
            raise ValueError("No active response genes are available for downstream analysis.")
        return selected.copy(), "active"

    return gene_names.copy(), "all"


def _format_gene_scope_key(selected_genes: pd.Index) -> tuple[str, ...]:
    return tuple(str(v) for v in selected_genes.tolist())


def _build_gene_response_analysis_space(result: "CRANEResult") -> dict[str, Any]:
    if result.gene_scores is None:
        raise ValueError("gene_pair()/gene_module() require a gene_response result.")

    if (
        result.result_ad is not None
        and isinstance(result.result_ad, ad.AnnData)
        and "affinity" in result.result_ad.obsp
    ):
        if result.result_ad.X is not None:
            expression = np.asarray(result.result_ad.X, dtype=np.float32)
        elif "exp_mixed_by_cell_graph" in result.result_ad.layers:
            expression = np.asarray(result.result_ad.layers["exp_mixed_by_cell_graph"], dtype=np.float32)
        else:
            expression = None
        if expression is not None:
            gene_names = pd.Index(result.result_ad.var_names.copy(), name="gene")
            response_identity = (
                result.result_ad.var["response_identity"].reindex(gene_names).fillna(0).astype(np.int8)
                if "response_identity" in result.result_ad.var.columns
                else (
                    result.response_identity.reindex(gene_names).fillna(0).astype(np.int8)
                    if result.response_identity is not None
                    else None
                )
            )
            return {
                "gene_names": gene_names,
                "response_identity": response_identity,
                "expression": expression,
                "label": (
                    result.result_ad.obs["label_last"].to_numpy(dtype=np.float32)
                    if "label_last" in result.result_ad.obs.columns
                    else result.result_ad.obs["label_raw"].to_numpy(dtype=np.float32)
                ),
                "affinity": sparse.csr_matrix(result.result_ad.obsp["affinity"]),
                "obs": result.result_ad.obs.copy(),
                "var": result.result_ad.var.copy(),
                "representative_sample_indices": tuple(
                    int(v)
                    for v in getattr(result.metadata.get("step2"), "representative_sample_indices", ())
                ),
            }

    step1_result = result.metadata.get("step1")
    step2_result = result.metadata.get("step2")
    if step1_result is None or step2_result is None:
        raise ValueError(
            "gene_pair()/gene_module() require the full gene_response result context from crane.tl.gene_response(...)."
        )

    from ..step2.runner import prepare_step2_packs

    sample_layer = result.metadata.get("step2_summary", {}).get("sample_layer")
    sample_packs = prepare_step2_packs(
        step1_result.sampling_plan,
        fs_input=step1_result.sampling_plan.init_feature_selection,
        guide_fs_input=step1_result.sampling_plan.guide_feature_selection,
        sample_layer=sample_layer,
    )
    sample_outputs = tuple(step2_result.sample_outputs)
    if not sample_packs or not sample_outputs:
        raise ValueError("The gene_response result does not carry Step2 sample outputs for downstream analysis.")

    gene_names = pd.Index(result.gene_scores.index.copy(), name="gene")
    if result.result_ad is not None and hasattr(result.result_ad, "var") and "response_identity" in result.result_ad.var:
        response_identity = result.result_ad.var["response_identity"].reindex(gene_names).fillna(0).astype(np.int8)
    elif result.response_identity is not None:
        response_identity = result.response_identity.reindex(gene_names).fillna(0).astype(np.int8)
    else:
        response_identity = None

    selected_indices = [
        int(idx) for idx in getattr(step2_result, "representative_sample_indices", ())[: min(3, len(sample_outputs))]
    ]
    if not selected_indices:
        selected_indices = list(range(min(3, len(sample_outputs))))

    raw_blocks: list[np.ndarray] = []
    smooth_blocks: list[np.ndarray] = []
    obs_frames: list[pd.DataFrame] = []
    for sample_idx in selected_indices:
        pack = sample_packs[sample_idx]
        output = sample_outputs[sample_idx]
        raw_blocks.append(np.asarray(pack.exp_raw, dtype=np.float32))
        smooth_blocks.append(np.asarray(output.exp_last_next, dtype=np.float32))
        label_raw = np.asarray(pack.label_raw, dtype=np.float32).reshape(-1)
        label_mixed = np.asarray(output.label_last_next, dtype=np.float32).reshape(-1)
        control_cells = list(pack.control_cells)
        case_cells = list(pack.case_cells)
        ordered_cells = control_cells + case_cells
        sample_id = output.metadata.get("sample_id") or pack.sample_id or f"sample_{sample_idx}"
        obs_frames.append(
            pd.DataFrame(
                {
                    "sample_id": sample_id,
                    "original_cell_id": ordered_cells,
                    "group_label": ["control"] * len(control_cells) + ["case"] * len(case_cells),
                    "label_raw": label_raw,
                    "label_mixed": label_mixed,
                },
                index=[f"{sample_id}:{cell_id}" for cell_id in ordered_cells],
            )
        )

    stacked_raw = np.vstack(raw_blocks).astype(np.float32, copy=False)
    stacked_smooth = np.vstack(smooth_blocks).astype(np.float32, copy=False)
    stacked_obs = pd.concat(obs_frames, axis=0)
    merged_obs, merged_raw, merged_smooth = _merge_selected_samples_exact(
        stacked_obs=stacked_obs,
        stacked_raw=stacked_raw,
        stacked_smooth=stacked_smooth,
    )
    rebuilt_exp, rebuilt_label, _, gene_self_cor, affinity, gene_label_cor = _build_reconstructed_result_space(
        gene_names=gene_names,
        response_identity=response_identity,
        merged_obs=merged_obs,
        merged_raw=merged_raw,
        merged_smooth=merged_smooth,
        compute_gene_pair=False,
    )

    result_obs = merged_obs.copy()
    result_obs["label_last"] = rebuilt_label
    keep_obs_cols = [
        col
        for col in ("sample_id", "group_label", "label_raw", "label_last", "original_cell_id")
        if col in result_obs.columns
    ]
    result_obs = result_obs.loc[:, keep_obs_cols].copy()

    result_var = pd.DataFrame(index=gene_names.copy())
    if result.result_ad is not None and hasattr(result.result_ad, "var"):
        result_var = result.result_ad.var.reindex(gene_names).copy()
    if "response_score" not in result_var.columns:
        result_var["response_score"] = result.gene_scores.reindex(gene_names).astype(np.float32)
    if "response_identity" not in result_var.columns and response_identity is not None:
        result_var["response_identity"] = response_identity.reindex(gene_names).astype(np.int8)
    if "gene_self_cor" not in result_var.columns:
        result_var["gene_self_cor"] = gene_self_cor.astype(np.float32)
    if "gene_label_cor" not in result_var.columns:
        result_var["gene_label_cor"] = gene_label_cor.astype(np.float32)

    return {
        "gene_names": gene_names,
        "response_identity": response_identity,
        "expression": np.asarray(rebuilt_exp, dtype=np.float32),
        "label": np.asarray(rebuilt_label, dtype=np.float32),
        "affinity": sparse.csr_matrix(affinity.astype(np.float32, copy=False)),
        "obs": result_obs,
        "var": result_var,
        "representative_sample_indices": tuple(selected_indices),
    }


def _build_gene_pair_analysis(
    result: "CRANEResult",
    *,
    selected_genes: pd.Index,
    gene_scope: str,
) -> CRANEGenePairResult:
    analysis_space = result._get_gene_response_analysis_space()
    selected_index = analysis_space["gene_names"].get_indexer(selected_genes)
    exp_selected = analysis_space["expression"][:, selected_index]
    affinity_dense = np.asarray(analysis_space["affinity"].toarray(), dtype=np.float32)
    gene_cor = _compute_sp_moran_between(exp_selected, exp_selected, affinity_dense.copy()).astype(np.float32, copy=False)

    pair_ad = ad.AnnData(
        X=np.zeros((0, len(selected_genes)), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index([], dtype=str)),
        var=pd.DataFrame(index=selected_genes.copy()),
    )
    pair_ad.varp["cor"] = gene_cor
    pair_ad.uns["crane_info"] = {
        "kind": "crane_gene_pair_result",
        "pair_method": "moran",
        "gene_scope": gene_scope,
        "source_result_space": (
            dict(result.result_ad.uns.get("crane_info", {}))
            if result.result_ad is not None and hasattr(result.result_ad, "uns")
            else {}
        ),
        "representative_sample_indices": analysis_space["representative_sample_indices"],
    }
    return CRANEGenePairResult(
        pair_ad=pair_ad,
        source_result_ad=result.result_ad if isinstance(result.result_ad, ad.AnnData) else None,
        metadata={"mode": "gene_pair", "gene_scope": gene_scope},
    )


def _build_gene_module_analysis(
    result: "CRANEResult",
    *,
    pair_result: CRANEGenePairResult,
    method: str,
    min_correlation: float,
    min_size: int,
) -> CRANEGeneModuleResult:
    if min_size < 2:
        raise ValueError("gene_module() requires min_size >= 2.")

    analysis_space = result._get_gene_response_analysis_space()
    selected_genes = pair_result.pair_ad.var_names.copy()
    selected_index = analysis_space["gene_names"].get_indexer(selected_genes)
    exp_selected = np.asarray(analysis_space["expression"][:, selected_index], dtype=np.float32)
    gene_cor = np.asarray(pair_result.pair_ad.varp["cor"], dtype=np.float32)
    response_identity = (
        analysis_space["response_identity"].reindex(selected_genes).fillna(0).astype(np.int8)
        if analysis_space["response_identity"] is not None
        else pd.Series(np.ones(len(selected_genes), dtype=np.int8), index=selected_genes)
    )
    backend_method, backend_labels, _ = run_gene_module_backend(
        method=method,
        gene_cor=gene_cor,
        fs_ind=response_identity,
        min_correlation=float(min_correlation),
    )
    backend_labels = pd.Series(np.asarray(backend_labels, dtype=object), index=selected_genes, dtype=object)

    invalid_labels = {"0", "-1", "", "nan", "none", "unassigned"}
    valid_modules: list[tuple[str, np.ndarray]] = []
    for backend_label in backend_labels.astype(str).unique().tolist():
        if str(backend_label).strip().lower() in invalid_labels:
            continue
        member_idx = np.flatnonzero(backend_labels.astype(str).to_numpy() == str(backend_label))
        if member_idx.size >= int(min_size):
            valid_modules.append((str(backend_label), member_idx))
    valid_modules.sort(key=lambda item: (-len(item[1]), item[0]))
    if not valid_modules:
        raise ValueError("No gene modules passed the current min_correlation/min_size thresholds.")

    exp_mean = exp_selected.mean(axis=0)
    exp_std = exp_selected.std(axis=0, ddof=0)
    exp_std[exp_std == 0] = 1.0
    exp_scaled = ((exp_selected - exp_mean) / exp_std).astype(np.float32, copy=False)
    label = np.asarray(analysis_space["label"], dtype=np.float32).reshape(-1, 1)
    affinity_dense = np.asarray(analysis_space["affinity"].toarray(), dtype=np.float32)
    module_scores: list[np.ndarray] = []
    module_meta_rows: list[dict[str, Any]] = []
    gene_module_map = analysis_space["var"].reindex(selected_genes).copy()
    gene_module_map["module_backend_label"] = backend_labels.reindex(selected_genes).astype(str)
    gene_module_map["module_label"] = "unassigned"

    for module_idx, (backend_label, member_idx) in enumerate(valid_modules, start=1):
        module_label = f"M{module_idx}"
        module_genes = selected_genes[member_idx]
        module_signal = exp_scaled[:, member_idx].mean(axis=1).astype(np.float32, copy=False)
        module_signal_2d = module_signal.reshape(-1, 1)
        module_self = float(_compute_sp_moran_between(module_signal_2d, module_signal_2d, affinity_dense.copy())[0, 0])
        module_label_cor = float(_compute_sp_moran_between(module_signal_2d, label, affinity_dense.copy())[0, 0])
        module_response_score = float(np.sqrt(module_self**2 + module_label_cor**2) / np.sqrt(2.0))
        module_cor_block = gene_cor[np.ix_(member_idx, member_idx)]
        if module_cor_block.shape[0] > 1:
            upper = np.triu_indices(module_cor_block.shape[0], k=1)
            mean_within_cor = float(np.mean(module_cor_block[upper])) if upper[0].size else 0.0
        else:
            mean_within_cor = 0.0
        module_scores.append(module_signal)
        gene_module_map.loc[module_genes, "module_label"] = module_label
        module_meta_rows.append(
            {
                "module_label": module_label,
                "gene_count": int(len(module_genes)),
                "module_backend_label": str(backend_label),
                "response_gene_count": int(
                    gene_module_map.loc[module_genes, "response_identity"].fillna(0).astype(np.int8).sum()
                )
                if "response_identity" in gene_module_map.columns
                else int(len(module_genes)),
                "module_self_cor": float(module_self),
                "module_label_cor": float(module_label_cor),
                "module_response_score": float(module_response_score),
                "mean_within_cor": float(mean_within_cor),
                "mean_gene_score": float(gene_module_map.loc[module_genes, "response_score"].astype(np.float32).mean())
                if "response_score" in gene_module_map.columns
                else 0.0,
            }
        )

    module_matrix = np.column_stack(module_scores).astype(np.float32, copy=False)
    module_var = pd.DataFrame(module_meta_rows).set_index("module_label")
    module_ad = ad.AnnData(
        X=module_matrix,
        obs=analysis_space["obs"].copy(),
        var=module_var,
    )
    module_ad.obsp["affinity"] = analysis_space["affinity"].copy()
    module_ad.uns["gene_module_map"] = gene_module_map
    module_ad.uns["crane_info"] = {
        "kind": "crane_gene_module_result",
        "module_method": backend_method,
        "module_method_requested": str(method).strip().lower(),
        "min_correlation": float(min_correlation),
        "min_size": int(min_size),
        "source_gene_pair": dict(pair_result.pair_ad.uns.get("crane_info", {})),
    }
    return CRANEGeneModuleResult(
        module_ad=module_ad,
        source_result_ad=result.result_ad if isinstance(result.result_ad, ad.AnnData) else None,
        metadata={
            "mode": "gene_module",
            "method": backend_method,
            "method_requested": str(method).strip().lower(),
            "min_correlation": float(min_correlation),
            "min_size": int(min_size),
        },
    )


@dataclass
class CRANEResult:
    """Unified result object for first-version CRANE public API."""

    adata: Any = None
    result_ad: Any = None
    gene_scores: Any = None
    response_identity: Any = None
    cell_scores: Any = None
    graph: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    logger_config: LoggerConfig | None = None
    _analysis_cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def _display_text(self) -> str:
        mode = self.metadata.get("mode", "gene_response")
        if self.gene_scores is not None:
            if mode == "gene_response":
                return (
                    "CRANE gene-response analysis completed.\n"
                    "Use .summary() to view the gene-response summary.\n"
                    "Use .gene_pair() or .gene_module() for gene-pair / gene-module analysis.\n"
                    "Use crane.tl.cell_response(...), crane.tl.extension_response(...), or crane.tl.function_response(...) for other downstream analyses.\n"
                    "Tutorial: CRANE GitHub repository URL will be added after sync."
                )
            return (
                "CRANE result completed.\n"
                "Use .summary() to view the available summary."
            )
        if self.cell_scores is not None:
            if mode == "cell_response":
                return (
                    "CRANE cell-response analysis completed.\n"
                    "Use .summary() to view the cell-level summary."
                )
            return (
                "CRANE result completed.\n"
                "Use .summary() to view the available summary."
            )
        return "CRANE result completed."

    def __repr__(self) -> str:
        return self._display_text()

    __str__ = __repr__

    def _metadata_summary_for_uns(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if "mode" in self.metadata:
            summary["mode"] = self.metadata["mode"]
        if "step1_summary" in self.metadata:
            summary["step1_summary"] = self.metadata["step1_summary"]
        if "step2_summary" in self.metadata:
            summary["step2_summary"] = self.metadata["step2_summary"]
        if "cell_response" in self.metadata:
            summary["cell_response"] = self.metadata["cell_response"]
        return summary

    def _get_gene_response_analysis_space(self) -> dict[str, Any]:
        cache_key = ("gene_response_space",)
        if cache_key not in self._analysis_cache:
            self._analysis_cache[cache_key] = _build_gene_response_analysis_space(self)
        return self._analysis_cache[cache_key]

    def _get_step2_mean_gene_correlation(self) -> pd.DataFrame | None:
        cache_key = ("step2_mean_gene_correlation",)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        if self.gene_scores is None:
            self._analysis_cache[cache_key] = None
            return None
        step2_result = self.metadata.get("step2")
        if step2_result is None:
            self._analysis_cache[cache_key] = None
            return None

        sample_outputs = tuple(getattr(step2_result, "sample_outputs", ()) or ())
        if not sample_outputs:
            self._analysis_cache[cache_key] = None
            return None

        gene_names = pd.Index(self.gene_scores.index.copy(), name="gene")
        gene_self_blocks = []
        gene_label_blocks = []
        for output in sample_outputs:
            gene_self = np.asarray(getattr(output, "gene_self_cor", ()), dtype=np.float32).reshape(-1)
            gene_label = np.asarray(getattr(output, "gene_label_cor", ()), dtype=np.float32).reshape(-1)
            if gene_self.shape[0] != len(gene_names) or gene_label.shape[0] != len(gene_names):
                self._analysis_cache[cache_key] = None
                return None
            gene_self_blocks.append(gene_self)
            gene_label_blocks.append(gene_label)

        frame = pd.DataFrame(
            {
                "gene_self_cor": np.mean(np.vstack(gene_self_blocks), axis=0, dtype=np.float32),
                "gene_label_cor": np.mean(np.vstack(gene_label_blocks), axis=0, dtype=np.float32),
            },
            index=gene_names,
        )
        self._analysis_cache[cache_key] = frame
        return frame

    def _compose_summary_response_score(
        self,
        *,
        gene_self_cor: pd.Series,
        gene_label_cor: pd.Series,
        normalized: bool,
    ) -> pd.Series:
        score = np.sqrt(np.square(gene_self_cor.astype(np.float32)) + np.square(gene_label_cor.astype(np.float32)))
        if normalized:
            score = score / np.sqrt(2.0)
        return pd.Series(np.asarray(score, dtype=np.float32), index=gene_self_cor.index.copy(), name="response_score")

    def evaluate_function(
        self,
        adata: Any | None = None,
        gene_set: Any = None,
        **metadata: Any,
    ):
        from ..functional.query import evaluate_function

        source_adata = self.adata if adata is None else adata
        if source_adata is None:
            raise ValueError("evaluate_function() requires adata because functional gene sets use input expression.")
        logger_config = self.logger_config or LoggerConfig()
        logger = build_logger(logger_config).bind("functional")
        logger.event(
            "functional.request.bound",
            "Functional evaluation request captured.",
            audience="debug",
            level="DEBUG",
            has_gene_set=gene_set is not None,
        )
        return evaluate_function(
            source_adata,
            result=self,
            gene_set=gene_set,
            logger=logger,
            metadata=metadata,
        )

    def gene_pair(
        self,
        *,
        genes: Any = None,
        active_only: bool = False,
        use_cache: bool = True,
    ) -> CRANEGenePairResult:
        """Run explicit gene-pair Moran correlation on the gene-response result space."""

        analysis_space = self._get_gene_response_analysis_space()
        selected_genes, gene_scope = _resolve_gene_scope(
            gene_names=analysis_space["gene_names"],
            response_identity=analysis_space["response_identity"],
            genes=genes,
            active_only=active_only,
        )
        cache_key = ("gene_pair", gene_scope, _format_gene_scope_key(selected_genes))
        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        pair_result = _build_gene_pair_analysis(
            self,
            selected_genes=selected_genes,
            gene_scope=gene_scope,
        )
        if use_cache:
            self._analysis_cache[cache_key] = pair_result
        return pair_result

    def gene_module(
        self,
        *,
        method: str = "auto",
        genes: Any = None,
        active_only: bool = False,
        pair: CRANEGenePairResult | None = None,
        min_correlation: float = 0.2,
        min_size: int = 5,
        use_cache: bool = True,
    ) -> CRANEGeneModuleResult:
        """Run explicit gene-module analysis on the gene-response result space."""

        if pair is not None and genes is not None:
            raise ValueError("gene_module() accepts either pair=... or genes=..., not both.")
        pair_result = pair or self.gene_pair(genes=genes, active_only=active_only, use_cache=use_cache)
        pair_key = tuple(str(v) for v in pair_result.pair_ad.var_names.tolist())
        method_normalized = resolve_gene_module_method(method, len(pair_key))
        cache_key = ("gene_module", method_normalized, pair_key, float(min_correlation), int(min_size))
        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        module_result = _build_gene_module_analysis(
            self,
            pair_result=pair_result,
            method=method,
            min_correlation=float(min_correlation),
            min_size=int(min_size),
        )
        if use_cache:
            self._analysis_cache[cache_key] = module_result
        return module_result

    def summary(
        self,
        *,
        responsive_only: bool = False,
        sort_by: str = "response_score",
        ascending: bool = False,
        represent: str = "mean",
        normalized: bool = False,
        active_only: bool | None = None,
    ) -> pd.DataFrame:
        """Return the user-facing CRANE summary table.

        For gene-response runs, this is the gene-level response summary rather
        than an internal runtime/state dump. By default it focuses on genes in
        the final response identity.
        """

        if self.gene_scores is not None:
            frame = pd.DataFrame(index=self.gene_scores.index.copy())
            if active_only is not None:
                responsive_only = bool(active_only)
            represent_normalized = str(represent).strip().lower()
            if represent_normalized not in {"mean", "reconst"}:
                raise ValueError("summary(represent=...) must be 'mean' or 'reconst'.")
            if self.result_ad is not None and hasattr(self.result_ad, "var"):
                result_var = self.result_ad.var.reindex(frame.index)
                if "response_identity" in result_var.columns:
                    frame["response_identity"] = result_var["response_identity"].astype(np.int8)
                elif self.response_identity is not None:
                    frame["response_identity"] = self.response_identity.astype(np.int8)
            else:
                result_var = pd.DataFrame(index=frame.index.copy())
                if self.response_identity is not None:
                    frame["response_identity"] = self.response_identity.astype(np.int8)

            if represent_normalized == "reconst":
                if "gene_self_cor" in result_var.columns:
                    frame["gene_self_cor"] = result_var["gene_self_cor"].astype(np.float32)
                if "gene_label_cor" in result_var.columns:
                    frame["gene_label_cor"] = result_var["gene_label_cor"].astype(np.float32)
                if "gene_self_cor" in frame.columns and "gene_label_cor" in frame.columns:
                    frame["response_score"] = self._compose_summary_response_score(
                        gene_self_cor=frame["gene_self_cor"],
                        gene_label_cor=frame["gene_label_cor"],
                        normalized=normalized,
                    )
                else:
                    frame["response_score"] = self.gene_scores.astype(np.float32)
            else:
                step2_gene_corr = self._get_step2_mean_gene_correlation()
                if step2_gene_corr is not None:
                    frame["gene_self_cor"] = step2_gene_corr["gene_self_cor"].reindex(frame.index).astype(np.float32)
                    frame["gene_label_cor"] = step2_gene_corr["gene_label_cor"].reindex(frame.index).astype(np.float32)
                if "gene_self_cor" in frame.columns and "gene_label_cor" in frame.columns:
                    frame["response_score"] = self._compose_summary_response_score(
                        gene_self_cor=frame["gene_self_cor"],
                        gene_label_cor=frame["gene_label_cor"],
                        normalized=normalized,
                    )
                else:
                    frame["response_score"] = self.gene_scores.astype(np.float32)
            if responsive_only and "response_identity" in frame.columns:
                frame = frame.loc[frame["response_identity"] == 1].copy()
            if sort_by in frame.columns:
                frame = frame.sort_values(by=sort_by, ascending=ascending)
            frame.index.name = "gene"
            return frame
        if self.cell_scores is not None:
            frame = pd.DataFrame({"cell_score": self.cell_scores.astype(np.float32)})
            frame.index.name = "cell"
            if sort_by in frame.columns:
                frame = frame.sort_values(by=sort_by, ascending=ascending)
            return frame
        return pd.DataFrame()

    def to_anndata(self, adata: Any | None = None, key_added: str = "crane") -> Any:
        """Export broadcastable CRANE summaries onto a copy of the input AnnData.

        This does not reconstruct CRANE's internal result-space graph or any
        subsampled Step 2 matrices. It only writes fields that can be safely
        aligned back onto the input observation / variable axes.
        """

        base_adata = self.adata if adata is None else adata
        if base_adata is None:
            raise ValueError("to_anndata() requires an AnnData object when the result does not carry one.")
        if not hasattr(base_adata, "copy"):
            raise TypeError("to_anndata() requires an AnnData-like object with a .copy() method.")
        return self.write_to_adata(base_adata.copy(), key_added=key_added)

    def write_to_adata(self, adata: Any, key_added: str = "crane") -> Any:
        """Write input-space CRANE summaries into an AnnData object.

        This helper is intentionally conservative:
        - gene-level outputs are expanded back to the input `var` axis
        - cell-level outputs are expanded back to the input `obs` axis
        - internal graph / affinity objects are *not* written into `obsp`
          because Step 2 may operate on subsampled cell sets that do not match
          the input cell axis
        """

        uns_block = dict(getattr(adata, "uns", {}).get(key_added, {}))
        uns_block["kind"] = "crane"
        uns_block["latest"] = self.metadata.get("mode", "gene_response")
        uns_block["input_space_writeback"] = True
        uns_block["result_object"] = {
            "has_result_ad": self.result_ad is not None,
            "has_gene_scores": self.gene_scores is not None,
            "has_response_identity": self.response_identity is not None,
            "has_cell_scores": self.cell_scores is not None,
            "has_graph": self.graph is not None,
        }
        if self.gene_scores is not None:
            adata.var[f"{key_added}_score"] = self.gene_scores.reindex(adata.var_names, fill_value=0.0)
            uns_block["gene_response"] = {
                "score_key": f"{key_added}_score",
                "missing_gene_default_score": 0.0,
            }
        if self.response_identity is not None:
            adata.var[f"{key_added}_ri"] = self.response_identity.reindex(adata.var_names, fill_value=0).astype(
                "int8"
            )
            gene_block = dict(uns_block.get("gene_response", {}))
            gene_block["ri_key"] = f"{key_added}_ri"
            gene_block["missing_gene_default_ri"] = 0
            uns_block["gene_response"] = gene_block
        if self.cell_scores is not None:
            adata.obs[f"{key_added}_cell_score"] = self.cell_scores.reindex(adata.obs_names, fill_value=0.0)
            uns_block["cell_response"] = {
                "score_key": f"{key_added}_cell_score",
                "missing_cell_default_score": 0.0,
            }
        if self.graph is not None:
            uns_block["graph"] = {
                "available": True,
                "type": self.graph.get("type") if isinstance(self.graph, dict) else type(self.graph).__name__,
                "stored_in_input": False,
                "reason": (
                    "CRANE internal graph is result-space data and may use subsampled cells; "
                    "it is therefore not written into input-space AnnData axes."
                ),
            }
        metadata_summary = self._metadata_summary_for_uns()
        if metadata_summary:
            uns_block["run"] = metadata_summary
        adata.uns[key_added] = uns_block
        return adata

    def save(self, path: str | Path) -> None:
        """Save the current result object using a simple skeleton serializer."""
        target = Path(path)
        with target.open("wb") as handle:
            pickle.dump(self, handle)


def load_result(path: str | Path) -> CRANEResult:
    target = Path(path)
    with target.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, CRANEResult):
        raise TypeError("Loaded object is not a CRANEResult instance.")
    return obj
