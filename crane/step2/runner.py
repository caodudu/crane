"""Serial Step 2 runner for the default ndarray-first mainline."""

from __future__ import annotations

import time
import concurrent.futures as cf
from typing import Any, Sequence

import numpy as np

from ..core.sampling import SamplingPlan, WeightedSample, require_sampled_cells
from .contracts import (
    Step2Options,
    Step2RunResult,
    Step2SampleInput,
    Step2SampleOutput,
    Step2SamplePack,
    Step2State,
)
from .kernels import run_sample_core, update_response_identity
from .kernels import update_response_identity_with_stage


def _build_result_space_identity(
    *,
    ri_history: Sequence[np.ndarray],
    score_history: Sequence[np.ndarray],
    response_identity: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the internal Step2->result-space handoff identity.

    Step2 should hand off the converged identity directly. Extra history-based
    rescue belongs in diagnostics, not in the default runtime contract.
    """

    base_identity = np.asarray(response_identity, dtype=bool)
    return base_identity.astype(np.int8), {
        "mode": "response_identity",
        "active_genes": int(base_identity.sum()),
        "fallback": None,
    }


def _select_representative_sample_indices(score_columns: np.ndarray, n: int = 3) -> tuple[int, ...]:
    """Mirror the legacy representative-sample selection on score columns."""

    score_columns = np.asarray(score_columns, dtype=np.float64)
    if score_columns.ndim != 2:
        raise ValueError("score_columns must be a 2D array.")
    sample_count = int(score_columns.shape[1])
    if sample_count == 0:
        return ()
    if sample_count <= n:
        return tuple(range(sample_count))

    centered = score_columns.T
    sq_norms = np.sum(centered * centered, axis=1, keepdims=True)
    dist_sq = np.maximum(sq_norms + sq_norms.T - 2.0 * (centered @ centered.T), 0.0)
    dist = np.sqrt(dist_sq).astype(np.float64, copy=False)
    centrality = dist.mean(axis=1)
    initial = int(np.argmin(centrality))
    selected = [initial]
    while len(selected) < n:
        if len(selected) == 1:
            distances_to_selected = dist[selected[0]].copy()
        else:
            distance_vectors = dist[:, selected]
            distances_to_selected = np.linalg.norm(distance_vectors, axis=1)
        distances_to_selected[selected] = -np.inf
        next_sample = int(np.argmax(distances_to_selected))
        selected.append(next_sample)
    return tuple(sorted(set(selected)))


def _resolve_threshold_k(
    *,
    base_threshold_k: float,
    total_input_genes: int,
    weak_pert: bool,
    options: Step2Options,
) -> tuple[float, bool, bool]:
    """Optionally relax RI threshold under the legacy weak-perturbation guard."""

    if not options._relaxed_threshold_on_weak_pert:
        return float(base_threshold_k), False, False

    small_input_gene_pool = total_input_genes < (options.max_iterations * 10)
    if weak_pert or small_input_gene_pool:
        relaxed_k = float(base_threshold_k) * float(options._relaxed_threshold_scale)
        if options._relaxed_threshold_min_k is not None:
            relaxed_k = max(relaxed_k, float(options._relaxed_threshold_min_k))
        return float(relaxed_k), True, bool(small_input_gene_pool)
    return float(base_threshold_k), False, bool(small_input_gene_pool)


def _resolve_update_stage(
    *,
    previous_stage: str,
    previous_ri: np.ndarray,
    next_ri: np.ndarray,
    delta_threshold: int,
) -> tuple[str, np.ndarray | None]:
    """Mirror legacy strict/wave stage switching using active-count deltas."""

    nondeg_delta = int(np.sum(previous_ri)) - int(np.sum(next_ri))
    if nondeg_delta <= delta_threshold:
        if previous_stage == "strict":
            return "wave", np.asarray(next_ri, dtype=bool).copy()
        return "wave", None
    return "strict", None


def build_sample_input(
    sample: WeightedSample,
    expression_matrix: Any,
    obs_names: Sequence[Any],
) -> Step2SampleInput:
    """Materialize one ``WeightedSample`` into control-case expression order."""

    obs_lookup = {cell_id: idx for idx, cell_id in enumerate(obs_names)}
    ordered_cells = list(sample.control_cells) + list(sample.case_cells)
    row_ids = [obs_lookup[cell_id] for cell_id in ordered_cells]
    exp_raw = np.asarray(expression_matrix[row_ids, :], dtype=np.float32)
    return Step2SampleInput(
        exp_raw=exp_raw,
        control_cells=tuple(sample.control_cells),
        case_cells=tuple(sample.case_cells),
        sample_id=sample.sample_id,
    )


def pack_sample(sample_input: Step2SampleInput, fs_mask: Any) -> Step2SamplePack:
    """Pack a sample into the ndarray-first Step 2 runtime payload."""

    control_n = len(sample_input.control_cells)
    case_n = len(sample_input.case_cells)
    label_raw = np.concatenate(
        (np.zeros(control_n, dtype=np.float32), np.ones(case_n, dtype=np.float32)),
        axis=0,
    )
    group_labels = np.concatenate(
        (np.repeat("control", control_n), np.repeat("case", case_n)),
        axis=0,
    )
    fs_array = _coerce_feature_mask(fs_mask)
    return Step2SamplePack(
        exp_raw=np.asarray(sample_input.exp_raw, dtype=np.float32),
        label_raw=label_raw,
        fs_mask=fs_array,
        group_labels=group_labels,
        sample_id=sample_input.sample_id,
        control_cells=tuple(sample_input.control_cells),
        case_cells=tuple(sample_input.case_cells),
    )


def _coerce_feature_mask(fs_mask: Any) -> np.ndarray:
    """Accept legacy/new Step 1 masks without depending on either caller shape."""

    if hasattr(fs_mask, "columns"):
        if "i_0" in fs_mask.columns:
            fs_array = fs_mask["i_0"].to_numpy()
        elif fs_mask.shape[1] == 1:
            fs_array = fs_mask.iloc[:, 0].to_numpy()
        else:
            numeric_columns = [
                col
                for col in fs_mask.columns
                if np.issubdtype(fs_mask[col].to_numpy().dtype, np.number)
            ]
            if len(numeric_columns) != 1:
                raise ValueError("Step 2 feature mask DataFrame must contain one numeric mask column or 'i_0'.")
            fs_array = fs_mask[numeric_columns[0]].to_numpy()
    elif hasattr(fs_mask, "to_numpy"):
        fs_array = fs_mask.to_numpy()
    else:
        fs_array = np.asarray(fs_mask)

    fs_array = np.asarray(fs_array).reshape(-1)
    if fs_array.dtype == bool:
        return fs_array
    if not np.issubdtype(fs_array.dtype, np.number):
        raise ValueError("Step 2 feature mask must be boolean or numeric.")
    return (fs_array == 1)


def prepare_step2_packs(
    sampling_plan: SamplingPlan,
    fs_input: Any | None = None,
    guide_fs_input: Any | None = None,
    sample_layer: str | None = None,
) -> tuple[Step2SamplePack, ...]:
    """Create compact Step 2 packs from a Step 1-like sampling handoff."""

    require_sampled_cells(sampling_plan)
    if sampling_plan.working_adata is None:
        raise ValueError("sampling_plan.working_adata is required for Step 2 packing.")
    if fs_input is None:
        fs_input = sampling_plan.init_feature_selection
    if fs_input is None:
        raise ValueError("Step 2 requires an initial feature-selection mask.")
    if guide_fs_input is None:
        guide_fs_input = sampling_plan.guide_feature_selection

    working_adata = sampling_plan.working_adata
    layer_name = sample_layer
    if layer_name is not None and layer_name in working_adata.layers:
        expression_matrix = working_adata.layers[layer_name]
    else:
        expression_matrix = working_adata.X
    if hasattr(expression_matrix, "toarray"):
        expression_matrix = expression_matrix.toarray()

    sample_inputs = [
        build_sample_input(sample, expression_matrix=expression_matrix, obs_names=working_adata.obs_names)
        for sample in sampling_plan.control_case_samples
    ]
    guide_fs_array = _coerce_feature_mask(guide_fs_input) if guide_fs_input is not None else None
    packs: list[Step2SamplePack] = []
    for sample_input in sample_inputs:
        pack = pack_sample(sample_input, fs_mask=fs_input)
        packs.append(
            Step2SamplePack(
                exp_raw=pack.exp_raw,
                label_raw=pack.label_raw,
                fs_mask=pack.fs_mask,
                group_labels=pack.group_labels,
                guide_fs_mask=guide_fs_array,
                sample_id=pack.sample_id,
                control_cells=pack.control_cells,
                case_cells=pack.case_cells,
            )
        )
    return tuple(packs)


def initial_step2_state(sample_count: int, ri_mask: Any) -> Step2State:
    """Create the empty cross-iteration Step 2 state."""

    ri_array = _coerce_feature_mask(ri_mask)
    if not np.any(ri_array):
        raise ValueError("Initial Step 2 response identity must retain at least one gene.")
    return Step2State(
        exp_last_list=tuple([None] * sample_count),
        label_last_list=tuple([None] * sample_count),
        guide_pass_list=tuple([False] * sample_count),
        ri_mask=ri_array,
        iteration=0,
    )


def run_step2_serial(
    packs: Sequence[Step2SamplePack],
    iterations: int | None = None,
    options: Step2Options | None = None,
    initial_state: Step2State | None = None,
) -> Step2RunResult:
    """Run the default no-module Step 2 RI-refinement loop as a serial baseline."""

    options = options or Step2Options()
    if not packs:
        raise ValueError("At least one Step 2 sample pack is required.")
    max_iterations = options.max_iterations if iterations is None else iterations
    if max_iterations < 1:
        raise ValueError("iterations must be >= 1")
    initial_ri = packs[0].fs_mask
    state = initial_state or initial_step2_state(len(packs), initial_ri)
    if (
        len(state.exp_last_list) != len(packs)
        or len(state.label_last_list) != len(packs)
        or len(state.guide_pass_list) != len(packs)
    ):
        raise ValueError("Initial Step2State length must match packs length.")
    if state.ri_mask.shape[0] != packs[0].exp_raw.shape[1]:
        raise ValueError("Initial Step2State ri_mask length must match gene count.")

    iter_times: list[float] = []
    sample_outputs: tuple[Step2SampleOutput, ...] = ()
    exp_last_list = list(state.exp_last_list)
    label_last_list = list(state.label_last_list)
    guide_pass_list = list(state.guide_pass_list)
    ri_mask = np.asarray(state.ri_mask, dtype=bool).copy()
    stable_count = 0
    stable_tail_count = 0
    weak_pert_latched = False
    update_stage = "strict"
    fs_strict_mask: np.ndarray | None = None
    ri_history: list[np.ndarray] = [ri_mask.copy()]
    score_history: list[np.ndarray] = []
    update_history: list[dict[str, Any]] = []
    response_score = np.zeros_like(ri_mask, dtype=np.float32)

    for iter_offset in range(max_iterations):
        start = time.perf_counter()
        next_outputs: list[Step2SampleOutput] = []
        for sample_idx, pack in enumerate(packs):
            iter_pack = Step2SamplePack(
                exp_raw=pack.exp_raw,
                label_raw=pack.label_raw,
                fs_mask=ri_mask,
                group_labels=pack.group_labels,
                guide_fs_mask=pack.guide_fs_mask,
                sample_id=pack.sample_id,
                control_cells=pack.control_cells,
                case_cells=pack.case_cells,
            )
            iter_options = Step2Options(
                n_pcs=options.n_pcs,
                cell_k=options.cell_k,
                max_iterations=options.max_iterations,
                stable_rounds=options.stable_rounds,
                score_mode=options.score_mode,
                threshold_k=options.threshold_k,
                drop_limit=options.drop_limit,
                dtype=options.dtype,
                force_connect=options.force_connect,
                _smooth_weight=options._smooth_weight,
                _smooth_alpha=options._smooth_alpha,
                _binary_delta_threshold=options._binary_delta_threshold,
                _guide_compare_rounds=options._guide_compare_rounds,
                _relaxed_threshold_on_weak_pert=options._relaxed_threshold_on_weak_pert,
                _relaxed_threshold_latch_weak_pert=options._relaxed_threshold_latch_weak_pert,
                _relaxed_threshold_min_k=options._relaxed_threshold_min_k,
                _relaxed_threshold_scale=options._relaxed_threshold_scale,
                _legacy_post_stable_rounds=options._legacy_post_stable_rounds,
                extras={**dict(options.extras), "iteration": state.iteration + iter_offset + 1},
            )
            next_outputs.append(
                run_sample_core(
                    pack=iter_pack,
                    exp_last=exp_last_list[sample_idx],
                    label_last=label_last_list[sample_idx],
                    guide_pass=guide_pass_list[sample_idx],
                    options=iter_options,
                )
            )
        exp_last_list = [output.exp_last_next for output in next_outputs]
        label_last_list = [output.label_last_next for output in next_outputs]
        guide_pass_list = [bool(output.guide_pass_next) for output in next_outputs]
        sample_outputs = tuple(next_outputs)
        weak_pert = not all(bool(v) for v in guide_pass_list)
        if options._relaxed_threshold_latch_weak_pert and weak_pert:
            weak_pert_latched = True
        weak_pert_effective = weak_pert_latched or weak_pert
        response_score = np.mean(
            np.vstack([output.norm_combined_score for output in sample_outputs]),
            axis=0,
            dtype=np.float32,
        )
        threshold_k_iter, threshold_relaxed, small_input_gene_pool = _resolve_threshold_k(
            base_threshold_k=options.threshold_k,
            total_input_genes=int(packs[0].exp_raw.shape[1]),
            weak_pert=weak_pert_effective,
            options=options,
        )
        if options._legacy_wave_compare:
            next_ri, update_meta = update_response_identity_with_stage(
                current_ri=ri_mask,
                response_score=response_score,
                threshold_k=threshold_k_iter,
                drop_limit=options.drop_limit,
                update_stage=update_stage,
                fs_strict_mask=fs_strict_mask,
            )
        else:
            next_ri, update_meta = update_response_identity(
                current_ri=ri_mask,
                response_score=response_score,
                threshold_k=threshold_k_iter,
                drop_limit=options.drop_limit,
            )
        update_meta["threshold_k"] = float(threshold_k_iter)
        update_meta["threshold_relaxed"] = bool(threshold_relaxed)
        update_meta["small_input_gene_pool"] = bool(small_input_gene_pool)
        update_meta["total_input_genes"] = int(packs[0].exp_raw.shape[1])
        update_meta["weak_pert"] = bool(weak_pert)
        update_meta["weak_pert_effective"] = bool(weak_pert_effective)
        update_meta["stage_before"] = str(update_stage)
        next_stage, next_fs_strict = _resolve_update_stage(
            previous_stage=update_stage,
            previous_ri=ri_mask,
            next_ri=next_ri,
            delta_threshold=options._legacy_wave_delta_threshold,
        )
        if next_stage == "strict":
            fs_strict_mask = None
        elif next_fs_strict is not None:
            fs_strict_mask = next_fs_strict
        update_stage = next_stage
        update_meta["stage_after"] = str(update_stage)
        update_meta["fs_strict_active_after"] = int(np.sum(fs_strict_mask)) if fs_strict_mask is not None else 0
        if np.array_equal(next_ri, ri_mask):
            stable_count += 1
            if stable_count >= options.stable_rounds:
                stable_tail_count += 1
        else:
            stable_count = 0
            stable_tail_count = 0
        ri_mask = next_ri
        ri_history.append(ri_mask.copy())
        score_history.append(response_score.copy())
        update_history.append(update_meta)
        iter_times.append(round(time.perf_counter() - start, 4))
        if (
            (stable_count >= options.stable_rounds and stable_tail_count > options._legacy_post_stable_rounds)
            or np.sum(ri_mask) < 1
        ):
            break

    final_state = Step2State(
        exp_last_list=tuple(exp_last_list),
        label_last_list=tuple(label_last_list),
        guide_pass_list=tuple(guide_pass_list),
        ri_mask=ri_mask,
        iteration=state.iteration + len(iter_times),
    )
    retained_exp_last_mb = sum(arr.nbytes for arr in exp_last_list if arr is not None) / 1024 / 1024
    representative_sample_indices = _select_representative_sample_indices(
        np.vstack([output.norm_combined_score for output in sample_outputs]).T,
        n=3,
    )
    result_space_identity, handoff_meta = _build_result_space_identity(
        ri_history=tuple(ri_history),
        score_history=tuple(score_history),
        response_identity=ri_mask,
    )
    return Step2RunResult(
        state=final_state,
        sample_outputs=sample_outputs,
        response_score=response_score,
        response_identity=ri_mask,
        result_space_identity=result_space_identity,
        representative_sample_indices=representative_sample_indices,
        iter_times_s=tuple(iter_times),
        ri_history=tuple(ri_history),
        score_history=tuple(score_history),
        metadata={
            "sample_count": len(packs),
            "iterations": len(iter_times),
            "total_s": round(sum(iter_times), 4),
            "retained_exp_last_mb": round(retained_exp_last_mb, 2),
            "runner": "serial",
            "converged": stable_count >= options.stable_rounds,
            "stable_rounds": stable_count,
            "stable_tail_rounds": stable_tail_count,
            "active_genes": int(np.sum(ri_mask)),
            "final_update_stage": update_stage,
            "final_fs_strict_active": int(np.sum(fs_strict_mask)) if fs_strict_mask is not None else 0,
            "result_space_identity_active_genes": int(np.sum(result_space_identity)),
            "representative_sample_indices": tuple(int(i) for i in representative_sample_indices),
            "result_space_handoff": handoff_meta,
            "update_history": update_history,
        },
    )


def run_step2_threaded(
    packs: Sequence[Step2SamplePack],
    iterations: int | None = None,
    options: Step2Options | None = None,
    initial_state: Step2State | None = None,
    max_workers: int = 2,
) -> Step2RunResult:
    """Run Step 2 with a small sample-level thread pool."""

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_workers == 1:
        return run_step2_serial(packs, iterations=iterations, options=options, initial_state=initial_state)

    options = options or Step2Options()
    if not packs:
        raise ValueError("At least one Step 2 sample pack is required.")
    max_iterations = options.max_iterations if iterations is None else iterations
    if max_iterations < 1:
        raise ValueError("iterations must be >= 1")
    state = initial_state or initial_step2_state(len(packs), packs[0].fs_mask)
    if (
        len(state.exp_last_list) != len(packs)
        or len(state.label_last_list) != len(packs)
        or len(state.guide_pass_list) != len(packs)
    ):
        raise ValueError("Initial Step2State length must match packs length.")

    iter_times: list[float] = []
    sample_outputs: tuple[Step2SampleOutput, ...] = ()
    exp_last_list = list(state.exp_last_list)
    label_last_list = list(state.label_last_list)
    guide_pass_list = list(state.guide_pass_list)
    ri_mask = np.asarray(state.ri_mask, dtype=bool).copy()
    stable_count = 0
    stable_tail_count = 0
    weak_pert_latched = False
    update_stage = "strict"
    fs_strict_mask: np.ndarray | None = None
    ri_history: list[np.ndarray] = [ri_mask.copy()]
    score_history: list[np.ndarray] = []
    update_history: list[dict[str, Any]] = []
    response_score = np.zeros_like(ri_mask, dtype=np.float32)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for iter_offset in range(max_iterations):
            start = time.perf_counter()
            iter_options = Step2Options(
                n_pcs=options.n_pcs,
                cell_k=options.cell_k,
                max_iterations=options.max_iterations,
                stable_rounds=options.stable_rounds,
                score_mode=options.score_mode,
                threshold_k=options.threshold_k,
                drop_limit=options.drop_limit,
                dtype=options.dtype,
                force_connect=options.force_connect,
                _smooth_weight=options._smooth_weight,
                _smooth_alpha=options._smooth_alpha,
                _binary_delta_threshold=options._binary_delta_threshold,
                _guide_compare_rounds=options._guide_compare_rounds,
                _relaxed_threshold_on_weak_pert=options._relaxed_threshold_on_weak_pert,
                _relaxed_threshold_latch_weak_pert=options._relaxed_threshold_latch_weak_pert,
                _relaxed_threshold_min_k=options._relaxed_threshold_min_k,
                _relaxed_threshold_scale=options._relaxed_threshold_scale,
                _legacy_post_stable_rounds=options._legacy_post_stable_rounds,
                extras={**dict(options.extras), "iteration": state.iteration + iter_offset + 1},
            )
            tasks = [
                (
                    Step2SamplePack(
                        exp_raw=pack.exp_raw,
                        label_raw=pack.label_raw,
                        fs_mask=ri_mask,
                        group_labels=pack.group_labels,
                        guide_fs_mask=pack.guide_fs_mask,
                        sample_id=pack.sample_id,
                        control_cells=pack.control_cells,
                        case_cells=pack.case_cells,
                    ),
                    exp_last_list[sample_idx],
                    label_last_list[sample_idx],
                    guide_pass_list[sample_idx],
                    iter_options,
                )
                for sample_idx, pack in enumerate(packs)
            ]
            next_outputs = list(executor.map(lambda args: run_sample_core(*args), tasks))
            exp_last_list = [output.exp_last_next for output in next_outputs]
            label_last_list = [output.label_last_next for output in next_outputs]
            guide_pass_list = [bool(output.guide_pass_next) for output in next_outputs]
            sample_outputs = tuple(next_outputs)
            weak_pert = not all(bool(v) for v in guide_pass_list)
            if options._relaxed_threshold_latch_weak_pert and weak_pert:
                weak_pert_latched = True
            weak_pert_effective = weak_pert_latched or weak_pert
            response_score = np.mean(
                np.vstack([output.norm_combined_score for output in sample_outputs]),
                axis=0,
                dtype=np.float32,
            )
            threshold_k_iter, threshold_relaxed, small_input_gene_pool = _resolve_threshold_k(
                base_threshold_k=options.threshold_k,
                total_input_genes=int(packs[0].exp_raw.shape[1]),
                weak_pert=weak_pert_effective,
                options=options,
            )
            if options._legacy_wave_compare:
                next_ri, update_meta = update_response_identity_with_stage(
                    current_ri=ri_mask,
                    response_score=response_score,
                    threshold_k=threshold_k_iter,
                    drop_limit=options.drop_limit,
                    update_stage=update_stage,
                    fs_strict_mask=fs_strict_mask,
                )
            else:
                next_ri, update_meta = update_response_identity(
                    current_ri=ri_mask,
                    response_score=response_score,
                    threshold_k=threshold_k_iter,
                    drop_limit=options.drop_limit,
                )
            update_meta["threshold_k"] = float(threshold_k_iter)
            update_meta["threshold_relaxed"] = bool(threshold_relaxed)
            update_meta["small_input_gene_pool"] = bool(small_input_gene_pool)
            update_meta["total_input_genes"] = int(packs[0].exp_raw.shape[1])
            update_meta["weak_pert"] = bool(weak_pert)
            update_meta["weak_pert_effective"] = bool(weak_pert_effective)
            update_meta["stage_before"] = str(update_stage)
            next_stage, next_fs_strict = _resolve_update_stage(
                previous_stage=update_stage,
                previous_ri=ri_mask,
                next_ri=next_ri,
                delta_threshold=options._legacy_wave_delta_threshold,
            )
            if next_stage == "strict":
                fs_strict_mask = None
            elif next_fs_strict is not None:
                fs_strict_mask = next_fs_strict
            update_stage = next_stage
            update_meta["stage_after"] = str(update_stage)
            update_meta["fs_strict_active_after"] = int(np.sum(fs_strict_mask)) if fs_strict_mask is not None else 0
            if np.array_equal(next_ri, ri_mask):
                stable_count += 1
                if stable_count >= options.stable_rounds:
                    stable_tail_count += 1
            else:
                stable_count = 0
                stable_tail_count = 0
            ri_mask = next_ri
            ri_history.append(ri_mask.copy())
            score_history.append(response_score.copy())
            update_history.append(update_meta)
            iter_times.append(round(time.perf_counter() - start, 4))
            if (
                (stable_count >= options.stable_rounds and stable_tail_count > options._legacy_post_stable_rounds)
                or np.sum(ri_mask) < 1
            ):
                break

    final_state = Step2State(
        exp_last_list=tuple(exp_last_list),
        label_last_list=tuple(label_last_list),
        guide_pass_list=tuple(guide_pass_list),
        ri_mask=ri_mask,
        iteration=state.iteration + len(iter_times),
    )
    retained_exp_last_mb = sum(arr.nbytes for arr in exp_last_list if arr is not None) / 1024 / 1024
    representative_sample_indices = _select_representative_sample_indices(
        np.vstack([output.norm_combined_score for output in sample_outputs]).T,
        n=3,
    )
    result_space_identity, handoff_meta = _build_result_space_identity(
        ri_history=tuple(ri_history),
        score_history=tuple(score_history),
        response_identity=ri_mask,
    )
    return Step2RunResult(
        state=final_state,
        sample_outputs=sample_outputs,
        response_score=response_score,
        response_identity=ri_mask,
        result_space_identity=result_space_identity,
        representative_sample_indices=representative_sample_indices,
        iter_times_s=tuple(iter_times),
        ri_history=tuple(ri_history),
        score_history=tuple(score_history),
        metadata={
            "sample_count": len(packs),
            "iterations": len(iter_times),
            "total_s": round(sum(iter_times), 4),
            "retained_exp_last_mb": round(retained_exp_last_mb, 2),
            "runner": "threaded",
            "max_workers": max_workers,
            "converged": stable_count >= options.stable_rounds,
            "stable_rounds": stable_count,
            "stable_tail_rounds": stable_tail_count,
            "active_genes": int(np.sum(ri_mask)),
            "final_update_stage": update_stage,
            "final_fs_strict_active": int(np.sum(fs_strict_mask)) if fs_strict_mask is not None else 0,
            "result_space_identity_active_genes": int(np.sum(result_space_identity)),
            "representative_sample_indices": tuple(int(i) for i in representative_sample_indices),
            "result_space_handoff": handoff_meta,
            "update_history": update_history,
        },
    )
