"""Step 1 contracts: weighted perturbation-control multi-sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .preprocess import PreparedInput


@dataclass(frozen=True)
class PerturbationTendency:
    """Cell-level tendency used to weight candidate perturbation cells."""

    values: Any
    source: str = "graph_neighbor_proportion"
    clipped: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightedSample:
    """One balanced control-case sample used by Step 2."""

    control_cells: Sequence[Any]
    case_cells: Sequence[Any]
    pseudo_case_cells: Sequence[Any] = field(default_factory=tuple)
    sample_id: str | None = None


@dataclass(frozen=True)
class SamplingOptions:
    """Step 1 options with paper-facing names and hidden legacy hooks."""

    n_cells: int = 50
    n_subsamples: int = 5
    weight_method: str | None = "softmax"
    random_state: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingPlan:
    """Complete Step 1 output consumed by closed-loop refinement."""

    prepared_input: PreparedInput
    tendency: PerturbationTendency | None
    control_case_samples: Sequence[WeightedSample]
    control_control_samples: Sequence[WeightedSample] = field(default_factory=tuple)
    working_adata: Any = None
    init_feature_selection: Any = None
    guide_feature_selection: Any = None
    margin_evaluation: Any = None
    options: SamplingOptions = field(default_factory=SamplingOptions)


def process_sampling_weights(
    case_neighbor_proportions: pd.Series,
    method: str = "softmax",
    epsilon: float = 0.0000001,
    alpha: float = 5,
) -> pd.Series:
    if method == "uniform":
        n = len(case_neighbor_proportions)
        return pd.Series(np.ones(n) / n, index=case_neighbor_proportions.index)

    bool_zero = case_neighbor_proportions == 0
    adjusted_proportions = case_neighbor_proportions.copy()
    adjusted_proportions[bool_zero] = epsilon
    if method == "linear":
        normalized_weights = adjusted_proportions / adjusted_proportions.sum()
    elif method == "sigmoid":
        weights = 1 / (1 + np.exp(-alpha * (adjusted_proportions - 0.5)))
        weights[bool_zero] = epsilon
        normalized_weights = weights / weights.sum()
    elif method == "softmax":
        exp_values = np.exp(alpha * adjusted_proportions)
        exp_values[bool_zero] = epsilon
        normalized_weights = exp_values / np.sum(exp_values)
    else:
        raise ValueError("method must be 'linear', 'sigmoid', 'softmax', or 'uniform'.")
    return pd.Series(normalized_weights, index=case_neighbor_proportions.index)


def _weighted_rswr_samples(
    adata: Any,
    cell_weight: pd.Series,
    group_obs: str,
    control_label: Any,
    case_label: Any,
    n_cells: int,
    n_subsamples: int,
    weight_method: str | None,
) -> list[WeightedSample]:
    control_bool = adata.obs[group_obs] == control_label
    case_bool = adata.obs[group_obs] == case_label
    if weight_method is None:
        weight_method = "softmax"
    case_weight = cell_weight[case_bool]
    case_weight_total = process_sampling_weights(
        case_neighbor_proportions=case_weight,
        method=weight_method,
        epsilon=1e-9,
        alpha=10,
    )

    n_co = int(control_bool.sum())
    n_ca = int(case_bool.sum())
    if n_co >= n_cells:
        n_coca_co = n_cells if n_ca >= n_cells else n_ca
    else:
        n_coca_co = int(n_co * 0.8) if n_ca >= n_co else n_ca
    n_coca_ca = min(n_ca, n_coca_co)
    control_case_samples: list[WeightedSample] = []
    for sample_idx in range(n_subsamples):
        np.random.seed(sample_idx)
        control_sampling = np.random.choice(control_bool[control_bool].index, n_coca_co, replace=False)
        case_sampling = np.random.choice(
            case_bool[case_bool].index,
            n_coca_ca,
            replace=False,
            p=case_weight_total[case_bool].values,
        )
        control_case_samples.append(
            WeightedSample(
                control_cells=tuple(control_sampling.tolist()),
                case_cells=tuple(case_sampling.tolist()),
                sample_id=f"co_ca_{sample_idx}",
            )
        )
    return control_case_samples


def build_sampling_plan(
    prepared_input: PreparedInput,
    tendency: PerturbationTendency | None = None,
    options: SamplingOptions | None = None,
    working_adata: Any = None,
    init_feature_selection: Any = None,
    guide_feature_selection: Any = None,
    margin_evaluation: Any = None,
) -> SamplingPlan:
    """Create the Step 1 weighted sampling handoff."""

    options = options or SamplingOptions()
    control_case_samples: Sequence[WeightedSample] = ()
    if tendency is not None and working_adata is not None:
        tendency_values = pd.Series(tendency.values, index=working_adata.obs_names)
        control_case_samples = _weighted_rswr_samples(
            adata=working_adata,
            cell_weight=np.clip(tendency_values, 0, 1),
            group_obs=prepared_input.contract.perturbation_key,
            control_label=prepared_input.contract.control_value,
            case_label=prepared_input.case_value,
            n_cells=options.n_cells,
            n_subsamples=options.n_subsamples,
            weight_method=options.weight_method,
        )
    return SamplingPlan(
        prepared_input=prepared_input,
        tendency=tendency,
        control_case_samples=control_case_samples,
        working_adata=working_adata,
        init_feature_selection=init_feature_selection,
        guide_feature_selection=guide_feature_selection,
        margin_evaluation=margin_evaluation,
        options=options,
    )


def require_sampled_cells(plan: SamplingPlan) -> None:
    """Fail clearly when Step 2 is called before real Step 1 sampling exists."""

    if not plan.control_case_samples:
        raise NotImplementedError(
            "Step 1 weighted multi-sampling is not migrated yet. "
            "SamplingPlan defines the internal contract, but contains no samples."
        )
