"""Step 2 contracts: closed-loop graph and response refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .sampling import SamplingPlan, require_sampled_cells


@dataclass(frozen=True)
class ResponseIdentityState:
    """Binary response identity vector at one refinement point."""

    values: Any
    iteration: int
    source: str = "initial_feature_selection"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementOptions:
    """Closed-loop refinement options and low-visibility safeguards."""

    max_iterations: int = 50
    convergence_patience: int = 4
    delta_response_threshold: int = 1
    graph_smoothing_weight: float = 0.15
    graph_smoothing_alpha: float = 0.1
    threshold_scale: float = 2.0
    enable_early_graph_safeguard: bool = True
    enable_weak_perturbation_safeguard: bool = True
    background_correction_weight: float = 0.0
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementTrace:
    """Trace of Step 2 state updates."""

    states: Sequence[ResponseIdentityState]
    representative_sample_ids: Sequence[str] = field(default_factory=tuple)
    converged: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementResult:
    """Step 2 output passed to scoring and output assembly."""

    sampling_plan: SamplingPlan
    final_response_identity: ResponseIdentityState
    converged_graph: Any = None
    trace: RefinementTrace | None = None
    options: RefinementOptions = field(default_factory=RefinementOptions)


def run_closed_loop_refinement(
    sampling_plan: SamplingPlan,
    initial_response_identity: ResponseIdentityState | None = None,
    options: RefinementOptions | None = None,
) -> RefinementResult:
    """Run Step 2 once the legacy algorithm has been migrated."""

    require_sampled_cells(sampling_plan)
    options = options or RefinementOptions()
    initial_response_identity = initial_response_identity or ResponseIdentityState(
        values=None,
        iteration=0,
    )
    raise NotImplementedError(
        "Step 2 closed-loop refinement is not migrated yet. "
        "The internal CRANE refinement contract is now defined."
    )
