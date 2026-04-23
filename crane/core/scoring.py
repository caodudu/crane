"""Graph-aware response scoring contracts for CRANE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .iteration import RefinementResult


@dataclass(frozen=True)
class MoranComponents:
    """Moran-style components used to form CRANE response score."""

    gene_self_correlation: Any
    gene_label_correlation: Any
    gene_module_correlation: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringOptions:
    """Score options including paper-hidden protective normalization hooks."""

    response_score_mode: str = "l2_moran"
    subtract_background_median: bool = True
    clip_negative_scores: bool = True
    enable_false_positive_filter: bool = False
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreResult:
    """Gene-level CRANE scoring output."""

    gene_scores: Any
    response_identity: Any
    moran_components: MoranComponents | None = None
    background: Mapping[str, Any] = field(default_factory=dict)
    options: ScoringOptions = field(default_factory=ScoringOptions)


def compute_response_scores(
    refinement: RefinementResult,
    options: ScoringOptions | None = None,
) -> ScoreResult:
    """Compute RS and RI from a converged refinement result."""

    options = options or ScoringOptions()
    raise NotImplementedError(
        "Graph-aware Moran scoring is not migrated yet. "
        "ScoreResult defines the target RS/RI boundary."
    )
