"""Output assembly contracts for CRANE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..io.result import CRANEResult
from ..io.schema import LoggerConfig
from .iteration import RefinementResult
from .scoring import ScoreResult


@dataclass(frozen=True)
class OutputOptions:
    """Final output options and low-visibility protective refinements."""

    return_anndata: bool = True
    refine_response_identity: bool = True
    refine_score_multiplier: int = 3
    filter_sparse_outliers: bool = False
    balanced_merge: bool = True
    deduplicate_cells: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputBundle:
    """Internal output payload before it is wrapped as ``CRANEResult``."""

    adata: Any
    gene_scores: Any
    response_identity: Any
    graph: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def assemble_result(
    refinement: RefinementResult,
    scores: ScoreResult,
    options: OutputOptions | None = None,
    logger_config: LoggerConfig | None = None,
) -> CRANEResult:
    """Wrap migrated CRANE outputs in the public result object."""

    options = options or OutputOptions()
    adata = refinement.sampling_plan.prepared_input.adata if options.return_anndata else None
    return CRANEResult(
        adata=adata,
        gene_scores=scores.gene_scores,
        response_identity=scores.response_identity,
        graph=refinement.converged_graph,
        metadata={
            "output_options": options,
            "scoring_background": dict(scores.background),
        },
        logger_config=logger_config,
    )
