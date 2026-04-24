"""Step1-only cell-response execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..internal.logger import CRANELogger
from ..io.result import CRANEResult
from ..io.schema import CRANEConfig
from ..step1 import (
    InputContract,
    PreprocessOptions,
    Step1Options,
    prepare_input,
    run_feature_selection,
    run_tendency_evaluation,
)


@dataclass(frozen=True)
class CellResponseExecution:
    prepared_input: Any
    feature_selection: Any
    tendency: Any
    graph_method: str
    n_neighbors: int
    compute_diagnostics: bool


def _build_feature_selection_options(
    *,
    config: CRANEConfig,
    graph_method: str,
    n_neighbors: int,
    advanced: dict[str, Any],
) -> Step1Options:
    internal = config.internal_options
    return Step1Options(
        n_top=internal.get("n_top"),
        n_bottom=internal.get("n_bottom"),
        extra_genes_dict=internal.get("function_gene_set") or internal.get("extra_genes_dict"),
        init_ratio=advanced.get("init_ratio"),
        init_ratio_case_cap=advanced.get("init_ratio_case_cap", 320),
        init_ratio_case_cap_min_cells=advanced.get("init_ratio_case_cap_min_cells", 640),
        init_ratio_case_cap_random_state=advanced.get("init_ratio_case_cap_random_state", 20260420),
        fix_point_factor=advanced.get("fix_point_factor", 0.3),
        weak_ratio=advanced.get("weak_ratio", 0.03),
        n_neighbors=n_neighbors,
        graph_method=graph_method,
        n_bg_permutations=advanced.get("n_set", 0),
    )


def _build_tendency_options(
    *,
    graph_method: str,
    n_neighbors: int,
    advanced: dict[str, Any],
) -> tuple[Step1Options, bool]:
    compute_diagnostics = bool(advanced.get("compute_diagnostics", False))
    return (
        Step1Options(
            n_neighbors=n_neighbors,
            graph_method=graph_method,
            compute_tendency_diagnostics=compute_diagnostics,
        ),
        compute_diagnostics,
    )


def execute_cell_response(
    *,
    adata: Any,
    perturbation_key: str,
    control_value: Any,
    case_value: Any | None,
    layer: str | None,
    graph_method: str,
    n_neighbors: int,
    config: CRANEConfig | None,
    logger: CRANELogger,
    advanced: dict[str, Any],
) -> CellResponseExecution:
    run_config = config or CRANEConfig()
    logger.user("Started.")
    logger.event("cell_response.entry", "CRANE cell-response entry initialized.", audience="reviewer")
    logger.event(
        "cell_response.stage.preprocess",
        "Preparing CRANE input contract for cell response.",
        audience="reviewer",
    )
    prepared_input = prepare_input(
        adata=adata,
        contract=InputContract(
            perturbation_key=perturbation_key,
            control_value=control_value,
            case_value=case_value,
            expression_layer=layer,
        ),
        options=PreprocessOptions(
            batch_key=advanced.get("batch_key"),
            preprocess_mode="baseline",
            extras={
                key: value
                for key, value in advanced.items()
                if key.startswith("preprocess_")
            },
        ),
    )

    logger.event(
        "cell_response.stage.feature_selection",
        "Running Step1 feature selection for cell response.",
        audience="reviewer",
        n_cells=prepared_input.n_cells,
        n_genes=prepared_input.n_genes,
        case_value=prepared_input.case_value,
        expression_layer=prepared_input.expression_layer,
    )
    feature_selection = run_feature_selection(
        prepared_input=prepared_input,
        options=_build_feature_selection_options(
            config=run_config,
            graph_method=graph_method,
            n_neighbors=n_neighbors,
            advanced=advanced,
        ),
    )
    logger.debug(
        "Step1 feature selection completed.",
        strong_feature_count=feature_selection.metadata["strong_feature_count"],
        reserve_deg_count=feature_selection.metadata["reserve_deg_count"],
        guide_feature_count=feature_selection.metadata.get("guide_feature_count"),
    )
    tendency_options, compute_diagnostics = _build_tendency_options(
        graph_method=graph_method,
        n_neighbors=n_neighbors,
        advanced=advanced,
    )
    logger.user("Score cell perturbation tendency.")
    logger.event(
        "cell_response.stage.tendency",
        "Running Step1 cell perturbation tendency scoring.",
        audience="reviewer",
        graph_method=graph_method,
        n_neighbors=n_neighbors,
    )
    tendency = run_tendency_evaluation(
        prepared_input=prepared_input,
        feature_selection=feature_selection,
        options=tendency_options,
    )
    return CellResponseExecution(
        prepared_input=prepared_input,
        feature_selection=feature_selection,
        tendency=tendency,
        graph_method=graph_method,
        n_neighbors=n_neighbors,
        compute_diagnostics=compute_diagnostics,
    )


def build_cell_response_result(
    execution: CellResponseExecution,
    *,
    logger: CRANELogger,
) -> CRANEResult:
    cell_response_metadata = {
        "graph_method": execution.graph_method,
        "n_neighbors": execution.n_neighbors,
        "ident_method": execution.tendency.metadata.get("ident_method"),
        "moran_I": execution.tendency.metadata.get("moran_I"),
    }
    if execution.compute_diagnostics:
        cell_response_metadata["moran_adj_I"] = execution.tendency.metadata.get("moran_adj_I")
    result = CRANEResult(
        adata=None,
        cell_scores=execution.tendency.values.copy(),
        metadata={
            "mode": "cell_response",
            "step1_summary": {
                "case_value": execution.prepared_input.case_value,
                "expression_layer": execution.prepared_input.expression_layer,
                "strong_feature_count": execution.feature_selection.metadata["strong_feature_count"],
                "reserve_deg_count": execution.feature_selection.metadata["reserve_deg_count"],
            },
            "cell_response": cell_response_metadata,
        },
        logger_config=logger.config,
    )
    logger.event(
        "cell_response.complete",
        "CRANE cell response completed.",
        audience="reviewer",
        graph_method=execution.graph_method,
        case_value=execution.prepared_input.case_value,
    )
    logger.user("Finished.")
    return result
