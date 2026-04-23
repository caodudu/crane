"""Tool-level public API for CRANE."""

from __future__ import annotations

from typing import Any

from .core.pipeline import run_pipeline
from .core.preprocess import InputContract, PreprocessOptions, prepare_input
from .core.step1 import Step1Options, run_feature_selection, run_tendency_evaluation
from .functional.query import CRANEExtensionResult, evaluate_extension, evaluate_function
from .internal.logger import build_logger
from .internal.precheck import maybe_warn_step1_cold_start
from .io.result import CRANEResult
from .io.schema import CRANEConfig, LoggerConfig, RuntimeOptions


def _copy_adata_if_requested(adata: Any, copy: bool) -> Any:
    if not copy:
        return adata
    if not hasattr(adata, "copy"):
        raise TypeError("copy=True requires an AnnData-like object with a .copy() method.")
    return adata.copy()


def _normalize_runtime(
    *,
    random_state: int | None,
    copy: bool,
    inplace: bool,
) -> RuntimeOptions:
    return RuntimeOptions(
        random_state=random_state,
        verbose=False,
        return_anndata=copy or inplace,
    )


def _write_tendency_to_adata(
    *,
    adata: Any,
    tendency,
    prepared_input,
    key_added: str,
    graph_method: str,
) -> Any:
    obs_key = f"{key_added}_cell_score"
    uns_block = dict(getattr(adata, "uns", {}).get(key_added, {}))
    adata.obs[obs_key] = tendency.values.reindex(adata.obs_names).astype("float32")
    uns_block.update(
        {
            "kind": "crane",
            "latest": "cell_response",
            "cell_response": {
                "score_key": obs_key,
                "perturbation_key": prepared_input.contract.perturbation_key,
                "control_value": prepared_input.contract.control_value,
                "case_value": prepared_input.case_value,
                "layer": prepared_input.expression_layer,
                "graph_method": graph_method,
            },
        }
    )
    adata.uns[key_added] = uns_block
    return adata


def gene_response(
    adata: Any,
    perturbation_key: str,
    control_value: Any,
    *,
    case_value: Any | None = None,
    layer: str | None = None,
    key_added: str = "crane",
    inplace: bool = False,
    copy: bool = False,
    random_state: int | None = None,
    config: CRANEConfig | None = None,
    logger_config: LoggerConfig | None = None,
    **advanced: Any,
) -> CRANEResult | Any | None:
    """Run CRANE gene-level response evaluation on an AnnData object.

    Default behavior returns a `CRANEResult` instead of mutating the input
    object. This keeps CRANE's internal result-space objects separate from the
    original AnnData unless the user explicitly requests input-space summary
    write-back with `inplace=True`.
    """

    target = _copy_adata_if_requested(adata, copy=copy)
    run_config = (config or CRANEConfig()).with_runtime_inputs(
        perturbation_key=perturbation_key,
        control_value=control_value,
        overrides={
            "case_value": case_value,
            "expression_layer": layer,
            **advanced,
        },
    )
    runtime = _normalize_runtime(random_state=random_state, copy=copy, inplace=inplace)
    logger_cfg = logger_config or LoggerConfig()
    result = run_pipeline(
        adata=target,
        config=run_config,
        runtime=runtime,
        logger=build_logger(logger_cfg).bind("tl.gene_response"),
    )

    if not inplace:
        return result

    adata_out = result.write_to_adata(target, key_added=key_added)
    if copy:
        return adata_out
    return None


def cell_response(
    adata: Any,
    perturbation_key: str,
    control_value: Any,
    *,
    case_value: Any | None = None,
    layer: str | None = None,
    key_added: str = "crane",
    inplace: bool = False,
    copy: bool = False,
    random_state: int | None = None,
    config: CRANEConfig | None = None,
    logger_config: LoggerConfig | None = None,
    graph_method: str = "umap",
    n_neighbors: int = 20,
    init_ratio: float | None = None,
    init_ratio_case_cap: int | None = 320,
    compute_diagnostics: bool = False,
    **advanced: Any,
) -> CRANEResult | Any | None:
    """Run the Step 1 tendency path and stop at cell-level perturbation response."""

    target = _copy_adata_if_requested(adata, copy=copy)
    logger = build_logger(logger_config or LoggerConfig()).bind("tl.cell_response")
    logger.user("Started.")
    logger.event("cell_response.entry", "CRANE cell-response entry initialized.", audience="reviewer")
    logger.event(
        "cell_response.stage.preprocess",
        "Preparing CRANE input contract for cell response.",
        audience="reviewer",
    )
    prepared_input = prepare_input(
        adata=target,
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
    maybe_warn_step1_cold_start(graph_method=graph_method, logger=logger)
    base_config = config or CRANEConfig()
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
        options=Step1Options(
            n_top=base_config.advanced_options.get("n_top"),
            n_bottom=base_config.advanced_options.get("n_bottom"),
            extra_genes_dict=base_config.advanced_options.get("function_gene_set")
            or base_config.advanced_options.get("extra_genes_dict"),
            init_ratio=init_ratio,
            init_ratio_case_cap=init_ratio_case_cap,
            init_ratio_case_cap_min_cells=advanced.get("init_ratio_case_cap_min_cells", 640),
            init_ratio_case_cap_random_state=advanced.get("init_ratio_case_cap_random_state", 20260420),
            fix_point_factor=advanced.get("fix_point_factor", 0.3),
            weak_ratio=advanced.get("weak_ratio", 0.03),
            n_neighbors=n_neighbors,
            graph_method=graph_method,
            n_bg_permutations=advanced.get("n_set", 0),
        ),
    )
    logger.debug(
        "Step1 feature selection completed.",
        strong_feature_count=feature_selection.metadata["strong_feature_count"],
        reserve_deg_count=feature_selection.metadata["reserve_deg_count"],
        guide_feature_count=feature_selection.metadata.get("guide_feature_count"),
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
        options=Step1Options(
            n_neighbors=n_neighbors,
            graph_method=graph_method,
            compute_tendency_diagnostics=compute_diagnostics,
        ),
    )
    cell_response_metadata = {
        "graph_method": graph_method,
        "n_neighbors": n_neighbors,
        "ident_method": tendency.metadata.get("ident_method"),
        "moran_I": tendency.metadata.get("moran_I"),
    }
    if compute_diagnostics:
        cell_response_metadata["moran_adj_I"] = tendency.metadata.get("moran_adj_I")
    result = CRANEResult(
        adata=target,
        cell_scores=tendency.values.copy(),
        metadata={
            "mode": "cell_response",
            "step1_summary": {
                "case_value": prepared_input.case_value,
                "expression_layer": prepared_input.expression_layer,
                "strong_feature_count": feature_selection.metadata["strong_feature_count"],
                "reserve_deg_count": feature_selection.metadata["reserve_deg_count"],
            },
            "cell_response": cell_response_metadata,
        },
        logger_config=logger.config,
    )

    logger.event(
        "cell_response.complete",
        "CRANE cell response completed.",
        audience="reviewer",
        graph_method=graph_method,
        case_value=prepared_input.case_value,
    )
    logger.user("Finished.")
    if not inplace:
        return result

    adata_out = _write_tendency_to_adata(
        adata=target,
        tendency=tendency,
        prepared_input=prepared_input,
        key_added=key_added,
        graph_method=graph_method,
    )
    if copy:
        return adata_out
    return None


def extension_response(
    adata: Any,
    *,
    result: CRANEResult | Any,
    gene_set: Any = None,
    gene_vector: Any = None,
    cell_vector: Any = None,
    layer: str | None = None,
    label_key: str | None = None,
    affinity_key: str = "affinity",
    set_min_genes_count: int = 10,
    set_loading_threshold: float = 0.5,
    set_embedding_threshold: float = 0.1,
    vector_min_genes_count: int = 50,
    logger_config: LoggerConfig | None = None,
    **metadata: Any,
) -> CRANEExtensionResult:
    """Evaluate extra gene-set/gene-vector/cell-vector inputs on a CRANE result graph.

    ``adata`` supplies the expression space for gene-set and gene-vector inputs,
    while ``result.result_ad`` supplies the converged CRANE graph and labels.
    This mirrors the legacy EctBottle query split without rerunning gene_response.
    """

    logger = build_logger(logger_config or getattr(result, "logger_config", None) or LoggerConfig()).bind(
        "tl.extension_response"
    )
    return evaluate_extension(
        adata,
        result=result,
        gene_set=gene_set,
        gene_vector=gene_vector,
        cell_vector=cell_vector,
        logger=logger,
        layer=layer,
        label_key=label_key,
        affinity_key=affinity_key,
        set_min_genes_count=set_min_genes_count,
        set_loading_threshold=set_loading_threshold,
        set_embedding_threshold=set_embedding_threshold,
        vector_min_genes_count=vector_min_genes_count,
        metadata=metadata,
    )


def function_response(
    adata: Any,
    *,
    result: CRANEResult | Any,
    gene_set: Any = None,
    label_key: str | None = None,
    affinity_key: str = "affinity",
    layer: str | None = None,
    set_min_genes_count: int = 10,
    set_loading_threshold: float = 0.5,
    set_embedding_threshold: float = 0.1,
    logger_config: LoggerConfig | None = None,
    **metadata: Any,
) -> CRANEExtensionResult:
    """Evaluate functional gene sets from an AnnData plus a CRANE gene-response result."""

    logger = build_logger(logger_config or getattr(result, "logger_config", None) or LoggerConfig()).bind(
        "tl.function_response"
    )
    return evaluate_function(
        adata,
        result=result,
        gene_set=gene_set,
        logger=logger,
        label_key=label_key,
        affinity_key=affinity_key,
        layer=layer,
        set_min_genes_count=set_min_genes_count,
        set_loading_threshold=set_loading_threshold,
        set_embedding_threshold=set_embedding_threshold,
        metadata=metadata,
    )
