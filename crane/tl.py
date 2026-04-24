"""Tool-level public API for CRANE."""

from __future__ import annotations

from typing import Any

from .core.cell_response import build_cell_response_result, execute_cell_response
from .core.pipeline import run_pipeline
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
    copy: bool,
    inplace: bool,
) -> RuntimeOptions:
    return RuntimeOptions(
        random_state=None,
        verbose=False,
        return_anndata=copy or inplace,
    )


def _drop_legacy_runtime_overrides(advanced: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(advanced)
    cleaned.pop("random_state", None)
    return cleaned


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
    config: CRANEConfig | None = None,
    logger_config: LoggerConfig | None = None,
    n_neighbors: int | None = None,
    n_cells: int | None = None,
    n_subsamples: int | None = None,
    step2_cell_k: int | None = None,
    **advanced: Any,
) -> CRANEResult | Any | None:
    """Run CRANE gene-level response evaluation on an AnnData object.

    Recommended public controls are kept intentionally small:
    `perturbation_key`, `control_value`, optional `case_value`, `layer`,
    `n_neighbors`, `n_cells`, `n_subsamples`, and `step2_cell_k`.
    Other legacy/internal kwargs remain accepted through `**advanced` for
    compatibility, but are not part of the recommended public surface.
    """

    target = _copy_adata_if_requested(adata, copy=copy)
    advanced = _drop_legacy_runtime_overrides(advanced)
    overrides: dict[str, Any] = {
        "case_value": case_value,
        "expression_layer": layer,
        **advanced,
    }
    if n_neighbors is not None:
        overrides["n_neighbors"] = n_neighbors
    if n_cells is not None:
        overrides["n_cells"] = n_cells
    if n_subsamples is not None:
        overrides["n_subsamples"] = n_subsamples
    if step2_cell_k is not None:
        overrides["step2_cell_k"] = step2_cell_k
    run_config = (config or CRANEConfig()).with_runtime_inputs(
        perturbation_key=perturbation_key,
        control_value=control_value,
        overrides=overrides,
    )
    runtime = _normalize_runtime(copy=copy, inplace=inplace)
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
    config: CRANEConfig | None = None,
    logger_config: LoggerConfig | None = None,
    graph_method: str = "umap",
    n_neighbors: int = 20,
    **advanced: Any,
) -> CRANEResult | Any | None:
    """Run the Step 1 tendency path and stop at cell-level perturbation response.

    The returned result is intentionally lightweight and does not retain the
    input AnnData handle unless the user explicitly requests write-back via
    ``inplace=True``.
    """

    target = _copy_adata_if_requested(adata, copy=copy)
    advanced = _drop_legacy_runtime_overrides(advanced)
    logger = build_logger(logger_config or LoggerConfig()).bind("tl.cell_response")
    maybe_warn_step1_cold_start(graph_method=graph_method, logger=logger)
    execution = execute_cell_response(
        adata=target,
        perturbation_key=perturbation_key,
        control_value=control_value,
        case_value=case_value,
        layer=layer,
        graph_method=graph_method,
        n_neighbors=n_neighbors,
        config=config,
        logger=logger,
        advanced=advanced,
    )
    result = build_cell_response_result(execution, logger=logger)
    if not inplace:
        return result

    adata_out = _write_tendency_to_adata(
        adata=target,
        tendency=execution.tendency,
        prepared_input=execution.prepared_input,
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
    """Unified graph-backed extension evaluator.

    ``result.result_ad`` provides the converged graph/labels/core result space.
    ``adata`` remains the raw-input handle for supplementing extra genes or
    modalities that are not carried in the core result space.
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
    """Convenience wrapper over ``extension_response(..., gene_set=...)``."""

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
