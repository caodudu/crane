"""Pipeline orchestration skeleton for CRANE."""

from __future__ import annotations

from typing import Any

from ..internal.logger import CRANELogger
from ..internal.precheck import maybe_warn_step1_cold_start
from ..io.result import (
    CRANEResult,
    build_result_anndata,
    resolve_public_response_score,
)
from ..io.schema import CRANEConfig, RuntimeOptions
from .preprocess import InputContract, PreprocessOptions, prepare_input
from .sampling import SamplingOptions
from .step2_adapter import Step2AdapterOptions, build_step2_public_outputs, run_step2_from_step1
from .step1 import Step1Options, run_step1
from ..step2 import Step2Options


def run_pipeline(
    adata: Any,
    config: CRANEConfig,
    runtime: RuntimeOptions,
    logger: CRANELogger,
) -> CRANEResult:
    """Run the formal Step 1 mainline for the new CRANE package."""

    logger.user("Started.")
    logger.event("pipeline.entry", "CRANE pipeline entry initialized.", audience="reviewer")
    logger.event(
        "pipeline.stage.preprocess",
        "Preparing CRANE input contract.",
        audience="reviewer",
    )
    contract = InputContract(
        perturbation_key=config.perturbation_key or "",
        control_value=config.control_value,
        case_value=config.advanced_options.get("case_value"),
        expression_layer=config.advanced_options.get("expression_layer"),
    )
    prepared_input = prepare_input(
        adata=adata,
        contract=contract,
        options=PreprocessOptions(
            batch_key=config.advanced_options.get("batch_key"),
            preprocess_mode="baseline",
            extras={
                key: value
                for key, value in config.advanced_options.items()
                if key.startswith("preprocess_")
            },
        ),
    )
    maybe_warn_step1_cold_start(
        graph_method=config.advanced_options.get("graph_method", "umap"),
        logger=logger,
    )
    logger.user("Step1: Balanced sampling (Perturbation-Control).")
    logger.event(
        "pipeline.stage.step1",
        "Running formal Step 1 mainline.",
        audience="reviewer",
        n_cells=prepared_input.n_cells,
        n_genes=prepared_input.n_genes,
        case_value=prepared_input.case_value,
        expression_layer=prepared_input.expression_layer,
    )
    step1_result = run_step1(
        prepared_input=prepared_input,
        step1_options=Step1Options(
            n_top=config.advanced_options.get("n_top"),
            n_bottom=config.advanced_options.get("n_bottom"),
            extra_genes_dict=config.advanced_options.get("function_gene_set")
            or config.advanced_options.get("extra_genes_dict"),
            init_ratio=config.advanced_options.get("init_ratio"),
            init_ratio_case_cap=config.advanced_options.get("init_ratio_case_cap", 320),
            init_ratio_case_cap_min_cells=config.advanced_options.get("init_ratio_case_cap_min_cells", 640),
            init_ratio_case_cap_random_state=config.advanced_options.get(
                "init_ratio_case_cap_random_state",
                20260420,
            ),
            fix_point_factor=config.advanced_options.get("fix_point_factor", 0.3),
            weak_ratio=config.advanced_options.get("weak_ratio", 0.03),
            n_neighbors=config.advanced_options.get("n_neighbors", 20),
            graph_method=config.advanced_options.get("graph_method", "umap"),
            n_bg_permutations=config.advanced_options.get("n_set", 0),
        ),
        sampling_options=SamplingOptions(
            n_cells=config.advanced_options.get("n_cells", 50),
            n_subsamples=config.advanced_options.get("n_subsamples", 5),
            weight_method=config.advanced_options.get("weight_method", "softmax"),
            random_state=runtime.random_state,
        ),
    )
    logger.user("Step2: Co-refinement (Cell graph <-> Gene evaluation).")
    logger.event(
        "pipeline.stage.step2",
        "Running default Step 2 no-module MVP.",
        audience="reviewer",
        strong_feature_count=step1_result.metadata["strong_feature_count"],
        sample_count=step1_result.metadata["sample_count"],
    )
    sample_packs, step2_result = run_step2_from_step1(
        step1_result=step1_result,
        adapter_options=Step2AdapterOptions(
            sample_layer=config.advanced_options.get("sample_layer", "central_norm"),
            runner=config.advanced_options.get("step2_runner", "serial"),
            max_workers=config.advanced_options.get("step2_max_workers", 2),
            iterations=config.advanced_options.get("step2_iterations"),
            step2_options=Step2Options(
                n_pcs=config.advanced_options.get("step2_n_pcs", 50),
                cell_k=config.advanced_options.get("step2_cell_k", 10),
                max_iterations=config.advanced_options.get("step2_max_iterations", 50),
                stable_rounds=config.advanced_options.get("step2_stable_rounds", 4),
                score_mode=config.advanced_options.get("step2_score_mode", "self_n_label"),
                threshold_k=config.advanced_options.get("step2_threshold_k", 2.0),
                drop_limit=config.advanced_options.get("step2_drop_limit", True),
                dtype=config.advanced_options.get("step2_dtype", "float32"),
                force_connect=config.advanced_options.get("step2_force_connect", True),
                _smooth_weight=config.advanced_options.get("step2_smooth_weight", 0.15),
                _smooth_alpha=config.advanced_options.get("step2_smooth_alpha", 0.12),
                _binary_delta_threshold=config.advanced_options.get("step2_binary_delta_threshold", 0.1),
                _guide_compare_rounds=config.advanced_options.get("_step2_guide_compare_rounds"),
                _relaxed_threshold_on_weak_pert=config.advanced_options.get(
                    "_step2_relaxed_threshold_on_weak_pert",
                    True,
                ),
                _relaxed_threshold_latch_weak_pert=config.advanced_options.get(
                    "_step2_relaxed_threshold_latch_weak_pert",
                    False,
                ),
                _relaxed_threshold_min_k=config.advanced_options.get("_step2_relaxed_threshold_min_k"),
                _relaxed_threshold_scale=config.advanced_options.get("_step2_relaxed_threshold_scale", 0.6),
                _legacy_post_stable_rounds=config.advanced_options.get("_step2_legacy_post_stable_rounds", 3),
            ),
        ),
        logger=logger,
    )
    gene_scores, response_identity, graph = build_step2_public_outputs(
        step1_result=step1_result,
        step2_result=step2_result,
    )
    response_score_merge_enabled = bool(config.advanced_options.get("_result_score_history_merge", False))
    gene_scores, response_score_meta = resolve_public_response_score(
        gene_scores=gene_scores,
        step2_result=step2_result,
        enable_history_merge=response_score_merge_enabled,
    )
    result_ad = None
    if config.advanced_options.get("_result_build", True):
        result_ad = build_result_anndata(
            gene_names=gene_scores.index,
            gene_scores=gene_scores,
            response_identity=response_identity,
            sample_outputs=step2_result.sample_outputs,
            sample_packs=sample_packs,
            step2_result=step2_result,
            metadata={
                "mode": "gene_response",
                "runner": step2_result.metadata.get("runner"),
                "iterations": step2_result.metadata.get("iterations"),
                "sample_count": step2_result.metadata.get("sample_count"),
                "active_genes": step2_result.metadata.get("active_genes"),
                "step2_threshold_k": config.advanced_options.get("step2_threshold_k", 2.0),
                "post_step2_refine_k": config.advanced_options.get("_result_refine_k", 3.0),
                "response_score_source": response_score_meta.get("source"),
                "response_score_history_merge": bool(response_score_meta.get("history_merge_enabled", False)),
                "response_score_history_merge_applied": bool(response_score_meta.get("history_merge_applied", False)),
                "response_score_history_rounds": response_score_meta.get("history_rounds"),
            },
            merge_top_n=config.advanced_options.get("_result_merge_top_n", 3),
            merge_mode=(
                "all"
                if config.advanced_options.get("n_cells", 50) > 80
                else config.advanced_options.get("_result_merge_mode", "legacy_representative")
            ),
            graph_method=config.advanced_options.get("_result_graph_method", "gauss"),
            compute_gene_pair=config.advanced_options.get("_result_compute_gene_pair", False),
            graph_fs_mode=(
                config.advanced_options.get("_result_graph_fs_mode", "default")
                if config.advanced_options.get("_result_merge_mode", "legacy_representative") == "legacy_representative"
                else config.advanced_options.get("_result_graph_fs_mode", "default")
            ),
        )
    logger.user("Finished.")
    logger.event(
        "pipeline.complete",
        "CRANE Step1-Step2 pipeline completed.",
        audience="reviewer",
        active_genes=int(response_identity.sum()),
        step2_iterations=step2_result.metadata.get("iterations"),
        step2_runner=step2_result.metadata.get("runner"),
    )
    return CRANEResult(
        adata=prepared_input.adata if runtime.return_anndata else None,
        result_ad=result_ad,
        gene_scores=gene_scores,
        response_identity=response_identity,
        graph=graph,
        metadata={
            "step1": step1_result,
            "step2": step2_result,
            "step1_summary": {
                "case_value": prepared_input.case_value,
                "expression_layer": prepared_input.expression_layer,
                "strong_feature_count": step1_result.metadata["strong_feature_count"],
                "sample_count": step1_result.metadata["sample_count"],
            },
            "step2_summary": {
                "runner": step2_result.metadata.get("runner"),
                "iterations": step2_result.metadata.get("iterations"),
                "converged": step2_result.metadata.get("converged"),
                "active_genes": step2_result.metadata.get("active_genes"),
                "score_mode": config.advanced_options.get("step2_score_mode", "self_n_label"),
                "sample_layer": config.advanced_options.get("sample_layer", "central_norm"),
                "response_score_source": response_score_meta.get("source"),
                "response_score_history_merge": bool(response_score_meta.get("history_merge_enabled", False)),
                "response_score_history_merge_applied": bool(response_score_meta.get("history_merge_applied", False)),
                "response_score_history_rounds": response_score_meta.get("history_rounds"),
            },
        },
        logger_config=logger.config,
    )
