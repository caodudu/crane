"""Step1-to-Step2 bridge for the default CRANE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..internal.logger import CRANELogger
from ..step2 import Step2Options, Step2RunResult, prepare_step2_packs, run_step2_serial, run_step2_threaded
from ..step1.step1 import Step1Result


@dataclass(frozen=True)
class Step2BridgeOptions:
    """Pipeline-level controls for Step 2 integration."""

    sample_layer: str | None = None
    runner: str = "serial"
    max_workers: int = 2
    iterations: int | None = None
    step2_options: Step2Options = Step2Options()


def _normalize_runner(runner: str) -> str:
    normalized = str(runner).strip().lower()
    if normalized not in {"serial", "threaded"}:
        raise ValueError("step2 runner must be 'serial' or 'threaded'.")
    return normalized


def run_step2_from_step1(
    step1_result: Step1Result,
    bridge_options: Step2BridgeOptions | None = None,
    logger: CRANELogger | None = None,
) -> tuple[tuple[Any, ...], Step2RunResult]:
    """Build Step 2 sample packs from Step 1 outputs and run Step 2."""

    bridge_options = bridge_options or Step2BridgeOptions()
    runner = _normalize_runner(bridge_options.runner)
    packs = prepare_step2_packs(
        sampling_plan=step1_result.sampling_plan,
        fs_input=step1_result.sampling_plan.init_feature_selection,
        aux_fs_input=step1_result.sampling_plan.aux_feature_selection,
        sample_layer=bridge_options.sample_layer,
    )
    if logger is not None:
        aux_fs = step1_result.sampling_plan.aux_feature_selection
        guide_active_genes = None
        if aux_fs is not None:
            guide_active_genes = int(np.sum(np.asarray(aux_fs.to_numpy()) == 1))
        logger.event(
            "pipeline.bridge.step1_to_step2",
            "Prepared Step 2 inputs from Step 1 outputs.",
            audience="reviewer",
            sample_count=len(packs),
            sample_layer=bridge_options.sample_layer or "X",
            initial_active_genes=int(np.sum(packs[0].fs_mask)),
            guide_active_genes=guide_active_genes,
        )
    if runner == "threaded":
        result = run_step2_threaded(
            packs,
            iterations=bridge_options.iterations,
            options=bridge_options.step2_options,
            max_workers=bridge_options.max_workers,
        )
    else:
        result = run_step2_serial(
            packs,
            iterations=bridge_options.iterations,
            options=bridge_options.step2_options,
        )
    return packs, result


def build_step2_public_outputs(
    step1_result: Step1Result,
    step2_result: Step2RunResult,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Convert Step 2 outputs into the package-level result objects."""

    gene_names = pd.Index(step1_result.feature_selection.working_adata.var_names.copy(), name="gene")
    gene_scores = pd.Series(
        np.asarray(step2_result.response_score, dtype=np.float32),
        index=gene_names,
        name="response_score",
    )
    response_identity = pd.Series(
        np.asarray(step2_result.response_identity, dtype=np.int8),
        index=gene_names,
        name="response_identity",
    )
    graph = {
        "type": "per_sample_affinity",
        "sample_ids": tuple(output.metadata.get("sample_id") for output in step2_result.sample_outputs),
        "affinity_matrices": tuple(output.affinity for output in step2_result.sample_outputs),
        "runner": step2_result.metadata.get("runner"),
    }
    return gene_scores, response_identity, graph


Step2AdapterOptions = Step2BridgeOptions
