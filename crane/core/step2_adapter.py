"""Step1-to-Step2 adapter for the default CRANE mainline.

This module owns the integration boundary between the paper-facing Step1 output
objects and the ndarray-first Step2 MVP runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..internal.logger import CRANELogger
from ..step2 import Step2Options, Step2RunResult, prepare_step2_packs, run_step2_serial, run_step2_threaded
from .step1 import Step1Result


@dataclass(frozen=True)
class Step2AdapterOptions:
    """Pipeline-level controls for Step2 integration."""

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
    adapter_options: Step2AdapterOptions | None = None,
    logger: CRANELogger | None = None,
) -> tuple[tuple[Any, ...], Step2RunResult]:
    """Bridge Step1 handoff objects into the default Step2 runtime."""

    adapter_options = adapter_options or Step2AdapterOptions()
    runner = _normalize_runner(adapter_options.runner)
    packs = prepare_step2_packs(
        sampling_plan=step1_result.sampling_plan,
        fs_input=step1_result.sampling_plan.init_feature_selection,
        guide_fs_input=step1_result.sampling_plan.guide_feature_selection,
        sample_layer=adapter_options.sample_layer,
    )
    if logger is not None:
        guide_fs = step1_result.sampling_plan.guide_feature_selection
        guide_active_genes = None
        if guide_fs is not None:
            guide_active_genes = int(np.sum(np.asarray(guide_fs.to_numpy()) == 1))
        logger.event(
            "pipeline.handoff.step1_to_step2",
            "Prepared Step1 handoff for Step2.",
            audience="reviewer",
            sample_count=len(packs),
            sample_layer=adapter_options.sample_layer or "X",
            initial_active_genes=int(np.sum(packs[0].fs_mask)),
            guide_active_genes=guide_active_genes,
        )
    if runner == "threaded":
        result = run_step2_threaded(
            packs,
            iterations=adapter_options.iterations,
            options=adapter_options.step2_options,
            max_workers=adapter_options.max_workers,
        )
    else:
        result = run_step2_serial(
            packs,
            iterations=adapter_options.iterations,
            options=adapter_options.step2_options,
        )
    return packs, result


def build_step2_public_outputs(
    step1_result: Step1Result,
    step2_result: Step2RunResult,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Convert ndarray-first Step2 outputs into public package objects."""

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
