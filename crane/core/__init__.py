"""Core CRANE pipeline layer."""

from .iteration import (
    RefinementOptions,
    RefinementResult,
    RefinementTrace,
    ResponseIdentityState,
    run_closed_loop_refinement,
)
from .output import OutputBundle, OutputOptions, assemble_result
from .pipeline import run_pipeline
from .preprocess import (
    InputContract,
    PreparedInput,
    PreprocessOptions,
    prepare_input,
    summarize_input,
)
from .sampling import (
    PerturbationTendency,
    SamplingOptions,
    SamplingPlan,
    WeightedSample,
    build_sampling_plan,
    require_sampled_cells,
)
from .feature_screen import (
    KSFeatureScreenOptions,
    KSFeatureScreenResult,
    compute_ks_statistics,
    compute_guide_prefix_raw_pvalues,
    find_raw_pvalue_boundary,
    screen_ks_features,
)
from .step1 import Step1FeatureSelectionResult, Step1Options, Step1Result, run_feature_selection, run_step1
from .step2_adapter import Step2AdapterOptions, build_step2_public_outputs, run_step2_from_step1
from .scoring import (
    MoranComponents,
    ScoreResult,
    ScoringOptions,
    compute_response_scores,
)

__all__ = [
    "InputContract",
    "KSFeatureScreenOptions",
    "KSFeatureScreenResult",
    "MoranComponents",
    "OutputOptions",
    "OutputBundle",
    "PerturbationTendency",
    "PreparedInput",
    "PreprocessOptions",
    "RefinementOptions",
    "RefinementResult",
    "RefinementTrace",
    "ResponseIdentityState",
    "SamplingOptions",
    "SamplingPlan",
    "ScoreResult",
    "ScoringOptions",
    "Step1FeatureSelectionResult",
    "Step1Options",
    "Step1Result",
    "Step2AdapterOptions",
    "WeightedSample",
    "assemble_result",
    "build_step2_public_outputs",
    "build_sampling_plan",
    "compute_ks_statistics",
    "compute_guide_prefix_raw_pvalues",
    "compute_response_scores",
    "find_raw_pvalue_boundary",
    "prepare_input",
    "require_sampled_cells",
    "run_feature_selection",
    "run_closed_loop_refinement",
    "run_pipeline",
    "run_step1",
    "run_step2_from_step1",
    "screen_ks_features",
    "summarize_input",
]
