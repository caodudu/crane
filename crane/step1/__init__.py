"""Step 1 implementation layer for the normalized CRANE package layout."""

from .feature_screen import (
    KSFeatureScreenOptions,
    KSFeatureScreenResult,
    compute_ks_statistics,
    compute_prefix_raw_pvalues,
    find_raw_pvalue_boundary,
    screen_ks_features,
)
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
from .step1 import (
    Step1FeatureSelectionResult,
    Step1Options,
    Step1Result,
    run_feature_selection,
    run_step1,
    run_tendency_evaluation,
)

__all__ = [
    "InputContract",
    "KSFeatureScreenOptions",
    "KSFeatureScreenResult",
    "PerturbationTendency",
    "PreparedInput",
    "PreprocessOptions",
    "SamplingOptions",
    "SamplingPlan",
    "Step1FeatureSelectionResult",
    "Step1Options",
    "Step1Result",
    "WeightedSample",
    "build_sampling_plan",
    "compute_ks_statistics",
    "compute_prefix_raw_pvalues",
    "find_raw_pvalue_boundary",
    "prepare_input",
    "require_sampled_cells",
    "run_feature_selection",
    "run_step1",
    "run_tendency_evaluation",
    "screen_ks_features",
    "summarize_input",
]
