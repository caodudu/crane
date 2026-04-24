"""Step 2 ndarray-first contracts for the default CRANE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Step2Options:
    """Default Step 2 controls for the current CRANE path."""

    n_pcs: int = 50
    cell_k: int = 10
    max_iterations: int = 50
    stable_rounds: int = 4
    score_mode: str = "self_n_label"
    threshold_k: float = 2.0
    drop_limit: bool = True
    dtype: str = "float32"
    force_connect: bool = True
    _smooth_weight: float = 0.15
    _smooth_alpha: float = 0.12
    _binary_delta_threshold: float = 0.1
    _guide_compare_rounds: int | None = None
    _relaxed_threshold_on_weak_pert: bool = True
    _relaxed_threshold_latch_weak_pert: bool = False
    _relaxed_threshold_min_k: float | None = None
    _relaxed_threshold_scale: float = 0.6
    _legacy_wave_compare: bool = True
    _legacy_wave_delta_threshold: int = 0
    _legacy_post_stable_rounds: int = 3
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step2SampleInput:
    """One balanced control-case sample before packing."""

    exp_raw: np.ndarray
    control_cells: Sequence[Any]
    case_cells: Sequence[Any]
    sample_id: str | None = None


@dataclass(frozen=True)
class Step2SamplePack:
    """One sample in the compact ndarray-first runtime form."""

    exp_raw: np.ndarray
    label_raw: np.ndarray
    fs_mask: np.ndarray
    group_labels: np.ndarray
    aux_fs_mask: np.ndarray | None = None
    sample_id: str | None = None
    control_cells: Sequence[Any] = field(default_factory=tuple)
    case_cells: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class Step2SampleOutput:
    """Single-sample output retained across Step 2 iterations."""

    exp_last_next: np.ndarray
    label_last_next: np.ndarray
    affinity: np.ndarray
    gene_self_cor: np.ndarray
    gene_label_cor: np.ndarray
    combined_score: np.ndarray
    norm_combined_score: np.ndarray
    branch_ready_next: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Step2State:
    """Minimal cross-iteration state for the default Step 2 path."""

    exp_last_list: tuple[np.ndarray | None, ...]
    label_last_list: tuple[np.ndarray | None, ...]
    ri_mask: np.ndarray
    branch_ready_list: tuple[bool, ...] = field(default_factory=tuple)
    iteration: int = 0


@dataclass(frozen=True)
class Step2RunResult:
    """Step 2 result package shared by the serial and threaded runners."""

    state: Step2State
    sample_outputs: tuple[Step2SampleOutput, ...]
    response_score: np.ndarray
    response_identity: np.ndarray
    result_space_identity: np.ndarray | None = None
    representative_sample_indices: tuple[int, ...] = field(default_factory=tuple)
    iter_times_s: tuple[float, ...] = field(default_factory=tuple)
    ri_history: tuple[np.ndarray, ...] = field(default_factory=tuple)
    score_history: tuple[np.ndarray, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
