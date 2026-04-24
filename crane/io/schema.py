"""Shared interface schema objects for CRANE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_PUBLIC_CONFIG_DEFAULTS: dict[str, Any] = {
    "case_value": None,
    "expression_layer": None,
    "n_neighbors": 20,
    "n_cells": 50,
    "n_subsamples": 5,
    "step2_cell_k": 10,
}


def _extract_public_option_updates(config: "CRANEConfig", options: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key, default in _PUBLIC_CONFIG_DEFAULTS.items():
        if key not in options:
            continue
        value = options.pop(key)
        if getattr(config, key) == default:
            updates[key] = value
    return updates


@dataclass(frozen=True)
class LogFileConfig:
    enabled: bool = False
    directory: str | Path | None = None
    filename: str | None = None


@dataclass(frozen=True)
class LoggerConfig:
    name: str = "crane"
    console_level: str = "INFO"
    debug_console: bool = False
    reviewer_console: bool = False
    propagate: bool = False
    user_file: LogFileConfig = field(default_factory=LogFileConfig)
    reviewer_file: LogFileConfig = field(
        default_factory=lambda: LogFileConfig(filename="crane.reviewer.log")
    )
    debug_file: LogFileConfig = field(
        default_factory=lambda: LogFileConfig(filename="crane.debug.log")
    )


@dataclass(frozen=True)
class RuntimeOptions:
    random_state: int | None = None
    verbose: bool = False
    return_anndata: bool = True


@dataclass(frozen=True, init=False)
class CRANEConfig:
    """Visible CRANE configuration for paper parameters and common user controls."""

    perturbation_key: str | None = None
    control_value: Any = None
    case_value: Any = None
    expression_layer: str | None = None
    n_neighbors: int = 20
    n_cells: int = 50
    n_subsamples: int = 5
    step2_cell_k: int = 10
    internal_options: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __init__(
        self,
        perturbation_key: str | None = None,
        control_value: Any = None,
        *,
        case_value: Any = None,
        expression_layer: str | None = None,
        n_neighbors: int = 20,
        n_cells: int = 50,
        n_subsamples: int = 5,
        step2_cell_k: int = 10,
        **legacy_options: Any,
    ) -> None:
        advanced_options = legacy_options.pop("advanced_options", None)
        internal_options = legacy_options.pop("internal_options", None)
        if legacy_options:
            unexpected = ", ".join(sorted(legacy_options))
            raise TypeError(f"Unexpected CRANEConfig keyword(s): {unexpected}.")
        merged_internal: dict[str, Any] = {}
        if internal_options:
            merged_internal.update(dict(internal_options))
        if advanced_options:
            merged_internal.update(dict(advanced_options))
        object.__setattr__(self, "perturbation_key", perturbation_key)
        object.__setattr__(self, "control_value", control_value)
        object.__setattr__(self, "case_value", case_value)
        object.__setattr__(self, "expression_layer", expression_layer)
        object.__setattr__(self, "n_neighbors", n_neighbors)
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "n_subsamples", n_subsamples)
        object.__setattr__(self, "step2_cell_k", step2_cell_k)
        public_updates = _extract_public_option_updates(self, merged_internal)
        for key, value in public_updates.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "internal_options", merged_internal)

    @property
    def advanced_options(self) -> dict[str, Any]:
        return self.internal_options

    def with_runtime_inputs(
        self,
        perturbation_key: str,
        control_value: Any,
        overrides: dict[str, Any] | None = None,
    ) -> "CRANEConfig":
        overrides = dict(overrides or {})
        public_updates = {
            key: overrides.pop(key)
            for key in _PUBLIC_CONFIG_DEFAULTS
            if key in overrides
        }
        merged = dict(self.internal_options)
        merged.update(overrides)
        return CRANEConfig(
            perturbation_key=perturbation_key,
            control_value=control_value,
            case_value=public_updates.get("case_value", self.case_value),
            expression_layer=public_updates.get("expression_layer", self.expression_layer),
            n_neighbors=public_updates.get("n_neighbors", self.n_neighbors),
            n_cells=public_updates.get("n_cells", self.n_cells),
            n_subsamples=public_updates.get("n_subsamples", self.n_subsamples),
            step2_cell_k=public_updates.get("step2_cell_k", self.step2_cell_k),
            internal_options=merged,
        )


@dataclass(frozen=True)
class FunctionalInput:
    gene_set: Any = None
    gene_vector: Any = None
    cell_vector: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
