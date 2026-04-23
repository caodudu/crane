"""Shared interface schema objects for CRANE."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class CRANEConfig:
    perturbation_key: str | None = None
    control_value: Any = None
    advanced_options: dict[str, Any] = field(default_factory=dict)

    def with_runtime_inputs(
        self,
        perturbation_key: str,
        control_value: Any,
        overrides: dict[str, Any] | None = None,
    ) -> "CRANEConfig":
        overrides = overrides or {}
        merged = dict(self.advanced_options)
        merged.update(overrides)
        return replace(
            self,
            perturbation_key=perturbation_key,
            control_value=control_value,
            advanced_options=merged,
        )


@dataclass(frozen=True)
class FunctionalInput:
    gene_set: Any = None
    gene_vector: Any = None
    cell_vector: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
