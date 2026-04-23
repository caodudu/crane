"""Compatibility wrappers for the top-level CRANE API."""

from __future__ import annotations

from typing import Any

from .io.result import CRANEResult
from .io.schema import CRANEConfig, LoggerConfig, RuntimeOptions
from .tl import gene_response


class CRANE:
    """Compatibility entry class that delegates to the tool-level API."""

    def __init__(
        self,
        config: CRANEConfig | None = None,
        runtime: RuntimeOptions | None = None,
        logger_config: LoggerConfig | None = None,
    ) -> None:
        self.config = config or CRANEConfig()
        self.runtime = runtime or RuntimeOptions()
        self.logger_config = logger_config or LoggerConfig()

    def fit(
        self,
        adata: Any,
        perturbation_key: str,
        control_value: Any,
        **overrides: Any,
    ) -> CRANEResult:
        """Run CRANE through the stable gene-response API."""
        return gene_response(
            adata=adata,
            perturbation_key=perturbation_key,
            control_value=control_value,
            case_value=overrides.pop("case_value", None),
            layer=overrides.pop("expression_layer", None),
            key_added=overrides.pop("key_added", "crane"),
            inplace=False,
            copy=False,
            random_state=overrides.pop("random_state", self.runtime.random_state),
            config=self.config,
            logger_config=self.logger_config,
            **overrides,
        )


def run_crane(
    adata: Any,
    perturbation_key: str,
    control_value: Any,
    config: CRANEConfig | None = None,
    runtime: RuntimeOptions | None = None,
    logger_config: LoggerConfig | None = None,
    **overrides: Any,
) -> CRANEResult:
    """Compatibility wrapper around :func:`crane.tl.gene_response`."""
    return gene_response(
        adata=adata,
        perturbation_key=perturbation_key,
        control_value=control_value,
        case_value=overrides.pop("case_value", None),
        layer=overrides.pop("expression_layer", None),
        key_added=overrides.pop("key_added", "crane"),
        inplace=False,
        copy=False,
        random_state=overrides.pop(
            "random_state",
            None if runtime is None else runtime.random_state,
        ),
        config=config,
        logger_config=logger_config,
        **overrides,
    )
