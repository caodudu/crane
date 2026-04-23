"""Input and preprocessing contracts for CRANE.

This module defines the first internal boundary of the paper-aligned CRANE
pipeline. It does not import or execute the legacy ``script/couture`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


@dataclass(frozen=True)
class InputContract:
    """User input fields required before CRANE can run."""

    perturbation_key: str
    control_value: Any
    case_value: Any | None = None
    expression_layer: str | None = None


@dataclass(frozen=True)
class PreprocessOptions:
    """Low-visibility preprocessing options reserved for the migration layer."""

    batch_key: str | None = None
    min_cells: int = 0
    normalize: bool = True
    centralize: bool = True
    graph_n_pcs: int = 30
    graph_n_neighbors: int = 20
    preprocess_mode: str = "baseline"
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedInput:
    """Validated input package handed from public API into Step 1."""

    adata: Any
    step1_adata: Any
    contract: InputContract
    options: PreprocessOptions
    expression_layer: str
    case_value: Any
    control_mask: Any = None
    case_mask: Any = None
    n_cells: int | None = None
    n_genes: int | None = None
    perturbation_counts: Mapping[Any, int] = field(default_factory=dict)


def summarize_input(adata: Any, contract: InputContract) -> dict[str, Any]:
    """Return lightweight input metadata without mutating ``adata``."""

    n_cells = getattr(adata, "n_obs", None)
    n_genes = getattr(adata, "n_vars", None)
    obs = getattr(adata, "obs", None)
    counts: Mapping[Any, int] = {}
    if obs is not None and contract.perturbation_key in obs:
        value_counts = obs[contract.perturbation_key].value_counts()
        counts = value_counts.to_dict()
    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "perturbation_counts": counts,
    }


def _resolve_expression_layer(adata: Any, requested_layer: str | None) -> str:
    if requested_layer is not None:
        if requested_layer != "X" and requested_layer not in getattr(adata, "layers", {}):
            raise KeyError(f"expression_layer {requested_layer!r} is not present in adata.layers.")
        return requested_layer
    if "count" in getattr(adata, "layers", {}):
        return "count"
    return "X"


def _resolve_case_value(obs: Any, contract: InputContract) -> Any:
    if contract.case_value is not None:
        return contract.case_value
    labels = obs[contract.perturbation_key]
    candidates = [value for value in labels.dropna().unique().tolist() if value != contract.control_value]
    if len(candidates) != 1:
        raise ValueError(
            "case_value is required when adata contains multiple non-control perturbation labels."
        )
    return candidates[0]


def _copy_step1_adata(adata: Any, expression_layer: str) -> ad.AnnData:
    matrix = adata.X if expression_layer == "X" else adata.layers[expression_layer]
    matrix = matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)
    step1_adata = ad.AnnData(matrix)
    step1_adata.obs = adata.obs.copy()
    step1_adata.var = adata.var.copy()
    return step1_adata


def _run_baseline_preprocess(adata: ad.AnnData, expression_layer: str) -> ad.AnnData:
    if expression_layer == "count":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers["log1p_norm"] = adata.X.copy()
    else:
        adata.layers["log1p_norm"] = adata.X.copy()
    if issparse(adata.layers["log1p_norm"]):
        adata.layers["log1p_norm"] = adata.layers["log1p_norm"].toarray()
    log1p_norm = np.asarray(adata.layers["log1p_norm"], dtype=np.float32)
    centered_expr = log1p_norm - np.mean(log1p_norm, axis=0, keepdims=True, dtype=np.float32)
    adata.layers["log1p_norm"] = log1p_norm
    adata.X = centered_expr.astype(np.float32, copy=False)
    adata.uns["crane_setting"] = {"preprocess": "baseline"}
    return adata


def prepare_input(
    adata: Any,
    contract: InputContract,
    options: PreprocessOptions | None = None,
) -> PreparedInput:
    """Validate the CRANE input boundary and return a prepared package.

    The legacy preprocessing algorithm is intentionally not wired here yet.
    This function only checks the public-to-core contract and records metadata
    needed by later stages.
    """

    options = options or PreprocessOptions()
    if adata is None:
        raise ValueError("adata must not be None.")
    if not contract.perturbation_key:
        raise ValueError("perturbation_key must be provided.")

    obs = getattr(adata, "obs", None)
    if obs is not None and contract.perturbation_key not in obs:
        raise KeyError(
            f"perturbation_key {contract.perturbation_key!r} is not present in adata.obs."
        )

    summary = summarize_input(adata, contract)
    expression_layer = _resolve_expression_layer(adata, contract.expression_layer)
    case_value = _resolve_case_value(obs, contract)
    control_mask = obs[contract.perturbation_key].values == contract.control_value
    case_mask = obs[contract.perturbation_key].values == case_value
    if int(np.sum(control_mask)) < options.min_cells:
        raise ValueError("control group does not meet min_cells requirement.")
    if int(np.sum(case_mask)) < options.min_cells:
        raise ValueError("case group does not meet min_cells requirement.")
    if options.preprocess_mode not in {"baseline", "no_correct"}:
        raise NotImplementedError(
            "The formal Step1 mainline currently supports only the baseline preprocessing path."
        )
    step1_adata = _copy_step1_adata(adata, expression_layer=expression_layer)
    step1_adata = _run_baseline_preprocess(step1_adata, expression_layer=expression_layer)
    return PreparedInput(
        adata=adata,
        step1_adata=step1_adata,
        contract=contract,
        options=options,
        expression_layer=expression_layer,
        case_value=case_value,
        control_mask=control_mask,
        case_mask=case_mask,
        n_cells=summary["n_cells"],
        n_genes=summary["n_genes"],
        perturbation_counts=summary["perturbation_counts"],
    )
