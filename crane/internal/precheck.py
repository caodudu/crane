"""Lightweight runtime prechecks for CRANE."""

from __future__ import annotations

import time
from typing import Any


_STEP1_GRAPH_BACKEND_CHECKED = False
_STEP1_COLD_START_REPORTED = False


def maybe_warn_step1_cold_start(
    graph_method: str,
    logger: Any,
    threshold_s: float = 1.0,
) -> dict[str, Any]:
    """Probe Step1 graph backend once and emit a user-facing cold-start notice if needed."""

    global _STEP1_GRAPH_BACKEND_CHECKED
    global _STEP1_COLD_START_REPORTED

    normalized = str(graph_method).strip().lower()
    if normalized != "umap":
        return {"checked": False, "detected": False, "graph_method": normalized}
    if _STEP1_GRAPH_BACKEND_CHECKED:
        return {"checked": False, "detected": False, "graph_method": normalized}

    _STEP1_GRAPH_BACKEND_CHECKED = True

    import anndata as ad
    import numpy as np
    import scanpy as sc

    dummy = ad.AnnData(np.random.default_rng(0).random((12, 20), dtype=np.float32))
    sc.tl.pca(dummy, n_comps=5)

    t0 = time.perf_counter()
    sc.pp.neighbors(dummy, n_neighbors=3, n_pcs=5, metric="cosine", method="umap")
    elapsed_s = time.perf_counter() - t0
    detected = elapsed_s > threshold_s

    if detected and not _STEP1_COLD_START_REPORTED:
        logger.user("First run may start slowly with graph_method='umap'. Use 'gauss' for zero-delay startup.")
        logger.event(
            "precheck.cold_start",
            "Step1 cold start detected for graph_method='umap'.",
            audience="reviewer",
            level="INFO",
        )
        _STEP1_COLD_START_REPORTED = True

    return {
        "checked": True,
        "detected": detected,
        "graph_method": normalized,
        "probe_time_s": round(elapsed_s, 4),
        "threshold_s": threshold_s,
    }
