
"""
Belief contraction helpers (v1).

Used mainly for experiment-level aggregation (E2 door mismatch).
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np


def summarize_contraction(trace_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute simple contraction stats from trace rows.
    Returns append-only dict (safe for external use).
    """
    if not trace_rows:
        return {"entropy_final": float("nan"), "credible_volume_final": float("nan")}
    last = trace_rows[-1]
    return {
        "entropy_final": float(last.get("entropy", float("nan"))),
        "credible_volume_final": float(last.get("credible_volume", float("nan"))),
    }
