"""Constraint behaviour RMSE used by the reference-form baseline.

Extracted from the Refai--Ahmed re-implementation so the EquiCEval method repo
carries no dependency on the baseline repository.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def compute_cons_rmse(
    matched_constraints: List[
        Tuple[
            Callable[[Dict[str, float]], float],
            Callable[[Dict[str, float]], float],
        ]
    ],
    samples: List[Dict[str, float]],
) -> float:
    """Cons-RMSE over matched constraints (Eq. 7, arXiv:2510.16943).

    Computed only over matched constraints; omitted/hallucinated constraints are
    excluded.
    """
    if not matched_constraints or not samples:
        return 0.0

    num_matched = len(matched_constraints)
    num_samples = len(samples)
    total_sq_diff = 0.0

    for gt_func, llm_func in matched_constraints:
        for x_sample in samples:
            try:
                gt_val = float(gt_func(x_sample))
                llm_val = float(llm_func(x_sample))
                diff = gt_val - llm_val
                total_sq_diff += diff * diff
            except Exception:
                total_sq_diff += 1e4

    mean_sq_diff = total_sq_diff / (num_matched * num_samples)
    return float(np.sqrt(mean_sq_diff))
