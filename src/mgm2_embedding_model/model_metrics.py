"""Regression metrics used identically for both embedding representations."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted values must be same-length 1D arrays")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    residual = observed - predicted
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(np.square(residual))) / denominator
    return {
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": r2,
        "bias_c": float(np.mean(residual)),
    }


def mean_baseline(
    development_target: np.ndarray,
    evaluation_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = np.full(
        len(evaluation_target), float(np.mean(development_target)), dtype=np.float64
    )
    return prediction, regression_metrics(evaluation_target, prediction)


def per_cruise_metrics(
    sample_ids: Sequence[str],
    cruise_ids: Sequence[str],
    observed: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cruise in enumerate(cruise_ids):
        groups[str(cruise)].append(index)
    rows: list[dict[str, object]] = []
    for cruise in sorted(groups):
        indices = np.asarray(groups[cruise], dtype=np.int64)
        metrics = regression_metrics(observed[indices], predicted[indices])
        rows.append(
            {
                "cruise": cruise,
                "samples": int(len(indices)),
                **metrics,
            }
        )
    return rows


def macro_cruise_mae(
    cruise_ids: Sequence[str], observed: np.ndarray, predicted: np.ndarray
) -> float:
    rows = per_cruise_metrics(
        [str(index) for index in range(len(observed))],
        cruise_ids,
        observed,
        predicted,
    )
    return float(np.mean([float(row["mae_c"]) for row in rows]))
