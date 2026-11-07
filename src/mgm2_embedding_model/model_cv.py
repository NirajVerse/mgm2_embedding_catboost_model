"""Five-fold frozen cruise-grouped tuning for one embedding representation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

import numpy as np

from .model_catboost import fit_candidate
from .model_config import ModelSettings
from .model_data import ModelingDataset
from .model_metrics import macro_cruise_mae, per_cruise_metrics, regression_metrics


@dataclass(frozen=True)
class CVSelection:
    candidate_name: str
    parameters: dict[str, Any]
    best_tree_count: int
    pooled_oof_metrics: dict[str, float]
    macro_cruise_mae_c: float
    fold_tree_counts: tuple[int, ...]
    evaluation_histories: dict[str, Any]


@dataclass(frozen=True)
class CVResult:
    selection: CVSelection
    fold_results: list[dict[str, Any]]
    candidate_summary: list[dict[str, Any]]
    oof_predictions: list[dict[str, Any]]
    oof_cruise_metrics: list[dict[str, Any]]


def run_frozen_cross_validation(
    dataset: ModelingDataset,
    candidates: tuple[dict[str, Any], ...],
    settings: ModelSettings,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> CVResult:
    development = dataset.development
    candidate_order = {str(candidate["name"]): index for index, candidate in enumerate(candidates)}
    parameters_by_name = {
        str(candidate["name"]): {key: value for key, value in candidate.items() if key != "name"}
        for candidate in candidates
    }
    predictions = {
        name: np.full(len(dataset.target), np.nan, dtype=np.float64)
        for name in candidate_order
    }
    histories: dict[str, dict[str, Any]] = {name: {} for name in candidate_order}
    fold_results: list[dict[str, Any]] = []

    for fold in range(1, 6):
        validation = development[dataset.cv_fold[development] == fold]
        training = development[dataset.cv_fold[development] != fold]
        if not len(validation) or not len(training):
            raise ValueError(f"Frozen fold {fold} is empty")
        if set(dataset.cruise_ids[training]) & set(dataset.cruise_ids[validation]):
            raise ValueError(f"Cruises overlap in frozen fold {fold}")
        for candidate in candidates:
            fit = fit_candidate(
                dataset.features[training],
                dataset.target[training],
                dataset.features[validation],
                dataset.target[validation],
                dataset.feature_names,
                candidate,
                settings,
            )
            predictions[fit.candidate_name][validation] = fit.validation_predictions
            histories[fit.candidate_name][f"fold_{fold}"] = fit.evaluation_history
            fold_results.append(
                {
                    "candidate": fit.candidate_name,
                    "candidate_order": candidate_order[fit.candidate_name],
                    "parameters": json.dumps(fit.parameters, sort_keys=True),
                    "cv_fold": fold,
                    "training_samples": len(training),
                    "validation_samples": len(validation),
                    "training_cruises": len(set(dataset.cruise_ids[training])),
                    "validation_cruises": len(set(dataset.cruise_ids[validation])),
                    "feature_count": len(dataset.feature_names),
                    "tree_count": fit.tree_count,
                    **{f"training_{key}": value for key, value in fit.training_metrics.items()},
                    **{f"validation_{key}": value for key, value in fit.validation_metrics.items()},
                }
            )
            if progress_callback is not None:
                progress_callback(fold_results)

    fold_results.sort(key=lambda row: (int(row["candidate_order"]), int(row["cv_fold"])))
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        rows = [row for row in fold_results if row["candidate"] == name]
        if len(rows) != 5:
            raise RuntimeError(f"Candidate {name} completed {len(rows)}/5 folds")
        oof = predictions[name][development]
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Candidate {name} has missing OOF predictions")
        pooled = regression_metrics(dataset.target[development], oof)
        tree_counts = np.asarray([row["tree_count"] for row in rows], dtype=np.int64)
        fold_mae = np.asarray([row["validation_mae_c"] for row in rows], dtype=np.float64)
        fold_rmse = np.asarray([row["validation_rmse_c"] for row in rows], dtype=np.float64)
        fold_r2 = np.asarray([row["validation_r2"] for row in rows], dtype=np.float64)
        summaries.append(
            {
                "candidate": name,
                "candidate_order": candidate_order[name],
                "parameters": json.dumps(parameters_by_name[name], sort_keys=True),
                "completed_folds": 5,
                "pooled_oof_mae_c": pooled["mae_c"],
                "pooled_oof_rmse_c": pooled["rmse_c"],
                "pooled_oof_r2": pooled["r2"],
                "macro_cruise_mae_c": macro_cruise_mae(
                    dataset.cruise_ids[development], dataset.target[development], oof
                ),
                "fold_mae_mean_c": float(np.mean(fold_mae)),
                "fold_mae_sd_c": float(np.std(fold_mae, ddof=1)),
                "fold_rmse_mean_c": float(np.mean(fold_rmse)),
                "fold_rmse_sd_c": float(np.std(fold_rmse, ddof=1)),
                "fold_r2_mean": float(np.mean(fold_r2)),
                "fold_r2_sd": float(np.std(fold_r2, ddof=1)),
                "tree_count_median": float(np.median(tree_counts)),
                "tree_count_min": int(np.min(tree_counts)),
                "tree_count_max": int(np.max(tree_counts)),
            }
        )
    summaries.sort(
        key=lambda row: (
            float(row["pooled_oof_mae_c"]),
            float(row["pooled_oof_rmse_c"]),
            int(row["candidate_order"]),
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["selection_rank"] = rank
    winner = summaries[0]
    winner_name = str(winner["candidate"])
    winner_rows = sorted(
        [row for row in fold_results if row["candidate"] == winner_name],
        key=lambda row: int(row["cv_fold"]),
    )
    tree_counts = tuple(int(row["tree_count"]) for row in winner_rows)
    fixed_tree_count = max(1, int(np.floor(float(np.median(tree_counts)) + 0.5)))
    winner_oof = predictions[winner_name][development]
    pooled = regression_metrics(dataset.target[development], winner_oof)
    selection = CVSelection(
        candidate_name=winner_name,
        parameters=parameters_by_name[winner_name],
        best_tree_count=fixed_tree_count,
        pooled_oof_metrics=pooled,
        macro_cruise_mae_c=float(winner["macro_cruise_mae_c"]),
        fold_tree_counts=tree_counts,
        evaluation_histories=histories[winner_name],
    )

    oof_rows: list[dict[str, Any]] = []
    for local, index in enumerate(development):
        prediction = float(winner_oof[local])
        observed = float(dataset.target[index])
        oof_rows.append(
            {
                "material_sample": str(dataset.sample_ids[index]),
                "cruise": str(dataset.cruise_ids[index]),
                "cv_fold": int(dataset.cv_fold[index]),
                "training_row": int(dataset.training_row[index]),
                "observed_temperature_c": observed,
                "predicted_temperature_c": prediction,
                "residual_c": observed - prediction,
                "absolute_error_c": abs(observed - prediction),
            }
        )
    oof_cruise = per_cruise_metrics(
        dataset.sample_ids[development],
        dataset.cruise_ids[development],
        dataset.target[development],
        winner_oof,
    )
    return CVResult(selection, fold_results, summaries, oof_rows, oof_cruise)
