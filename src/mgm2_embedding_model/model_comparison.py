"""Paired test comparison with cruise-block bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import read_csv_rows
from .model_config import ModelRunConfig
from .model_data import ModelingDataset
from .model_metrics import macro_cruise_mae, per_cruise_metrics, regression_metrics


MODEL_ORDER = ("original", "mapseq", "filtered_asv", "mean_baseline")
BOOTSTRAP_METRICS = ("mae_c", "rmse_c", "r2", "macro_cruise_mae_c", "bias_c")
CONTRASTS = (
    ("mapseq_minus_original", True, "mapseq", "original"),
    ("mapseq_minus_filtered_asv", False, "mapseq", "filtered_asv"),
    ("original_minus_filtered_asv", False, "original", "filtered_asv"),
)


@dataclass(frozen=True)
class ComparisonResult:
    paired_predictions: list[dict[str, Any]]
    metric_summary: list[dict[str, Any]]
    paired_differences: list[dict[str, Any]]
    cruise_metrics: list[dict[str, Any]]
    bootstrap_wide: dict[str, np.ndarray]
    direct_metrics: dict[str, dict[str, float]]


def _all_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    cruise_ids: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(observed, predicted)
    metrics["macro_cruise_mae_c"] = macro_cruise_mae(cruise_ids, observed, predicted)
    return metrics


def load_asv_reference_prediction(
    config: ModelRunConfig,
    dataset: ModelingDataset,
) -> np.ndarray:
    _, rows = read_csv_rows(config.inputs.asv_reference_predictions)
    by_id = {row["materialSampleID"].strip(): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("Filtered-ASV reference predictions contain duplicate material IDs")
    test = dataset.test
    test_ids = dataset.sample_ids[test].astype(str).tolist()
    if set(by_id) != set(test_ids) or len(by_id) != len(test_ids):
        raise ValueError("Filtered-ASV reference test IDs do not match embedding test IDs")
    predictions: list[float] = []
    for index in test:
        material = str(dataset.sample_ids[index])
        row = by_id[material]
        if row["sfer_cruise_id"].strip() != str(dataset.cruise_ids[index]):
            raise ValueError(f"Filtered-ASV cruise mismatch for {material}")
        if not np.isclose(
            float(row["observed_temperature_c"]),
            float(dataset.target[index]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Filtered-ASV target mismatch for {material}")
        predictions.append(float(row["predicted_temperature_c"]))

    prediction = np.asarray(predictions, dtype=np.float64)
    computed = regression_metrics(dataset.target[test], prediction)
    _, metric_rows = read_csv_rows(config.inputs.asv_reference_metrics)
    model_rows = [row for row in metric_rows if row.get("model") == "catboost"]
    if len(model_rows) != 1:
        raise ValueError("Filtered-ASV metric file must contain one catboost row")
    frozen = model_rows[0]
    for metric in ("mae_c", "rmse_c", "r2"):
        if not np.isclose(computed[metric], float(frozen[metric]), rtol=0, atol=1e-12):
            raise ValueError(f"Filtered-ASV {metric} does not reproduce its frozen metric")
    return prediction


def _bootstrap_distributions(
    observed: np.ndarray,
    cruise_ids: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    unique_cruises = np.asarray(sorted(set(cruise_ids.tolist())), dtype=object)
    indices = {
        str(cruise): np.flatnonzero(cruise_ids == cruise) for cruise in unique_cruises
    }
    model_values = {
        model: {metric: np.full(replicates, np.nan, dtype=np.float64) for metric in BOOTSTRAP_METRICS}
        for model in predictions
    }
    draw_codes = np.empty((replicates, len(unique_cruises)), dtype=np.int16)
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        selected_codes = generator.integers(0, len(unique_cruises), size=len(unique_cruises))
        draw_codes[replicate] = selected_codes
        selected_cruises = unique_cruises[selected_codes]
        combined = np.concatenate([indices[str(cruise)] for cruise in selected_cruises])
        sampled_observed = observed[combined]
        for model, prediction in predictions.items():
            sampled_prediction = prediction[combined]
            metrics = regression_metrics(sampled_observed, sampled_prediction)
            metrics["macro_cruise_mae_c"] = float(
                np.mean(
                    [
                        np.mean(
                            np.abs(
                                observed[indices[str(cruise)]]
                                - prediction[indices[str(cruise)]]
                            )
                        )
                        for cruise in selected_cruises
                    ]
                )
            )
            for metric in BOOTSTRAP_METRICS:
                model_values[model][metric][replicate] = metrics[metric]
    bootstrap_wide: dict[str, np.ndarray] = {
        "bootstrap_replicate": np.arange(1, replicates + 1, dtype=np.int64)
    }
    for position in range(draw_codes.shape[1]):
        bootstrap_wide[f"cruise_draw_{position + 1}"] = unique_cruises[draw_codes[:, position]].astype(str)
    for contrast, _, model_a, model_b in CONTRASTS:
        for metric in ("mae_c", "rmse_c", "r2", "macro_cruise_mae_c"):
            bootstrap_wide[f"{contrast}__{metric}"] = (
                model_values[model_a][metric] - model_values[model_b][metric]
            )
    return model_values, bootstrap_wide


def _percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper), int(len(finite))


def compare_predictions(
    dataset: ModelingDataset,
    original_prediction: np.ndarray,
    mapseq_prediction: np.ndarray,
    asv_prediction: np.ndarray,
    config: ModelRunConfig,
) -> ComparisonResult:
    test = dataset.test
    observed = dataset.target[test]
    cruises = dataset.cruise_ids[test].astype(str)
    baseline = np.full(
        len(test), float(np.mean(dataset.target[dataset.development])), dtype=np.float64
    )
    predictions = {
        "original": np.asarray(original_prediction, dtype=np.float64),
        "mapseq": np.asarray(mapseq_prediction, dtype=np.float64),
        "filtered_asv": np.asarray(asv_prediction, dtype=np.float64),
        "mean_baseline": baseline,
    }
    if any(value.shape != observed.shape for value in predictions.values()):
        raise ValueError("All paired predictions must have the held-out target shape")
    direct = {
        model: _all_metrics(observed, prediction, cruises)
        for model, prediction in predictions.items()
    }
    distributions, bootstrap_wide = _bootstrap_distributions(
        observed,
        cruises,
        predictions,
        config.settings.bootstrap_replicates,
        config.settings.bootstrap_seed,
    )

    paired_rows: list[dict[str, Any]] = []
    for local, index in enumerate(test):
        row: dict[str, Any] = {
            "material_sample": str(dataset.sample_ids[index]),
            "cruise": str(dataset.cruise_ids[index]),
            "observed_temperature_c": float(observed[local]),
        }
        for model in MODEL_ORDER:
            prediction = float(predictions[model][local])
            residual = float(observed[local] - prediction)
            row[f"{model}_prediction_c"] = prediction
            row[f"{model}_residual_c"] = residual
            row[f"{model}_absolute_error_c"] = abs(residual)
        paired_rows.append(row)

    metric_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for metric in BOOTSTRAP_METRICS:
            lower, upper, valid = _percentile_interval(distributions[model][metric])
            metric_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": direct[model][metric],
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "samples": len(test),
                    "cruises": len(set(cruises.tolist())),
                    "bootstrap_replicates": config.settings.bootstrap_replicates,
                    "valid_bootstrap_replicates": valid,
                    "block_unit": "cruise",
                    "ci_method": "percentile",
                }
            )

    difference_rows: list[dict[str, Any]] = []
    for contrast, primary, model_a, model_b in CONTRASTS:
        for metric in ("mae_c", "rmse_c", "r2", "macro_cruise_mae_c"):
            values = distributions[model_a][metric] - distributions[model_b][metric]
            lower, upper, valid = _percentile_interval(values)
            negative_favors_a = metric != "r2"
            finite = values[np.isfinite(values)]
            fraction = float(np.mean(finite < 0 if negative_favors_a else finite > 0))
            difference_rows.append(
                {
                    "contrast": contrast,
                    "is_primary": primary,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "delta_definition": "model_a - model_b",
                    "estimate": direct[model_a][metric] - direct[model_b][metric],
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "favorable_direction": "negative" if negative_favors_a else "positive",
                    "bootstrap_fraction_favorable": fraction,
                    "valid_bootstrap_replicates": valid,
                }
            )

    cruise_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for row in per_cruise_metrics(
            dataset.sample_ids[test], cruises, observed, predictions[model]
        ):
            cruise_rows.append({"model": model, **row})
    return ComparisonResult(
        paired_predictions=paired_rows,
        metric_summary=metric_rows,
        paired_differences=difference_rows,
        cruise_metrics=cruise_rows,
        bootstrap_wide=bootstrap_wide,
        direct_metrics=direct,
    )


def write_bootstrap_gzip(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    names = list(columns)
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Bootstrap output columns have inconsistent lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for values in zip(*(columns[name] for name in names), strict=True):
            writer.writerow(values)
