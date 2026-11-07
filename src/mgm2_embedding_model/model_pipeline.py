"""Orchestration for paired original-versus-MAPseq CatBoost training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback
from typing import Any

import numpy as np

from . import __version__
from .io import sha256_file, write_csv_rows, write_json, write_text
from .model_catboost import (
    EXPECTED_CATBOOST_VERSION,
    calculate_shap,
    fit_final_model,
    make_pool,
    require_catboost,
    verify_saved_model,
)
from .model_comparison import (
    ComparisonResult,
    compare_predictions,
    load_asv_reference_prediction,
    write_bootstrap_gzip,
)
from .model_config import ModelRunConfig, serializable_model_config
from .model_cv import CVResult, run_frozen_cross_validation
from .model_data import ModelingDataset, load_paired_modeling_data, model_data_summary
from .model_metrics import mean_baseline, per_cruise_metrics, regression_metrics
from .model_plots import (
    plot_absolute_error_distribution,
    plot_absolute_error_ecdf,
    plot_candidate_tuning,
    plot_cruise_mapseq_difference,
    plot_feature_importance,
    plot_metric_comparison,
    plot_observed_vs_predicted,
    plot_oof_metrics,
    plot_oof_vs_test_metrics,
    plot_paired_sample_error_difference,
    plot_per_cruise_mae,
    plot_residuals,
    plot_shap_importance,
    plot_single_model_accuracy,
    plot_single_model_cruise_mae,
    plot_single_model_residuals,
)
from .model_reporting import comparison_report, environment_manifest


@dataclass(frozen=True)
class RepresentationRun:
    representation: str
    cv: CVResult
    test_predictions: np.ndarray
    summary: dict[str, Any]
    root: Path


FOLD_RESULT_FIELDS = [
    "candidate",
    "candidate_order",
    "parameters",
    "cv_fold",
    "training_samples",
    "validation_samples",
    "training_cruises",
    "validation_cruises",
    "feature_count",
    "tree_count",
    "training_mae_c",
    "training_rmse_c",
    "training_r2",
    "training_bias_c",
    "validation_mae_c",
    "validation_rmse_c",
    "validation_r2",
    "validation_bias_c",
]

CANDIDATE_SUMMARY_FIELDS = [
    "selection_rank",
    "candidate",
    "candidate_order",
    "parameters",
    "completed_folds",
    "pooled_oof_mae_c",
    "pooled_oof_rmse_c",
    "pooled_oof_r2",
    "macro_cruise_mae_c",
    "fold_mae_mean_c",
    "fold_mae_sd_c",
    "fold_rmse_mean_c",
    "fold_rmse_sd_c",
    "fold_r2_mean",
    "fold_r2_sd",
    "tree_count_median",
    "tree_count_min",
    "tree_count_max",
]

PREDICTION_FIELDS = [
    "material_sample",
    "cruise",
    "observed_temperature_c",
    "predicted_temperature_c",
    "residual_c",
    "absolute_error_c",
]

OOF_PREDICTION_FIELDS = [
    "material_sample",
    "cruise",
    "cv_fold",
    "training_row",
    "observed_temperature_c",
    "predicted_temperature_c",
    "residual_c",
    "absolute_error_c",
]

CRUISE_METRIC_FIELDS = ["cruise", "samples", "mae_c", "rmse_c", "r2", "bias_c"]


def _safe_run_id(run_id: str) -> str:
    cleaned = run_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", cleaned):
        raise ValueError(
            "run_id must be 1-80 characters and contain only letters, numbers, dot, underscore, or hyphen"
        )
    return cleaned


def modeling_preflight(config: ModelRunConfig) -> dict[str, Any]:
    data = load_paired_modeling_data(config)
    asv_prediction = load_asv_reference_prediction(config, data.original)
    test = data.original.test
    asv_metrics = regression_metrics(data.original.target[test], asv_prediction)
    summary = model_data_summary(data)
    summary.update(
        {
            "status": "PASS",
            "candidate_count": len(config.candidates),
            "candidate_names": [candidate["name"] for candidate in config.candidates],
            "asv_reference_test_metrics_reproduced": asv_metrics,
            "catboost_required_version": EXPECTED_CATBOOST_VERSION,
            "real_training_started": False,
        }
    )
    return summary


def _prepare_run_root(config: ModelRunConfig, run_id: str) -> Path:
    root = config.results_dir / _safe_run_id(run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {root}. Choose a new run ID."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _representation_directories(root: Path, representation: str) -> dict[str, Path]:
    paths = {
        "root": root / representation,
        "tables": root / representation / "tables",
        "figures": root / representation / "figures",
        "models": root / representation / "models",
        "logs": root / representation / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _test_prediction_rows(
    dataset: ModelingDataset, predictions: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local, index in enumerate(dataset.test):
        observed = float(dataset.target[index])
        predicted = float(predictions[local])
        rows.append(
            {
                "material_sample": str(dataset.sample_ids[index]),
                "cruise": str(dataset.cruise_ids[index]),
                "observed_temperature_c": observed,
                "predicted_temperature_c": predicted,
                "residual_c": observed - predicted,
                "absolute_error_c": abs(observed - predicted),
            }
        )
    return rows


def _train_representation(
    root: Path,
    dataset: ModelingDataset,
    config: ModelRunConfig,
) -> RepresentationRun:
    paths = _representation_directories(root, dataset.representation)

    def save_progress(rows: list[dict[str, Any]]) -> None:
        write_csv_rows(paths["tables"] / "cv_progress.csv", FOLD_RESULT_FIELDS, rows)

    cv = run_frozen_cross_validation(
        dataset,
        config.candidates,
        config.settings,
        progress_callback=save_progress,
    )
    write_csv_rows(paths["tables"] / "cv_fold_results.csv", FOLD_RESULT_FIELDS, cv.fold_results)
    write_csv_rows(
        paths["tables"] / "cv_candidate_summary.csv",
        CANDIDATE_SUMMARY_FIELDS,
        cv.candidate_summary,
    )
    write_csv_rows(
        paths["tables"] / "selected_candidate_oof_predictions.csv",
        OOF_PREDICTION_FIELDS,
        cv.oof_predictions,
    )
    write_csv_rows(
        paths["tables"] / "selected_candidate_oof_cruise_metrics.csv",
        CRUISE_METRIC_FIELDS,
        cv.oof_cruise_metrics,
    )
    write_json(
        paths["tables"] / "selected_candidate_evaluation_histories.json",
        cv.selection.evaluation_histories,
    )

    development = dataset.development
    test = dataset.test
    model, _ = fit_final_model(
        dataset.features[development],
        dataset.target[development],
        dataset.feature_names,
        cv.selection.parameters,
        cv.selection.best_tree_count,
        config.settings,
    )
    model_path = paths["models"] / "catboost_temperature.cbm"
    model.save_model(str(model_path), format="cbm")
    test_pool = make_pool(dataset.features[test], dataset.target[test], dataset.feature_names)
    test_predictions = np.asarray(model.predict(test_pool), dtype=np.float64)
    verify_saved_model(
        model_path,
        dataset.features[test],
        dataset.target[test],
        dataset.feature_names,
        test_predictions,
    )

    test_metrics = regression_metrics(dataset.target[test], test_predictions)
    _, baseline_metrics = mean_baseline(dataset.target[development], dataset.target[test])
    metric_rows = [
        {"model": "development_mean_baseline", **baseline_metrics},
        {"model": "catboost", **test_metrics},
    ]
    write_csv_rows(
        paths["tables"] / "test_metrics.csv",
        ["model", "mae_c", "rmse_c", "r2", "bias_c"],
        metric_rows,
    )
    test_rows = _test_prediction_rows(dataset, test_predictions)
    write_csv_rows(paths["tables"] / "test_predictions.csv", PREDICTION_FIELDS, test_rows)
    cruise_rows = per_cruise_metrics(
        dataset.sample_ids[test], dataset.cruise_ids[test], dataset.target[test], test_predictions
    )
    write_csv_rows(
        paths["tables"] / "test_cruise_metrics.csv", CRUISE_METRIC_FIELDS, cruise_rows
    )
    plot_single_model_accuracy(
        test_rows,
        dataset.representation,
        test_metrics,
        paths["figures"] / "test_observed_vs_predicted.png",
    )
    plot_single_model_residuals(
        test_rows,
        dataset.representation,
        paths["figures"] / "test_residuals_vs_observed.png",
    )
    plot_absolute_error_distribution(
        test_rows,
        dataset.representation,
        paths["figures"] / "test_absolute_error_distribution.png",
    )
    plot_single_model_cruise_mae(
        cruise_rows,
        dataset.representation,
        paths["figures"] / "test_cruise_mae.png",
    )
    plot_oof_vs_test_metrics(
        cv.selection.pooled_oof_metrics,
        test_metrics,
        dataset.representation,
        paths["figures"] / "development_oof_vs_test.png",
    )

    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    importance_rows = [
        {"feature": feature, "prediction_values_change": float(value)}
        for feature, value in zip(dataset.feature_names, importance, strict=True)
    ]
    importance_rows.sort(key=lambda row: float(row["prediction_values_change"]), reverse=True)
    for rank, row in enumerate(importance_rows, start=1):
        row["importance_rank"] = rank
    write_csv_rows(
        paths["tables"] / "feature_importance.csv",
        ["importance_rank", "feature", "prediction_values_change"],
        importance_rows,
    )
    plot_feature_importance(
        importance_rows, paths["figures"] / "feature_importance_top20.png"
    )

    shap_calculated = False
    shap_error: str | None = None
    if config.settings.calculate_shap:
        try:
            shap_rows = calculate_shap(
                model,
                dataset.features[test],
                dataset.target[test],
                dataset.sample_ids[test],
                dataset.feature_names,
                config.settings,
                paths["tables"] / "test_shap_values.npz",
            )
            write_csv_rows(
                paths["tables"] / "test_shap_summary.csv",
                [
                    "shap_rank",
                    "feature",
                    "mean_absolute_shap",
                    "mean_signed_shap",
                    "maximum_absolute_shap",
                ],
                shap_rows,
            )
            plot_shap_importance(
                shap_rows,
                dataset.representation,
                paths["figures"] / "test_shap_importance_top20.png",
            )
            shap_calculated = True
        except Exception as error:  # Retain completed model and predictions.
            shap_error = f"{type(error).__name__}: {error}"
            write_text(paths["logs"] / "shap_error.txt", shap_error + "\n")

    summary = {
        "representation": dataset.representation,
        "selected_candidate": cv.selection.candidate_name,
        "selected_parameters": cv.selection.parameters,
        "selection_metric": "pooled_oof_mae_c",
        "oof_metrics": cv.selection.pooled_oof_metrics,
        "oof_macro_cruise_mae_c": cv.selection.macro_cruise_mae_c,
        "fold_tree_counts": cv.selection.fold_tree_counts,
        "final_tree_count": cv.selection.best_tree_count,
        "feature_count": len(dataset.feature_names),
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "shap_calculated": shap_calculated,
        "shap_error": shap_error,
    }
    write_json(paths["root"] / "result_summary.json", summary)
    write_json(paths["root"] / "environment.json", environment_manifest())
    return RepresentationRun(
        dataset.representation, cv, test_predictions, summary, paths["root"]
    )


def _comparison_fields(result: ComparisonResult) -> dict[str, list[str]]:
    paired = [
        "material_sample",
        "cruise",
        "observed_temperature_c",
    ]
    for model in ("original", "mapseq", "filtered_asv", "mean_baseline"):
        paired.extend(
            [
                f"{model}_prediction_c",
                f"{model}_residual_c",
                f"{model}_absolute_error_c",
            ]
        )
    return {
        "paired": paired,
        "metrics": [
            "model",
            "metric",
            "estimate",
            "ci_lower_95",
            "ci_upper_95",
            "samples",
            "cruises",
            "bootstrap_replicates",
            "valid_bootstrap_replicates",
            "block_unit",
            "ci_method",
        ],
        "differences": [
            "contrast",
            "is_primary",
            "model_a",
            "model_b",
            "metric",
            "delta_definition",
            "estimate",
            "ci_lower_95",
            "ci_upper_95",
            "favorable_direction",
            "bootstrap_fraction_favorable",
            "valid_bootstrap_replicates",
        ],
        "cruise": ["model", *CRUISE_METRIC_FIELDS],
    }


def _write_comparison(
    root: Path,
    run_id: str,
    dataset: ModelingDataset,
    original: RepresentationRun,
    mapseq: RepresentationRun,
    comparison: ComparisonResult,
) -> None:
    comparison_root = root / "comparison"
    tables = comparison_root / "tables"
    figures = comparison_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    fields = _comparison_fields(comparison)
    write_csv_rows(tables / "paired_test_predictions.csv", fields["paired"], comparison.paired_predictions)
    write_csv_rows(tables / "test_metric_summary.csv", fields["metrics"], comparison.metric_summary)
    write_csv_rows(
        tables / "paired_test_metric_differences.csv",
        fields["differences"],
        comparison.paired_differences,
    )
    write_csv_rows(tables / "test_cruise_metrics.csv", fields["cruise"], comparison.cruise_metrics)
    write_bootstrap_gzip(
        tables / "bootstrap_metric_differences.csv.gz", comparison.bootstrap_wide
    )
    oof_rows = [
        {
            "representation": "original",
            **original.cv.selection.pooled_oof_metrics,
            "macro_cruise_mae_c": original.cv.selection.macro_cruise_mae_c,
        },
        {
            "representation": "mapseq",
            **mapseq.cv.selection.pooled_oof_metrics,
            "macro_cruise_mae_c": mapseq.cv.selection.macro_cruise_mae_c,
        },
    ]
    write_csv_rows(
        tables / "development_oof_metric_summary.csv",
        ["representation", "mae_c", "rmse_c", "r2", "bias_c", "macro_cruise_mae_c"],
        oof_rows,
    )
    write_json(
        comparison_root / "result_summary.json",
        {
            "run_id": run_id,
            "primary_contrast": "mapseq_minus_original",
            "direct_test_metrics": comparison.direct_metrics,
            "paired_test_metric_differences": comparison.paired_differences,
            "bootstrap_unit": "cruise",
            "test_cruises": len(set(dataset.cruise_ids[dataset.test].tolist())),
        },
    )

    plot_observed_vs_predicted(
        comparison.paired_predictions, figures / "observed_vs_predicted_comparison.png"
    )
    plot_metric_comparison(comparison.metric_summary, figures / "test_metric_comparison.png")
    plot_per_cruise_mae(comparison.cruise_metrics, figures / "per_cruise_mae.png")
    plot_cruise_mapseq_difference(
        comparison.cruise_metrics, figures / "cruise_mapseq_minus_original_mae.png"
    )
    plot_residuals(comparison.paired_predictions, figures / "residuals_vs_observed.png")
    plot_absolute_error_ecdf(
        comparison.paired_predictions, figures / "absolute_error_ecdf_comparison.png"
    )
    plot_paired_sample_error_difference(
        comparison.paired_predictions, figures / "paired_sample_error_difference.png"
    )
    plot_oof_metrics(
        original.cv.selection.pooled_oof_metrics,
        mapseq.cv.selection.pooled_oof_metrics,
        figures / "development_oof_comparison.png",
    )
    plot_candidate_tuning(
        original.cv.candidate_summary,
        mapseq.cv.candidate_summary,
        figures / "candidate_tuning_comparison.png",
    )
    report = comparison_report(
        run_id,
        original.summary,
        mapseq.summary,
        comparison.metric_summary,
        comparison.paired_differences,
        len(dataset.sample_ids),
        len(dataset.development),
        len(dataset.test),
        len(set(dataset.cruise_ids[dataset.test].tolist())),
    )
    write_text(comparison_root / "report.md", report)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "training_state.json"}:
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "files": files,
    }


def run_modeling(config: ModelRunConfig, run_id: str) -> dict[str, Any]:
    # Fail before creating a results directory if CatBoost is absent or the wrong version.
    catboost_version = require_catboost()
    data = load_paired_modeling_data(config)
    asv_prediction = load_asv_reference_prediction(config, data.original)
    root = _prepare_run_root(config, run_id)
    state_path = root / "training_state.json"
    write_json(
        state_path,
        {
            "status": "RUNNING",
            "run_id": run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        write_json(root / "protocol.json", serializable_model_config(config, run_id))
        write_json(root / "environment.json", environment_manifest())
        write_json(root / "data_summary.json", model_data_summary(data))

        original = _train_representation(root, data.original, config)
        mapseq = _train_representation(root, data.mapseq, config)
        comparison = compare_predictions(
            data.original,
            original.test_predictions,
            mapseq.test_predictions,
            asv_prediction,
            config,
        )
        _write_comparison(root, run_id, data.original, original, mapseq, comparison)
        write_json(root / "artifact_manifest.json", _artifact_manifest(root))
        summary = {
            "status": "COMPLETE",
            "run_id": run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": __version__,
            "catboost_version": catboost_version,
            "original_test_metrics": original.summary["test_metrics"],
            "mapseq_test_metrics": mapseq.summary["test_metrics"],
            "comparison_report": str(root / "comparison" / "report.md"),
        }
        write_json(state_path, summary)
        return summary
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "FAILED",
                "run_id": run_id,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
