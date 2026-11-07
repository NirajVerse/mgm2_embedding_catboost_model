"""Lazy CatBoost fitting for dense MGM2 embedding matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model_config import ModelSettings
from .model_metrics import regression_metrics


EXPECTED_CATBOOST_VERSION = "1.2.10"


class CatBoostDependencyError(RuntimeError):
    """Raised when the pinned CatBoost dependency is unavailable."""


@dataclass(frozen=True)
class CandidateFit:
    candidate_name: str
    parameters: dict[str, Any]
    tree_count: int
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    validation_predictions: np.ndarray
    evaluation_history: dict[str, Any]


def _catboost_objects() -> tuple[Any, Any, Any, str]:
    try:
        import catboost
        from catboost import CatBoostRegressor, EFstrType, Pool
    except ImportError as error:
        raise CatBoostDependencyError(
            "CatBoost is not installed. Activate the project environment and run "
            "python -m pip install -r requirements.txt before training."
        ) from error
    return CatBoostRegressor, Pool, EFstrType, str(catboost.__version__)


def require_catboost() -> str:
    _, _, _, version = _catboost_objects()
    if version != EXPECTED_CATBOOST_VERSION:
        raise CatBoostDependencyError(
            f"This protocol requires CatBoost {EXPECTED_CATBOOST_VERSION}; found {version}."
        )
    return version


def _base_parameters(settings: ModelSettings) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "custom_metric": ["RMSE", "R2"],
        "random_seed": settings.random_seed,
        "thread_count": settings.thread_count,
        "task_type": settings.task_type,
        "allow_writing_files": False,
        "nan_mode": "Forbidden",
    }
    if settings.task_type == "GPU" and settings.devices:
        parameters["devices"] = settings.devices
    return parameters


def _verbose(settings: ModelSettings) -> bool | int:
    return False if settings.verbose == 0 else settings.verbose


def make_pool(
    features: np.ndarray,
    target: np.ndarray | None,
    feature_names: Sequence[str],
) -> Any:
    _, Pool, _, _ = _catboost_objects()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Feature matrix width does not match feature-name count")
    if len(feature_names) != len(set(feature_names)) or any(not name for name in feature_names):
        raise ValueError("Feature names must be unique and nonempty")
    return Pool(data=features, label=target, feature_names=list(feature_names))


def fit_candidate(
    training_features: np.ndarray,
    training_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    feature_names: Sequence[str],
    candidate: dict[str, Any],
    settings: ModelSettings,
) -> CandidateFit:
    CatBoostRegressor, _, _, _ = _catboost_objects()
    name = str(candidate["name"])
    candidate_parameters = {key: value for key, value in candidate.items() if key != "name"}
    forbidden = {
        "iterations",
        "random_seed",
        "loss_function",
        "eval_metric",
        "use_best_model",
        "early_stopping_rounds",
    } & set(candidate_parameters)
    if forbidden:
        raise ValueError(f"Candidate {name} overrides controlled keys: {sorted(forbidden)}")
    train_pool = make_pool(training_features, training_target, feature_names)
    validation_pool = make_pool(validation_features, validation_target, feature_names)
    model = CatBoostRegressor(
        **_base_parameters(settings),
        **candidate_parameters,
        iterations=settings.maximum_iterations,
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=settings.early_stopping_rounds,
        verbose=_verbose(settings),
    )
    training_prediction = np.asarray(model.predict(train_pool), dtype=np.float64)
    validation_prediction = np.asarray(model.predict(validation_pool), dtype=np.float64)
    return CandidateFit(
        candidate_name=name,
        parameters=candidate_parameters,
        tree_count=int(model.tree_count_),
        training_metrics=regression_metrics(training_target, training_prediction),
        validation_metrics=regression_metrics(validation_target, validation_prediction),
        validation_predictions=validation_prediction,
        evaluation_history=model.get_evals_result(),
    )


def fit_final_model(
    development_features: np.ndarray,
    development_target: np.ndarray,
    feature_names: Sequence[str],
    parameters: dict[str, Any],
    tree_count: int,
    settings: ModelSettings,
) -> tuple[Any, Any]:
    CatBoostRegressor, _, _, _ = _catboost_objects()
    development_pool = make_pool(development_features, development_target, feature_names)
    model = CatBoostRegressor(
        **_base_parameters(settings),
        **parameters,
        iterations=int(tree_count),
    )
    model.fit(development_pool, use_best_model=False, verbose=_verbose(settings))
    return model, development_pool


def verify_saved_model(
    path: Path,
    features: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    expected_predictions: np.ndarray,
) -> None:
    CatBoostRegressor, _, _, _ = _catboost_objects()
    reloaded = CatBoostRegressor()
    reloaded.load_model(str(path), format="cbm")
    predictions = np.asarray(
        reloaded.predict(make_pool(features, target, feature_names)), dtype=np.float64
    )
    if not np.allclose(predictions, expected_predictions, rtol=1e-12, atol=1e-12):
        error = float(np.max(np.abs(predictions - expected_predictions)))
        raise RuntimeError(f"Reloaded model predictions changed; max error={error}")


def calculate_shap(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: Sequence[str],
    settings: ModelSettings,
    output_npz: Path,
) -> list[dict[str, Any]]:
    _, _, EFstrType, _ = _catboost_objects()
    if len(features) <= settings.shap_max_samples:
        selected = np.arange(len(features), dtype=np.int64)
    else:
        generator = np.random.default_rng(settings.random_seed)
        selected = np.sort(
            generator.choice(len(features), size=settings.shap_max_samples, replace=False)
        )
    pool = make_pool(features[selected], target[selected], feature_names)
    raw = np.asarray(
        model.get_feature_importance(
            data=pool,
            type=EFstrType.ShapValues,
            thread_count=settings.thread_count,
            verbose=False,
        ),
        dtype=np.float64,
    )
    expected_shape = (len(selected), len(feature_names) + 1)
    if raw.shape != expected_shape:
        raise RuntimeError(f"Unexpected SHAP shape {raw.shape}; expected {expected_shape}")
    values = raw[:, :-1]
    expected_values = raw[:, -1]
    predictions = np.asarray(model.predict(pool), dtype=np.float64)
    reconstructed = expected_values + values.sum(axis=1)
    if not np.allclose(reconstructed, predictions, rtol=1e-5, atol=1e-5):
        raise RuntimeError("SHAP additivity check failed")
    np.savez_compressed(
        output_npz,
        shap_values=values,
        expected_values=expected_values,
        sample_ids=sample_ids[selected].astype(str),
        feature_ids=np.asarray(feature_names, dtype=str),
        observed_temperature=target[selected],
        predicted_temperature=predictions,
    )
    rows = []
    for index, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "mean_absolute_shap": float(np.mean(np.abs(values[:, index]))),
                "mean_signed_shap": float(np.mean(values[:, index])),
                "maximum_absolute_shap": float(np.max(np.abs(values[:, index]))),
            }
        )
    rows.sort(key=lambda row: float(row["mean_absolute_shap"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["shap_rank"] = rank
    return rows
