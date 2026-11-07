"""Configuration for the paired MGM2 embedding CatBoost experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInputPaths:
    prepared_manifest: Path
    original_table: Path
    mapseq_table: Path
    feature_columns: Path
    training_schema: Path
    candidate_config: Path
    asv_reference_predictions: Path
    asv_reference_metrics: Path
    asv_reference_summary: Path


@dataclass(frozen=True)
class ModelSettings:
    maximum_iterations: int
    early_stopping_rounds: int
    random_seed: int
    thread_count: int
    task_type: str
    devices: str | None
    verbose: int
    bootstrap_replicates: int
    bootstrap_seed: int
    calculate_shap: bool
    shap_max_samples: int


@dataclass(frozen=True)
class ModelRunConfig:
    project_root: Path
    config_path: Path
    inputs: ModelInputPaths
    results_dir: Path
    expected_sha256: dict[str, str]
    expected: dict[str, int]
    settings: ModelSettings
    candidates: tuple[dict[str, Any], ...]


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_candidates(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate configuration must contain a nonempty candidates list")
    required = {"name", "depth", "learning_rate", "l2_leaf_reg"}
    allowed = required | {"random_strength"}
    names: set[str] = set()
    clean: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {position} is not an object")
        missing = required - set(candidate)
        unexpected = set(candidate) - allowed
        if missing or unexpected:
            raise ValueError(
                f"Candidate {position} has missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}"
            )
        name = str(candidate["name"]).strip()
        if not name or name in names:
            raise ValueError(f"Candidate name is blank or duplicated: {name!r}")
        depth = candidate["depth"]
        learning_rate = candidate["learning_rate"]
        l2_leaf_reg = candidate["l2_leaf_reg"]
        random_strength = candidate.get("random_strength", 1.0)
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 12:
            raise ValueError(f"Candidate {name}: depth must be an integer in [1, 12]")
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError(f"Candidate {name}: learning_rate must be in (0, 1]")
        if (
            not isinstance(l2_leaf_reg, (int, float))
            or isinstance(l2_leaf_reg, bool)
            or l2_leaf_reg < 0
        ):
            raise ValueError(f"Candidate {name}: l2_leaf_reg must be nonnegative")
        if (
            not isinstance(random_strength, (int, float))
            or isinstance(random_strength, bool)
            or random_strength < 0
        ):
            raise ValueError(f"Candidate {name}: random_strength must be nonnegative")
        names.add(name)
        clean.append(
            {
                "name": name,
                "depth": depth,
                "learning_rate": float(learning_rate),
                "l2_leaf_reg": float(l2_leaf_reg),
                "random_strength": float(random_strength),
            }
        )
    return tuple(clean)


def load_model_config(path: str | Path) -> ModelRunConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "prepared_manifest",
        "original_table",
        "mapseq_table",
        "feature_columns",
        "training_schema",
        "candidate_config",
        "asv_reference_predictions",
        "asv_reference_metrics",
        "asv_reference_summary",
        "results_dir",
        "expected_sha256",
        "expected",
        "settings",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing modeling configuration keys: {sorted(missing)}")
    project_root = config_path.parent.parent
    inputs = ModelInputPaths(
        prepared_manifest=_resolve(project_root, raw["prepared_manifest"]),
        original_table=_resolve(project_root, raw["original_table"]),
        mapseq_table=_resolve(project_root, raw["mapseq_table"]),
        feature_columns=_resolve(project_root, raw["feature_columns"]),
        training_schema=_resolve(project_root, raw["training_schema"]),
        candidate_config=_resolve(project_root, raw["candidate_config"]),
        asv_reference_predictions=_resolve(project_root, raw["asv_reference_predictions"]),
        asv_reference_metrics=_resolve(project_root, raw["asv_reference_metrics"]),
        asv_reference_summary=_resolve(project_root, raw["asv_reference_summary"]),
    )
    settings_raw = raw["settings"]
    if not isinstance(settings_raw.get("calculate_shap"), bool):
        raise ValueError("settings.calculate_shap must be a JSON boolean")
    settings = ModelSettings(
        maximum_iterations=int(settings_raw["maximum_iterations"]),
        early_stopping_rounds=int(settings_raw["early_stopping_rounds"]),
        random_seed=int(settings_raw["random_seed"]),
        thread_count=int(settings_raw["thread_count"]),
        task_type=str(settings_raw["task_type"]).upper(),
        devices=(None if settings_raw.get("devices") is None else str(settings_raw["devices"])),
        verbose=int(settings_raw["verbose"]),
        bootstrap_replicates=int(settings_raw["bootstrap_replicates"]),
        bootstrap_seed=int(settings_raw["bootstrap_seed"]),
        calculate_shap=settings_raw["calculate_shap"],
        shap_max_samples=int(settings_raw["shap_max_samples"]),
    )
    _validate_settings(settings)
    candidate_path = inputs.candidate_config
    return ModelRunConfig(
        project_root=project_root,
        config_path=config_path,
        inputs=inputs,
        results_dir=_resolve(project_root, raw["results_dir"]),
        expected_sha256={str(k): str(v).lower() for k, v in raw["expected_sha256"].items()},
        expected={str(k): int(v) for k, v in raw["expected"].items()},
        settings=settings,
        candidates=_load_candidates(candidate_path),
    )


def _validate_settings(settings: ModelSettings) -> None:
    if settings.maximum_iterations <= 0 or settings.early_stopping_rounds <= 0:
        raise ValueError("Iteration limits and early stopping must be positive")
    if settings.task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU")
    if settings.thread_count == 0 or settings.thread_count < -1:
        raise ValueError("thread_count must be -1 or a positive integer")
    if settings.verbose < 0:
        raise ValueError("verbose must be nonnegative")
    if settings.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if settings.shap_max_samples <= 0:
        raise ValueError("shap_max_samples must be positive")


def apply_runtime_overrides(
    config: ModelRunConfig,
    *,
    task_type: str | None = None,
    devices: str | None = None,
    thread_count: int | None = None,
    verbose: int | None = None,
) -> ModelRunConfig:
    settings = replace(
        config.settings,
        task_type=(task_type.upper() if task_type else config.settings.task_type),
        devices=(devices if devices is not None else config.settings.devices),
        thread_count=(thread_count if thread_count is not None else config.settings.thread_count),
        verbose=(verbose if verbose is not None else config.settings.verbose),
    )
    _validate_settings(settings)
    return replace(config, settings=settings)


def serializable_model_config(config: ModelRunConfig, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_path": str(config.config_path),
        "inputs": {key: str(value) for key, value in asdict(config.inputs).items()},
        "results_dir": str(config.results_dir),
        "expected_sha256": config.expected_sha256,
        "expected": config.expected,
        "settings": asdict(config.settings),
        "candidates": list(config.candidates),
    }
