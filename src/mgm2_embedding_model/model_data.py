"""Safe loading of model-ready embedding tables and frozen fold assignments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_csv_rows, read_nonempty_lines, sha256_file
from .model_config import ModelRunConfig


NONFEATURE_COLUMNS = [
    "material_sample",
    "cruise",
    "temperature",
    "outer_partition",
    "development_cv_fold",
    "training_row",
    "source_asv_processed_matrix_row",
]


@dataclass(frozen=True)
class ModelingDataset:
    representation: str
    sample_ids: np.ndarray
    cruise_ids: np.ndarray
    target: np.ndarray
    outer_partition: np.ndarray
    cv_fold: np.ndarray
    training_row: np.ndarray
    source_asv_processed_matrix_row: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray

    @property
    def development(self) -> np.ndarray:
        return np.flatnonzero(self.outer_partition == "development")

    @property
    def test(self) -> np.ndarray:
        return np.flatnonzero(self.outer_partition == "test")


@dataclass(frozen=True)
class PairedModelingData:
    original: ModelingDataset
    mapseq: ModelingDataset
    prepared_manifest: dict[str, Any]
    training_schema: dict[str, Any]


def _manifest_hash(manifest: dict[str, Any], role: str) -> str:
    matches = [row for row in manifest.get("outputs", []) if row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Prepared manifest must contain exactly one output role {role!r}")
    return str(matches[0]["sha256"])


def _verify_prepared_inputs(config: ModelRunConfig) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    paths = config.inputs
    required_files = {
        "prepared_manifest": paths.prepared_manifest,
        "original_table": paths.original_table,
        "mapseq_table": paths.mapseq_table,
        "feature_columns": paths.feature_columns,
        "training_schema": paths.training_schema,
        "candidate_config": paths.candidate_config,
        "asv_reference_predictions": paths.asv_reference_predictions,
        "asv_reference_metrics": paths.asv_reference_metrics,
        "asv_reference_summary": paths.asv_reference_summary,
    }
    missing = [f"{role}: {path}" for role, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing modeling inputs: " + "; ".join(missing))

    manifest = json.loads(paths.prepared_manifest.read_text(encoding="utf-8"))
    prepared_roles = {
        "original_temperature_training_table": paths.original_table,
        "mapseq_temperature_training_table": paths.mapseq_table,
        "feature_columns": paths.feature_columns,
        "training_schema": paths.training_schema,
    }
    for role, path in prepared_roles.items():
        observed = sha256_file(path)
        expected = _manifest_hash(manifest, role)
        if observed != expected:
            raise ValueError(
                f"Prepared input hash drift for {role}: observed={observed}, expected={expected}"
            )

    pinned_paths = {
        "prepared_manifest": paths.prepared_manifest,
        "asv_reference_predictions": paths.asv_reference_predictions,
        "asv_reference_metrics": paths.asv_reference_metrics,
        "asv_reference_summary": paths.asv_reference_summary,
    }
    if set(pinned_paths) != set(config.expected_sha256):
        raise ValueError("Pinned input hash roles are incomplete in modeling.json")
    for role, path in pinned_paths.items():
        observed = sha256_file(path)
        expected = config.expected_sha256[role]
        if observed != expected:
            raise ValueError(
                f"ASV reference hash drift for {role}: observed={observed}, expected={expected}"
            )

    feature_names = tuple(read_nonempty_lines(paths.feature_columns))
    schema = json.loads(paths.training_schema.read_text(encoding="utf-8"))
    if schema.get("feature_columns") != list(feature_names):
        raise ValueError("training_schema.json and feature_columns.txt disagree")
    if schema.get("feature_count") != len(feature_names):
        raise ValueError("training schema feature count is incorrect")
    return manifest, schema, feature_names


def _load_table(
    path: Path,
    representation: str,
    feature_names: tuple[str, ...],
) -> ModelingDataset:
    fields, rows = read_csv_rows(path)
    expected_fields = [*NONFEATURE_COLUMNS, *feature_names]
    if fields != expected_fields:
        raise ValueError(
            f"{representation} table columns do not exactly match the safe schema"
        )
    if not rows:
        raise ValueError(f"{representation} table is empty")
    sample_ids = np.asarray([row["material_sample"].strip() for row in rows], dtype=object)
    cruise_ids = np.asarray([row["cruise"].strip() for row in rows], dtype=object)
    target = np.asarray([float(row["temperature"]) for row in rows], dtype=np.float64)
    partitions = np.asarray([row["outer_partition"].strip() for row in rows], dtype=object)
    folds = np.asarray(
        [int(row["development_cv_fold"]) if row["development_cv_fold"].strip() else -1 for row in rows],
        dtype=np.int64,
    )
    training_rows = np.asarray([int(row["training_row"]) for row in rows], dtype=np.int64)
    source_rows = np.asarray(
        [int(row["source_asv_processed_matrix_row"]) for row in rows], dtype=np.int64
    )
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    dataset = ModelingDataset(
        representation=representation,
        sample_ids=sample_ids,
        cruise_ids=cruise_ids,
        target=target,
        outer_partition=partitions,
        cv_fold=folds,
        training_row=training_rows,
        source_asv_processed_matrix_row=source_rows,
        feature_names=feature_names,
        features=features,
    )
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: ModelingDataset) -> None:
    count = len(dataset.sample_ids)
    if len(set(dataset.sample_ids.tolist())) != count or any(not value for value in dataset.sample_ids):
        raise ValueError(f"{dataset.representation} material IDs must be unique and nonempty")
    if not np.array_equal(dataset.training_row, np.arange(count, dtype=np.int64)):
        raise ValueError(f"{dataset.representation} training_row is not contiguous")
    if dataset.features.shape != (count, len(dataset.feature_names)):
        raise ValueError(f"{dataset.representation} feature matrix shape is incorrect")
    if not np.isfinite(dataset.features).all() or not np.isfinite(dataset.target).all():
        raise ValueError(f"{dataset.representation} contains non-finite values")
    if set(dataset.outer_partition.tolist()) != {"development", "test"}:
        raise ValueError(f"{dataset.representation} has invalid outer partition labels")
    development = dataset.development
    test = dataset.test
    if set(dataset.cv_fold[development].tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{dataset.representation} development folds must be 1-5")
    if not np.all(dataset.cv_fold[test] == -1):
        raise ValueError(f"{dataset.representation} test rows must not have CV folds")
    dev_cruises = set(dataset.cruise_ids[development].tolist())
    test_cruises = set(dataset.cruise_ids[test].tolist())
    if dev_cruises & test_cruises:
        raise ValueError(f"{dataset.representation} cruises cross development and test")
    for fold in range(1, 6):
        validation = development[dataset.cv_fold[development] == fold]
        training = development[dataset.cv_fold[development] != fold]
        if set(dataset.cruise_ids[validation]) & set(dataset.cruise_ids[training]):
            raise ValueError(f"{dataset.representation} cruise crosses CV fold {fold}")


def load_paired_modeling_data(config: ModelRunConfig) -> PairedModelingData:
    manifest, schema, feature_names = _verify_prepared_inputs(config)
    original = _load_table(config.inputs.original_table, "original", feature_names)
    mapseq = _load_table(config.inputs.mapseq_table, "mapseq", feature_names)
    comparable_fields = (
        "sample_ids",
        "cruise_ids",
        "target",
        "outer_partition",
        "cv_fold",
        "training_row",
        "source_asv_processed_matrix_row",
    )
    for field in comparable_fields:
        if not np.array_equal(getattr(original, field), getattr(mapseq, field)):
            raise ValueError(f"Original and MAPseq data disagree in {field}")
    if np.array_equal(original.features, mapseq.features):
        raise ValueError(
            "Original and MAPseq feature matrices are identical; refusing a meaningless comparison"
        )
    expected = config.expected
    observed = {
        "samples": len(original.sample_ids),
        "features": original.features.shape[1],
        "development_samples": len(original.development),
        "test_samples": len(original.test),
        "development_cruises": len(set(original.cruise_ids[original.development])),
        "test_cruises": len(set(original.cruise_ids[original.test])),
        "cv_folds": len(set(original.cv_fold[original.development].tolist())),
    }
    mismatches = {key: (value, expected.get(key)) for key, value in observed.items() if value != expected.get(key)}
    if mismatches:
        raise ValueError(f"Prepared modeling counts differ from config: {mismatches}")
    return PairedModelingData(original, mapseq, manifest, schema)


def model_data_summary(data: PairedModelingData) -> dict[str, Any]:
    dataset = data.original
    return {
        "samples": len(dataset.sample_ids),
        "features_per_representation": dataset.features.shape[1],
        "development_samples": len(dataset.development),
        "test_samples": len(dataset.test),
        "development_cruises": len(set(dataset.cruise_ids[dataset.development])),
        "test_cruises": len(set(dataset.cruise_ids[dataset.test])),
        "fold_sample_counts": {
            str(fold): int(np.sum(dataset.cv_fold == fold)) for fold in range(1, 6)
        },
        "target_min_c": float(np.min(dataset.target)),
        "target_max_c": float(np.max(dataset.target)),
        "target_mean_c": float(np.mean(dataset.target)),
    }
