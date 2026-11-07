"""Validation primitives and structured audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io import EmbeddingData


EXPECTED_METADATA_COLUMNS = [
    "sample_id",
    "temperature",
    "salinity",
    "station",
    "cruise",
    "material_sample",
    "replicate",
    "date_utc",
    "depth_m",
    "depth_category",
    "lat",
    "lon",
    "is_biological_sample",
]

REQUIRED_SPLIT_COLUMNS = [
    "materialSampleID",
    "sfer_cruise_id",
    "sfer_temp",
    "outer_partition",
    "development_cv_fold",
    "processed_matrix_row",
]


@dataclass
class ValidationRecorder:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def require(
        self,
        name: str,
        condition: bool,
        *,
        observed: Any = "",
        expected: Any = "",
        details: str = "",
    ) -> None:
        self.checks.append(
            {
                "check": name,
                "status": "PASS" if bool(condition) else "FAIL",
                "severity": "error",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
        )

    def warn(
        self,
        name: str,
        condition: bool,
        *,
        observed: Any = "",
        expected: Any = "",
        details: str = "",
    ) -> None:
        self.checks.append(
            {
                "check": name,
                "status": "PASS" if bool(condition) else "WARN",
                "severity": "warning",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if row["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if row["status"] == "WARN"]

    @property
    def status(self) -> str:
        return "FAIL" if self.failures else "PASS"

    def raise_for_failures(self) -> None:
        if self.failures:
            names = ", ".join(row["check"] for row in self.failures)
            raise ValidationError(f"Validation failed: {names}", self)


class ValidationError(RuntimeError):
    def __init__(self, message: str, recorder: ValidationRecorder):
        super().__init__(message)
        self.recorder = recorder


def canonical_replicate(raw_value: str) -> str | None:
    """Map the approved raw labels to A/B/C without inventing new groups."""

    mapping = {
        "A": "A",
        "AA": "A",
        "AB": "A",
        "B": "B",
        "C": "C",
        "CB": "C",
    }
    cleaned = raw_value.strip().upper()
    return mapping.get(cleaned)


def duplicate_embedding_rows(
    representation: str,
    data: EmbeddingData,
) -> list[dict[str, Any]]:
    """Return an audit row for every sample involved in an exact vector duplicate."""

    _, inverse, counts = np.unique(
        data.embeddings,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    duplicate_labels = [int(label) for label, count in enumerate(counts) if count > 1]
    rows: list[dict[str, Any]] = []
    for group_number, label in enumerate(duplicate_labels, start=1):
        indices = np.flatnonzero(inverse == label)
        for index in indices:
            rows.append(
                {
                    "representation": representation,
                    "duplicate_group": group_number,
                    "row_index": int(index),
                    "sample_id": data.sample_ids[int(index)],
                    "group_size": int(len(indices)),
                }
            )
    return rows


def validate_embedding_pair(
    original: EmbeddingData,
    mapseq: EmbeddingData,
    expected: Mapping[str, int],
    recorder: ValidationRecorder,
) -> list[dict[str, Any]]:
    expected_rows = expected["embedding_rows"]
    expected_features = expected["embedding_features"]

    for label, data in (("original", original), ("mapseq", mapseq)):
        recorder.require(
            f"{label}_hdf5_keys",
            set(data.root_keys) == {"embeddings", "sample_ids"},
            observed="|".join(data.root_keys),
            expected="embeddings|sample_ids",
        )
        recorder.require(
            f"{label}_embedding_shape",
            data.embeddings.shape == (expected_rows, expected_features),
            observed=str(tuple(data.embeddings.shape)),
            expected=str((expected_rows, expected_features)),
        )
        recorder.require(
            f"{label}_embedding_dtype",
            data.embedding_dtype == "float32",
            observed=data.embedding_dtype,
            expected="float32",
        )
        recorder.require(
            f"{label}_sample_id_count",
            len(data.sample_ids) == expected_rows,
            observed=len(data.sample_ids),
            expected=expected_rows,
        )
        recorder.require(
            f"{label}_sample_ids_unique",
            len(set(data.sample_ids)) == len(data.sample_ids),
            observed=len(set(data.sample_ids)),
            expected=len(data.sample_ids),
        )
        recorder.require(
            f"{label}_sample_ids_nonempty_and_stripped",
            all(value and value == value.strip() for value in data.sample_ids),
            details="IDs must be nonempty and must not contain surrounding whitespace.",
        )
        recorder.require(
            f"{label}_embeddings_finite",
            bool(np.isfinite(data.embeddings).all()),
            observed=int(np.isfinite(data.embeddings).sum()),
            expected=int(data.embeddings.size),
        )
        row_norms = np.linalg.norm(data.embeddings.astype(np.float64), axis=1)
        recorder.require(
            f"{label}_no_zero_embedding_rows",
            bool(np.all(row_norms > 0)),
            observed=int(np.count_nonzero(row_norms == 0)),
            expected=0,
        )
        recorder.require(
            f"{label}_hidden_size_attribute",
            int(data.attributes.get("hidden_size", -1)) == expected_features,
            observed=data.attributes.get("hidden_size", "missing"),
            expected=expected_features,
        )
        recorder.require(
            f"{label}_embedding_strategy_attribute",
            data.attributes.get("embedding_strategy") == "cls",
            observed=data.attributes.get("embedding_strategy", "missing"),
            expected="cls",
        )

    recorder.require(
        "paired_sample_ids_same_set",
        set(original.sample_ids) == set(mapseq.sample_ids),
        observed=f"original={len(set(original.sample_ids))}; mapseq={len(set(mapseq.sample_ids))}",
        expected="identical sets",
    )
    recorder.require(
        "paired_sample_ids_same_order",
        original.sample_ids == mapseq.sample_ids,
        observed="same" if original.sample_ids == mapseq.sample_ids else "different",
        expected="same",
        details="The pipeline still joins by ID instead of relying on row position.",
    )
    for attribute in ("embedding_strategy", "hidden_size", "model_class", "model_checkpoint", "model_config"):
        recorder.require(
            f"paired_{attribute}_matches",
            original.attributes.get(attribute) == mapseq.attributes.get(attribute),
            observed=f"original={original.attributes.get(attribute)}; mapseq={mapseq.attributes.get(attribute)}",
            expected="matching values",
        )
    recorder.require(
        "representations_are_not_identical",
        not np.array_equal(original.embeddings, mapseq.embeddings),
        observed="different" if not np.array_equal(original.embeddings, mapseq.embeddings) else "identical",
        expected="different",
    )

    duplicates = duplicate_embedding_rows("original", original)
    duplicates.extend(duplicate_embedding_rows("mapseq", mapseq))
    duplicate_groups = {
        (row["representation"], row["duplicate_group"]) for row in duplicates
    }
    recorder.warn(
        "exact_duplicate_embedding_vectors_reviewed",
        len(duplicate_groups) == 0,
        observed=len(duplicate_groups),
        expected=0,
        details=(
            "Exact duplicate vectors are reported for review; they are not automatically "
            "removed because sample IDs remain distinct."
        ),
    )
    return duplicates


def validate_metadata_pair(
    original_fields: Sequence[str],
    original_rows: Sequence[Mapping[str, str]],
    mapseq_fields: Sequence[str],
    mapseq_rows: Sequence[Mapping[str, str]],
    embedding_ids: Iterable[str],
    expected: Mapping[str, int],
    recorder: ValidationRecorder,
) -> None:
    recorder.require(
        "original_metadata_columns",
        list(original_fields) == EXPECTED_METADATA_COLUMNS,
        observed="|".join(original_fields),
        expected="|".join(EXPECTED_METADATA_COLUMNS),
    )
    recorder.require(
        "mapseq_metadata_columns",
        list(mapseq_fields) == EXPECTED_METADATA_COLUMNS,
        observed="|".join(mapseq_fields),
        expected="|".join(EXPECTED_METADATA_COLUMNS),
    )
    recorder.require(
        "metadata_tables_identical",
        list(original_fields) == list(mapseq_fields) and list(original_rows) == list(mapseq_rows),
        observed="identical" if list(original_rows) == list(mapseq_rows) else "different",
        expected="identical",
    )
    recorder.require(
        "metadata_row_count",
        len(original_rows) == expected["metadata_rows"],
        observed=len(original_rows),
        expected=expected["metadata_rows"],
    )
    sample_ids = [row.get("sample_id", "").strip() for row in original_rows]
    recorder.require(
        "metadata_sample_ids_unique",
        len(sample_ids) == len(set(sample_ids)),
        observed=len(set(sample_ids)),
        expected=len(sample_ids),
    )
    recorder.require(
        "metadata_sample_ids_nonempty",
        all(sample_ids),
        observed=sum(not value for value in sample_ids),
        expected=0,
    )
    recorder.require(
        "metadata_rows_are_biological",
        {row.get("is_biological_sample", "").strip().lower() for row in original_rows} == {"yes"},
        observed="|".join(sorted({row.get("is_biological_sample", "") for row in original_rows})),
        expected="yes",
    )
    embedding_set = set(embedding_ids)
    metadata_set = set(sample_ids)
    recorder.require(
        "all_metadata_ids_present_in_embeddings",
        metadata_set <= embedding_set,
        observed=len(metadata_set - embedding_set),
        expected=0,
    )
    hdf5_only = len(embedding_set - metadata_set)
    recorder.require(
        "hdf5_only_sample_count",
        hdf5_only == expected["hdf5_only_rows"],
        observed=hdf5_only,
        expected=expected["hdf5_only_rows"],
        details="These rows are excluded because they are absent from biological metadata.",
    )
    raw_replicates = {
        row.get("replicate", "").strip()
        for row in original_rows
        if row.get("replicate", "").strip()
    }
    approved = {"A", "Aa", "Ab", "B", "C", "Cb"}
    recorder.require(
        "replicate_labels_approved",
        raw_replicates == approved,
        observed="|".join(sorted(raw_replicates)),
        expected="|".join(sorted(approved)),
    )


def validate_split_table(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    expected: Mapping[str, int],
    recorder: ValidationRecorder,
) -> None:
    recorder.require(
        "split_columns",
        list(fields) == REQUIRED_SPLIT_COLUMNS,
        observed="|".join(fields),
        expected="|".join(REQUIRED_SPLIT_COLUMNS),
    )
    material_ids = [row.get("materialSampleID", "").strip() for row in rows]
    recorder.require(
        "split_row_count",
        len(rows) == expected["temperature_samples"],
        observed=len(rows),
        expected=expected["temperature_samples"],
    )
    recorder.require(
        "split_material_ids_unique_nonempty",
        all(material_ids) and len(material_ids) == len(set(material_ids)),
        observed=len(set(material_ids)),
        expected=len(material_ids),
    )
    partitions = {row.get("outer_partition", "").strip() for row in rows}
    recorder.require(
        "split_partition_labels",
        partitions == {"development", "test"},
        observed="|".join(sorted(partitions)),
        expected="development|test",
    )
    development = [row for row in rows if row.get("outer_partition", "").strip() == "development"]
    test = [row for row in rows if row.get("outer_partition", "").strip() == "test"]
    recorder.require(
        "development_sample_count",
        len(development) == expected["development_samples"],
        observed=len(development),
        expected=expected["development_samples"],
    )
    recorder.require(
        "test_sample_count",
        len(test) == expected["test_samples"],
        observed=len(test),
        expected=expected["test_samples"],
    )
    folds = {row.get("development_cv_fold", "").strip() for row in development}
    expected_folds = {str(number) for number in range(1, expected["cv_folds"] + 1)}
    recorder.require(
        "development_fold_labels",
        folds == expected_folds,
        observed="|".join(sorted(folds)),
        expected="|".join(sorted(expected_folds)),
    )
    recorder.require(
        "test_fold_is_blank",
        all(not row.get("development_cv_fold", "").strip() for row in test),
        observed=sum(bool(row.get("development_cv_fold", "").strip()) for row in test),
        expected=0,
    )

    cruise_partitions: dict[str, set[str]] = {}
    cruise_folds: dict[str, set[str]] = {}
    for row in rows:
        cruise = row.get("sfer_cruise_id", "").strip()
        cruise_partitions.setdefault(cruise, set()).add(row.get("outer_partition", "").strip())
        if row.get("outer_partition", "").strip() == "development":
            cruise_folds.setdefault(cruise, set()).add(row.get("development_cv_fold", "").strip())
    recorder.require(
        "cruises_do_not_cross_outer_partitions",
        all(len(values) == 1 for values in cruise_partitions.values()),
        observed=sum(len(values) > 1 for values in cruise_partitions.values()),
        expected=0,
    )
    recorder.require(
        "development_cruises_do_not_cross_folds",
        all(len(values) == 1 for values in cruise_folds.values()),
        observed=sum(len(values) > 1 for values in cruise_folds.values()),
        expected=0,
    )
    development_cruises = {
        row.get("sfer_cruise_id", "").strip() for row in development
    }
    test_cruises = {row.get("sfer_cruise_id", "").strip() for row in test}
    recorder.require(
        "development_cruise_count",
        len(development_cruises) == expected["development_cruises"],
        observed=len(development_cruises),
        expected=expected["development_cruises"],
    )
    recorder.require(
        "test_cruise_count",
        len(test_cruises) == expected["test_cruises"],
        observed=len(test_cruises),
        expected=expected["test_cruises"],
    )
    recorder.require(
        "development_test_cruises_disjoint",
        development_cruises.isdisjoint(test_cruises),
        observed="|".join(sorted(development_cruises & test_cruises)),
        expected="no overlap",
    )
    processed_rows: list[int] = []
    valid_rows = True
    for row in rows:
        try:
            processed_rows.append(int(row.get("processed_matrix_row", "")))
        except ValueError:
            valid_rows = False
    recorder.require(
        "processed_matrix_rows_are_unique_integers",
        valid_rows and len(processed_rows) == len(set(processed_rows)),
        observed=len(set(processed_rows)) if valid_rows else "non-integer present",
        expected=len(rows),
    )


def validate_processed_row_mapping(
    split_rows: Sequence[Mapping[str, str]],
    processed_material_ids: Sequence[str],
    expected: Mapping[str, int],
    recorder: ValidationRecorder,
) -> None:
    recorder.require(
        "processed_material_id_count",
        len(processed_material_ids) == expected["material_samples"],
        observed=len(processed_material_ids),
        expected=expected["material_samples"],
    )
    recorder.require(
        "processed_material_ids_unique_nonempty",
        all(processed_material_ids)
        and len(processed_material_ids) == len(set(processed_material_ids)),
        observed=len(set(processed_material_ids)),
        expected=len(processed_material_ids),
    )
    mismatches = 0
    invalid_indices = 0
    for row in split_rows:
        try:
            index = int(row.get("processed_matrix_row", ""))
        except ValueError:
            invalid_indices += 1
            continue
        if index < 0 or index >= len(processed_material_ids):
            invalid_indices += 1
            continue
        if processed_material_ids[index] != row.get("materialSampleID", "").strip():
            mismatches += 1
    recorder.require(
        "split_processed_rows_in_range",
        invalid_indices == 0,
        observed=invalid_indices,
        expected=0,
    )
    recorder.require(
        "split_processed_rows_match_material_ids",
        mismatches == 0,
        observed=mismatches,
        expected=0,
        details="Each frozen processed_matrix_row must point to the same material ID in sample_ids.txt.",
    )


def checks_summary(recorder: ValidationRecorder) -> dict[str, int | str]:
    return {
        "status": recorder.status,
        "total_checks": len(recorder.checks),
        "passed_checks": sum(row["status"] == "PASS" for row in recorder.checks),
        "warning_checks": len(recorder.warnings),
        "failed_checks": len(recorder.failures),
    }
