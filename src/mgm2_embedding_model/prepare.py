"""End-to-end preparation of paired original and MAPseq embedding tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from . import __version__
from .config import PreparationConfig
from .io import (
    EmbeddingData,
    load_embedding_h5,
    read_csv_rows,
    read_nonempty_lines,
    sha256_file,
    write_csv_rows,
    write_json,
    write_text,
)
from .validation import (
    REQUIRED_SPLIT_COLUMNS,
    ValidationRecorder,
    canonical_replicate,
    checks_summary,
    validate_embedding_pair,
    validate_metadata_pair,
    validate_processed_row_mapping,
    validate_split_table,
)


EMBEDDING_COLUMNS = [f"embedding_{number:03d}" for number in range(1, 225)]
CONSENSUS_FIELDS = [
    "temperature",
    "salinity",
    "station",
    "cruise",
    "date_utc",
    "depth_m",
    "depth_category",
    "lat",
    "lon",
    "is_biological_sample",
]
STRICT_CONSENSUS_FIELDS = [field for field in CONSENSUS_FIELDS if field != "date_utc"]


@dataclass(frozen=True)
class LoadedSources:
    original: EmbeddingData
    mapseq: EmbeddingData
    metadata_fields: list[str]
    metadata_rows: list[dict[str, str]]
    split_fields: list[str]
    split_rows: list[dict[str, str]]
    processed_material_ids: list[str]
    duplicate_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class AggregationResult:
    vectors: dict[str, np.ndarray]
    stats: dict[str, dict[str, Any]]
    replicate_rows: list[dict[str, Any]]


def load_and_validate_sources(
    config: PreparationConfig,
    recorder: ValidationRecorder | None = None,
) -> tuple[LoadedSources, ValidationRecorder]:
    recorder = recorder or ValidationRecorder()
    source_paths = {
        "original_embeddings": config.original_embeddings,
        "mapseq_embeddings": config.mapseq_embeddings,
        "original_metadata": config.original_metadata,
        "mapseq_metadata": config.mapseq_metadata,
        "frozen_splits": config.frozen_splits,
        "processed_material_ids": config.processed_material_ids,
    }
    for role, path in source_paths.items():
        recorder.require(
            f"source_exists_{role}",
            path.is_file(),
            observed=str(path),
            expected="existing file",
        )
    recorder.raise_for_failures()
    recorder.require(
        "expected_hash_roles_complete",
        set(config.expected_sha256) == set(source_paths),
        observed="|".join(sorted(config.expected_sha256)),
        expected="|".join(sorted(source_paths)),
    )
    for role, path in source_paths.items():
        observed_hash = sha256_file(path)
        recorder.require(
            f"source_sha256_{role}",
            observed_hash == config.expected_sha256.get(role),
            observed=observed_hash,
            expected=config.expected_sha256.get(role, "missing"),
            details="Pinned hashes prevent silent source drift, including changes under a latest/ path.",
        )
    recorder.raise_for_failures()

    original = load_embedding_h5(config.original_embeddings)
    mapseq = load_embedding_h5(config.mapseq_embeddings)
    original_fields, original_rows = read_csv_rows(config.original_metadata)
    mapseq_fields, mapseq_rows = read_csv_rows(config.mapseq_metadata)
    split_fields, split_rows = read_csv_rows(config.frozen_splits)
    processed_material_ids = read_nonempty_lines(config.processed_material_ids)

    duplicate_rows = validate_embedding_pair(
        original, mapseq, config.expected, recorder
    )
    validate_metadata_pair(
        original_fields,
        original_rows,
        mapseq_fields,
        mapseq_rows,
        original.sample_ids,
        config.expected,
        recorder,
    )
    validate_split_table(split_fields, split_rows, config.expected, recorder)
    validate_processed_row_mapping(
        split_rows, processed_material_ids, config.expected, recorder
    )
    recorder.raise_for_failures()
    return (
        LoadedSources(
            original=original,
            mapseq=mapseq,
            metadata_fields=original_fields,
            metadata_rows=original_rows,
            split_fields=split_fields,
            split_rows=split_rows,
            processed_material_ids=processed_material_ids,
            duplicate_rows=duplicate_rows,
        ),
        recorder,
    )


def _included_metadata_rows(metadata_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    included: list[dict[str, str]] = []
    for row in metadata_rows:
        material = row.get("material_sample", "").strip()
        replicate = canonical_replicate(row.get("replicate", ""))
        biological = row.get("is_biological_sample", "").strip().lower() == "yes"
        if material and replicate and biological:
            included.append(dict(row))
    return included


def aggregate_embeddings(
    metadata_rows: Sequence[Mapping[str, str]],
    embedding: EmbeddingData,
) -> AggregationResult:
    """Average technical reruns, then give each A/B/C replicate equal weight."""

    id_to_index = {sample_id: index for index, sample_id in enumerate(embedding.sample_ids)}
    grouped: dict[str, dict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _included_metadata_rows(metadata_rows):
        sample_id = row["sample_id"].strip()
        if sample_id not in id_to_index:
            raise ValueError(f"Metadata sample is absent from embeddings: {sample_id}")
        material = row["material_sample"].strip()
        replicate = canonical_replicate(row["replicate"])
        if replicate is None:  # guarded by _included_metadata_rows
            raise AssertionError("Canonical replicate unexpectedly missing")
        grouped[material][replicate].append(row)

    vectors: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, Any]] = {}
    replicate_rows: list[dict[str, Any]] = []
    for material in sorted(grouped):
        replicate_groups = grouped[material]
        replicate_means: list[np.ndarray] = []
        source_count = 0
        technical_rerun_group_count = 0
        biological_replicate_count = len(replicate_groups)
        for replicate in sorted(replicate_groups):
            rows = replicate_groups[replicate]
            indices = [id_to_index[row["sample_id"].strip()] for row in rows]
            source_count += len(indices)
            if len(indices) > 1:
                technical_rerun_group_count += 1
            replicate_mean = np.mean(
                embedding.embeddings[indices].astype(np.float64), axis=0
            )
            replicate_means.append(replicate_mean)
            for row in rows:
                technical_weight = 1.0 / len(rows)
                biological_weight = 1.0 / biological_replicate_count
                replicate_rows.append(
                    {
                        "sample_id": row["sample_id"].strip(),
                        "material_sample": material,
                        "raw_replicate": row["replicate"].strip(),
                        "canonical_replicate": replicate,
                        "technical_run_count": len(rows),
                        "biological_replicate_count": biological_replicate_count,
                        "technical_run_weight_within_replicate": technical_weight,
                        "canonical_replicate_weight_within_material": biological_weight,
                        "final_source_row_weight": technical_weight * biological_weight,
                    }
                )
        vectors[material] = np.mean(np.stack(replicate_means), axis=0)
        stats[material] = {
            "source_sample_count": source_count,
            "biological_replicate_count": biological_replicate_count,
            "canonical_replicates": "|".join(sorted(replicate_groups)),
            "technical_rerun_group_count": technical_rerun_group_count,
        }
    return AggregationResult(vectors=vectors, stats=stats, replicate_rows=replicate_rows)


def _metadata_by_material(
    metadata_rows: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in _included_metadata_rows(metadata_rows):
        grouped[row["material_sample"].strip()].append(row)
    return dict(grouped)


def build_material_metadata(
    metadata_rows: Sequence[Mapping[str, str]],
    aggregation: AggregationResult,
    recorder: ValidationRecorder,
    expected: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped = _metadata_by_material(metadata_rows)
    rows: list[dict[str, Any]] = []
    inconsistent_counts: dict[str, int] = {field: 0 for field in CONSENSUS_FIELDS}
    for material in sorted(grouped):
        source_rows = grouped[material]
        output: dict[str, Any] = {"material_sample": material}
        inconsistent_fields: list[str] = []
        for field in CONSENSUS_FIELDS:
            values = sorted(
                {
                    row.get(field, "").strip()
                    for row in source_rows
                    if row.get(field, "").strip()
                }
            )
            if len(values) > 1:
                inconsistent_fields.append(field)
                inconsistent_counts[field] += 1
                output[field] = ""
            else:
                output[field] = values[0] if values else ""
            if field == "date_utc":
                output["date_utc_values"] = "|".join(values)

        output.update(aggregation.stats[material])
        output["all_metadata_consistent"] = not inconsistent_fields
        output["inconsistent_metadata_fields"] = "|".join(inconsistent_fields)
        output["temperature_model_eligible"] = bool(
            output["temperature"] and output["cruise"]
        )
        rows.append(output)

    recorder.require(
        "usable_source_row_count",
        sum(row["source_sample_count"] for row in rows) == expected["usable_source_rows"],
        observed=sum(row["source_sample_count"] for row in rows),
        expected=expected["usable_source_rows"],
    )
    recorder.require(
        "material_sample_count",
        len(rows) == expected["material_samples"],
        observed=len(rows),
        expected=expected["material_samples"],
    )
    for field in STRICT_CONSENSUS_FIELDS:
        recorder.require(
            f"material_{field}_consistent",
            inconsistent_counts[field] == 0,
            observed=inconsistent_counts[field],
            expected=0,
        )
    recorder.warn(
        "material_date_utc_consistent",
        inconsistent_counts["date_utc"] == 0,
        observed=inconsistent_counts["date_utc"],
        expected=0,
        details=(
            "Date conflicts are retained in date_utc_values and date_utc is left blank. "
            "Date is not used in the temperature model."
        ),
    )
    eligible = sum(bool(row["temperature_model_eligible"]) for row in rows)
    recorder.require(
        "temperature_eligible_material_count",
        eligible == expected["temperature_samples"],
        observed=eligible,
        expected=expected["temperature_samples"],
    )
    return rows


def build_sample_alignment(
    sources: LoadedSources,
) -> list[dict[str, Any]]:
    metadata = {row["sample_id"].strip(): row for row in sources.metadata_rows}
    mapseq_indices = {
        sample_id: index for index, sample_id in enumerate(sources.mapseq.sample_ids)
    }
    rows: list[dict[str, Any]] = []
    for original_index, sample_id in enumerate(sources.original.sample_ids):
        row = metadata.get(sample_id)
        if row is None:
            material = ""
            raw_replicate = ""
            canonical = None
            biological = ""
            included = False
            reason = "not_in_biological_metadata"
        else:
            material = row.get("material_sample", "").strip()
            raw_replicate = row.get("replicate", "").strip()
            canonical = canonical_replicate(raw_replicate)
            biological = row.get("is_biological_sample", "").strip()
            if biological.lower() != "yes":
                included = False
                reason = "not_biological"
            elif not material:
                included = False
                reason = "missing_material_sample"
            elif not raw_replicate:
                included = False
                reason = "missing_replicate"
            elif canonical is None:
                included = False
                reason = "unrecognized_replicate"
            else:
                included = True
                reason = ""
        rows.append(
            {
                "sample_id": sample_id,
                "original_present": True,
                "mapseq_present": sample_id in mapseq_indices,
                "metadata_present": row is not None,
                "original_row_index": original_index,
                "mapseq_row_index": mapseq_indices.get(sample_id, ""),
                "is_biological_sample": biological,
                "material_sample": material,
                "raw_replicate": raw_replicate,
                "canonical_replicate": canonical or "",
                "included_for_material_aggregation": included,
                "exclusion_reason": reason,
            }
        )
    return rows


def _material_embedding_rows(
    aggregation: AggregationResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for material in sorted(aggregation.vectors):
        row: dict[str, Any] = {"material_sample": material}
        row.update(aggregation.stats[material])
        row.update(
            dict(zip(EMBEDDING_COLUMNS, aggregation.vectors[material], strict=True))
        )
        rows.append(row)
    return rows


def _normalize_split_rows(split_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return sorted(
        (dict(row) for row in split_rows),
        key=lambda row: int(row["processed_matrix_row"]),
    )


def build_training_tables(
    split_rows: Sequence[Mapping[str, str]],
    material_metadata: Sequence[Mapping[str, Any]],
    original: AggregationResult,
    mapseq: AggregationResult,
    recorder: ValidationRecorder,
    expected: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_by_id = {str(row["material_sample"]): row for row in material_metadata}
    eligible_ids = {
        material
        for material, row in metadata_by_id.items()
        if bool(row["temperature_model_eligible"])
    }
    split_by_id = {row["materialSampleID"].strip(): row for row in split_rows}
    recorder.require(
        "temperature_materials_match_frozen_split",
        eligible_ids == set(split_by_id),
        observed=f"eligible={len(eligible_ids)}; split={len(split_by_id)}",
        expected="identical sets",
    )
    recorder.require(
        "original_mapseq_material_sets_match",
        set(original.vectors) == set(mapseq.vectors) == set(metadata_by_id),
        observed=(
            f"original={len(original.vectors)}; mapseq={len(mapseq.vectors)}; "
            f"metadata={len(metadata_by_id)}"
        ),
        expected="identical sets",
    )

    original_rows: list[dict[str, Any]] = []
    mapseq_rows: list[dict[str, Any]] = []
    target_mismatches = 0
    cruise_mismatches = 0
    for training_row, split in enumerate(_normalize_split_rows(split_rows)):
        material = split["materialSampleID"].strip()
        metadata = metadata_by_id[material]
        if not np.isclose(float(metadata["temperature"]), float(split["sfer_temp"]), rtol=0, atol=1e-12):
            target_mismatches += 1
        if str(metadata["cruise"]) != split["sfer_cruise_id"].strip():
            cruise_mismatches += 1
        base: dict[str, Any] = {
            "material_sample": material,
            "cruise": str(metadata["cruise"]),
            "temperature": float(metadata["temperature"]),
            "outer_partition": split["outer_partition"].strip(),
            "development_cv_fold": split["development_cv_fold"].strip(),
            "training_row": training_row,
            "source_asv_processed_matrix_row": int(split["processed_matrix_row"]),
        }
        original_row = dict(base)
        mapseq_row = dict(base)
        original_row.update(
            dict(zip(EMBEDDING_COLUMNS, original.vectors[material], strict=True))
        )
        mapseq_row.update(
            dict(zip(EMBEDDING_COLUMNS, mapseq.vectors[material], strict=True))
        )
        original_rows.append(original_row)
        mapseq_rows.append(mapseq_row)

    recorder.require(
        "frozen_split_temperature_matches_metadata",
        target_mismatches == 0,
        observed=target_mismatches,
        expected=0,
    )
    recorder.require(
        "frozen_split_cruise_matches_metadata",
        cruise_mismatches == 0,
        observed=cruise_mismatches,
        expected=0,
    )
    recorder.require(
        "paired_training_table_row_count",
        len(original_rows) == len(mapseq_rows) == expected["temperature_samples"],
        observed=f"original={len(original_rows)}; mapseq={len(mapseq_rows)}",
        expected=expected["temperature_samples"],
    )
    recorder.require(
        "paired_training_table_nonfeature_columns_match",
        all(
            all(left[key] == right[key] for key in left if not key.startswith("embedding_"))
            for left, right in zip(original_rows, mapseq_rows, strict=True)
        ),
        expected="identical",
    )
    original_matrix = np.vstack(
        [[row[column] for column in EMBEDDING_COLUMNS] for row in original_rows]
    )
    mapseq_matrix = np.vstack(
        [[row[column] for column in EMBEDDING_COLUMNS] for row in mapseq_rows]
    )
    recorder.require(
        "aggregated_training_embeddings_finite",
        bool(np.isfinite(original_matrix).all() and np.isfinite(mapseq_matrix).all()),
        expected="all finite",
    )
    recorder.require(
        "aggregated_representations_are_not_identical",
        not np.array_equal(original_matrix, mapseq_matrix),
        observed="different" if not np.array_equal(original_matrix, mapseq_matrix) else "identical",
        expected="different",
    )
    return original_rows, mapseq_rows


def _validate_replicate_weights(
    replicate_rows: Sequence[Mapping[str, Any]],
    recorder: ValidationRecorder,
) -> None:
    sums: dict[str, float] = defaultdict(float)
    for row in replicate_rows:
        sums[str(row["material_sample"])] += float(row["final_source_row_weight"])
    bad = [material for material, value in sums.items() if not np.isclose(value, 1.0)]
    recorder.require(
        "two_stage_replicate_weights_sum_to_one",
        not bad,
        observed=len(bad),
        expected=0,
        details="Technical reruns share one canonical-replicate weight; A/B/C are equal-weighted.",
    )


def _validation_report(
    summary: Mapping[str, Any],
    recorder: ValidationRecorder,
) -> str:
    counts = summary["counts"]
    warnings = recorder.warnings
    lines = [
        "# MGM2 embedding preparation validation",
        "",
        f"**Overall status: {summary['status']}**",
        "",
        "## Prepared cohort",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| HDF5 rows per representation | {counts['embedding_rows']:,} |",
        f"| Metadata-matched rows | {counts['metadata_rows']:,} |",
        f"| Excluded HDF5-only rows | {counts['hdf5_only_rows']:,} |",
        f"| Usable source rows | {counts['usable_source_rows']:,} |",
        f"| Averaged material samples | {counts['material_samples']:,} |",
        f"| Temperature cohort | {counts['temperature_samples']:,} |",
        f"| Development samples | {counts['development_samples']:,} |",
        f"| Held-out test samples | {counts['test_samples']:,} |",
        f"| Embedding features retained | {counts['embedding_features']:,} |",
        "",
        "## Validation checks",
        "",
        f"- Passed: {summary['passed_checks']}",
        f"- Warnings: {summary['warning_checks']}",
        f"- Failed: {summary['failed_checks']}",
        "",
        "## Warnings requiring documentation",
        "",
    ]
    if warnings:
        for warning in warnings:
            lines.append(
                f"- **{warning['check']}**: observed {warning['observed']}; "
                f"{warning['details']}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The original and MAPseq temperature tables contain the same materials in the "
            "same frozen order, with identical targets and cruise assignments. Their only "
            "model-feature difference is the 224-dimensional MGM2 representation. This "
            "makes the later prediction comparison paired and fair.",
            "",
            "No model was trained during this stage.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_manifest_entry(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare(config: PreparationConfig) -> dict[str, Any]:
    sources, recorder = load_and_validate_sources(config)
    alignment_rows = build_sample_alignment(sources)

    original_aggregation = aggregate_embeddings(
        sources.metadata_rows, sources.original
    )
    mapseq_aggregation = aggregate_embeddings(sources.metadata_rows, sources.mapseq)
    recorder.require(
        "replicate_mapping_identical_between_representations",
        original_aggregation.replicate_rows == mapseq_aggregation.replicate_rows,
        expected="identical",
    )
    _validate_replicate_weights(original_aggregation.replicate_rows, recorder)
    material_metadata = build_material_metadata(
        sources.metadata_rows, original_aggregation, recorder, config.expected
    )
    original_training, mapseq_training = build_training_tables(
        sources.split_rows,
        material_metadata,
        original_aggregation,
        mapseq_aggregation,
        recorder,
        config.expected,
    )
    recorder.raise_for_failures()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_fields = [
        "sample_id",
        "original_present",
        "mapseq_present",
        "metadata_present",
        "original_row_index",
        "mapseq_row_index",
        "is_biological_sample",
        "material_sample",
        "raw_replicate",
        "canonical_replicate",
        "included_for_material_aggregation",
        "exclusion_reason",
    ]
    replicate_fields = [
        "sample_id",
        "material_sample",
        "raw_replicate",
        "canonical_replicate",
        "technical_run_count",
        "biological_replicate_count",
        "technical_run_weight_within_replicate",
        "canonical_replicate_weight_within_material",
        "final_source_row_weight",
    ]
    material_metadata_fields = [
        "material_sample",
        "temperature",
        "salinity",
        "station",
        "cruise",
        "date_utc",
        "date_utc_values",
        "depth_m",
        "depth_category",
        "lat",
        "lon",
        "is_biological_sample",
        "source_sample_count",
        "biological_replicate_count",
        "canonical_replicates",
        "technical_rerun_group_count",
        "all_metadata_consistent",
        "inconsistent_metadata_fields",
        "temperature_model_eligible",
    ]
    material_embedding_fields = [
        "material_sample",
        "source_sample_count",
        "biological_replicate_count",
        "canonical_replicates",
        "technical_rerun_group_count",
        *EMBEDDING_COLUMNS,
    ]
    training_fields = [
        "material_sample",
        "cruise",
        "temperature",
        "outer_partition",
        "development_cv_fold",
        "training_row",
        "source_asv_processed_matrix_row",
        *EMBEDDING_COLUMNS,
    ]
    duplicate_fields = [
        "representation",
        "duplicate_group",
        "row_index",
        "sample_id",
        "group_size",
    ]
    check_fields = [
        "check",
        "status",
        "severity",
        "observed",
        "expected",
        "details",
    ]

    output_paths = {
        "sample_alignment_audit": output_dir / "sample_alignment_audit.csv",
        "replicate_mapping": output_dir / "replicate_mapping.csv",
        "material_metadata": output_dir / "material_metadata.csv",
        "original_material_embeddings": output_dir / "original_material_embeddings.csv",
        "mapseq_material_embeddings": output_dir / "mapseq_material_embeddings.csv",
        "original_temperature_training_table": output_dir / "original_temperature_training_table.csv",
        "mapseq_temperature_training_table": output_dir / "mapseq_temperature_training_table.csv",
        "split_assignments": output_dir / "split_assignments.csv",
        "feature_columns": output_dir / "feature_columns.txt",
        "training_schema": output_dir / "training_schema.json",
        "exact_duplicate_embedding_rows": output_dir / "exact_duplicate_embedding_rows.csv",
        "validation_checks": output_dir / "validation_checks.csv",
        "validation_summary": output_dir / "validation_summary.json",
        "validation_report": output_dir / "validation_report.md",
        "input_manifest": output_dir / "input_manifest.json",
    }

    write_csv_rows(output_paths["sample_alignment_audit"], alignment_fields, alignment_rows)
    write_csv_rows(
        output_paths["replicate_mapping"],
        replicate_fields,
        original_aggregation.replicate_rows,
    )
    write_csv_rows(
        output_paths["material_metadata"], material_metadata_fields, material_metadata
    )
    write_csv_rows(
        output_paths["original_material_embeddings"],
        material_embedding_fields,
        _material_embedding_rows(original_aggregation),
    )
    write_csv_rows(
        output_paths["mapseq_material_embeddings"],
        material_embedding_fields,
        _material_embedding_rows(mapseq_aggregation),
    )
    write_csv_rows(
        output_paths["original_temperature_training_table"],
        training_fields,
        original_training,
    )
    write_csv_rows(
        output_paths["mapseq_temperature_training_table"],
        training_fields,
        mapseq_training,
    )
    write_csv_rows(
        output_paths["split_assignments"],
        REQUIRED_SPLIT_COLUMNS,
        _normalize_split_rows(sources.split_rows),
    )
    write_text(
        output_paths["feature_columns"],
        "\n".join(EMBEDDING_COLUMNS) + "\n",
    )
    write_json(
        output_paths["training_schema"],
        {
            "schema_version": 1,
            "id_column": "material_sample",
            "target_column": "temperature",
            "group_column": "cruise",
            "outer_partition_column": "outer_partition",
            "cv_fold_column": "development_cv_fold",
            "training_row_column": "training_row",
            "audit_only_columns": ["source_asv_processed_matrix_row"],
            "feature_columns": EMBEDDING_COLUMNS,
            "feature_count": len(EMBEDDING_COLUMNS),
            "rule": "Only feature_columns may be passed to a predictive model.",
        },
    )
    write_csv_rows(
        output_paths["exact_duplicate_embedding_rows"],
        duplicate_fields,
        sources.duplicate_rows,
    )
    write_csv_rows(output_paths["validation_checks"], check_fields, recorder.checks)

    summary: dict[str, Any] = {
        **checks_summary(recorder),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "embedding_rows": len(sources.original.sample_ids),
            "embedding_features": int(sources.original.embeddings.shape[1]),
            "metadata_rows": len(sources.metadata_rows),
            "hdf5_only_rows": sum(
                row["exclusion_reason"] == "not_in_biological_metadata"
                for row in alignment_rows
            ),
            "usable_source_rows": len(original_aggregation.replicate_rows),
            "material_samples": len(original_aggregation.vectors),
            "temperature_samples": len(original_training),
            "development_samples": sum(
                row["outer_partition"] == "development" for row in original_training
            ),
            "test_samples": sum(
                row["outer_partition"] == "test" for row in original_training
            ),
            "exact_duplicate_groups": len(
                {
                    (row["representation"], row["duplicate_group"])
                    for row in sources.duplicate_rows
                }
            ),
            "date_conflict_materials": sum(
                "date_utc" in str(row["inconsistent_metadata_fields"]).split("|")
                for row in material_metadata
            ),
        },
        "warnings": recorder.warnings,
        "statement": (
            "Preparation and validation completed; no predictive model was trained."
        ),
    }
    write_json(output_paths["validation_summary"], summary)
    write_text(output_paths["validation_report"], _validation_report(summary, recorder))

    sources_manifest = [
        _source_manifest_entry("original_embeddings", config.original_embeddings),
        _source_manifest_entry("mapseq_embeddings", config.mapseq_embeddings),
        _source_manifest_entry("original_metadata", config.original_metadata),
        _source_manifest_entry("mapseq_metadata", config.mapseq_metadata),
        _source_manifest_entry("frozen_splits", config.frozen_splits),
        _source_manifest_entry("processed_material_ids", config.processed_material_ids),
    ]
    sources_manifest[0].update(
        {
            "shape": list(sources.original.embeddings.shape),
            "dtype": sources.original.embedding_dtype,
            "attributes": sources.original.attributes,
        }
    )
    sources_manifest[1].update(
        {
            "shape": list(sources.mapseq.embeddings.shape),
            "dtype": sources.mapseq.embedding_dtype,
            "attributes": sources.mapseq.attributes,
        }
    )
    output_manifest = []
    for role, path in output_paths.items():
        if role == "input_manifest":
            continue
        output_manifest.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": summary["generated_at_utc"],
        "pipeline_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "config_path": str(config.config_path),
        "expected_counts": config.expected,
        "expected_source_sha256": config.expected_sha256,
        "sources": sources_manifest,
        "pipeline_files": [
            _source_manifest_entry(
                str(path.relative_to(config.project_root)), path
            )
            for path in sorted(
                [
                    config.config_path,
                    config.project_root / "pyproject.toml",
                    config.project_root / "requirements.txt",
                    *list((config.project_root / "scripts").glob("*.sh")),
                    *list((config.project_root / "src").rglob("*.py")),
                    *list((config.project_root / "tests").glob("*.py")),
                ]
            )
        ],
        "outputs": output_manifest,
    }
    write_json(output_paths["input_manifest"], manifest)
    return {
        "summary": summary,
        "output_dir": str(output_dir),
        "output_paths": {key: str(value) for key, value in output_paths.items()},
    }


def with_output_override(config: PreparationConfig, output_dir: Path | None) -> PreparationConfig:
    if output_dir is None:
        return config
    resolved = output_dir.expanduser()
    if not resolved.is_absolute():
        resolved = config.project_root / resolved
    return replace(config, output_dir=resolved.resolve())
