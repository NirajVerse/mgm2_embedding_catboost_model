"""I/O helpers for HDF5 embeddings and auditable flat tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class EmbeddingData:
    path: Path
    sample_ids: tuple[str, ...]
    embeddings: np.ndarray
    attributes: dict[str, Any]
    root_keys: tuple[str, ...]
    embedding_dtype: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_embedding_h5(path: str | Path) -> EmbeddingData:
    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        root_keys = tuple(sorted(handle.keys()))
        if "sample_ids" not in handle or "embeddings" not in handle:
            missing = sorted({"sample_ids", "embeddings"} - set(root_keys))
            raise ValueError(f"{path} is missing required datasets: {missing}")
        raw_ids = handle["sample_ids"][:]
        sample_ids = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_ids
        )
        embeddings = np.asarray(handle["embeddings"][:])
        attributes = {key: _json_safe(handle.attrs[key]) for key in handle.attrs}
        embedding_dtype = str(handle["embeddings"].dtype)
    return EmbeddingData(
        path=path,
        sample_ids=sample_ids,
        embeddings=embeddings,
        attributes=attributes,
        root_keys=root_keys,
        embedding_dtype=embedding_dtype,
    )


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def read_nonempty_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text_path(destination: Path) -> tuple[tempfile._TemporaryFileWrapper, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    return temporary, Path(temporary.name)


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    handle, temporary_path = _atomic_text_path(destination)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _format_csv_value(row.get(key, "")) for key in fieldnames})
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            # Undefined metrics such as R² for a constant-target group are
            # represented as an empty cell; infinite values remain errors.
            return ""
        if np.isinf(value):
            raise ValueError(f"Cannot write non-finite CSV value: {value}")
        return format(float(value), ".10g")
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    handle, temporary_path = _atomic_text_path(destination)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_safe)
            handle.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    handle, temporary_path = _atomic_text_path(destination)
    try:
        with handle:
            handle.write(text)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
