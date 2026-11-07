"""Configuration loading with paths resolved relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparationConfig:
    project_root: Path
    config_path: Path
    original_embeddings: Path
    mapseq_embeddings: Path
    original_metadata: Path
    mapseq_metadata: Path
    frozen_splits: Path
    processed_material_ids: Path
    output_dir: Path
    expected_sha256: dict[str, str]
    expected: dict[str, int]


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_config(path: str | Path) -> PreparationConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    required = {
        "original_embeddings",
        "mapseq_embeddings",
        "original_metadata",
        "mapseq_metadata",
        "frozen_splits",
        "processed_material_ids",
        "output_dir",
        "expected_sha256",
        "expected",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    # configs/prepare.json lives one directory below the project root.
    project_root = config_path.parent.parent
    expected = {str(key): int(value) for key, value in raw["expected"].items()}
    expected_sha256 = {
        str(key): str(value).lower() for key, value in raw["expected_sha256"].items()
    }
    return PreparationConfig(
        project_root=project_root,
        config_path=config_path,
        original_embeddings=_resolve(project_root, raw["original_embeddings"]),
        mapseq_embeddings=_resolve(project_root, raw["mapseq_embeddings"]),
        original_metadata=_resolve(project_root, raw["original_metadata"]),
        mapseq_metadata=_resolve(project_root, raw["mapseq_metadata"]),
        frozen_splits=_resolve(project_root, raw["frozen_splits"]),
        processed_material_ids=_resolve(project_root, raw["processed_material_ids"]),
        output_dir=_resolve(project_root, raw["output_dir"]),
        expected_sha256=expected_sha256,
        expected=expected,
    )
