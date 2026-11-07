from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from mgm2_embedding_model.config import load_config
from mgm2_embedding_model.io import sha256_file
from mgm2_embedding_model.prepare import EMBEDDING_COLUMNS, prepare, with_output_override


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "prepare.json"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


@unittest.skipUnless(CONFIG_PATH.is_file(), "real preparation config is unavailable")
class EndToEndPreparationTests(unittest.TestCase):
    def test_real_preparation_is_paired_and_deterministic(self) -> None:
        config = load_config(CONFIG_PATH)
        source_paths = [
            config.original_embeddings,
            config.mapseq_embeddings,
            config.original_metadata,
            config.mapseq_metadata,
            config.frozen_splits,
            config.processed_material_ids,
        ]
        if not all(path.is_file() for path in source_paths):
            self.skipTest("one or more external source files are unavailable")

        with tempfile.TemporaryDirectory(prefix="mgm2_embedding_test_") as temporary:
            output = Path(temporary)
            test_config = with_output_override(config, output)
            first = prepare(test_config)
            original_path = output / "original_temperature_training_table.csv"
            mapseq_path = output / "mapseq_temperature_training_table.csv"
            first_hashes = (sha256_file(original_path), sha256_file(mapseq_path))

            second = prepare(test_config)
            second_hashes = (sha256_file(original_path), sha256_file(mapseq_path))
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(first["summary"]["status"], "PASS")
            self.assertEqual(second["summary"]["status"], "PASS")

            original_fields, original_rows = read_csv(original_path)
            mapseq_fields, mapseq_rows = read_csv(mapseq_path)
            self.assertEqual(original_fields, mapseq_fields)
            self.assertEqual(len(original_rows), 1191)
            self.assertEqual(len(mapseq_rows), 1191)
            self.assertEqual(
                [field for field in original_fields if field.startswith("embedding_")],
                EMBEDDING_COLUMNS,
            )
            nonfeatures = [
                field for field in original_fields if not field.startswith("embedding_")
            ]
            self.assertTrue(
                all(
                    all(left[field] == right[field] for field in nonfeatures)
                    for left, right in zip(original_rows, mapseq_rows, strict=True)
                )
            )
            self.assertEqual(
                [int(row["training_row"]) for row in original_rows],
                list(range(1191)),
            )

            schema = json.loads((output / "training_schema.json").read_text())
            self.assertEqual(schema["feature_columns"], EMBEDDING_COLUMNS)
            self.assertEqual(schema["feature_count"], 224)
            self.assertEqual(schema["target_column"], "temperature")


if __name__ == "__main__":
    unittest.main()
