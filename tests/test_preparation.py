from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mgm2_embedding_model.io import EmbeddingData
from mgm2_embedding_model.prepare import aggregate_embeddings
from mgm2_embedding_model.validation import canonical_replicate


def embedding_data(sample_ids: list[str], values: list[list[float]]) -> EmbeddingData:
    return EmbeddingData(
        path=Path("synthetic.h5"),
        sample_ids=tuple(sample_ids),
        embeddings=np.asarray(values, dtype=np.float32),
        attributes={"hidden_size": len(values[0]), "embedding_strategy": "cls"},
        root_keys=("embeddings", "sample_ids"),
        embedding_dtype="float32",
    )


class CanonicalReplicateTests(unittest.TestCase):
    def test_approved_labels_are_canonicalized(self) -> None:
        expected = {
            "A": "A",
            "Aa": "A",
            "Ab": "A",
            "B": "B",
            "C": "C",
            "Cb": "C",
        }
        for raw, canonical in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(canonical_replicate(raw), canonical)

    def test_unknown_or_blank_labels_are_rejected(self) -> None:
        for raw in ("", "D", "A1", "control"):
            with self.subTest(raw=raw):
                self.assertIsNone(canonical_replicate(raw))


class AggregationTests(unittest.TestCase):
    def test_technical_reruns_are_averaged_before_biological_replicates(self) -> None:
        data = embedding_data(
            ["sample_A", "sample_Aa", "sample_B"],
            [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]],
        )
        metadata = [
            {
                "sample_id": "sample_A",
                "material_sample": "material_1",
                "replicate": "A",
                "is_biological_sample": "yes",
            },
            {
                "sample_id": "sample_Aa",
                "material_sample": "material_1",
                "replicate": "Aa",
                "is_biological_sample": "yes",
            },
            {
                "sample_id": "sample_B",
                "material_sample": "material_1",
                "replicate": "B",
                "is_biological_sample": "yes",
            },
        ]

        result = aggregate_embeddings(metadata, data)

        # A mean = (0 + 2) / 2 = 1; material mean = (A mean + B) / 2 = 5.5.
        np.testing.assert_allclose(result.vectors["material_1"], [5.5, 5.5])
        self.assertEqual(result.stats["material_1"]["source_sample_count"], 3)
        self.assertEqual(result.stats["material_1"]["biological_replicate_count"], 2)
        self.assertEqual(result.stats["material_1"]["technical_rerun_group_count"], 1)

        weights = {
            row["sample_id"]: row["final_source_row_weight"]
            for row in result.replicate_rows
        }
        self.assertAlmostEqual(weights["sample_A"], 0.25)
        self.assertAlmostEqual(weights["sample_Aa"], 0.25)
        self.assertAlmostEqual(weights["sample_B"], 0.50)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_material_order_is_deterministic(self) -> None:
        data = embedding_data(
            ["z_A", "a_A"],
            [[2.0], [1.0]],
        )
        metadata = [
            {
                "sample_id": "z_A",
                "material_sample": "z_material",
                "replicate": "A",
                "is_biological_sample": "yes",
            },
            {
                "sample_id": "a_A",
                "material_sample": "a_material",
                "replicate": "A",
                "is_biological_sample": "yes",
            },
        ]
        result = aggregate_embeddings(metadata, data)
        self.assertEqual(list(result.vectors), ["a_material", "z_material"])


if __name__ == "__main__":
    unittest.main()
