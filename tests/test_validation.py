from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mgm2_embedding_model.io import EmbeddingData
from mgm2_embedding_model.validation import (
    ValidationRecorder,
    duplicate_embedding_rows,
    validate_embedding_pair,
)


def make_data(name: str, ids: list[str], matrix: np.ndarray) -> EmbeddingData:
    return EmbeddingData(
        path=Path(name),
        sample_ids=tuple(ids),
        embeddings=np.asarray(matrix, dtype=np.float32),
        attributes={
            "hidden_size": int(matrix.shape[1]),
            "embedding_strategy": "cls",
            "model_class": "MGM2LightningModule",
            "model_checkpoint": "same/model.ckpt",
            "model_config": "same/config.json",
        },
        root_keys=("embeddings", "sample_ids"),
        embedding_dtype="float32",
    )


class DuplicateAuditTests(unittest.TestCase):
    def test_exact_duplicates_are_reported_with_ids(self) -> None:
        data = make_data(
            "duplicate.h5",
            ["one", "two", "three"],
            np.asarray([[1, 2], [1, 2], [4, 5]], dtype=np.float32),
        )
        rows = duplicate_embedding_rows("original", data)
        self.assertEqual({row["sample_id"] for row in rows}, {"one", "two"})
        self.assertTrue(all(row["group_size"] == 2 for row in rows))


class PairValidationTests(unittest.TestCase):
    def test_nonfinite_value_fails_validation(self) -> None:
        original = make_data("original.h5", ["a", "b"], np.asarray([[1, 2], [3, np.nan]]))
        mapseq = make_data("mapseq.h5", ["a", "b"], np.asarray([[2, 3], [4, 5]]))
        recorder = ValidationRecorder()
        validate_embedding_pair(
            original,
            mapseq,
            {"embedding_rows": 2, "embedding_features": 2},
            recorder,
        )
        failed_names = {row["check"] for row in recorder.failures}
        self.assertIn("original_embeddings_finite", failed_names)

    def test_mismatched_id_order_fails_order_check(self) -> None:
        original = make_data("original.h5", ["a", "b"], np.asarray([[1, 2], [3, 4]]))
        mapseq = make_data("mapseq.h5", ["b", "a"], np.asarray([[2, 3], [4, 5]]))
        recorder = ValidationRecorder()
        validate_embedding_pair(
            original,
            mapseq,
            {"embedding_rows": 2, "embedding_features": 2},
            recorder,
        )
        failed_names = {row["check"] for row in recorder.failures}
        self.assertIn("paired_sample_ids_same_order", failed_names)
        self.assertNotIn("paired_sample_ids_same_set", failed_names)


if __name__ == "__main__":
    unittest.main()
