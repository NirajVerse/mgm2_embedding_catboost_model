from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from mgm2_embedding_model.model_config import load_model_config
from mgm2_embedding_model.model_data import load_paired_modeling_data
from mgm2_embedding_model.model_pipeline import modeling_preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealPreparedModelDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_model_config(PROJECT_ROOT / "configs" / "modeling.json")
        cls.data = load_paired_modeling_data(cls.config)

    def test_paired_data_have_frozen_shape_and_order(self) -> None:
        original = self.data.original
        mapseq = self.data.mapseq

        self.assertEqual(original.features.shape, (1191, 224))
        self.assertEqual(mapseq.features.shape, (1191, 224))
        self.assertEqual(len(original.development), 946)
        self.assertEqual(len(original.test), 245)
        np.testing.assert_array_equal(original.sample_ids, mapseq.sample_ids)
        np.testing.assert_array_equal(original.target, mapseq.target)
        self.assertFalse(np.array_equal(original.features, mapseq.features))

    def test_cruise_groups_are_disjoint(self) -> None:
        data = self.data.original
        development_cruises = set(data.cruise_ids[data.development])
        test_cruises = set(data.cruise_ids[data.test])

        self.assertEqual(len(development_cruises), 27)
        self.assertEqual(len(test_cruises), 8)
        self.assertFalse(development_cruises & test_cruises)

    def test_preflight_reproduces_frozen_asv_benchmark(self) -> None:
        summary = modeling_preflight(self.config)

        self.assertEqual(summary["status"], "PASS")
        self.assertFalse(summary["real_training_started"])
        self.assertAlmostEqual(
            summary["asv_reference_test_metrics_reproduced"]["mae_c"],
            1.6862365697866848,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
