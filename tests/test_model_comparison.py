from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from mgm2_embedding_model.model_comparison import compare_predictions
from mgm2_embedding_model.model_config import load_model_config
from mgm2_embedding_model.model_data import ModelingDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def comparison_dataset() -> ModelingDataset:
    target = np.arange(18, dtype=np.float64) + 10.0
    return ModelingDataset(
        representation="synthetic",
        sample_ids=np.asarray([f"sample_{index}" for index in range(18)], dtype=object),
        cruise_ids=np.asarray(
            ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d5", "d5",
             "t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"],
            dtype=object,
        ),
        target=target,
        outer_partition=np.asarray(["development"] * 10 + ["test"] * 8, dtype=object),
        cv_fold=np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5] + [-1] * 8),
        training_row=np.arange(18, dtype=np.int64),
        source_asv_processed_matrix_row=np.arange(18, dtype=np.int64),
        feature_names=("embedding_001",),
        features=target[:, None].astype(np.float32),
    )


class ComparisonTests(unittest.TestCase):
    def test_paired_cruise_bootstrap_preserves_known_advantage(self) -> None:
        dataset = comparison_dataset()
        observed = dataset.target[dataset.test]
        original = observed + 1.0
        mapseq = observed + 0.5
        asv = observed + 0.8
        config = load_model_config(PROJECT_ROOT / "configs" / "modeling.json")
        config = replace(
            config,
            settings=replace(config.settings, bootstrap_replicates=200, bootstrap_seed=11),
        )

        result = compare_predictions(dataset, original, mapseq, asv, config)

        self.assertAlmostEqual(result.direct_metrics["original"]["mae_c"], 1.0)
        self.assertAlmostEqual(result.direct_metrics["mapseq"]["mae_c"], 0.5)
        primary_mae = next(
            row
            for row in result.paired_differences
            if row["contrast"] == "mapseq_minus_original" and row["metric"] == "mae_c"
        )
        self.assertAlmostEqual(primary_mae["estimate"], -0.5)
        self.assertAlmostEqual(primary_mae["bootstrap_fraction_favorable"], 1.0)
        self.assertEqual(len(result.bootstrap_wide["bootstrap_replicate"]), 200)


if __name__ == "__main__":
    unittest.main()
