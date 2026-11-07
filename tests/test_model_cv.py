from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch
import unittest

import numpy as np

from mgm2_embedding_model.model_catboost import CandidateFit
from mgm2_embedding_model.model_config import ModelSettings
from mgm2_embedding_model.model_cv import run_frozen_cross_validation
from mgm2_embedding_model.model_data import ModelingDataset
from mgm2_embedding_model.model_metrics import regression_metrics


def synthetic_dataset() -> ModelingDataset:
    target = np.arange(12, dtype=np.float64) + 10.0
    folds = np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, -1, -1])
    return ModelingDataset(
        representation="synthetic",
        sample_ids=np.asarray([f"sample_{index}" for index in range(12)], dtype=object),
        cruise_ids=np.asarray(
            ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d5", "d5", "t1", "t1"],
            dtype=object,
        ),
        target=target,
        outer_partition=np.asarray(["development"] * 10 + ["test"] * 2, dtype=object),
        cv_fold=folds,
        training_row=np.arange(12, dtype=np.int64),
        source_asv_processed_matrix_row=np.arange(12, dtype=np.int64),
        feature_names=("embedding_001", "embedding_002"),
        features=np.column_stack([target, target / 2]).astype(np.float32),
    )


class FrozenCrossValidationTests(unittest.TestCase):
    def test_lowest_pooled_oof_mae_wins_and_tree_count_is_median(self) -> None:
        calls: defaultdict[str, int] = defaultdict(int)
        good_trees = [3, 5, 7, 9, 11]

        def fake_fit(
            training_features,
            training_target,
            validation_features,
            validation_target,
            feature_names,
            candidate,
            settings,
        ) -> CandidateFit:
            name = candidate["name"]
            call = calls[name]
            calls[name] += 1
            error = 0.1 if name == "good" else 0.4
            training_prediction = np.asarray(training_target) + error
            validation_prediction = np.asarray(validation_target) + error
            return CandidateFit(
                candidate_name=name,
                parameters={key: value for key, value in candidate.items() if key != "name"},
                tree_count=good_trees[call] if name == "good" else 20 + call,
                training_metrics=regression_metrics(training_target, training_prediction),
                validation_metrics=regression_metrics(validation_target, validation_prediction),
                validation_predictions=validation_prediction,
                evaluation_history={"validation": {"MAE": [error]}},
            )

        candidates = (
            {"name": "good", "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
            {"name": "worse", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
        )
        settings = ModelSettings(100, 10, 42, -1, "CPU", None, 0, 100, 7, False, 20)
        with patch("mgm2_embedding_model.model_cv.fit_candidate", side_effect=fake_fit):
            result = run_frozen_cross_validation(synthetic_dataset(), candidates, settings)

        self.assertEqual(result.selection.candidate_name, "good")
        self.assertEqual(result.selection.fold_tree_counts, tuple(good_trees))
        self.assertEqual(result.selection.best_tree_count, 7)
        self.assertAlmostEqual(result.selection.pooled_oof_metrics["mae_c"], 0.1)
        self.assertEqual(len(result.fold_results), 10)
        self.assertEqual(len(result.oof_predictions), 10)


if __name__ == "__main__":
    unittest.main()
