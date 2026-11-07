from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mgm2_embedding_model.io import write_csv_rows
from mgm2_embedding_model.model_metrics import regression_metrics


class ModelMetricTests(unittest.TestCase):
    def test_regression_metrics_have_expected_definitions(self) -> None:
        observed = np.asarray([1.0, 2.0, 3.0])
        predicted = np.asarray([1.0, 3.0, 2.0])

        metrics = regression_metrics(observed, predicted)

        self.assertAlmostEqual(metrics["mae_c"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse_c"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(metrics["r2"], 0.0)
        self.assertAlmostEqual(metrics["bias_c"], 0.0)

    def test_undefined_r2_is_written_as_blank_not_infinity(self) -> None:
        metrics = regression_metrics(np.asarray([2.0, 2.0]), np.asarray([1.0, 3.0]))
        self.assertTrue(np.isnan(metrics["r2"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            write_csv_rows(path, ["r2"], [{"r2": metrics["r2"]}])
            self.assertEqual(path.read_text(encoding="utf-8"), "r2\n\"\"\n")


if __name__ == "__main__":
    unittest.main()
