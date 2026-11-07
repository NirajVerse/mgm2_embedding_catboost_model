from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mgm2_embedding_model.model_plots import plot_metric_comparison


class ModelPlotTests(unittest.TestCase):
    def test_percentile_interval_not_containing_estimate_still_plots(self) -> None:
        rows = []
        for model_index, model in enumerate(
            ("original", "mapseq", "filtered_asv", "mean_baseline")
        ):
            for metric_index, metric in enumerate(
                ("mae_c", "rmse_c", "macro_cruise_mae_c")
            ):
                estimate = 1.0 + model_index + metric_index / 10
                # Deliberately place the interval above the direct estimate.
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower_95": estimate + 0.1,
                        "ci_upper_95": estimate + 0.3,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metric_comparison.png"
            plot_metric_comparison(rows, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
