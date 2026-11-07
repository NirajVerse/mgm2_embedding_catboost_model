"""Regenerate report-ready figures from a completed modeling run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .io import read_csv_rows, sha256_file, write_json
from .model_plots import (
    plot_absolute_error_distribution,
    plot_absolute_error_ecdf,
    plot_oof_vs_test_metrics,
    plot_paired_sample_error_difference,
    plot_shap_importance,
    plot_single_model_accuracy,
    plot_single_model_cruise_mae,
    plot_single_model_residuals,
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed-run file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed-run file: {path}")
    _, rows = read_csv_rows(path)
    return rows


def generate_additional_figures(run_dir: str | Path) -> list[Path]:
    """Create additional figures without refitting either CatBoost model."""
    root = Path(run_dir).expanduser().resolve()
    state = _json(root / "training_state.json")
    if state.get("status") != "COMPLETE":
        raise ValueError(f"Run is not complete: status={state.get('status')!r}")

    outputs: list[Path] = []
    for representation in ("original", "mapseq"):
        representation_root = root / representation
        tables = representation_root / "tables"
        figures = representation_root / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        summary = _json(representation_root / "result_summary.json")
        predictions = _rows(tables / "test_predictions.csv")
        cruise_metrics = _rows(tables / "test_cruise_metrics.csv")
        shap_summary = _rows(tables / "test_shap_summary.csv")

        targets = [
            ("test_observed_vs_predicted.png", plot_single_model_accuracy),
            ("test_residuals_vs_observed.png", plot_single_model_residuals),
            ("test_absolute_error_distribution.png", plot_absolute_error_distribution),
            ("test_cruise_mae.png", plot_single_model_cruise_mae),
            ("development_oof_vs_test.png", plot_oof_vs_test_metrics),
            ("test_shap_importance_top20.png", plot_shap_importance),
        ]
        paths = {name: figures / name for name, _ in targets}
        plot_single_model_accuracy(
            predictions, representation, summary["test_metrics"], paths["test_observed_vs_predicted.png"]
        )
        plot_single_model_residuals(
            predictions, representation, paths["test_residuals_vs_observed.png"]
        )
        plot_absolute_error_distribution(
            predictions, representation, paths["test_absolute_error_distribution.png"]
        )
        plot_single_model_cruise_mae(
            cruise_metrics, representation, paths["test_cruise_mae.png"]
        )
        plot_oof_vs_test_metrics(
            summary["oof_metrics"],
            summary["test_metrics"],
            representation,
            paths["development_oof_vs_test.png"],
        )
        plot_shap_importance(
            shap_summary, representation, paths["test_shap_importance_top20.png"]
        )
        outputs.extend(paths.values())

    comparison_tables = root / "comparison" / "tables"
    comparison_figures = root / "comparison" / "figures"
    comparison_figures.mkdir(parents=True, exist_ok=True)
    paired = _rows(comparison_tables / "paired_test_predictions.csv")
    ecdf = comparison_figures / "absolute_error_ecdf_comparison.png"
    paired_difference = comparison_figures / "paired_sample_error_difference.png"
    plot_absolute_error_ecdf(paired, ecdf)
    plot_paired_sample_error_difference(paired, paired_difference)
    outputs.extend([ecdf, paired_difference])
    write_json(
        root / "additional_figures_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(root),
            "models_retrained": False,
            "figure_count": len(outputs),
            "figures": [
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in outputs
            ],
        },
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate additional figures from a completed embedding-model run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = generate_additional_figures(args.run_dir)
    print(f"Generated {len(outputs)} figures without retraining:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
