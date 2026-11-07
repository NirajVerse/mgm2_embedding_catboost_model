"""Environment capture and Markdown reporting for the comparison run."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import platform
from typing import Any, Sequence

def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "h5py", "catboost", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def comparison_report(
    run_id: str,
    original_summary: dict[str, Any],
    mapseq_summary: dict[str, Any],
    metric_rows: Sequence[dict[str, Any]],
    difference_rows: Sequence[dict[str, Any]],
    samples: int,
    development_samples: int,
    test_samples: int,
    cruises: int,
) -> str:
    metric_lookup = {(row["model"], row["metric"]): row for row in metric_rows}
    primary = {
        row["metric"]: row
        for row in difference_rows
        if row["contrast"] == "mapseq_minus_original"
    }
    mae = primary["mae_c"]
    if float(mae["ci_upper_95"]) < 0:
        conclusion = "MAPseq had clearly lower held-out MAE than the original-taxonomy embeddings under this protocol."
    elif float(mae["ci_lower_95"]) > 0:
        conclusion = "MAPseq had clearly higher held-out MAE than the original-taxonomy embeddings under this protocol."
    else:
        conclusion = "The cruise-block interval crossed zero, so there was no clear held-out MAE advantage for either embedding representation."
    lines = [
        f"# MGM2 embedding CatBoost comparison: {run_id}",
        "",
        "## Outcome",
        "",
        conclusion,
        "",
        "## Locked held-out test performance",
        "",
        "| Model | MAE °C | RMSE °C | R² | Macro-cruise MAE °C |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "original": "Original MGM2",
        "mapseq": "MAPseq MGM2",
        "filtered_asv": "5%-filtered ASV benchmark",
        "mean_baseline": "Development-mean baseline",
    }
    for model in ("original", "mapseq", "filtered_asv", "mean_baseline"):
        lines.append(
            f"| {labels[model]} | {metric_lookup[(model, 'mae_c')]['estimate']:.3f} | "
            f"{metric_lookup[(model, 'rmse_c')]['estimate']:.3f} | "
            f"{metric_lookup[(model, 'r2')]['estimate']:.3f} | "
            f"{metric_lookup[(model, 'macro_cruise_mae_c')]['estimate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Development-only model selection",
            "",
            "| Representation | Selected candidate | Final trees | Pooled OOF MAE °C | Pooled OOF RMSE °C | Pooled OOF R² |",
            "|---|---|---:|---:|---:|---:|",
            f"| Original MGM2 | {original_summary['selected_candidate']} | {original_summary['final_tree_count']} | "
            f"{original_summary['oof_metrics']['mae_c']:.3f} | {original_summary['oof_metrics']['rmse_c']:.3f} | "
            f"{original_summary['oof_metrics']['r2']:.3f} |",
            f"| MAPseq MGM2 | {mapseq_summary['selected_candidate']} | {mapseq_summary['final_tree_count']} | "
            f"{mapseq_summary['oof_metrics']['mae_c']:.3f} | {mapseq_summary['oof_metrics']['rmse_c']:.3f} | "
            f"{mapseq_summary['oof_metrics']['r2']:.3f} |",
            "",
            "## Primary paired contrast",
            "",
            "MAPseq minus original; negative MAE/RMSE values favor MAPseq.",
            "",
            "| Metric | Difference | 95% cruise-block interval | Bootstrap fraction favoring MAPseq |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("mae_c", "rmse_c", "r2", "macro_cruise_mae_c"):
        row = primary[metric]
        lines.append(
            f"| {metric} | {row['estimate']:.4f} | [{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] | "
            f"{row['bootstrap_fraction_favorable']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Design and limitations",
            "",
            f"- The same {samples} materials, including {development_samples} development and {test_samples} held-out samples, were used for both embeddings.",
            f"- The held-out set contains {cruises} cruises; uncertainty therefore uses cruise-block resampling.",
            "- All tuning used development-only out-of-fold predictions. The test set was used only after each model configuration and tree count were fixed.",
            "- The ASV benchmark test result was already known before this embedding experiment. This is a locked benchmark comparison, not a fully prospective untouched test.",
            "- MGM2 dimensions are latent features. Feature importance or SHAP values rank dimensions, not bacterial taxa, and should not be interpreted as causal organisms.",
            "- The results do not establish causality or universal generalization to new regions or monitoring programs.",
            "",
        ]
    )
    return "\n".join(lines)
