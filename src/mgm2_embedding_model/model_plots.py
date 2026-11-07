"""Static figures for the embedding-model comparison report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


COLORS = {
    "original": "#4C78A8",
    "mapseq": "#F58518",
    "filtered_asv": "#54A24B",
    "mean_baseline": "#9D9D9D",
}
LABELS = {
    "original": "Original MGM2",
    "mapseq": "MAPseq MGM2",
    "filtered_asv": "Filtered ASVs",
    "mean_baseline": "Development mean",
}


def _representation_style(representation: str) -> tuple[str, str]:
    if representation not in {"original", "mapseq"}:
        raise ValueError(f"Unknown embedding representation: {representation}")
    return LABELS[representation], COLORS[representation]


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    _plt().close(fig)


def plot_observed_vs_predicted(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    observed = np.asarray([row["observed_temperature_c"] for row in rows], dtype=float)
    models = ("original", "mapseq", "filtered_asv")
    lower = float(observed.min())
    upper = float(observed.max())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for axis, model in zip(axes, models, strict=True):
        predicted = np.asarray([row[f"{model}_prediction_c"] for row in rows], dtype=float)
        axis.scatter(observed, predicted, s=22, alpha=0.72, color=COLORS[model], edgecolors="none")
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
        axis.set_title(LABELS[model])
        axis.set_xlabel("Observed temperature (°C)")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Predicted temperature (°C)")
    fig.suptitle("Held-out temperature predictions: same 245 samples and 8 cruises")
    _save(fig, path)


def plot_metric_comparison(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    metrics = (
        ("mae_c", "MAE (°C)"),
        ("rmse_c", "RMSE (°C)"),
        ("macro_cruise_mae_c", "Macro-cruise MAE (°C)"),
    )
    models = ("original", "mapseq", "filtered_asv", "mean_baseline")
    lookup = {(row["model"], row["metric"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        estimates = np.asarray([lookup[(model, metric)]["estimate"] for model in models], dtype=float)
        lower = np.asarray([lookup[(model, metric)]["ci_lower_95"] for model in models], dtype=float)
        upper = np.asarray([lookup[(model, metric)]["ci_upper_95"] for model in models], dtype=float)
        positions = np.arange(len(models))
        axis.bar(positions, estimates, color=[COLORS[model] for model in models], alpha=0.88)
        # A percentile interval need not contain the direct point estimate, so
        # draw its endpoints directly instead of passing potentially negative
        # error-bar distances to Matplotlib.
        axis.vlines(positions, lower, upper, color="black", linewidth=1)
        cap_width = 0.08
        axis.hlines(lower, positions - cap_width, positions + cap_width, color="black")
        axis.hlines(upper, positions - cap_width, positions + cap_width, color="black")
        axis.set_xticks(positions, [LABELS[model] for model in models], rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Held-out error with 95% cruise-block bootstrap intervals")
    _save(fig, path)


def plot_per_cruise_mae(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    models = ("original", "mapseq", "filtered_asv")
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["model"] in models:
            grouped[str(row["cruise"])][str(row["model"])] = float(row["mae_c"])
    cruises = sorted(grouped)
    positions = np.arange(len(cruises))
    fig, axis = plt.subplots(figsize=(11, 5))
    for model in models:
        axis.plot(
            positions,
            [grouped[cruise][model] for cruise in cruises],
            marker="o",
            linewidth=1.7,
            label=LABELS[model],
            color=COLORS[model],
        )
    axis.set_xticks(positions, cruises, rotation=35, ha="right")
    axis.set_ylabel("MAE (°C)")
    axis.set_title("Held-out MAE for every cruise")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save(fig, path)


def plot_cruise_mapseq_difference(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["model"] in {"original", "mapseq"}:
            grouped[str(row["cruise"])][str(row["model"])] = float(row["mae_c"])
    cruises = sorted(grouped)
    values = [grouped[cruise]["mapseq"] - grouped[cruise]["original"] for cruise in cruises]
    colors = ["#2E8B57" if value < 0 else "#C44E52" for value in values]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(np.arange(len(cruises)), values, color=colors)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(cruises)), cruises, rotation=35, ha="right")
    axis.set_ylabel("MAPseq MAE − original MAE (°C)")
    axis.set_title("Cruise-level MAPseq change: negative values favor MAPseq")
    axis.grid(axis="y", alpha=0.2)
    _save(fig, path)


def plot_residuals(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    observed = np.asarray([row["observed_temperature_c"] for row in rows], dtype=float)
    models = ("original", "mapseq", "filtered_asv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for axis, model in zip(axes, models, strict=True):
        residual = np.asarray([row[f"{model}_residual_c"] for row in rows], dtype=float)
        axis.scatter(observed, residual, s=22, alpha=0.72, color=COLORS[model], edgecolors="none")
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(LABELS[model])
        axis.set_xlabel("Observed temperature (°C)")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Residual: observed − predicted (°C)")
    fig.suptitle("Held-out residuals")
    _save(fig, path)


def plot_oof_metrics(
    original_metrics: dict[str, float],
    mapseq_metrics: dict[str, float],
    path: Path,
) -> None:
    plt = _plt()
    metrics = (("mae_c", "OOF MAE (°C)"), ("rmse_c", "OOF RMSE (°C)"), ("r2", "OOF R²"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [original_metrics[metric], mapseq_metrics[metric]]
        axis.bar([0, 1], values, color=[COLORS["original"], COLORS["mapseq"]])
        axis.set_xticks([0, 1], [LABELS["original"], LABELS["mapseq"]], rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Development-only pooled out-of-fold performance")
    _save(fig, path)


def plot_candidate_tuning(
    original_rows: Sequence[dict[str, Any]],
    mapseq_rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, label, rows, color in (
        (axes[0], "Original MGM2", original_rows, COLORS["original"]),
        (axes[1], "MAPseq MGM2", mapseq_rows, COLORS["mapseq"]),
    ):
        ordered = sorted(rows, key=lambda row: int(row["candidate_order"]))
        positions = np.arange(len(ordered))
        axis.barh(positions, [row["pooled_oof_mae_c"] for row in ordered], color=color)
        axis.set_yticks(positions, [row["candidate"] for row in ordered])
        axis.invert_yaxis()
        axis.set_xlabel("Pooled OOF MAE (°C)")
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.18)
    fig.suptitle("Same six CatBoost candidates evaluated on frozen folds")
    _save(fig, path)


def plot_feature_importance(rows: Sequence[dict[str, Any]], path: Path, top_n: int = 20) -> None:
    plt = _plt()
    selected = sorted(rows, key=lambda row: float(row["prediction_values_change"]), reverse=True)[:top_n]
    selected.reverse()
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        [row["feature"] for row in selected],
        [row["prediction_values_change"] for row in selected],
        color="#4C78A8",
    )
    axis.set_xlabel("CatBoost prediction-values-change importance")
    axis.set_title("Most influential latent MGM2 dimensions")
    axis.grid(axis="x", alpha=0.18)
    _save(fig, path)


def plot_single_model_accuracy(
    rows: Sequence[dict[str, Any]],
    representation: str,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """Plot held-out observed versus predicted values for one representation."""
    plt = _plt()
    label, color = _representation_style(representation)
    observed = np.asarray([row["observed_temperature_c"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_temperature_c"] for row in rows], dtype=float)
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    padding = max((upper - lower) * 0.05, 0.25)
    limits = (lower - padding, upper + padding)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.scatter(observed, predicted, s=30, alpha=0.72, color=color, edgecolors="none")
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.2, label="Perfect prediction")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed temperature (°C)")
    axis.set_ylabel("Predicted temperature (°C)")
    axis.set_title(f"{label}: held-out observed versus predicted")
    metric_text = (
        f"n = {len(rows)}\n"
        f"MAE = {metrics['mae_c']:.3f} °C\n"
        f"RMSE = {metrics['rmse_c']:.3f} °C\n"
        f"R² = {metrics['r2']:.3f}"
    )
    axis.text(
        0.04,
        0.96,
        metric_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    _save(fig, path)


def plot_single_model_residuals(
    rows: Sequence[dict[str, Any]], representation: str, path: Path
) -> None:
    """Plot signed held-out residuals against observed temperature."""
    plt = _plt()
    label, color = _representation_style(representation)
    observed = np.asarray([row["observed_temperature_c"] for row in rows], dtype=float)
    residual = np.asarray([row["residual_c"] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(7.5, 5.6))
    axis.scatter(observed, residual, s=30, alpha=0.72, color=color, edgecolors="none")
    axis.axhline(0, color="black", linestyle="--", linewidth=1.2)
    if np.ptp(observed) > 0:
        slope, intercept = np.polyfit(observed, residual, 1)
        grid = np.linspace(observed.min(), observed.max(), 100)
        axis.plot(grid, slope * grid + intercept, color="#333333", linewidth=1.4, label="Linear trend")
    axis.set_xlabel("Observed temperature (°C)")
    axis.set_ylabel("Residual: observed − predicted (°C)")
    axis.set_title(f"{label}: held-out residual pattern")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    _save(fig, path)


def plot_absolute_error_distribution(
    rows: Sequence[dict[str, Any]], representation: str, path: Path
) -> None:
    """Plot the distribution of held-out absolute errors for one model."""
    plt = _plt()
    label, color = _representation_style(representation)
    error = np.asarray([row["absolute_error_c"] for row in rows], dtype=float)
    median = float(np.median(error))
    percentile_90 = float(np.percentile(error, 90))

    fig, axis = plt.subplots(figsize=(7.5, 5.4))
    axis.hist(error, bins=20, color=color, alpha=0.84, edgecolor="white")
    axis.axvline(median, color="black", linewidth=1.5, label=f"Median = {median:.2f} °C")
    axis.axvline(
        percentile_90,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"90th percentile = {percentile_90:.2f} °C",
    )
    axis.set_xlabel("Absolute prediction error (°C)")
    axis.set_ylabel("Number of held-out samples")
    axis.set_title(f"{label}: held-out absolute-error distribution")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False)
    _save(fig, path)


def plot_single_model_cruise_mae(
    rows: Sequence[dict[str, Any]], representation: str, path: Path
) -> None:
    """Plot held-out MAE for each unseen cruise."""
    plt = _plt()
    label, color = _representation_style(representation)
    ordered = sorted(rows, key=lambda row: float(row["mae_c"]), reverse=True)
    cruises = [str(row["cruise"]) for row in ordered]
    values = np.asarray([row["mae_c"] for row in ordered], dtype=float)
    samples = [int(row["samples"]) for row in ordered]

    fig, axis = plt.subplots(figsize=(8.2, 5.7))
    positions = np.arange(len(ordered))
    bars = axis.barh(positions, values, color=color, alpha=0.88)
    axis.set_yticks(positions, cruises)
    axis.invert_yaxis()
    for bar, sample_count in zip(bars, samples, strict=True):
        axis.text(
            bar.get_width() + max(values.max() * 0.015, 0.03),
            bar.get_y() + bar.get_height() / 2,
            f"n={sample_count}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, values.max() * 1.18)
    axis.set_xlabel("MAE (°C)")
    axis.set_ylabel("Held-out cruise")
    axis.set_title(f"{label}: error across eight unseen cruises")
    axis.grid(axis="x", alpha=0.18)
    _save(fig, path)


def plot_oof_vs_test_metrics(
    oof_metrics: dict[str, float],
    test_metrics: dict[str, float],
    representation: str,
    path: Path,
) -> None:
    """Show development OOF performance beside the final held-out result."""
    plt = _plt()
    label, color = _representation_style(representation)
    metrics = (("mae_c", "MAE (°C)"), ("rmse_c", "RMSE (°C)"), ("r2", "R²"))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.1))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [float(oof_metrics[metric]), float(test_metrics[metric])]
        axis.bar([0, 1], values, color=["#A0A0A0", color], alpha=0.9)
        axis.set_xticks([0, 1], ["Development\nOOF", "Held-out\ntest"])
        axis.set_title(title)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.18)
        for position, value in enumerate(values):
            vertical = "bottom" if value >= 0 else "top"
            axis.text(position, value, f"{value:.3f}", ha="center", va=vertical, fontsize=9)
    fig.suptitle(f"{label}: development-to-test generalization")
    _save(fig, path)


def plot_shap_importance(
    rows: Sequence[dict[str, Any]], representation: str, path: Path, top_n: int = 20
) -> None:
    """Plot mean absolute SHAP effects for latent embedding dimensions."""
    plt = _plt()
    label, color = _representation_style(representation)
    selected = sorted(rows, key=lambda row: float(row["mean_absolute_shap"]), reverse=True)[:top_n]
    selected.reverse()
    fig, axis = plt.subplots(figsize=(8.2, 6.4))
    axis.barh(
        [row["feature"] for row in selected],
        [float(row["mean_absolute_shap"]) for row in selected],
        color=color,
    )
    axis.set_xlabel("Mean absolute SHAP contribution (°C)")
    axis.set_title(f"{label}: influential latent dimensions by SHAP")
    axis.text(
        0.99,
        0.01,
        "Latent MGM2 dimensions—not individual taxa",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    axis.grid(axis="x", alpha=0.18)
    _save(fig, path)


def plot_absolute_error_ecdf(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Compare the complete held-out absolute-error distributions."""
    plt = _plt()
    models = ("original", "mapseq", "filtered_asv", "mean_baseline")
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    for model in models:
        error = np.sort(np.asarray([row[f"{model}_absolute_error_c"] for row in rows], dtype=float))
        cumulative = np.arange(1, len(error) + 1, dtype=float) / len(error)
        axis.step(error, cumulative, where="post", color=COLORS[model], linewidth=2, label=LABELS[model])
    axis.set_xlabel("Absolute prediction error (°C)")
    axis.set_ylabel("Fraction of held-out samples")
    axis.set_title("Held-out absolute-error distributions")
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    _save(fig, path)


def plot_paired_sample_error_difference(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Show paired per-sample MAPseq minus original absolute error."""
    plt = _plt()
    difference = np.asarray(
        [row["mapseq_absolute_error_c"] for row in rows], dtype=float
    ) - np.asarray([row["original_absolute_error_c"] for row in rows], dtype=float)
    fraction_better = float(np.mean(difference < 0))
    median = float(np.median(difference))
    fig, axis = plt.subplots(figsize=(7.8, 5.5))
    axis.hist(difference, bins=24, color=COLORS["mapseq"], alpha=0.84, edgecolor="white")
    axis.axvline(0, color="black", linestyle="--", linewidth=1.3)
    axis.axvline(median, color="#333333", linewidth=1.5, label=f"Median = {median:.2f} °C")
    axis.set_xlabel("MAPseq absolute error − original absolute error (°C)")
    axis.set_ylabel("Number of held-out samples")
    axis.set_title("Paired sample-level error difference")
    axis.text(
        0.03,
        0.95,
        f"MAPseq lower error for {fraction_better:.1%} of samples\nNegative values favor MAPseq",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False)
    _save(fig, path)
