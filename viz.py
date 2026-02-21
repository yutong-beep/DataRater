"""
viz.py — Visualization utilities for experiment results.

Generates:
  - Training curves (loss, MSE, Pearson over epochs)
  - Score distribution histograms
  - P_accept distribution
  - Performance comparison bar charts
  - Performance-per-FLOP comparison
"""

import os
import json
import logging
from typing import Dict, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "figure.dpi": 150,
})

COLORS = {
    "baseline": "#2196F3",
    "retrained": "#4CAF50",
    "datarater": "#FF9800",
    "ablation": "#9C27B0",
}


def plot_training_curves(
    history: Dict,
    title: str = "Training Curves",
    save_path: str = "results/training_curves.png",
):
    """Plot train loss, val MSE, Pearson across epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Train loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=COLORS["baseline"], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Training Loss")

    # Val MSE
    ax = axes[1]
    ax.plot(epochs, history["val_mse"], color=COLORS["retrained"], linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE")
    ax.set_title("Validation MSE")

    # Pearson
    ax = axes[2]
    ax.plot(epochs, history["val_pearson"], color=COLORS["datarater"], linewidth=2, marker="^", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("Validation Pearson Correlation")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved -> {save_path}")


def plot_score_distribution(
    scores: np.ndarray,
    save_path: str = "results/score_distribution.png",
):
    """Plot DataRater score distribution with CDF."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(scores, bins=80, color=COLORS["datarater"], alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean={np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color="green", linestyle="--", label=f"Median={np.median(scores):.3f}")
    ax.set_xlabel("DataRater Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # CDF
    ax = axes[1]
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, color=COLORS["baseline"], linewidth=2)
    ax.set_xlabel("DataRater Score")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Empirical CDF")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Score distribution saved -> {save_path}")


def plot_comparison(
    baseline_metrics: Dict,
    retrained_metrics: Dict,
    baseline_flops: float,
    retrained_flops: float,
    save_path: str = "results/comparison.png",
):
    """
    Side-by-side comparison of baseline vs retrained:
      - MSE, RMSE, Pearson, Spearman bar charts
      - Performance-per-FLOP comparison
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ---- 1. MSE comparison ----
    ax = fig.add_subplot(gs[0, 0])
    names = ["Baseline", "Retrained\n(DataRater)"]
    mse_vals = [baseline_metrics["mse"], retrained_metrics["mse"]]
    bars = ax.bar(names, mse_vals, color=[COLORS["baseline"], COLORS["retrained"]], width=0.5)
    ax.set_ylabel("MSE (lower = better)")
    ax.set_title("Validation MSE")
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    # ---- 2. Correlation comparison ----
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    w = 0.3
    pearson_vals = [baseline_metrics.get("pearson_r", 0), retrained_metrics.get("pearson_r", 0)]
    spearman_vals = [baseline_metrics.get("spearman_r", 0), retrained_metrics.get("spearman_r", 0)]
    ax.bar(x - w / 2, pearson_vals, w, label="Pearson r", color=[COLORS["baseline"], COLORS["retrained"]])
    ax.bar(x + w / 2, spearman_vals, w, label="Spearman r", color=[COLORS["baseline"], COLORS["retrained"]], alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Retrained"])
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation Metrics")
    ax.legend()

    # ---- 3. FLOPs comparison ----
    ax = fig.add_subplot(gs[1, 0])
    flops_vals = [baseline_flops, retrained_flops]
    bars = ax.bar(names, flops_vals, color=[COLORS["baseline"], COLORS["retrained"]], width=0.5)
    ax.set_ylabel("Total FLOPs")
    ax.set_title("Computational Cost")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # ---- 4. Performance-per-FLOP ----
    ax = fig.add_subplot(gs[1, 1])
    # perf_per_flop = (1 / MSE) / FLOPs — higher is better
    ppf_baseline = (1.0 / max(baseline_metrics["mse"], 1e-10)) / max(baseline_flops, 1)
    ppf_retrained = (1.0 / max(retrained_metrics["mse"], 1e-10)) / max(retrained_flops, 1)
    bars = ax.bar(names, [ppf_baseline, ppf_retrained],
                  color=[COLORS["baseline"], COLORS["retrained"]], width=0.5)
    ax.set_ylabel("(1/MSE) / FLOPs")
    ax.set_title("Performance per FLOP (higher = better)")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    for bar, val in zip(bars, [ppf_baseline, ppf_retrained]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Baseline vs DataRater-Curated Retraining", fontsize=15, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison chart saved -> {save_path}")


def plot_multi_curve_overlay(
    histories: Dict[str, Dict],
    metric: str = "val_mse",
    ylabel: str = "Val MSE",
    title: str = "Training Comparison",
    save_path: str = "results/overlay.png",
):
    """Overlay training curves from multiple runs on one plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    color_list = list(COLORS.values())

    for i, (name, history) in enumerate(histories.items()):
        if metric in history:
            epochs = list(range(1, len(history[metric]) + 1))
            color = color_list[i % len(color_list)]
            ax.plot(epochs, history[metric], linewidth=2, marker="o", markersize=3,
                    label=name, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Overlay plot saved -> {save_path}")
