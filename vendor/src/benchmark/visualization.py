"""
Publication-ready visualizations for EquiCEval benchmark evaluation (Phase 9).
All plots are rendered with the matplotlib Agg backend and saved to disk.
"""

import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_dir(out_path) -> Path:
    out_path = Path(os.path.abspath(str(out_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_scatter_delta_vs_gap(x, y, out_path, title: str = ""):
    """Scatter of delta (metric delta) vs. objective gap with raw points."""
    plt = _mpl()
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.7, edgecolors="k")
    ax.set_xlabel("Objective Gap")
    ax.set_ylabel("Delta")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fpr_comparison(labels: List[str], fprs: List[float], out_path):
    """Bar chart comparing false positive rates across models/methods."""
    plt = _mpl()
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(labels))
    ax.bar(positions, fprs, color="crimson", alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_detection_cost_frontier(costs: List[float], recalls: List[float], out_path):
    """Detection-cost frontier: recall vs. solver cost curve."""
    plt = _mpl()
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(costs)
    ax.step([costs[i] for i in order], [recalls[i] for i in order], where="post", color="steelblue")
    ax.set_xlabel("Solver Calls (Cost)")
    ax.set_ylabel("Detection Recall")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_recall_by_error_family(families: List[str], recalls: List[float], out_path):
    """Bar chart of detection recall grouped by error family."""
    plt = _mpl()
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(families))
    ax.bar(positions, recalls, color="teal", alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("Detection Recall")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ranking_stability(ranks_by_model: Dict[str, Dict[str, float]], out_path):
    """Ranking stability plot with 95% CI error bars per model."""
    plt = _mpl()
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(ranks_by_model.keys())
    means = [ranks_by_model[m]["mean_rank"] for m in models]
    lows = [ranks_by_model[m]["ci_low"] for m in models]
    highs = [ranks_by_model[m]["ci_high"] for m in models]
    positions = np.arange(len(models))
    yerr = [
        [means[i] - lows[i] for i in range(len(models))],
        [highs[i] - means[i] for i in range(len(models))],
    ]
    ax.errorbar(positions, means, yerr=yerr, fmt="o", capsize=5, color="darkorange")
    ax.set_xticks(positions)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rank")
    ax.set_title("Ranking Stability (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
